# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MCP tool-use client with bounded context compaction and structured logs."""
import asyncio
import sys
import os
import json
import time
import re
import base64
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from contextlib import AsyncExitStack
import subprocess
import shlex
import traceback
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import openai
from dotenv import load_dotenv
from client.log import save_chat_log_html
try:
    from PIL import Image
    import io
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print('⚠️  Warning: PIL (Pillow) not available. Images will not be resized for logs.')
SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'server'))
CLIENT_DIR = os.path.abspath(os.path.dirname(__file__))
PROMPT_LIBRARY_DIR = os.path.join(CLIENT_DIR, 'prompts', 'prompt_library')
load_dotenv()

def get_task_definition_text(room_desc: str, asset_mode: str='generation', scene_type: str=''):
    template = Path(os.environ.get('SAGE_GENERATION_TASK_FILE', Path(__file__).resolve().parents[1] / 'configs/generation_prompt.txt'))
    return template.read_text(encoding='utf-8').replace('{room_desc}', room_desc)

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def resize_and_encode_image(image_path: str, max_size: tuple=(512, 512)) -> tuple:
    """Resize image and encode to base64 for logging
    
    Args:
        image_path: Path to image file
        max_size: Maximum dimensions (width, height) for resized image
    
    Returns:
        tuple: (base64_string, original_size, resized_size, file_size)
    """
    if not PIL_AVAILABLE:
        base64_str = encode_image_to_base64(image_path)
        file_size = os.path.getsize(image_path)
        return (base64_str, (0, 0), (0, 0), file_size)
    try:
        with Image.open(image_path) as img:
            original_size = img.size
            file_size = os.path.getsize(image_path)
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            resized_size = img.size
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            base64_str = base64.b64encode(buffer.read()).decode('utf-8')
            return (base64_str, original_size, resized_size, file_size)
    except Exception as e:
        print(f'⚠️  Failed to resize image {image_path}: {e}')
        base64_str = encode_image_to_base64(image_path)
        file_size = os.path.getsize(image_path)
        return (base64_str, (0, 0), (0, 0), file_size)

def get_image_mime_type(image_path: str) -> str:
    """Get MIME type based on file extension"""
    ext = os.path.splitext(image_path)[1].lower()
    mime_types = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.gif': 'image/gif', '.webp': 'image/webp', '.bmp': 'image/bmp'}
    return mime_types.get(ext, 'image/jpeg')

class ContextBudgetExhausted(Exception):
    """The next request cannot fit the configured policy context budget."""

class MCPClientOAI:
    """
    MCP Client for Qwen3-VL-235B-A22B-Thinking model
    
    This client uses the OpenAI chat completions API format to interact with
    the Qwen3-VL thinking model. Key features:
    - Uses chat.completions.create() instead of responses.create()
    - Supports tool calling in OpenAI format
    - Captures and displays model reasoning (Qwen thinking model feature)
    - Maintains conversation history in messages format (user/assistant/tool roles)
    - Tracks token usage and tool call limits
    """

    def __init__(self):
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.exit_stack = AsyncExitStack()
        self.layout_id = None
        key_dict_path = Path(os.environ['SAGE_CLIENT_KEY_FILE'])
        with open(key_dict_path, 'r') as f:
            key_dict = json.load(f)
        API_TOKEN = key_dict['API_TOKEN']
        API_URL_QWEN = key_dict['API_URL_QWEN']
        self.MODEL_NAME = key_dict['MODEL_NAME']
        self.openai_client = openai.OpenAI(base_url=API_URL_QWEN, api_key=API_TOKEN)
        self.messages: List[Dict[str, Any]] = []
        self.logs_dir = Path(os.environ.get('SAGE_LOG_DIR', 'logs'))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.session_start_time = datetime.now()
        self.log_filename = self._generate_log_filename()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.api_call_count = 0
        self.tool_call_count = 0
        self.tool_call_limit_reached = False
        self.max_tool_calls = 8

    def _generate_log_filename(self) -> str:
        """Generate a timestamped log filename"""
        timestamp = self.session_start_time.strftime('%Y%m%d_%H%M%S_%f')
        return f'chat_session_qwen3vl_{timestamp}.json'

    def _save_chat_log(self):
        """Save the current conversation history to a log file"""
        try:
            conversation_summary = {'user_messages': 0, 'user_images': 0, 'assistant_messages': 0, 'function_calls': 0, 'function_outputs': 0, 'reasoning_entries': 0}
            organized_conversation = []
            for msg in self.messages:
                if hasattr(msg, 'get'):
                    msg_type = msg.get('role', msg.get('type', 'unknown'))
                elif hasattr(msg, 'type'):
                    msg_type = msg.type
                elif hasattr(msg, 'role'):
                    msg_type = msg.role
                else:
                    msg_type = 'unknown'
                if msg_type == 'user':
                    conversation_summary['user_messages'] += 1
                    content = msg.get('content', '') if hasattr(msg, 'get') else getattr(msg, 'content', '')
                    timestamp = msg.get('timestamp', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'timestamp', 'unknown')
                    images_data = []
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get('type') == 'image_url':
                                conversation_summary['user_images'] += 1
                                if 'image_metadata' in item:
                                    images_data.append(item['image_metadata'])
                    images_metadata = msg.get('images_metadata', []) if hasattr(msg, 'get') else []
                    if images_metadata:
                        images_data.extend(images_metadata)
                    entry = {'type': 'user_message', 'content': content, 'timestamp': timestamp}
                    if images_data:
                        entry['images'] = images_data
                    organized_conversation.append(entry)
                elif msg_type == 'assistant':
                    conversation_summary['assistant_messages'] += 1
                    if hasattr(msg, 'get'):
                        entry = {'type': 'assistant_response', 'content': msg.get('content', ''), 'iteration': msg.get('iteration', 'unknown'), 'timestamp': msg.get('timestamp', 'unknown'), 'has_tool_calls': msg.get('has_tool_calls', False)}
                        if 'reasoning' in msg and msg['reasoning']:
                            entry['reasoning'] = msg['reasoning']
                            conversation_summary['reasoning_entries'] += 1
                        if 'tool_calls' in msg and msg['tool_calls']:
                            conversation_summary['function_calls'] += len(msg['tool_calls'])
                    else:
                        entry = {'type': 'assistant_response', 'content': getattr(msg, 'content', ''), 'iteration': getattr(msg, 'iteration', 'unknown'), 'timestamp': getattr(msg, 'timestamp', 'unknown'), 'has_tool_calls': getattr(msg, 'has_tool_calls', False)}
                        if hasattr(msg, 'reasoning') and msg.reasoning:
                            entry['reasoning'] = msg.reasoning
                            conversation_summary['reasoning_entries'] += 1
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            conversation_summary['function_calls'] += len(msg.tool_calls)
                    organized_conversation.append(entry)
                elif msg_type == 'function_call':
                    conversation_summary['function_calls'] += 1
                    organized_conversation.append({'type': 'function_call', 'name': getattr(msg, 'name', 'unknown'), 'arguments': getattr(msg, 'arguments', ''), 'call_id': getattr(msg, 'call_id', 'unknown')})
                elif msg_type == 'tool':
                    conversation_summary['function_outputs'] += 1
                    organized_conversation.append({'type': 'tool_result', 'tool_name': msg.get('tool_name', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'tool_name', 'unknown'), 'server': msg.get('server', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'server', 'unknown'), 'content': msg.get('content', '') if hasattr(msg, 'get') else getattr(msg, 'content', ''), 'tool_call_id': msg.get('tool_call_id', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'tool_call_id', 'unknown'), 'timestamp': msg.get('timestamp', 'unknown') if hasattr(msg, 'get') else getattr(msg, 'timestamp', 'unknown'), 'error': msg.get('error', False) if hasattr(msg, 'get') else getattr(msg, 'error', False)})
                elif msg_type == 'function_call_output':
                    conversation_summary['function_outputs'] += 1
                    if hasattr(msg, 'get'):
                        organized_conversation.append({'type': 'function_output', 'tool_name': msg.get('tool_name', 'unknown'), 'server': msg.get('server', 'unknown'), 'output': msg.get('output', ''), 'call_id': msg.get('call_id', 'unknown'), 'timestamp': msg.get('timestamp', 'unknown'), 'error': msg.get('error', False)})
                    else:
                        organized_conversation.append({'type': 'function_output', 'tool_name': getattr(msg, 'tool_name', 'unknown'), 'server': getattr(msg, 'server', 'unknown'), 'output': getattr(msg, 'output', ''), 'call_id': getattr(msg, 'call_id', 'unknown'), 'timestamp': getattr(msg, 'timestamp', 'unknown'), 'error': getattr(msg, 'error', False)})
                elif msg_type == 'reasoning':
                    conversation_summary['reasoning_entries'] += 1
                    content = getattr(msg, 'content', [])
                    if isinstance(content, list):
                        reasoning_texts = []
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                reasoning_texts.append(item['text'])
                            else:
                                reasoning_texts.append(str(item))
                        content_str = ' | '.join(reasoning_texts)
                    else:
                        content_str = str(content)
                    organized_conversation.append({'type': 'reasoning', 'id': getattr(msg, 'id', 'unknown'), 'content': content_str, 'raw_content': content})
            log_data = {'session_start': self.session_start_time.isoformat(), 'session_end': datetime.now().isoformat(), 'connected_servers': {name: info['path'] for (name, info) in self.servers.items()}, 'conversation_summary': conversation_summary, 'organized_conversation': organized_conversation, 'input_list': [self._serialize_message(msg) for msg in self.messages], 'total_messages': len(self.messages), 'client_type': 'qwen3vl_chat_completions', 'token_usage': {'total_input_tokens': self.total_input_tokens, 'total_output_tokens': self.total_output_tokens, 'total_tokens': self.total_tokens, 'api_call_count': self.api_call_count, 'average_tokens_per_call': self.total_tokens / max(1, self.api_call_count)}, 'tool_usage': {'total_tool_calls': self.tool_call_count, 'max_tool_calls': self.max_tool_calls, 'limit_reached': self.tool_call_count >= self.max_tool_calls}}
            log_path = self.logs_dir / self.log_filename
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)
            print(f'💾 Enhanced chat log saved to: {log_path}')
            print(f'📊 Token usage: {self.total_input_tokens} input + {self.total_output_tokens} output = {self.total_tokens} total tokens')
            print(f'🔧 Tool usage: {self.tool_call_count}/{self.max_tool_calls} tool calls')
            print(f'💬 Conversation summary:')
            print(f"   👤 User messages: {conversation_summary['user_messages']}")
            if conversation_summary['user_images'] > 0:
                print(f"   📸 User images: {conversation_summary['user_images']}")
            print(f"   🤖 Assistant messages: {conversation_summary['assistant_messages']}")
            print(f"   🧠 Reasoning entries: {conversation_summary['reasoning_entries']}")
            print(f"   🔧 Tool calls: {conversation_summary['function_calls']}")
            print(f"   📤 Tool results: {conversation_summary['function_outputs']}")
            save_chat_log_html(str(log_path))
            print(f"💾 Chat log saved to: {log_path.with_suffix('.html')}")
        except Exception as e:
            print(f'⚠️  Failed to save chat log: {str(e)}')

    def _serialize_message(self, msg):
        """Convert message objects to serializable dictionaries"""
        if hasattr(msg, 'get'):
            if 'content_for_api' in msg:
                msg_copy = msg.copy()
                msg_copy.pop('content_for_api', None)
                msg = msg_copy
            if msg.get('role') == 'assistant' and 'reasoning' in msg:
                enhanced_msg = msg.copy()
                reasoning_content = msg['reasoning']
                reasoning_items = []
                for reasoning_item in reasoning_content:
                    reasoning_str = str(reasoning_item)
                    if 'reasoning_text' in reasoning_str.lower() or 'text' in str(reasoning_item):
                        reasoning_items.append(f"ResponseReasoningItem: 'text': '{reasoning_str}', 'type': 'reasoning_text'")
                    else:
                        reasoning_items.append(f'ResponseReasoningItem: {reasoning_str}')
                enhanced_msg['reasoning_items'] = reasoning_items
                return enhanced_msg
            return msg
        elif hasattr(msg, '__dict__'):
            result = {}
            msg_type = getattr(msg, 'type', getattr(msg, 'role', 'unknown'))
            if msg_type == 'function_call' or (hasattr(msg, 'name') and hasattr(msg, 'arguments')):
                name = getattr(msg, 'name', 'unknown')
                arguments = getattr(msg, 'arguments', '{}')
                call_id = getattr(msg, 'call_id', 'unknown')
                return f"ResponseFunctionToolCall: name='{name}', arguments='{arguments}', call_id='{call_id}'"
            for attr in dir(msg):
                if not attr.startswith('_'):
                    try:
                        value = getattr(msg, attr)
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            result[attr] = value
                        elif callable(value):
                            continue
                        else:
                            result[attr] = str(value)
                    except:
                        continue
            if hasattr(msg, 'type'):
                result['type'] = msg.type
            elif hasattr(msg, 'role'):
                result['role'] = msg.role
            return result
        else:
            return {'raw_content': str(msg), 'type': 'unknown'}

    def _auto_save_log(self):
        """Auto-save the chat log periodically"""
        if len(self.messages) > 0:
            self._save_chat_log()

    async def connect_to_servers(self, server_script_paths: List[str]):
        """Connect to multiple MCP servers

        Args:
            server_script_paths: List of paths to server scripts (.py or .js)
        """
        for (i, server_script_path) in enumerate(server_script_paths):
            server_name = f'server_{i + 1}'
            print(f'\n🔌 Connecting to {server_name}: {server_script_path}')
            is_python = server_script_path.endswith('.py')
            is_js = server_script_path.endswith('.js')
            if not (is_python or is_js):
                print(f'⚠️  Skipping {server_script_path}: Server script must be a .py or .js file')
                continue
            try:
                if is_python:
                    import os
                    abs_script_path = os.path.abspath(server_script_path)
                    bash_command = f'cd {shlex.quote(SERVER_DIR)} && exec {shlex.quote(sys.executable)} {shlex.quote(abs_script_path)}'
                    command = 'bash'
                    args = ['-c', bash_command]
                else:
                    command = 'node'
                    args = [server_script_path]
                server_params = StdioServerParameters(command=command, args=args, env=os.environ.copy())
                print(f'🔌 Connecting to {server_name} with command: {command} {args}')
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                (stdio, write) = stdio_transport
                print(f'📡 Transport type: stdio (standard input/output pipes)')
                print(f'📡 Server process command: {command}')
                print(f'📡 Server process args: {args}')
                session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
                print(f'🔌 Connected to {server_name} (stdio communication established)')
                await session.initialize()
                print(f'🔌 Initializing {server_name}')
                response = await session.list_tools()
                tools = response.tools
                self.servers[server_name] = {'session': session, 'stdio': stdio, 'write': write, 'tools': tools, 'path': server_script_path}
                print(f'✅ Connected to {server_name} with {len(tools)} tools:')
                for tool in tools:
                    print(f'  - {tool.name}: {tool.description}')
            except Exception as e:
                print(f'❌ Failed to connect to {server_script_path}: {e!r}')
                traceback.print_exc()
                continue
        if not self.servers:
            raise Exception('Failed to connect to any servers')
        total_tools = sum((len(server_info['tools']) for server_info in self.servers.values()))
        print(f'\n🎉 Successfully connected to {len(self.servers)} servers with {total_tools} total tools')

    def get_all_tools(self) -> List[Dict[str, Any]]:
        """Get all tools from all connected servers in OpenAI chat completions format"""
        all_tools = []
        tool_server_map = {}
        for (server_name, server_info) in self.servers.items():
            for tool in server_info['tools']:
                original_name = tool.name
                tool_name = tool.name
                if tool_name in tool_server_map:
                    tool_name = f'{server_name}_{original_name}'
                    conflicting_server = tool_server_map[original_name]
                    for existing_tool in all_tools:
                        if existing_tool['function']['name'] == original_name:
                            existing_tool['function']['name'] = f'{conflicting_server}_{original_name}'
                            break
                    tool_server_map[f'{conflicting_server}_{original_name}'] = conflicting_server
                    del tool_server_map[original_name]
                tool_server_map[tool_name] = server_name
                all_tools.append({'type': 'function', 'function': {'name': tool_name, 'description': f'[{server_name}] {tool.description}', 'parameters': tool.inputSchema}})
        self.tool_server_map = tool_server_map
        return all_tools

    async def call_tool_on_server(self, tool_name: str, tool_input: Dict[str, Any]) -> Any:
        """Route tool call to the correct server"""
        server_name = self.tool_server_map.get(tool_name)
        if not server_name:
            raise Exception(f'Tool {tool_name} not found on any server')
        original_tool_name = tool_name
        if tool_name.startswith(f'{server_name}_'):
            original_tool_name = tool_name[len(f'{server_name}_'):]
        server_info = self.servers[server_name]
        session = server_info['session']
        return await session.call_tool(original_tool_name, tool_input)

    def parse_tool_calls_from_content(self, content: str) -> List[Dict[str, Any]]:
        """Parse <tool_call> blocks from Qwen's text output
        
        Qwen sometimes formats tool calls as:
        <tool_call>
        {"name": "tool_name", "arguments": {...}}
        </tool_call>
        
        This method extracts and parses these into a list of tool call dictionaries.
        """
        tool_calls = []
        pattern = '<tool_call>\\s*(.*?)\\s*</tool_call>'
        matches = re.finditer(pattern, content, re.DOTALL)
        for (i, match) in enumerate(matches):
            try:
                json_str = match.group(1).strip()
                tool_data = json.loads(json_str)
                tool_call = {'id': f'call_parsed_{i}_{int(time.time() * 1000)}', 'type': 'function', 'function': {'name': tool_data.get('name', ''), 'arguments': json.dumps(tool_data.get('arguments', {}))}}
                tool_calls.append(tool_call)
                print(f"🔍 Parsed tool call from content: {tool_data.get('name', 'unknown')}")
            except json.JSONDecodeError as e:
                print(f'⚠️  Failed to parse tool call JSON: {e}')
                print(f'   JSON string was: {json_str[:200]}...')
                continue
        return tool_calls

    async def process_query(self, query: str, image_paths: Optional[List[str]]=None, system_message: Optional[str]=None) -> Tuple[str, bool]:
        """Process a query using Qwen3-VL and available tools from all servers
        
        Args:
            query: Text query/instruction
            image_paths: Optional list of image file paths to include in the query
            system_message: Optional system prompt used by retrieval mode
        
        Returns:
            tuple: (final_response, intermediate_responses_shown)
        """
        if system_message and str(system_message).strip():
            self.messages.append({'role': 'system', 'content': system_message, 'timestamp': datetime.now().isoformat()})
        if image_paths:
            content_for_api = [{'type': 'text', 'text': query}]
            content_for_log = [{'type': 'text', 'text': query}]
            images_metadata = []
            for image_path in image_paths:
                try:
                    print(f'📸 Loading image: {image_path}')
                    base64_image_full = encode_image_to_base64(image_path)
                    mime_type = get_image_mime_type(image_path)
                    content_for_api.append({'type': 'image_url', 'image_url': {'url': f'data:{mime_type};base64,{base64_image_full}'}})
                    (base64_image_resized, original_size, resized_size, file_size) = resize_and_encode_image(image_path)
                    content_for_log.append({'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{base64_image_resized}'}, 'image_metadata': {'filename': os.path.basename(image_path), 'path': image_path, 'mime_type': mime_type, 'original_size': {'width': original_size[0], 'height': original_size[1]}, 'resized_size': {'width': resized_size[0], 'height': resized_size[1]}, 'file_size_bytes': file_size, 'file_size_kb': round(file_size / 1024, 2), 'note': 'Image resized to 512x512 max for log storage'}})
                    images_metadata.append({'filename': os.path.basename(image_path), 'path': image_path, 'mime_type': mime_type, 'original_size': {'width': original_size[0], 'height': original_size[1]}, 'resized_size': {'width': resized_size[0], 'height': resized_size[1]}, 'file_size_bytes': file_size, 'file_size_kb': round(file_size / 1024, 2)})
                    print(f'   ✅ Image loaded: {os.path.basename(image_path)} ({mime_type}, {round(file_size / 1024, 2)} KB)')
                    if PIL_AVAILABLE and resized_size[0] > 0:
                        print(f'      Original: {original_size[0]}x{original_size[1]} → Resized for log: {resized_size[0]}x{resized_size[1]}')
                except Exception as e:
                    print(f'   ⚠️  Failed to load image {image_path}: {e}')
            user_message = {'role': 'user', 'content': content_for_log, 'content_for_api': content_for_api, 'images_metadata': images_metadata, 'timestamp': datetime.now().isoformat()}
        else:
            user_message = {'role': 'user', 'content': query, 'timestamp': datetime.now().isoformat()}
        self.messages.append(user_message)
        self._auto_save_log()
        available_tools = self.get_all_tools()
        max_iterations = 200
        iteration = 0
        intermediate_responses_shown = False
        while iteration < max_iterations:
            iteration += 1
            print(f'\n🔄 Processing iteration {iteration}...')
            try:
                messages_for_api = []
                for msg in self.messages:
                    if isinstance(msg, dict) and 'content_for_api' in msg:
                        api_msg = msg.copy()
                        api_msg['content'] = msg['content_for_api']
                        api_msg.pop('content_for_api', None)
                        api_msg.pop('images_metadata', None)
                        messages_for_api.append(api_msg)
                    else:
                        messages_for_api.append(msg)
                call_params = {'model': self.MODEL_NAME, 'messages': messages_for_api, 'max_tokens': int(os.environ.get('SAGE_CLIENT_MAX_TOKENS', '8192')), 'temperature': 1.0}
                if available_tools:
                    call_params['tools'] = available_tools
                    call_params['tool_choice'] = 'auto'
                    print(f'🛠️  Available tools: {len(available_tools)}')
                max_retry = 3
                last_error = None
                for retry in range(max_retry):
                    try:
                        if retry > 0:
                            wait_time = 2 ** retry
                            print(f'⏳ Retry attempt {retry + 1}/{max_retry} after {wait_time}s delay...', file=sys.stderr)
                            time.sleep(wait_time)
                        response = self.openai_client.chat.completions.create(**call_params)
                        if hasattr(response, 'usage'):
                            usage = response.usage
                            input_tokens = getattr(usage, 'prompt_tokens', 0)
                            output_tokens = getattr(usage, 'completion_tokens', 0)
                            total_tokens = getattr(usage, 'total_tokens', input_tokens + output_tokens)
                            self.total_input_tokens += input_tokens
                            self.total_output_tokens += output_tokens
                            self.total_tokens += total_tokens
                            self.api_call_count += 1
                            print(f'🔢 Request processed successfully (tokens: {input_tokens} input + {output_tokens} output = {total_tokens} total)')
                        else:
                            print(f'🔢 Request processed successfully')
                        break
                    except ContextBudgetExhausted as e:
                        self.context_budget_exhausted = True
                        print(f'Context budget exhausted; keeping the current scene: {e}')
                        return (str(e), intermediate_responses_shown)
                    except Exception as e:
                        last_error = e
                        print(f'❌ API call failed: {str(e)}')
                        context_error = re.search('maximum context length is (\\d+).*prompt contains at least (\\d+)', str(e))
                        if context_error:
                            self.context_budget_exhausted = True
                            return (str(e), intermediate_responses_shown)
                        if retry == max_retry - 1:
                            error_msg = f'Error calling Qwen3-VL API after {max_retry} attempts: {str(e)}'
                            return (error_msg, intermediate_responses_shown)
            except Exception as e:
                error_msg = f'Error calling Qwen3-VL API: {str(e)}'
                return (error_msg, intermediate_responses_shown)
            text_responses = []
            tool_calls = []
            reasoning_text = None
            try:
                message = response.choices[0].message
                if hasattr(message, 'content') and message.content:
                    text_responses.append(message.content)
                    print(f'📝 Response content: {len(message.content)} chars')
                if hasattr(message, 'reasoning') and message.reasoning:
                    reasoning_text = message.reasoning
                    print(f'🧠 Reasoning available: {len(reasoning_text)} chars')
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    tool_calls = message.tool_calls
                    print(f'🔧 Found {len(tool_calls)} tool call(s) in API response')
                if not tool_calls and message.content and ('<tool_call>' in message.content):
                    print(f'🔍 No tool calls in API response, parsing from content...')
                    parsed_tool_calls = self.parse_tool_calls_from_content(message.content)
                    if parsed_tool_calls:

                        class ToolCall:

                            def __init__(self, data):
                                self.id = data['id']
                                self.type = data['type']
                                self.function = type('Function', (), {'name': data['function']['name'], 'arguments': data['function']['arguments']})()
                        tool_calls = [ToolCall(tc) for tc in parsed_tool_calls]
                        print(f'🔧 Parsed {len(tool_calls)} tool call(s) from content')
            except Exception as e:
                print(f'⚠️  Error processing response: {e}')
                import traceback
                traceback.print_exc()
                text_responses.append(f'Error processing response: {str(e)}')
            if reasoning_text:
                print(f'\n🧠 Qwen3-VL Reasoning/Thinking:')
                reasoning_preview = reasoning_text[:500] + '...' if len(reasoning_text) > 500 else reasoning_text
                print(f'  💭 {reasoning_preview}')
            assistant_msg = {'role': 'assistant', 'content': message.content if hasattr(message, 'content') else '', 'timestamp': datetime.now().isoformat(), 'iteration': iteration, 'has_tool_calls': len(tool_calls) > 0}
            if tool_calls:
                tool_calls = tool_calls[:1]
                assistant_msg['tool_calls'] = [{'id': tc.id, 'type': 'function', 'function': {'name': tc.function.name, 'arguments': tc.function.arguments}} for tc in tool_calls]
            if reasoning_text:
                assistant_msg['reasoning'] = reasoning_text
            self.messages.append(assistant_msg)
            self._auto_save_log()
            if text_responses:
                full_text = '\n'.join(text_responses)
                print(f'\n🤖 Qwen3-VL Response:')
                print(f'   📝 Content length: {len(full_text)} characters')
                print(f'   🔤 Response: {full_text}')
                intermediate_responses_shown = True
            if not tool_calls:
                final_response = '\n'.join(text_responses) if text_responses else 'No response from Qwen3-VL.'
                print(f'✅ Conversation complete after {iteration} iteration(s)')
                return (final_response, intermediate_responses_shown)
            if self.tool_call_count >= self.max_tool_calls:
                self.tool_call_limit_reached = True
                warning_msg = f'⚠️  Reached maximum tool call limit ({self.max_tool_calls}). Stopping execution.'
                print(f'\n{warning_msg}')
                final_response = '\n'.join(text_responses) if text_responses else warning_msg
                return (final_response, intermediate_responses_shown)
            print(f'\n🔧 Executing {len(tool_calls)} tool call(s)...')
            for (i, tool_call) in enumerate(tool_calls):
                if self.tool_call_count >= self.max_tool_calls:
                    self.tool_call_limit_reached = True
                    warning_msg = f'Reached maximum tool call limit ({self.max_tool_calls}). Skipping remaining {len(tool_calls) - i} tool call(s).'
                    print(f'\n  ⚠️  {warning_msg}')
                    tool_result_msg = {'role': 'tool', 'tool_call_id': tool_call.id, 'content': warning_msg, 'timestamp': datetime.now().isoformat()}
                    self.messages.append(tool_result_msg)
                    continue
                try:
                    tool_name = tool_call.function.name
                    tool_arguments = tool_call.function.arguments
                    server_name = self.tool_server_map.get(tool_name, 'unknown')
                    print(f'\n  🔧 Tool {i + 1}/{len(tool_calls)}: {tool_name} on {server_name}')
                    print(f'     📥 Arguments: {tool_arguments}')
                    self.tool_call_count += 1
                    print(f'     📊 Tool call count: {self.tool_call_count}/{self.max_tool_calls}')
                    tool_input = json.loads(tool_arguments)
                    tool_started_at = time.perf_counter()
                    try:
                        result = await self.call_tool_on_server(tool_name, tool_input)
                    finally:
                        tool_duration = time.perf_counter() - tool_started_at
                        print(f'     Tool duration: {tool_duration:.3f}s')
                    if hasattr(result, 'content'):
                        if isinstance(result.content, list):
                            content_blocks = []
                            for item in result.content:
                                if hasattr(item, 'type'):
                                    if item.type == 'text' and hasattr(item, 'text'):
                                        content_blocks.append(item.text)
                                    elif item.type == 'image':
                                        content_blocks.append(f"[Image: {getattr(item.source, 'media_type', 'unknown')}]")
                                        print(f"     📸 Image result: {getattr(item.source, 'media_type', 'unknown')}")
                                    else:
                                        content_blocks.append(str(item))
                                else:
                                    content_blocks.append(str(item))
                            result_content = '\n'.join(content_blocks)
                            print(f"     📤 Result: {result_content[:200]}{('...' if len(result_content) > 200 else '')}")
                        else:
                            result_content = str(result.content)
                            print(f"     📤 Result: {result_content[:200]}{('...' if len(result_content) > 200 else '')}")
                    else:
                        result_content = str(result)
                        print(f"     📤 Result: {result_content[:200]}{('...' if len(result_content) > 200 else '')}")
                    result_content_dict = json.loads(result_content)
                    if 'layout_id' in result_content_dict:
                        self.layout_id = result_content_dict['layout_id']
                    tool_result_msg = {'role': 'tool', 'tool_call_id': tool_call.id, 'content': result_content, 'timestamp': datetime.now().isoformat(), 'tool_name': tool_name, 'server': server_name}
                    self.messages.append(tool_result_msg)
                except Exception as e:
                    error_msg = f'Error executing tool {tool_name}: {str(e)}'
                    print(f'     ❌ Error: {error_msg}')
                    error_result_msg = {'role': 'tool', 'tool_call_id': tool_call.id, 'content': error_msg, 'timestamp': datetime.now().isoformat(), 'tool_name': tool_name, 'server': server_name, 'error': True}
                    self.messages.append(error_result_msg)
            self._auto_save_log()
            print(f'🔄 Continuing to next iteration...')
            print(f'🔍 Layout ID: {self.layout_id}')
            continue
        return ('Conversation exceeded maximum iterations. Please try a simpler query.', intermediate_responses_shown)

    async def cleanup(self):
        """Clean up resources"""
        self._save_chat_log()
        await self.exit_stack.aclose()
