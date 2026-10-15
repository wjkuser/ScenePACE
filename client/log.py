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
import json
import os
from datetime import datetime

def _convert_organized_to_conversation(organized_conversation):
    """
    Convert organized_conversation format to conversation_history format.
    
    Args:
        organized_conversation: List of organized conversation items with type field
        
    Returns:
        list: Conversation history in expected format
    """
    conversation = []
    current_assistant_content = []
    for item in organized_conversation:
        item_type = item.get('type', 'unknown')
        if item_type == 'user_message':
            if current_assistant_content:
                conversation.append({'role': 'assistant', 'content': current_assistant_content})
                current_assistant_content = []
            conversation.append({'role': 'user', 'content': item.get('content', '')})
        elif item_type == 'reasoning':
            content_text = item.get('content', '')
            raw_content = item.get('raw_content', [])
            thinking_text = content_text
            if 'Content(text=' in thinking_text:
                import re
                text_match = re.search('Content\\(text=(?:\\"((?:[^\\"]|\\\\\\")*)\\")|(?:\'((?:[^\'\\\\]|\\\\.)*)\')', thinking_text, re.DOTALL)
                if text_match:
                    thinking_text = text_match.group(1) or text_match.group(2)
                    thinking_text = thinking_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                    thinking_text = thinking_text.replace("\\'", "'").replace('\\"', '"').replace('\\\\', '\\')
            current_assistant_content.append({'type': 'thinking', 'thinking': thinking_text})
        elif item_type == 'function_call':
            tool_name = item.get('name', 'unknown')
            arguments = item.get('arguments', '{}')
            call_id = item.get('call_id', 'unknown')
            import json
            try:
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
            except:
                pass
            current_assistant_content.append({'type': 'tool_use', 'id': call_id, 'name': tool_name, 'input': arguments})
        elif item_type == 'tool_result':
            content = item.get('content', '')
            tool_call_id = item.get('tool_call_id', 'unknown')
            is_error = item.get('error', False)
            current_assistant_content.append({'type': 'tool_result', 'tool_use_id': tool_call_id, 'content': content, 'is_error': is_error})
        elif item_type == 'function_output':
            output = item.get('output', '')
            call_id = item.get('call_id', 'unknown')
            is_error = item.get('error', False)
            current_assistant_content.append({'type': 'tool_result', 'tool_use_id': call_id, 'content': output, 'is_error': is_error})
        elif item_type == 'assistant_response' or item_type == 'ai_response':
            if current_assistant_content:
                conversation.append({'role': 'assistant', 'content': current_assistant_content})
                current_assistant_content = []
            assistant_content = []
            if 'reasoning' in item:
                reasoning_value = item.get('reasoning', [])
                if isinstance(reasoning_value, str):
                    reasoning_items = [reasoning_value]
                else:
                    reasoning_items = reasoning_value
                for reasoning_item in reasoning_items:
                    reasoning_text = str(reasoning_item)
                    if 'Content(text=' in reasoning_text:
                        import re
                        text_match = re.search('Content\\(text=(?:\\"((?:[^\\"]|\\\\\\")*)\\")|(?:\'((?:[^\'\\\\]|\\\\.)*)\')', reasoning_text, re.DOTALL)
                        if text_match:
                            reasoning_text = text_match.group(1) or text_match.group(2)
                            reasoning_text = reasoning_text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
                            reasoning_text = reasoning_text.replace("\\'", "'").replace('\\"', '"').replace('\\\\', '\\')
                    assistant_content.append({'type': 'thinking', 'thinking': reasoning_text})
            content = item.get('content', '')
            if content:
                assistant_content.append({'type': 'text', 'text': content})
            if assistant_content:
                conversation.append({'role': 'assistant', 'content': assistant_content})
    if current_assistant_content:
        conversation.append({'role': 'assistant', 'content': current_assistant_content})
    return conversation

def _convert_input_list_to_conversation(input_list):
    """
    Convert input_list format to conversation_history format.
    
    Args:
        input_list: List of various message types from input_list format
        
    Returns:
        list: Conversation history in expected format
    """
    conversation = []
    current_assistant_content = []
    for item in input_list:
        if isinstance(item, dict):
            if item.get('role') == 'user':
                if current_assistant_content:
                    conversation.append({'role': 'assistant', 'content': current_assistant_content})
                    current_assistant_content = []
                conversation.append(item)
            elif item.get('role') == 'assistant':
                if current_assistant_content:
                    conversation.append({'role': 'assistant', 'content': current_assistant_content})
                    current_assistant_content = []
                assistant_content = []
                if 'reasoning' in item:
                    for reasoning_item in item['reasoning']:
                        reasoning_text = str(reasoning_item)
                        assistant_content.append({'type': 'thinking', 'thinking': reasoning_text})
                if 'content' in item and item['content']:
                    assistant_content.append({'type': 'text', 'text': item['content']})
                if assistant_content:
                    conversation.append({'role': 'assistant', 'content': assistant_content})
            elif item.get('type') == 'function_call_output':
                current_assistant_content.append({'type': 'tool_result', 'tool_use_id': item.get('call_id', ''), 'content': item.get('output', ''), 'is_error': item.get('error', False)})
        elif isinstance(item, str):
            if item.startswith('ResponseFunctionToolCall'):
                try:
                    import re
                    name_match = re.search("name='([^']+)'", item)
                    args_match = re.search("arguments='({.*?})'", item, re.DOTALL)
                    id_match = re.search("call_id='([^']+)'", item)
                    if name_match and args_match and id_match:
                        tool_name = name_match.group(1)
                        arguments_str = args_match.group(1)
                        tool_id = id_match.group(1)
                        arguments_str = arguments_str.replace('\\n', '\n').replace('\\"', '"')
                        try:
                            arguments = json.loads(arguments_str)
                        except:
                            arguments = arguments_str
                        current_assistant_content.append({'type': 'tool_use', 'id': tool_id, 'name': tool_name, 'input': arguments})
                except:
                    current_assistant_content.append({'type': 'text', 'text': item})
            elif item.startswith('ResponseReasoningItem'):
                try:
                    import re
                    text_match = re.search("'text': '(.*?)', 'type': 'reasoning_text'", item, re.DOTALL)
                    if text_match:
                        thinking_text = text_match.group(1)
                        thinking_text = thinking_text.replace("\\'", "'").replace('\\n', '\n').replace('\\\\', '\\')
                        current_assistant_content.append({'type': 'thinking', 'thinking': thinking_text})
                    else:
                        reasoning_text = item.replace('ResponseReasoningItem: ', '')
                        current_assistant_content.append({'type': 'thinking', 'thinking': reasoning_text})
                except:
                    current_assistant_content.append({'type': 'text', 'text': item})
            else:
                current_assistant_content.append({'type': 'text', 'text': item})
    if current_assistant_content:
        conversation.append({'role': 'assistant', 'content': current_assistant_content})
    return conversation

def chat_log_to_html(log_file):
    """
    Convert client chat logs to HTML format with inline CSS styling.
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        str: HTML content of the chat logs
    """
    try:
        if not os.path.exists(log_file):
            return f'<html><body><h1>Log file not found: {log_file}</h1></body></html>'
        with open(log_file, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        if not isinstance(log_data, dict):
            return '<html><body><h1>Invalid client log format</h1></body></html>'
        if 'conversation_history' not in log_data and 'input_list' not in log_data and ('raw_input_list' not in log_data):
            return '<html><body><h1>Invalid client log format - missing conversation data</h1></body></html>'
    except Exception as e:
        return f'<html><body><h1>Error reading log file: {str(e)}</h1></body></html>'
    session_start = log_data.get('session_start', 'Unknown')
    session_end = log_data.get('session_end', 'Unknown')
    connected_servers = log_data.get('connected_servers', {})
    tool_uses_count = 0
    if 'organized_conversation' in log_data:
        organized_conv = log_data.get('organized_conversation', [])
        conversation_history = _convert_organized_to_conversation(organized_conv)
        tool_uses_count = sum((1 for item in organized_conv if item.get('type') == 'tool_result'))
    elif 'conversation_history' in log_data:
        conversation_history = log_data.get('conversation_history', [])
    else:
        input_list = log_data.get('input_list', log_data.get('raw_input_list', []))
        conversation_history = _convert_input_list_to_conversation(input_list)
    total_messages = log_data.get('total_messages', len(conversation_history))
    try:
        if session_start != 'Unknown' and session_end != 'Unknown':
            start_dt = datetime.fromisoformat(session_start.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(session_end.replace('Z', '+00:00'))
            duration = end_dt - start_dt
            duration_str = str(duration).split('.')[0]
        else:
            duration_str = 'Unknown'
    except:
        duration_str = 'Unknown'
    log_filename = os.path.basename(log_file)
    html_content = f"""\n    <!DOCTYPE html>\n    <html lang="en">\n    <head>\n        <meta charset="UTF-8">\n        <meta name="viewport" content="width=device-width, initial-scale=1.0">\n        <title>MCP Client Chat Log</title>\n        <style>\n            body {{\n                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;\n                line-height: 1.6;\n                margin: 0;\n                padding: 20px;\n                background-color: #f5f5f5;\n                color: #333;\n            }}\n            .container {{\n                max-width: 1400px;\n                margin: 0 auto;\n                background: white;\n                border-radius: 8px;\n                box-shadow: 0 2px 10px rgba(0,0,0,0.1);\n                overflow: hidden;\n            }}\n            .header {{\n                background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);\n                color: white;\n                padding: 30px;\n                text-align: center;\n            }}\n            .header h1 {{\n                margin: 0 0 10px 0;\n                font-size: 2.2em;\n            }}\n            .header p {{\n                margin: 5px 0;\n                opacity: 0.9;\n            }}\n            .session-info {{\n                background: #f8f9fa;\n                padding: 20px;\n                border-bottom: 1px solid #eee;\n            }}\n            .info-grid {{\n                display: grid;\n                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));\n                gap: 15px;\n                margin-bottom: 15px;\n            }}\n            .info-item {{\n                background: white;\n                padding: 15px;\n                border-radius: 6px;\n                border-left: 4px solid #4CAF50;\n            }}\n            .info-label {{\n                font-weight: bold;\n                color: #495057;\n                margin-bottom: 5px;\n            }}\n            .info-value {{\n                color: #6c757d;\n                word-break: break-all;\n            }}\n            .servers-section {{\n                margin-top: 15px;\n            }}\n            .server-item {{\n                background: white;\n                padding: 10px 15px;\n                margin: 5px 0;\n                border-radius: 4px;\n                border-left: 3px solid #2196F3;\n            }}\n            .conversation {{\n                padding: 20px;\n            }}\n            .message {{\n                margin: 20px 0;\n                border-radius: 8px;\n                overflow: hidden;\n                box-shadow: 0 2px 4px rgba(0,0,0,0.1);\n            }}\n            .message-header {{\n                padding: 15px 20px;\n                font-weight: bold;\n                font-size: 1.1em;\n                display: flex;\n                align-items: center;\n                justify-content: space-between;\n            }}\n            .message-content {{\n                padding: 20px;\n                line-height: 1.6;\n            }}\n            .user-message {{\n                border-left: 4px solid #2196F3;\n            }}\n            .user-message .message-header {{\n                background: #e3f2fd;\n                color: #1976d2;\n            }}\n            .assistant-message {{\n                border-left: 4px solid #9c27b0;\n            }}\n            .assistant-message .message-header {{\n                background: #f3e5f5;\n                color: #7b1fa2;\n            }}\n            .content-block {{\n                margin: 15px 0;\n                padding: 15px;\n                border-radius: 6px;\n                border-left: 3px solid #dee2e6;\n            }}\n            .thinking-block {{\n                background: #fff3e0;\n                border-left-color: #ff9800;\n            }}\n            .thinking-header {{\n                font-weight: bold;\n                color: #f57c00;\n                margin-bottom: 10px;\n                display: flex;\n                align-items: center;\n            }}\n            .thinking-content {{\n                white-space: pre-wrap;\n                word-wrap: break-word;\n                font-family: 'Courier New', monospace;\n                font-size: 0.9em;\n                line-height: 1.4;\n            }}\n            .text-block {{\n                background: #f8f9fa;\n                border-left-color: #28a745;\n            }}\n            .text-content {{\n                white-space: pre-wrap;\n                word-wrap: break-word;\n            }}\n            .tool-use-block {{\n                background: #e8f5e8;\n                border-left-color: #4caf50;\n            }}\n            .tool-header {{\n                font-weight: bold;\n                color: #2e7d32;\n                margin-bottom: 10px;\n            }}\n            .tool-details {{\n                background: white;\n                padding: 10px;\n                border-radius: 4px;\n                margin: 10px 0;\n                font-family: 'Courier New', monospace;\n                font-size: 0.9em;\n            }}\n            .tool-result-block {{\n                background: #f0f4f8;\n                border-left-color: #607d8b;\n            }}\n            .tool-result-header {{\n                font-weight: bold;\n                color: #455a64;\n                margin-bottom: 10px;\n            }}\n            .tool-result-content {{\n                background: white;\n                padding: 15px;\n                border-radius: 4px;\n                white-space: pre-wrap;\n                word-wrap: break-word;\n                font-family: 'Courier New', monospace;\n                font-size: 0.9em;\n                line-height: 1.4;\n                max-height: 400px;\n                overflow-y: auto;\n            }}\n            .redacted {{\n                background: #ffebee;\n                border-left-color: #f44336;\n                color: #c62828;\n                font-style: italic;\n            }}\n            .json-content {{\n                background: #f8f9fa;\n                padding: 10px;\n                border-radius: 4px;\n                font-family: 'Courier New', monospace;\n                font-size: 0.85em;\n                overflow-x: auto;\n                border: 1px solid #e9ecef;\n            }}\n            .message-counter {{\n                background: rgba(255,255,255,0.2);\n                padding: 4px 12px;\n                border-radius: 12px;\n                font-size: 0.9em;\n            }}\n            .summary-stats {{\n                display: flex;\n                justify-content: space-around;\n                background: #e8f5e8;\n                padding: 15px;\n                margin-top: 20px;\n                border-radius: 6px;\n            }}\n            .stat-item {{\n                text-align: center;\n            }}\n            .stat-number {{\n                font-size: 1.5em;\n                font-weight: bold;\n                color: #2e7d32;\n            }}\n            .stat-label {{\n                color: #666;\n                font-size: 0.9em;\n            }}\n        </style>\n    </head>\n    <body>\n        <div class="container">\n            <div class="header">\n                <h1>🤖 MCP Client Chat Log</h1>\n                <p>Generated from: {log_filename}</p>\n            </div>\n            \n            <div class="session-info">\n                <div class="info-grid">\n                    <div class="info-item">\n                        <div class="info-label">Session Start</div>\n                        <div class="info-value">{session_start}</div>\n                    </div>\n                    <div class="info-item">\n                        <div class="info-label">Session End</div>\n                        <div class="info-value">{session_end}</div>\n                    </div>\n                    <div class="info-item">\n                        <div class="info-label">Duration</div>\n                        <div class="info-value">{duration_str}</div>\n                    </div>\n                    <div class="info-item">\n                        <div class="info-label">Total Messages</div>\n                        <div class="info-value">{total_messages}</div>\n                    </div>\n                </div>\n                \n                <div class="servers-section">\n                    <div class="info-label">Connected Servers:</div>\n    """
    if connected_servers:
        for (server_name, server_path) in connected_servers.items():
            html_content += f'<div class="server-item"><strong>{server_name}:</strong> {server_path}</div>'
    else:
        html_content += '<div class="server-item">No servers connected</div>'
    html_content += '\n                </div>\n            </div>\n            \n            <div class="conversation">\n    '
    for (i, message) in enumerate(conversation_history):
        role = message.get('role', 'unknown')
        content = message.get('content', '')
        message_class = 'user-message' if role == 'user' else 'assistant-message'
        html_content += f'\n            <div class="message {message_class}">\n                <div class="message-header">\n                    <span>👤 {role.title()}</span>\n                    <span class="message-counter">Message #{i + 1}</span>\n                </div>\n                <div class="message-content">\n        '
        if isinstance(content, str):
            formatted_text = _format_text_content(content)
            html_content += f'<div class="text-content">{formatted_text}</div>'
        elif isinstance(content, list):
            has_images = any((isinstance(item, dict) and item.get('type') == 'image_url' for item in content))
            if has_images and role == 'user':
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text = item.get('text', '')
                            formatted_text = _format_text_content(text)
                            html_content += f'\n                                <div class="content-block text-block">\n                                    <div class="text-content">{formatted_text}</div>\n                                </div>\n                            '
                        elif item.get('type') == 'image_url':
                            image_url = item.get('image_url', {}).get('url', '')
                            metadata = item.get('image_metadata', {})
                            html_content += '<div class="content-block" style="background: #f0f8ff; border-left-color: #2196F3;">'
                            html_content += '<div style="font-weight: bold; color: #1976d2; margin-bottom: 10px;">📸 Image</div>'
                            if image_url:
                                html_content += f'<img src="{image_url}" style="max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 10px 0;" alt="User uploaded image" />'
                            if metadata:
                                html_content += '<div style="font-size: 0.85em; color: #666; margin-top: 10px;">'
                                html_content += f"<strong>Filename:</strong> {_escape_html(metadata.get('filename', 'unknown'))}<br>"
                                orig_size = metadata.get('original_size', {})
                                if orig_size and orig_size.get('width', 0) > 0:
                                    html_content += f"<strong>Original size:</strong> {orig_size['width']}×{orig_size['height']}px<br>"
                                resized_size = metadata.get('resized_size', {})
                                if resized_size and resized_size.get('width', 0) > 0:
                                    html_content += f"<strong>Displayed size:</strong> {resized_size['width']}×{resized_size['height']}px<br>"
                                if 'file_size_kb' in metadata:
                                    html_content += f"<strong>File size:</strong> {metadata['file_size_kb']} KB<br>"
                                if 'note' in metadata:
                                    html_content += f"<em>{_escape_html(metadata['note'])}</em>"
                                html_content += '</div>'
                            html_content += '</div>'
            else:
                for item in content:
                    if isinstance(item, dict):
                        content_type = item.get('type', 'text')
                        if content_type == 'thinking':
                            thinking_text = item.get('thinking', '')
                            signature = item.get('signature', '')
                            html_content += f"""\n                                <div class="content-block thinking-block">\n                                    <div class="thinking-header">🧠 AI Reasoning / Thinking Process</div>\n                                    <div class="thinking-content">{_escape_html(thinking_text)}</div>\n                                    {(f'<div style="margin-top: 10px; font-size: 0.8em; color: #666;">Signature: {signature[:50]}...</div>' if signature else '')}\n                                </div>\n                            """
                        elif content_type == 'redacted_thinking':
                            html_content += f'\n                                <div class="content-block redacted">\n                                    <div class="thinking-header">🧠 AI Reasoning / Thinking Process (Redacted)</div>\n                                    <div>This thinking content was redacted for safety reasons.</div>\n                                </div>\n                            '
                        elif content_type == 'text':
                            text = item.get('text', '')
                            formatted_text = _format_text_content(text)
                            html_content += f'\n                                <div class="content-block text-block">\n                                    <div class="text-content">{formatted_text}</div>\n                                </div>\n                            '
                        elif content_type == 'tool_use':
                            tool_id = item.get('id', '')
                            tool_name = item.get('name', '')
                            tool_input = item.get('input', {})
                            html_content += f'\n                                <div class="content-block tool-use-block">\n                                    <div class="tool-header">🔧 Tool Use: {tool_name}</div>\n                                    <div class="tool-details">\n                                        <strong>ID:</strong> {tool_id}<br>\n                                        <strong>Arguments:</strong>\n                                        <div class="json-content">{_format_json(tool_input)}</div>\n                                    </div>\n                                </div>\n                            '
                        elif content_type == 'tool_result':
                            tool_use_id = item.get('tool_use_id', '')
                            result_content = item.get('content', '')
                            is_error = item.get('is_error', False)
                            if isinstance(result_content, list):
                                formatted_result = ''
                                for result_item in result_content:
                                    if isinstance(result_item, dict) and result_item.get('type') == 'text':
                                        formatted_result += _format_text_content(result_item.get('text', ''))
                                    else:
                                        formatted_result += _format_text_content(str(result_item))
                            else:
                                formatted_result = _format_text_content(str(result_content))
                            error_class = ' redacted' if is_error else ''
                            html_content += f"""\n                                <div class="content-block tool-result-block{error_class}">\n                                    <div class="tool-result-header">📋 Tool Result{(' (Error)' if is_error else '')}</div>\n                                    <div style="font-size: 0.9em; margin-bottom: 10px;">Tool Use ID: {tool_use_id}</div>\n                                    <div class="tool-result-content">{formatted_result}</div>\n                                </div>\n                            """
                    else:
                        formatted_text = _format_text_content(str(item))
                        html_content += f'<div class="text-content">{formatted_text}</div>'
        else:
            formatted_text = _format_text_content(str(content))
            html_content += f'<div class="text-content">{formatted_text}</div>'
        html_content += '</div></div>'
    user_messages = len([m for m in conversation_history if m.get('role') == 'user'])
    assistant_messages = len([m for m in conversation_history if m.get('role') == 'assistant'])
    tool_uses = tool_uses_count
    html_content += f'\n                <div class="summary-stats">\n                    <div class="stat-item">\n                        <div class="stat-number">{total_messages}</div>\n                        <div class="stat-label">Total Messages</div>\n                    </div>\n                    <div class="stat-item">\n                        <div class="stat-number">{user_messages}</div>\n                        <div class="stat-label">User Messages</div>\n                    </div>\n                    <div class="stat-item">\n                        <div class="stat-number">{assistant_messages}</div>\n                        <div class="stat-label">Assistant Messages</div>\n                    </div>\n                    <div class="stat-item">\n                        <div class="stat-number">{tool_uses}</div>\n                        <div class="stat-label">Tool Uses</div>\n                    </div>\n                </div>\n            </div>\n        </div>\n    </body>\n    </html>\n    '
    return html_content

def _escape_html(text):
    """Helper function to escape HTML characters"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')

def _format_text_content(text):
    """
    Format text content for HTML display with proper line breaks and spacing.
    """
    if not isinstance(text, str):
        text = str(text)
    text = _escape_html(text)
    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')
    import re
    text = re.sub('  +', lambda m: '&nbsp;' * len(m.group()), text)
    if not text.startswith('<p>') and (not text.startswith('</p>')):
        text = f'<p>{text}</p>'
    text = re.sub('<p></p>', '', text)
    text = re.sub('<p><br>', '<p>', text)
    text = re.sub('<br></p>', '</p>', text)
    return text

def _format_json(obj):
    """Format JSON object for display"""
    try:
        return _escape_html(json.dumps(obj, indent=2, ensure_ascii=False))
    except:
        return _escape_html(str(obj))

def save_chat_log_html(log_file, output_file=None):
    """
    Convert client chat logs to HTML and save to file.
    
    Args:
        log_file: Path to the JSON log file
        output_file: Optional output file path. If None, creates .html version of log_file
        
    Returns:
        str: Path to the generated HTML file
    """
    if output_file is None:
        base_name = os.path.splitext(log_file)[0]
        output_file = f'{base_name}.html'
    html_content = chat_log_to_html(log_file)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Client chat log saved to: {output_file}')
    return output_file
