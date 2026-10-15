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
import time
import random
import threading
from datetime import datetime
from key import ANTHROPIC_API_KEY, API_TOKEN, API_URL_DICT, MODEL_DICT
import anthropic
from constants import SERVER_ROOT_DIR
try:
    from openai import OpenAI
except ImportError:
    print('Warning: OpenAI library not installed. Install with: pip install openai')
    OpenAI = None
LOG_DIR = os.environ.get('SAGE_LOG_DIR', os.path.join(SERVER_ROOT_DIR, 'logs'))
_log_lock = threading.Lock()

def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0, backoff_factor=2.0, jitter=True):
    """
    Decorator to add retry logic with exponential backoff to API calls.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        backoff_factor: Factor to multiply delay by after each retry
        jitter: Whether to add random jitter to delay times
    """

    def decorator(func):

        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    if not _is_retryable_error(e):
                        print(f'Non-retryable error encountered: {type(e).__name__}: {str(e)}')
                        break
                    delay = min(base_delay * backoff_factor ** attempt, max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random() * 0.5)
                    print(f'API call failed (attempt {attempt + 1}/{max_retries + 1}): {type(e).__name__}: {str(e)}')
                    print(f'Retrying in {delay:.2f} seconds...')
                    time.sleep(delay)
            raise last_exception
        return wrapper
    return decorator

def _is_retryable_error(error):
    """
    Determine if an error is retryable.
    
    Args:
        error: The exception that was raised
        
    Returns:
        bool: True if the error should be retried, False otherwise
    """
    error_str = str(error).lower()
    error_type = type(error).__name__
    retryable_errors = ['ConnectionError', 'Timeout', 'TimeoutError', 'ConnectTimeout', 'ReadTimeout', 'HTTPSConnectionPool', 'NewConnectionError', 'MaxRetryError']
    if any((err_type in error_type for err_type in retryable_errors)):
        return True
    retryable_messages = ['connection', 'timeout', 'network', 'temporarily unavailable', 'service unavailable', 'internal server error', 'bad gateway', 'gateway timeout', 'rate limit', 'rate_limit', 'too many requests', 'overloaded', 'server error']
    if any((msg in error_str for msg in retryable_messages)):
        return True
    if hasattr(error, 'status_code'):
        retryable_status_codes = {429, 500, 502, 503, 504}
        if error.status_code in retryable_status_codes:
            return True
    if 'anthropic' in error_type.lower():
        if any((keyword in error_str for keyword in ['rate', 'server', 'internal', 'unavailable'])):
            return True
    if 'openai' in error_type.lower():
        if any((keyword in error_str for keyword in ['rate', 'server', 'internal', 'unavailable'])):
            return True
    return False

def convert_claude_to_openai_messages(claude_messages):
    """
    Convert Claude message format to OpenAI format.
    
    Claude format:
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "..."}}
        ]
    }
    
    OpenAI format:
    {
        "role": "user", 
        "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
        ]
    }
    """
    openai_messages = []
    for message in claude_messages:
        openai_message = {'role': message['role']}
        if isinstance(message['content'], str):
            openai_message['content'] = message['content']
        elif isinstance(message['content'], list):
            openai_content = []
            for content_item in message['content']:
                if content_item['type'] == 'text':
                    openai_content.append({'type': 'text', 'text': content_item['text']})
                elif content_item['type'] == 'image':
                    source = content_item['source']
                    if source['type'] == 'base64':
                        media_type = source['media_type']
                        data = source['data']
                        openai_content.append({'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{data}'}})
                elif content_item['type'] == 'video':
                    source = content_item['source']
                    if source['type'] == 'base64':
                        media_type = source['media_type']
                        data = source['data']
                        openai_content.append({'type': 'video_url', 'video_url': {'url': f'data:{media_type};base64,{data}'}})
            openai_message['content'] = openai_content
        else:
            openai_message['content'] = str(message['content'])
        openai_messages.append(openai_message)
    return openai_messages

class OpenAIResponseAdapter:
    """
    Adapter to make OpenAI response compatible with Claude response format.
    
    This allows existing code that expects Claude response structure to work
    with OpenAI/Qwen responses without modification.
    """

    def __init__(self, openai_response):
        self.openai_response = openai_response
        self._content = None

    @property
    def content(self):
        """Convert OpenAI choices to Claude content format."""
        if self._content is None:
            self._content = []
            if hasattr(self.openai_response, 'choices') and self.openai_response.choices:
                for choice in self.openai_response.choices:
                    if hasattr(choice, 'message') and hasattr(choice.message, 'content'):
                        content_obj = ClaudeContentItem(choice.message.content)
                        self._content.append(content_obj)
        return self._content

    @property
    def id(self):
        return getattr(self.openai_response, 'id', None)

    @property
    def model(self):
        return getattr(self.openai_response, 'model', None)

    @property
    def usage(self):
        return getattr(self.openai_response, 'usage', None)

    def __getattr__(self, name):
        """Forward any other attributes to the original response."""
        return getattr(self.openai_response, name)

class ClaudeContentItem:
    """Adapter to make OpenAI message content look like Claude content item."""

    def __init__(self, text_content):
        self._text = text_content
        self.type = 'text'

    @property
    def text(self):
        return self._text

def log_vlm_call(model, max_tokens, temperature, thinking, messages, response, log_file):
    """
    Log VLM API calls to a JSON file.
    
    Args:
        model: The model name used for the API call
        max_tokens: Maximum tokens parameter
        temperature: Temperature parameter
        thinking: Thinking parameter (if any)
        messages: List of messages sent to the API
        response: API response object
        log_file: Path to the log file
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_entry = {'timestamp': datetime.now().isoformat(), 'model': model, 'max_tokens': max_tokens, 'temperature': temperature, 'thinking': thinking, 'messages': messages, 'response': {'id': getattr(response, 'id', None), 'model': getattr(response, 'model', None), 'usage': getattr(response, 'usage', None).__dict__ if hasattr(response, 'usage') and response.usage else None, 'content': [{'type': content.type, 'text': content.text if hasattr(content, 'text') else str(content)} for content in response.content] if hasattr(response, 'content') else str(response)}}
        with _log_lock:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
                except (json.JSONDecodeError, IOError):
                    logs = []
            else:
                logs = []
            logs.append(log_entry)
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            save_chat_log_html(log_file)
    except Exception as e:
        print(f'Warning: Failed to log VLM call to {log_file}: {str(e)}')

def log_qwen_call(model, max_tokens, temperature, thinking, messages, response, log_file):
    """
    Log Qwen/OpenAI API calls to a JSON file.
    
    Args:
        model: The model name used for the API call
        max_tokens: Maximum tokens parameter
        temperature: Temperature parameter
        thinking: Thinking parameter (if any)
        messages: List of messages sent to the API
        response: OpenAI API response object
        log_file: Path to the log file
    """
    try:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_entry = {'timestamp': datetime.now().isoformat(), 'model': model, 'max_tokens': max_tokens, 'temperature': temperature, 'thinking': thinking, 'messages': messages, 'response': {'id': getattr(response, 'id', None), 'model': getattr(response, 'model', None), 'created': getattr(response, 'created', None), 'object': getattr(response, 'object', None), 'usage': getattr(response, 'usage', None).__dict__ if hasattr(response, 'usage') and response.usage else None, 'choices': [{'index': choice.index, 'message': {'role': choice.message.role, 'content': choice.message.content}, 'finish_reason': choice.finish_reason} for choice in response.choices] if hasattr(response, 'choices') else []}}
        with _log_lock:
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        logs = json.load(f)
                    if not isinstance(logs, list):
                        logs = []
                except (json.JSONDecodeError, IOError):
                    logs = []
            else:
                logs = []
            logs.append(log_entry)
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
            save_chat_log_html(log_file)
    except Exception as e:
        print(f'Warning: Failed to log Qwen VLM call to {log_file}: {str(e)}')

def call_vlm(vlm_type, max_tokens, temperature, messages, thinking=None, max_retries=3, retry_base_delay=1.0, retry_max_delay=60.0):
    """
    Call the configured VLM with retry logic.
    
    Args:
        vlm_type: Type of VLM to use ("claude", "qwen", "openai", "glmv")
        max_tokens: Maximum tokens to generate
        temperature: Temperature for generation
        messages: List of messages in Claude format
        thinking: Thinking parameter for Claude (optional)
        max_retries: Maximum number of retry attempts (default: 3)
        retry_base_delay: Base delay between retries in seconds (default: 1.0)
        retry_max_delay: Maximum delay between retries in seconds (default: 60.0)
    
    Returns:
        API response object
    """
    model = MODEL_DICT[vlm_type]
    if vlm_type == 'claude':
        return _call_claude_with_retry(model, max_tokens, temperature, messages, thinking, max_retries, retry_base_delay, retry_max_delay)
    elif vlm_type in ['qwen', 'openai', 'glmv']:
        return _call_openai_with_retry(vlm_type, model, max_tokens, temperature, messages, thinking, max_retries, retry_base_delay, retry_max_delay)
    else:
        raise ValueError(f'Invalid VLM type: {vlm_type}')

@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
def _call_claude_api(client, model, max_tokens, temperature, messages, thinking=None):
    """Internal function to make Claude API call - wrapped with retry logic."""
    if thinking is not None:
        return client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature, thinking=thinking, messages=messages)
    else:
        return client.messages.create(model=model, max_tokens=max_tokens, temperature=temperature, messages=messages)

@retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0)
def _call_openai_api(client, model, max_tokens, temperature, openai_messages, thinking=None):
    """Internal function to make OpenAI API call - wrapped with retry logic."""
    return client.chat.completions.create(model=model, max_tokens=max_tokens, temperature=temperature, messages=openai_messages, extra_body={'chat_template_kwargs': {'enable_thinking': thinking is True}})

def _call_claude_with_retry(model, max_tokens, temperature, messages, thinking, max_retries, retry_base_delay, retry_max_delay):
    """Call Claude API with retry logic."""
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    from layout_wo_robot import get_mcp_init_id
    log_file = f'{LOG_DIR}/{get_mcp_init_id()}.json'
    retry_decorator = retry_with_backoff(max_retries=max_retries, base_delay=retry_base_delay, max_delay=retry_max_delay)
    retryable_call = retry_decorator(_call_claude_api)
    response = retryable_call(client, model, max_tokens, temperature, messages, thinking)
    log_vlm_call(model, max_tokens, temperature, thinking, messages, response, log_file)
    return response

def _call_openai_with_retry(vlm_type, model, max_tokens, temperature, messages, thinking, max_retries, retry_base_delay, retry_max_delay):
    """Call OpenAI/Qwen API with retry logic."""
    if OpenAI is None:
        raise ImportError('OpenAI library is not installed. Cannot use Qwen.')
    client = OpenAI(api_key=API_TOKEN, base_url=API_URL_DICT[vlm_type])
    from layout_wo_robot import get_mcp_init_id
    log_file = f'{LOG_DIR}/{get_mcp_init_id()}.json'
    openai_messages = convert_claude_to_openai_messages(messages)
    retry_decorator = retry_with_backoff(max_retries=max_retries, base_delay=retry_base_delay, max_delay=retry_max_delay)
    retryable_call = retry_decorator(_call_openai_api)
    response = retryable_call(client, model, max_tokens, temperature, openai_messages, thinking)
    response = OpenAIResponseAdapter(response)
    log_vlm_call(model, max_tokens, temperature, thinking, messages, response, log_file)
    return response

def chat_log_to_html(log_file):
    """
    Convert JSON chat logs to HTML format with inline CSS styling.
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        str: HTML content of the chat logs
    """
    import base64
    import re
    try:
        if not os.path.exists(log_file):
            return f'<html><body><h1>Log file not found: {log_file}</h1></body></html>'
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        if not isinstance(logs, list):
            return '<html><body><h1>Invalid log format</h1></body></html>'
    except Exception as e:
        return f'<html><body><h1>Error reading log file: {str(e)}</h1></body></html>'
    log_filename = os.path.basename(log_file)
    html_content = f"""\n    <!DOCTYPE html>\n    <html lang="en">\n    <head>\n        <meta charset="UTF-8">\n        <meta name="viewport" content="width=device-width, initial-scale=1.0">\n        <title>VLM Chat Logs</title>\n        <style>\n            body {{\n                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;\n                line-height: 1.6;\n                margin: 0;\n                padding: 20px;\n                background-color: #f5f5f5;\n                color: #333;\n            }}\n            .container {{\n                max-width: 1200px;\n                margin: 0 auto;\n                background: white;\n                border-radius: 8px;\n                box-shadow: 0 2px 10px rgba(0,0,0,0.1);\n                overflow: hidden;\n            }}\n            .header {{\n                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\n                color: white;\n                padding: 20px;\n                text-align: center;\n            }}\n            .log-entry {{\n                border-bottom: 1px solid #eee;\n                padding: 20px;\n                margin-bottom: 0;\n            }}\n            .log-entry:last-child {{\n                border-bottom: none;\n            }}\n            .log-metadata {{\n                background: #f8f9fa;\n                padding: 15px;\n                border-radius: 6px;\n                margin-bottom: 15px;\n                border-left: 4px solid #007bff;\n            }}\n            .metadata-row {{\n                display: inline-block;\n                margin-right: 20px;\n                margin-bottom: 5px;\n            }}\n            .metadata-label {{\n                font-weight: bold;\n                color: #495057;\n            }}\n            .metadata-value {{\n                color: #6c757d;\n                margin-left: 5px;\n            }}\n            .messages-section {{\n                margin: 20px 0;\n            }}\n            .message {{\n                margin: 10px 0;\n                padding: 15px;\n                border-radius: 8px;\n                max-width: 80%;\n            }}\n            .message.user {{\n                background: #e3f2fd;\n                border-left: 4px solid #2196f3;\n                margin-left: auto;\n                margin-right: 0;\n            }}\n            .message.assistant {{\n                background: #f3e5f5;\n                border-left: 4px solid #9c27b0;\n                margin-right: auto;\n                margin-left: 0;\n            }}\n            .message.system {{\n                background: #fff3e0;\n                border-left: 4px solid #ff9800;\n                margin: 10px auto;\n                text-align: center;\n                max-width: 60%;\n            }}\n            .message-role {{\n                font-weight: bold;\n                text-transform: uppercase;\n                font-size: 0.8em;\n                margin-bottom: 8px;\n                opacity: 0.8;\n            }}\n            .message-content {{\n                line-height: 1.6;\n                white-space: pre-wrap;\n                word-wrap: break-word;\n            }}\n            .message-content p {{\n                margin: 8px 0;\n                line-height: 1.6;\n            }}\n            .message-content p:first-child {{\n                margin-top: 0;\n            }}\n            .message-content p:last-child {{\n                margin-bottom: 0;\n            }}\n            .message-image {{\n                max-width: 100%;\n                height: auto;\n                border-radius: 6px;\n                margin: 10px 0;\n                box-shadow: 0 2px 8px rgba(0,0,0,0.15);\n            }}\n            .response-section {{\n                background: #f8f9fa;\n                border-radius: 6px;\n                padding: 15px;\n                margin-top: 15px;\n                border-left: 4px solid #28a745;\n            }}\n            .response-header {{\n                font-weight: bold;\n                color: #495057;\n                margin-bottom: 10px;\n                font-size: 1.1em;\n            }}\n            .usage-info {{\n                background: #e9ecef;\n                padding: 10px;\n                border-radius: 4px;\n                margin: 10px 0;\n                font-family: 'Courier New', monospace;\n                font-size: 0.9em;\n            }}\n            .timestamp {{\n                color: #6c757d;\n                font-size: 0.9em;\n                font-style: italic;\n            }}\n            pre {{\n                background: #f8f9fa;\n                padding: 15px;\n                border-radius: 4px;\n                overflow-x: auto;\n                border-left: 3px solid #dee2e6;\n                white-space: pre-wrap;\n                word-wrap: break-word;\n                line-height: 1.5;\n                font-family: 'Courier New', monospace;\n                font-size: 0.9em;\n            }}\n            code {{\n                background: #f8f9fa;\n                padding: 2px 6px;\n                border-radius: 3px;\n                font-family: 'Courier New', monospace;\n            }}\n        </style>\n    </head>\n    <body>\n        <div class="container">\n            <div class="header">\n                <h1>VLM Chat Logs</h1>\n                <p>Generated from: {log_filename}</p>\n            </div>\n    """
    for (i, log_entry) in enumerate(logs):
        timestamp = log_entry.get('timestamp', 'Unknown')
        model = log_entry.get('model', 'Unknown')
        max_tokens = log_entry.get('max_tokens', 'Unknown')
        temperature = log_entry.get('temperature', 'Unknown')
        messages = log_entry.get('messages', [])
        response = log_entry.get('response', {})
        html_content += f'\n            <div class="log-entry">\n                <div class="log-metadata">\n                    <div class="metadata-row">\n                        <span class="metadata-label">Entry:</span>\n                        <span class="metadata-value">#{i + 1}</span>\n                    </div>\n                    <div class="metadata-row">\n                        <span class="metadata-label">Timestamp:</span>\n                        <span class="metadata-value timestamp">{timestamp}</span>\n                    </div>\n                    <div class="metadata-row">\n                        <span class="metadata-label">Model:</span>\n                        <span class="metadata-value">{model}</span>\n                    </div>\n                    <div class="metadata-row">\n                        <span class="metadata-label">Max Tokens:</span>\n                        <span class="metadata-value">{max_tokens}</span>\n                    </div>\n                    <div class="metadata-row">\n                        <span class="metadata-label">Temperature:</span>\n                        <span class="metadata-value">{temperature}</span>\n                    </div>\n                </div>\n                \n                <div class="messages-section">\n                    <h3>Messages:</h3>\n        '
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            html_content += f'<div class="message {role}">'
            html_content += f'<div class="message-role">{role}</div>'
            html_content += '<div class="message-content">'
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict):
                        content_type = content_item.get('type', 'text')
                        if content_type == 'text':
                            text = content_item.get('text', '')
                            html_content += _format_text_content(text)
                        elif content_type == 'image':
                            image_data = content_item.get('source', {})
                            if image_data.get('type') == 'base64':
                                media_type = image_data.get('media_type', 'image/png')
                                data = image_data.get('data', '')
                                html_content += f'<img src="data:{media_type};base64,{data}" class="message-image" alt="Chat Image" />'
                    else:
                        html_content += _format_text_content(str(content_item))
            else:
                html_content += _format_text_content(str(content))
            html_content += '</div></div>'
        html_content += '\n                </div>\n                \n                <div class="response-section">\n                    <div class="response-header">API Response:</div>\n        '
        if isinstance(response, dict):
            if response.get('id'):
                html_content += f"<p><strong>Response ID:</strong> {response['id']}</p>"
            if response.get('model'):
                html_content += f"<p><strong>Response Model:</strong> {response['model']}</p>"
            usage = response.get('usage')
            if usage:
                html_content += '<div class="usage-info">'
                html_content += '<strong>Token Usage:</strong><br>'
                for (key, value) in usage.items():
                    html_content += f'&nbsp;&nbsp;{key}: {value}<br>'
                html_content += '</div>'
            response_content = response.get('content')
            if response_content:
                html_content += '<div><strong>Response Content:</strong></div>'
                if isinstance(response_content, list):
                    for content_item in response_content:
                        if isinstance(content_item, dict):
                            content_type = content_item.get('type', 'text')
                            if content_type == 'text':
                                text = content_item.get('text', '')
                                formatted_text = _escape_html(text)
                                formatted_text = formatted_text.replace('\n', '<br>')
                                formatted_text = formatted_text.replace('  ', '&nbsp;&nbsp;')
                                html_content += f'<pre>{formatted_text}</pre>'
                        else:
                            formatted_text = _escape_html(str(content_item))
                            formatted_text = formatted_text.replace('\n', '<br>')
                            formatted_text = formatted_text.replace('  ', '&nbsp;&nbsp;')
                            html_content += f'<pre>{formatted_text}</pre>'
                else:
                    formatted_text = _escape_html(str(response_content))
                    formatted_text = formatted_text.replace('\n', '<br>')
                    formatted_text = formatted_text.replace('  ', '&nbsp;&nbsp;')
                    html_content += f'<pre>{formatted_text}</pre>'
        else:
            formatted_text = _escape_html(str(response))
            formatted_text = formatted_text.replace('\n', '<br>')
            formatted_text = formatted_text.replace('  ', '&nbsp;&nbsp;')
            html_content += f'<pre>{formatted_text}</pre>'
        html_content += '</div></div>'
    html_content += '\n        </div>\n    </body>\n    </html>\n    '
    return html_content

def _escape_html(text):
    """Helper function to escape HTML characters"""
    if not isinstance(text, str):
        text = str(text)
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;').replace("'", '&#x27;')

def _format_text_content(text):
    """
    Format text content for HTML display with proper line breaks and spacing.
    
    Args:
        text: Raw text content
        
    Returns:
        str: HTML-formatted text with proper line breaks and spacing
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

def save_chat_log_html(log_file, output_file=None):
    """
    Convert chat logs to HTML and save to file.
    
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
    print(f'Chat logs saved to: {output_file}')
    return output_file

def calculate_vlm_cost(log_file):
    """
    Calculate the cost of VLM API calls from a JSON log file.
    
    Args:
        log_file: Path to the JSON log file
        
    Returns:
        tuple: (num_input_tokens, num_output_tokens)
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        if not isinstance(logs, list):
            print(f'Warning: Log file format unexpected. Expected list, got {type(logs)}')
            return (0, 0)
        num_input_tokens = 0
        num_output_tokens = 0
        for log_entry in logs:
            response = log_entry.get('response', {})
            if not response:
                continue
            usage = response.get('usage')
            if not usage:
                continue
            input_tokens = usage.get('input_tokens', 0) or usage.get('prompt_tokens', 0)
            output_tokens = usage.get('output_tokens', 0) or usage.get('completion_tokens', 0)
            num_input_tokens += input_tokens
            num_output_tokens += output_tokens
        return (num_input_tokens, num_output_tokens)
    except FileNotFoundError:
        print(f'Error: Log file not found: {log_file}')
        return (0, 0)
    except json.JSONDecodeError as e:
        print(f'Error: Invalid JSON in log file {log_file}: {e}')
        return (0, 0)
    except Exception as e:
        print(f'Error calculating VLM cost from {log_file}: {e}')
        return (0, 0)
