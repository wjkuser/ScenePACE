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
"""Credentials and service URLs supplied by the launcher or environment."""
import json, os
from pathlib import Path
key_dict = {}
if os.environ.get('SAGE_KEY_FILE'):
    key_dict = json.loads(Path(os.environ['SAGE_KEY_FILE']).read_text())
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', key_dict.get('ANTHROPIC_API_KEY', 'EMPTY'))
API_TOKEN = os.environ.get('SAGE_VLM_API_KEY', key_dict.get('API_TOKEN', 'EMPTY'))
API_URL_QWEN = os.environ.get('SAGE_VLM_BASE_URL', key_dict.get('API_URL_QWEN', 'http://127.0.0.1:1081/v1'))
API_URL_OPENAI = API_URL_QWEN
API_URL_DICT = {'qwen': API_URL_QWEN, 'openai': API_URL_QWEN}
MODEL_DICT = {k: os.environ.get('SAGE_VLM_MODEL', 'qwen-vl-local') for k in ('qwen', 'openai', 'claude', 'glmv')}
SERVER_URL = os.environ.get('TEXT_TO_3D_SERVER_URL', 'http://127.0.0.1:10070')
FLUX_SERVER_URL = os.environ.get('FLUX_SERVER_URL', SERVER_URL)
