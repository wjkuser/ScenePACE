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
from constants import MATFUSE_ROOT_DIR
import os
import importlib.util
import sys
import random
generate_py_path = os.path.join(MATFUSE_ROOT_DIR, 'generate.py')
spec = importlib.util.spec_from_file_location('generate_module', generate_py_path)
generate_module = importlib.util.module_from_spec(spec)
sys.modules['generate_module'] = generate_module
spec.loader.exec_module(generate_module)
generate_texture_map_from_prompt = generate_module.generate_texture_map_from_prompt
generate_texture_map_from_prompt_and_sketch = generate_module.generate_texture_map_from_prompt_and_sketch
generate_texture_map_from_prompt_and_sketch_and_image = generate_module.generate_texture_map_from_prompt_and_sketch_and_image
generate_texture_map_from_prompt_and_color = generate_module.generate_texture_map_from_prompt_and_color
generate_texture_map_from_prompt_and_color_and_sketch = generate_module.generate_texture_map_from_prompt_and_color_and_sketch
generate_texture_map_from_prompt_and_color_palette = generate_module.generate_texture_map_from_prompt_and_color_palette

def material_generate_from_prompt(prompts):
    results = []
    for prompt in prompts:
        texture_map_pil = generate_texture_map_from_prompt(prompt)
        results.append(texture_map_pil)
    return results
