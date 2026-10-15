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
import os
import uuid
import tempfile
import time
import random
from models import Room, Object
import trimesh
from .object_generation import generate_model_from_text
from .object_attribute_inference import infer_attributes_from_claude
import sys
import numpy as np
from constants import RESULTS_DIR

def rotate_wall_mesh(mesh_dict):
    mesh = mesh_dict['mesh']
    mesh_vertices = mesh.vertices.copy()
    mesh_faces = mesh.faces.copy()
    height = mesh_vertices[:, 2].max() - mesh_vertices[:, 2].min()
    length = mesh_vertices[:, 0].max() - mesh_vertices[:, 0].min()
    width = mesh_vertices[:, 1].max() - mesh_vertices[:, 1].min()
    if height < min(length, width):
        mesh_vertices[:, 2] = mesh_vertices[:, 2] - 0.001
        mesh_vertices[:, 2] = mesh_vertices[:, 2] - 0.5 * (mesh_vertices[:, 2].max() + mesh_vertices[:, 2].min())
        temp_y = mesh_vertices[:, 1].copy()
        mesh_vertices[:, 1] = mesh_vertices[:, 2]
        mesh_vertices[:, 2] = -temp_y
        mesh_vertices[:, 2] = mesh_vertices[:, 2] - mesh_vertices[:, 2].min()
        mesh_dict['mesh'] = trimesh.Trimesh(mesh_vertices, mesh_faces)
        object_attributes = infer_attributes_from_claude(mesh_dict, caption=mesh_dict['object_attributes']['given_caption'])
        mesh_dict['object_attributes'] = object_attributes
        scale_factor = object_attributes['height'] / mesh_dict['mesh'].vertices[:, 2].max()
        mesh_dict['mesh'].vertices = mesh_dict['mesh'].vertices * scale_factor
        mesh_dict['mesh'].vertices[:, 2] = mesh_dict['mesh'].vertices[:, 2] + 0.001
        return mesh_dict
    elif width > length:
        temp_x = mesh_dict['mesh'].vertices[:, 0].copy()
        mesh_dict['mesh'].vertices[:, 0] = mesh_dict['mesh'].vertices[:, 1]
        mesh_dict['mesh'].vertices[:, 1] = -temp_x
        return mesh_dict
    else:
        return mesh_dict

def get_object_candidates(object_info: dict, source: str='generation', exclude_asset_ids=None, top_k: int=3):
    object_type = object_info['type']
    object_description = object_info['description']
    object_location = object_info['location']
    object_size = object_info['size']
    object_limit_size = object_info.get('limit_size', False)
    object_random_id = str(uuid.uuid4())[:8]
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, f'generated_object_{object_random_id}.glb')
    caption = f'{object_description}'
    time.sleep(random.random() * 4)
    mesh_dict = generate_model_from_text(caption, temp_file_path, reference_object_size=object_size)
    if object_location == 'wall':
        mesh_dict = rotate_wall_mesh(mesh_dict)
    if object_limit_size:
        print(f'limiting size to {object_size}', file=sys.stderr)
        generated_width = mesh_dict['mesh'].vertices[:, 0].max() - mesh_dict['mesh'].vertices[:, 0].min()
        generated_length = mesh_dict['mesh'].vertices[:, 1].max() - mesh_dict['mesh'].vertices[:, 1].min()
        generated_height = mesh_dict['mesh'].vertices[:, 2].max() - mesh_dict['mesh'].vertices[:, 2].min()
        print(f'generated_width: {generated_width}, generated_length: {generated_length}, generated_height: {generated_height}', file=sys.stderr)
        limit_width = object_size[0] / 100.0
        limit_length = object_size[1] / 100.0
        limit_height = object_size[2] / 100.0
        if generated_width > limit_width:
            scale_factor = limit_width / generated_width * np.random.uniform(0.9, 0.99)
            mesh_dict['mesh'].vertices[:, 0] = mesh_dict['mesh'].vertices[:, 0] * scale_factor
        if generated_length > limit_length:
            scale_factor = limit_length / generated_length * np.random.uniform(0.9, 0.99)
            mesh_dict['mesh'].vertices[:, 1] = mesh_dict['mesh'].vertices[:, 1] * scale_factor
        if generated_height > limit_height:
            scale_factor = limit_height / generated_height * np.random.uniform(0.9, 0.99)
            mesh_dict['mesh'].vertices[:, 2] = mesh_dict['mesh'].vertices[:, 2] * scale_factor
        print(f'reshape object if size is too small', file=sys.stderr)
        generated_width = mesh_dict['mesh'].vertices[:, 0].max() - mesh_dict['mesh'].vertices[:, 0].min()
        generated_length = mesh_dict['mesh'].vertices[:, 1].max() - mesh_dict['mesh'].vertices[:, 1].min()
        generated_height = mesh_dict['mesh'].vertices[:, 2].max() - mesh_dict['mesh'].vertices[:, 2].min()
        limit_width = object_size[0] / 100.0
        limit_length = object_size[1] / 100.0
        limit_height = object_size[2] / 100.0
        if generated_width < 0.9 * limit_width:
            mesh_dict['mesh'].vertices[:, 0] = mesh_dict['mesh'].vertices[:, 0] * (limit_width * np.random.uniform(0.9, 0.99)) / generated_width
        if generated_length < 0.9 * limit_length:
            mesh_dict['mesh'].vertices[:, 1] = mesh_dict['mesh'].vertices[:, 1] * (limit_length * np.random.uniform(0.9, 0.99)) / generated_length
        if generated_height < 0.9 * limit_height:
            mesh_dict['mesh'].vertices[:, 2] = mesh_dict['mesh'].vertices[:, 2] * (limit_height * np.random.uniform(0.9, 0.99)) / generated_height
        generated_width = mesh_dict['mesh'].vertices[:, 0].max() - mesh_dict['mesh'].vertices[:, 0].min()
        generated_length = mesh_dict['mesh'].vertices[:, 1].max() - mesh_dict['mesh'].vertices[:, 1].min()
        generated_height = mesh_dict['mesh'].vertices[:, 2].max() - mesh_dict['mesh'].vertices[:, 2].min()
        print(f'generated_width: {generated_width}, generated_length: {generated_length}, generated_height: {generated_height}', file=sys.stderr)
    os.remove(temp_file_path)
    return [{'source': 'generation', 'source_id': object_random_id, 'mesh': mesh_dict['mesh'], 'texture': mesh_dict['texture'], 'tex_coords': mesh_dict['tex_coords'], 'mass': float(mesh_dict['object_attributes']['weight']), 'pbr_parameters': mesh_dict['object_attributes']['pbr_parameters']}]

def get_object_mesh(source, source_id, layout_id):
    object_save_path = f'{RESULTS_DIR}/{layout_id}/{source}/{source_id}.ply'
    if os.path.exists(object_save_path):
        return trimesh.load(object_save_path)
    else:
        return None
