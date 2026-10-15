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
from objects.get_objects import get_object_candidates
from models import Room, Object, Point3D, Dimensions, FloorPlan, Euler
from typing import Dict, List, Optional, Set
import uuid
import os
import numpy as np
import pickle
from PIL import Image
from constants import RESULTS_DIR
import sys
import torch
from pytorch3d.io.obj_io import _save
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
_EXPORT_LOCKS: Dict[str, threading.Lock] = {}
_EXPORT_LOCKS_GUARD = threading.Lock()

def _enabled(name: str, default: bool=True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}

def _export_lock(path: str) -> threading.Lock:
    with _EXPORT_LOCKS_GUARD:
        return _EXPORT_LOCKS.setdefault(path, threading.Lock())

def _save_obj_reusing_texture(obj_path: str, mesh, tex_coords: dict, texture_path: str) -> None:
    """Write the same textured OBJ geometry while referencing an existing PNG."""
    mtl_path = os.path.splitext(obj_path)[0] + '.mtl'
    with open(obj_path, 'w', encoding='utf-8') as obj_file:
        obj_file.write(f'\nmtllib {os.path.basename(mtl_path)}\nusemtl mesh\n\n')
        _save(obj_file, torch.from_numpy(mesh.vertices), torch.from_numpy(mesh.faces), verts_uvs=torch.from_numpy(tex_coords['vts']), faces_uvs=torch.from_numpy(tex_coords['fts']), save_texture=True)
    texture_reference = os.path.relpath(texture_path, os.path.dirname(obj_path)).replace(os.sep, '/')
    with open(mtl_path, 'w', encoding='utf-8') as mtl_file:
        mtl_file.write(f'newmtl mesh\nmap_Kd {texture_reference}\n')

def _export_candidate_assets(object_save_dir: str, obj: Object, candidate: dict) -> None:
    asset_dir = os.path.join(object_save_dir, obj.source)
    ply_path = os.path.join(asset_dir, f'{obj.source_id}.ply')
    with _export_lock(ply_path):
        if os.path.exists(ply_path):
            return
        os.makedirs(asset_dir, exist_ok=True)
        candidate['mesh'].export(ply_path)
        watertight = candidate.get('mesh_watertight')
        if watertight is not None:
            watertight.export(os.path.join(asset_dir, f'{obj.source_id}_wt.ply'))
        texture_path = os.path.join(asset_dir, f'{obj.source_id}_texture.png')
        if not os.path.exists(texture_path):
            Image.fromarray((candidate['texture'] * 255).astype(np.uint8)).save(texture_path)
        tex_coords_path = os.path.join(asset_dir, f'{obj.source_id}_tex_coords.pkl')
        if not os.path.exists(tex_coords_path):
            with open(tex_coords_path, 'wb') as handle:
                pickle.dump(candidate['tex_coords'], handle)
        if 'pbr_parameters' in candidate:
            pbr_path = os.path.join(asset_dir, f'{obj.source_id}_pbr_parameters.json')
            if not os.path.exists(pbr_path):
                with open(pbr_path, 'w', encoding='utf-8') as handle:
                    json.dump(candidate['pbr_parameters'], handle)
        _save_obj_reusing_texture(os.path.join(asset_dir, f'{obj.source_id}.obj'), candidate['mesh'], candidate['tex_coords'], texture_path)

def process_single_object(object_name: str, object_info: dict, room: Room, object_save_dir: str, selection_source='generation'):
    """
    Process a single object type and return selected objects and updated recommendations.
    
    Args:
        object_name: Name of the object type
        object_info: Dictionary containing object information
        room: Room object with dimensions
        object_save_dir: Directory to save object files

    Returns:
        tuple: (selected_objects, updated_recommendations) for this object type
    """
    print(f'Selecting objects for {object_name}', file=sys.stderr)
    object_info['type'] = object_name
    object_description = object_info['description']
    object_location = object_info['location']
    object_size = object_info['size']
    object_quantity = object_info['quantity']
    object_variance_type = object_info['variance_type']
    object_place_guidance = object_info.get('place_guidance', f'Standard placement for {object_name}')
    candidates = get_object_candidates(object_info, selection_source)
    target_size_m = [s / 100.0 for s in object_size]
    room_dims = room.dimensions
    filtered_candidates = candidates * object_quantity

    def size_similarity_score(candidate):
        mesh = candidate['mesh']
        bounds = mesh.bounds
        mesh_dims = bounds[1] - bounds[0]
        score = 0
        for i in range(3):
            score += abs(mesh_dims[i] - target_size_m[i])
        return score
    filtered_candidates.sort(key=size_similarity_score)
    selected_candidates = filtered_candidates[:object_quantity]
    selected_objects = []
    updated_recommendations = []
    for (i, candidate) in enumerate(selected_candidates):
        mesh = candidate['mesh']
        bounds = mesh.bounds
        mesh_dims = bounds[1] - bounds[0]
        unique_object_id = f'{room.id}_{object_name}_{str(uuid.uuid4())[:8]}'
        obj = Object(id=unique_object_id, room_id=room.id, type=object_name, description=object_description, position=Point3D(0, 0, 0), rotation=Euler(0, 0, 0), dimensions=Dimensions(width=mesh_dims[0], length=mesh_dims[1], height=mesh_dims[2]), source=candidate['source'], source_id=candidate['source_id'], place_id=object_location, place_guidance=object_place_guidance, mass=candidate.get('mass', 1.0))
        selected_objects.append(obj)
        object_info_wo_quantity = {key: value for (key, value) in object_info.items() if not key.startswith('_')}
        object_info_wo_quantity.update({'object_id': unique_object_id, 'type': object_name, 'quantity': 1, 'size': [mesh_dims[0] * 100, mesh_dims[1] * 100, mesh_dims[2] * 100], 'actual_source': candidate['source'], 'actual_source_id': candidate['source_id']})
        updated_recommendations.append(object_info_wo_quantity)
        _export_candidate_assets(object_save_dir, obj, candidate)
    return (selected_objects, updated_recommendations)

def select_objects(object_info_dict: dict, room: Room, existing_objects: List[Object], current_layout: FloorPlan, selection_source='generation'):
    object_save_dir = f'{RESULTS_DIR}/{current_layout.id}'
    os.makedirs(object_save_dir, exist_ok=True)
    selected_objects = []
    updated_recommendations = []
    object_names_order = list(object_info_dict.keys())
    with ThreadPoolExecutor(max_workers=int(os.environ.get('SAGE_ASSET_WORKERS', '2'))) as executor:
        future_to_object_name = {}
        for object_name in object_names_order:
            object_info = object_info_dict[object_name]
            future = executor.submit(process_single_object, object_name, object_info, room, object_save_dir, selection_source)
            future_to_object_name[future] = object_name
        results = {}
        for future in as_completed(future_to_object_name):
            object_name = future_to_object_name[future]
            try:
                (objects, recommendations) = future.result()
                results[object_name] = (objects, recommendations)
                print(f'Object {object_name} selected {len(objects)} objects', file=sys.stderr)
            except Exception as exc:
                print(f'Object {object_name} generated an exception: {exc}', file=sys.stderr)
                results[object_name] = ([], [])
        for object_name in object_names_order:
            if object_name in results:
                (objects, recommendations) = results[object_name]
                selected_objects.extend(objects)
                updated_recommendations.extend(recommendations)
    return (selected_objects, updated_recommendations)
