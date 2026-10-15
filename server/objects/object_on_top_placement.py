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
import trimesh
import tempfile
from concurrent.futures import ThreadPoolExecutor
from constants import SERVER_ROOT_DIR, RESULTS_DIR, PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED
from models import FloorPlan, Object, Room
from tex_utils import get_textured_object_mesh, get_textured_object_mesh_from_object, apply_object_transform_direct
from vlm import call_vlm
from utils import extract_json_from_response
import numpy as np
from datetime import datetime
import json
from isaacsim.isaac_mcp.server import test_object_placements_batch_in_single_room, test_object_placements_in_single_room
from typing import Any, Dict, List, Optional, Sequence, Tuple
import open3d as o3d
import sys
import hashlib
import time
from dataclasses import asdict
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R

class PlacementCollisionContext:
    """Ray intersectors built once and reused for one object's candidates."""

    def __init__(self, mesh_list: Sequence[trimesh.Trimesh]):
        if not mesh_list:
            raise ValueError('At least one support mesh is required')
        self.base_face_count = mesh_list[0].faces.shape[0]
        self.base_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh_list[0])
        self.combined_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh.util.concatenate(mesh_list)) if len(mesh_list) > 1 else None
        self.collision_intersectors = [self.base_intersector] + [trimesh.ray.ray_pyembree.RayMeshIntersector(mesh) for mesh in mesh_list[1:]]

def get_random_placements_on_target_object(layout: FloorPlan, room: Room, target_object_id: str, object_to_place: Object, sample_count: int=100, place_location: str=None, layout_file_name: str=None, regular_rotation: bool=False, random_seed: Optional[int]=None, reuse_collision_context: bool=False, max_accepted_candidates: Optional[int]=None):
    if max_accepted_candidates is not None:
        max_accepted_candidates = int(max_accepted_candidates)
        if max_accepted_candidates <= 0:
            raise ValueError('max_accepted_candidates must be positive')
    room_id = room.id
    mesh_info_dict = get_textured_object_mesh(layout, room, room_id, target_object_id)
    mesh = mesh_info_dict['mesh']
    vts = mesh_info_dict['texture']['vts']
    fts = mesh_info_dict['texture']['fts']
    texture_map_path = mesh_info_dict['texture']['texture_map_path']
    mesh_and_objects_on_it_list = [mesh]
    mesh_and_objects_on_it_list_ids = [target_object_id]
    for object_in_room in room.objects:
        object_in_room_place_id_original = object_in_room.id
        while True:
            object_in_room_place_id = object_in_room.place_id
            if object_in_room_place_id == 'floor' or object_in_room_place_id == 'wall':
                break
            elif object_in_room_place_id == target_object_id:
                mesh_and_objects_on_it_list.append(get_textured_object_mesh(layout, room, room_id, object_in_room_place_id_original)['mesh'])
                mesh_and_objects_on_it_list_ids.append(object_in_room_place_id_original)
                break
            else:
                object_in_room = next((object for object in room.objects if object.id == object_in_room_place_id), None)
                if object_in_room is None:
                    raise ValueError(f'Object with id {object_in_room_place_id} not found')
    face_normals = mesh.face_normals.reshape(-1, 3)
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)
    up_axis = np.array([0, 0, 1]).reshape(1, 3)
    support_faces = face_normals @ up_axis.T > 0.5
    support_faces_idxs = np.where(support_faces)[0]
    support_mesh = mesh.submesh([support_faces_idxs], append=True)
    rng = np.random.default_rng(random_seed) if random_seed is not None else None
    (samples, sample_face_idxs) = trimesh.sample.sample_surface(support_mesh, sample_count, seed=rng)
    sample_face_normals = support_mesh.face_normals[sample_face_idxs]
    placement_locations = samples + sample_face_normals * 0.01
    if regular_rotation:
        regular_angles = np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        placement_z_rotations = np.random.choice(regular_angles, size=sample_count) if rng is None else rng.choice(regular_angles, size=sample_count)
    else:
        placement_z_rotations = np.random.uniform(0, 2 * np.pi, sample_count) if rng is None else rng.uniform(0, 2 * np.pi, sample_count)
    object_to_place_mesh_dict = get_textured_object_mesh_from_object(layout, object_to_place)
    object_to_place_mesh = object_to_place_mesh_dict['mesh']
    collision_context = PlacementCollisionContext(mesh_and_objects_on_it_list) if reuse_collision_context else None
    target_object = next((object for object in room.objects if object.id == target_object_id), None)
    detailed_prompt = f'You are an expert interior designer analyzing object placement relationships. You need to determine the most appropriate placement location for one object relative to another object.\n\nPLACEMENT OPTIONS:\n1. "top" - The object should be placed ON TOP of the target object\n2. "inside" - The object should be placed INSIDE the target object  \n3. "both" - Either placement is acceptable with no strong preference\n\nTARGET OBJECT INFORMATION:\n- Type: {target_object.type}\n- Description: {target_object.description}\n- Dimensions: {target_object.dimensions.width}m × {target_object.dimensions.length}m × {target_object.dimensions.height}m\n- Place ID: {target_object.place_id}\n- Place Guidance: {target_object.place_guidance}\n\nOBJECT TO PLACE INFORMATION:\n- Type: {object_to_place.type}\n- Description: {object_to_place.description}\n- Dimensions: {object_to_place.dimensions.width}m × {object_to_place.dimensions.length}m × {object_to_place.dimensions.height}m\n- Place Guidance: {object_to_place.place_guidance}\n\nPLACEMENT EXAMPLES:\n\nTOP PLACEMENT (place_location: "top"):\n- Laptop on top of desk\n- Pillow on top of sofa/bed\n- Vase on top of table\n- Book on top of nightstand\n- Remote control on top of coffee table\n- Plate on top of dining table\n- Lamp on top of side table\n- Phone on top of counter\n- Decorative items on top of shelf\n- Cup on top of coaster\n\nINSIDE PLACEMENT (place_location: "inside"):\n- Books inside bookshelf\n- Food inside refrigerator\n- Clothes inside wardrobe/closet\n- Dishes inside dishwasher\n- Toiletries inside medicine cabinet\n- Files inside filing cabinet\n- Shoes inside shoe rack\n- Utensils inside drawer\n- Groceries inside pantry\n- Tools inside toolbox\n\nBOTH PLACEMENT (place_location: "both"):\n- Storage containers on/in storage shelf\n- Decorative items on/in display cabinet\n- Linens on/in linen closet\n- Office supplies on/in desk organizer\n- Cleaning supplies on/in utility cabinet\n- Toys on/in toy chest\n- Craft supplies on/in craft cabinet\n- Personal items on/in dresser\n\nDECISION CRITERIA:\n- Consider the functional relationship between objects\n- Think about typical usage patterns and accessibility\n- Consider safety and stability\n- Think about the intended purpose of both objects\n- Consider size compatibility and physical constraints\n- Follow common interior design and organizational principles\n\nAnalyze the target object and object to place, then determine the most appropriate placement location.\n\nReturn your response as a JSON object with this exact format:\n```json\n{{\n    "place_location": "top|inside|both",\n    "reason": "Brief explanation of why this placement choice makes the most sense based on the object types, dimensions, functionality, and typical usage patterns"\n}}\n```\n'
    if place_location is None:
        claude_response = call_vlm(vlm_type='openai', max_tokens=4000, temperature=0.2, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': detailed_prompt}]}])
        response_text = claude_response.content[0].text
        response_text = response_text.strip()
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        response = json.loads(response_text)
        place_location = response['place_location']
    placements = []
    for i in range(sample_count):
        placement_location = placement_locations[i]
        placement_z_rotation = placement_z_rotations[i]
        placement = {'position': {'x': float(placement_location[0]), 'y': float(placement_location[1]), 'z': float(placement_location[2])}, 'rotation': {'x': 0.0, 'y': 0.0, 'z': float(placement_z_rotation)}}
        if filter_placements_by_support_rays_and_collisions(mesh_and_objects_on_it_list, object_to_place_mesh, placement, place_location=place_location, collision_context=collision_context):
            placements.append(placement)
            if max_accepted_candidates is not None and (not False) and (len(placements) >= max_accepted_candidates):
                break
    if max_accepted_candidates is not None:
        placements = placements[:max_accepted_candidates]
    return placements

def derive_on_object_candidate_seed(base_seed: int, item_index: int, object_id: str, target_object_id: str) -> int:
    """Derive a stable, independent RNG seed for one placement task."""
    payload = f'{base_seed}\x00{item_index}\x00{object_id}\x00{target_object_id}'.encode('utf-8')
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), 'little')

def generate_on_object_candidate_batch(layout: FloorPlan, room: Room, items: Sequence[Tuple[Object, str]], *, sample_count: int=150, regular_rotation: bool=False, place_location: str='both', max_workers: int=1, base_seed: int=0, reuse_collision_context: bool=False, max_accepted_candidates: Optional[int]=None) -> List[Dict[str, Any]]:
    """Generate independent candidate sets while preserving input order."""
    if not items:
        return []
    worker_count = max(1, min(len(items), int(max_workers)))

    def generate(item_index: int, item: Tuple[Object, str]) -> Dict[str, Any]:
        (object_to_place, target_object_id) = item
        random_seed = derive_on_object_candidate_seed(base_seed, item_index, object_to_place.id, target_object_id)
        started = time.perf_counter()
        try:
            placements = get_random_placements_on_target_object(layout, room, target_object_id, object_to_place, sample_count=sample_count, regular_rotation=regular_rotation, place_location=place_location, random_seed=random_seed, reuse_collision_context=reuse_collision_context, max_accepted_candidates=max_accepted_candidates)
            error = None
        except Exception as exc:
            placements = []
            error = f'{type(exc).__name__}: {exc}'
            print(f'Candidate generation failed for {object_to_place.id}: {error}', file=sys.stderr)
        return {'object_id': object_to_place.id, 'target_object_id': target_object_id, 'random_seed': random_seed, 'placements': placements, 'error': error, 'duration_s': time.perf_counter() - started}
    indexed_items = list(enumerate(items))
    if worker_count == 1:
        return [generate(item_index, item) for (item_index, item) in indexed_items]
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix='sage-on-object-candidates') as executor:
        futures = [executor.submit(generate, item_index, item) for (item_index, item) in indexed_items]
        return [future.result() for future in futures]

def filter_placements_by_support_rays_and_collisions(target_object_mesh_list: List[trimesh.Trimesh], object_to_place_mesh: trimesh.Trimesh, placement: Dict, place_location='both', collision_context: Optional[PlacementCollisionContext]=None):
    placement_location = placement['position']
    placement_rotation = placement['rotation']
    transformed_object_to_place_mesh = apply_object_transform_direct(object_to_place_mesh, placement_location, placement_rotation)
    transformed_vertices = transformed_object_to_place_mesh.vertices
    ray_origins = transformed_vertices
    ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)
    base_intersector = collision_context.base_intersector if collision_context is not None else trimesh.ray.ray_pyembree.RayMeshIntersector(target_object_mesh_list[0])
    (locations, index_ray, index_tri) = base_intersector.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
    num_vertices = len(transformed_vertices)
    intersected_num_rays = index_ray.shape[0]
    if intersected_num_rays < num_vertices:
        return False
    if len(target_object_mesh_list) > 1:
        ray_origins = transformed_vertices
        ray_directions = np.array([0, 0, -1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)
        combined_intersector = collision_context.combined_intersector if collision_context is not None else trimesh.ray.ray_pyembree.RayMeshIntersector(trimesh.util.concatenate(target_object_mesh_list))
        (locations, index_ray, index_tri) = combined_intersector.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
        base_face_count = collision_context.base_face_count if collision_context is not None else target_object_mesh_list[0].faces.shape[0]
        if np.max(index_tri) >= base_face_count:
            return False
    if place_location == 'top':
        ray_origins = transformed_vertices
        ray_directions = np.array([0, 0, 1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)
        (locations, index_ray, index_tri) = base_intersector.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
        num_vertices = len(transformed_vertices)
        intersected_num_rays = index_ray.shape[0]
        if intersected_num_rays > 0:
            return False
    elif place_location == 'inside':
        ray_origins = transformed_vertices
        ray_directions = np.array([0, 0, 1]).reshape(1, 3).repeat(len(transformed_vertices), axis=0)
        (locations, index_ray, index_tri) = base_intersector.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)
        num_vertices = len(transformed_vertices)
        intersected_num_rays = index_ray.shape[0]
        if intersected_num_rays == 0:
            return False
    is_collided = detect_collision(target_object_mesh_list, transformed_object_to_place_mesh, collision_intersectors=collision_context.collision_intersectors if collision_context is not None else None)
    if is_collided:
        return False
    return True

def detect_collision(base_meshes, test_mesh, collision_intersectors=None):
    """
    Detect collisions between a test mesh and a series of base meshes.
    Uses edge-based ray casting to detect intersections.

    Parameters:
    -----------
    base_meshes : List of trimesh.Trimesh
        List of base meshes to check against.


    test_mesh : trimesh.Trimesh
        The mesh to test for collisions.

    Returns:
    --------
    is_collided: bool
        True if the test mesh is collided with any of the base meshes, False otherwise
    """
    edges = test_mesh.edges_unique
    edge_points = test_mesh.vertices[edges]
    ray_origins = edge_points[:, 0]
    ray_directions = edge_points[:, 1] - edge_points[:, 0]
    ray_lengths = np.linalg.norm(ray_directions, axis=1)
    ray_directions = ray_directions / ray_lengths[:, np.newaxis]
    if collision_intersectors is None:
        collision_intersectors = [trimesh.ray.ray_pyembree.RayMeshIntersector(base_trimesh) for base_trimesh in base_meshes]
    is_collided = False
    for intersector in collision_intersectors:
        (locations, index_ray, index_tri) = intersector.intersects_location(ray_origins=ray_origins, ray_directions=ray_directions)
        if len(locations) > 0:
            distances = np.linalg.norm(locations - ray_origins[index_ray], axis=1)
            valid_indices = distances <= ray_lengths[index_ray]
            if np.any(valid_indices):
                is_collided = True
                break
    return is_collided

def filter_placements_by_physics_critic(layout: FloorPlan, room: Room, object: Object, placements: list):
    room_id = room.id
    placements_info = {'placements': placements, 'object': {'source': object.source, 'source_id': object.source_id, 'mass': getattr(object, 'mass', 1.0)}}
    if not PHYSICS_CRITIC_ENABLED:
        return placements
    layout_id = layout.id
    scene_save_dir = f'{RESULTS_DIR}/{layout_id}'
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        placements_info_path = f.name
        json.dump(placements_info, f, indent=4)
    room_dict = asdict(room)
    room_dict_save_path = os.path.join(scene_save_dir, f'{room_id}.json')
    with open(room_dict_save_path, 'w') as f:
        json.dump(room_dict, f, indent=4)
    sim_result = test_object_placements_in_single_room(room_dict_save_path, placements_info_path, only_need_one=True)
    safe_placements_path = sim_result['safe_placements_path']
    with open(safe_placements_path, 'r') as f:
        safe_placements = json.load(f)
    if os.path.exists(placements_info_path):
        os.remove(placements_info_path)
    if os.path.exists(safe_placements_path):
        os.remove(safe_placements_path)
    return safe_placements

def filter_placements_batch_by_physics_critic(layout: FloorPlan, room: Room, object_placements: Sequence[Tuple[Object, Sequence[Dict[str, Any]]]], parallel_width: int=16, max_candidates_per_object: int=10, scheduling_policy: str='work_conserving') -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Evaluate one tool call's candidates with one reusable Isaac scene build."""
    grouped = {obj.id: [] for (obj, _) in object_placements}
    if not PHYSICS_CRITIC_ENABLED:
        for (obj, placements) in object_placements:
            grouped[obj.id] = [{'object_id': obj.id, 'candidate_index': candidate_index, 'safe': True, 'invalid': False, 'position': placement['position'], 'rotation': placement['rotation'], 'position_delta_m': 0.0, 'rotation_delta_deg': 0.0, 'pass_index': 0, 'simulation_steps': 0, 'stability_cost': float(candidate_index) * 1e-09} for (candidate_index, placement) in enumerate(list(placements)[:max_candidates_per_object])]
        return (grouped, {'scene_build_count': 0, 'candidate_count': sum((len(values) for values in grouped.values())), 'safe_count': sum((len(values) for values in grouped.values())), 'physics_disabled': True})
    scene_save_dir = os.path.join(RESULTS_DIR, layout.id)
    os.makedirs(scene_save_dir, exist_ok=True)
    room_dict_save_path = os.path.join(scene_save_dir, f'{room.id}.json')
    with open(room_dict_save_path, 'w') as room_file:
        json.dump(asdict(room), room_file, indent=4)
    payload = [{'object_id': obj.id, 'source': obj.source, 'source_id': obj.source_id, 'mass': float(getattr(obj, 'mass', 1.0)), 'placements': list(placements)[:max_candidates_per_object]} for (obj, placements) in object_placements]
    response = test_object_placements_batch_in_single_room(room_dict_save_path, payload, parallel_width=parallel_width, max_candidates_per_object=max_candidates_per_object, scheduling_policy=scheduling_policy)
    if not isinstance(response, dict) or response.get('status') != 'success':
        raise RuntimeError(f'Batched Isaac placement check failed: {response}')
    expected = {(obj.id, candidate_index) for (obj, placements) in object_placements for (candidate_index, _) in enumerate(list(placements)[:max_candidates_per_object])}
    received = set()
    for result in response.get('results', []):
        try:
            object_id = str(result['object_id'])
            candidate_index = int(result['candidate_index'])
        except (KeyError, TypeError, ValueError):
            continue
        key = (object_id, candidate_index)
        if key not in expected or key in received:
            continue
        received.add(key)
        finite_pose = False
        try:
            pose_values = [float(result[section][axis]) for section in ('position', 'rotation') for axis in ('x', 'y', 'z')]
            finite_pose = bool(np.all(np.isfinite(pose_values)))
        except (KeyError, TypeError, ValueError):
            pass
        normalized = dict(result)
        normalized['safe'] = bool(result.get('safe')) and finite_pose
        normalized['invalid'] = bool(result.get('invalid')) or not finite_pose
        grouped[object_id].append(normalized)
    for values in grouped.values():
        values.sort(key=lambda value: int(value['candidate_index']))
    metrics = dict(response.get('metrics', {}))
    metrics.update({'expected_result_count': len(expected), 'received_result_count': len(received), 'missing_result_count': len(expected - received)})
    return (grouped, metrics)

def _load_candidate_collision_mesh(layout: FloorPlan, obj: Object) -> trimesh.Trimesh:
    object_dir = os.path.join(RESULTS_DIR, layout.id, obj.source)
    mesh_path = os.path.join(object_dir, f'{obj.source_id}.ply')
    watertight_path = os.path.join(object_dir, f'{obj.source_id}_wt.ply')
    if obj.source == 'asset_library' and os.path.exists(watertight_path):
        mesh_path = watertight_path
    loaded = trimesh.load(mesh_path, force='mesh')
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise ValueError(f'Unable to load collision mesh for {obj.id}: {mesh_path}')
    return loaded

def _transform_candidate_mesh(mesh: trimesh.Trimesh, placement: Dict[str, Any]) -> trimesh.Trimesh:
    transformed = mesh.copy()
    rotation = R.from_euler('xyz', [placement['rotation'][axis] for axis in ('x', 'y', 'z')], degrees=False).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = [placement['position'][axis] for axis in ('x', 'y', 'z')]
    transformed.apply_transform(transform)
    return transformed

def _candidate_meshes_conflict(first: Dict[str, Any], second: Dict[str, Any], clearance: float) -> bool:
    first_bounds = first['_mesh'].bounds
    second_bounds = second['_mesh'].bounds
    overlap = np.minimum(first_bounds[1], second_bounds[1]) - np.maximum(first_bounds[0], second_bounds[0])
    if np.any(overlap < -clearance):
        return False
    try:
        return bool(detect_collision([first['_mesh']], second['_mesh'], collision_intersectors=[first['_intersector']]) or detect_collision([second['_mesh']], first['_mesh'], collision_intersectors=[second['_intersector']]))
    except Exception:
        return True

def select_maximum_stability_placements(layout: FloorPlan, objects: Sequence[Object], candidates_by_object: Dict[str, Sequence[Dict[str, Any]]], beam_width: int=512, clearance: float=0.0001) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Prefer maximum placed count, then minimum physical motion."""
    prepared: Dict[str, List[Dict[str, Any]]] = {}
    preparation_errors = {}
    for obj in objects:
        safe_values = [dict(value) for value in candidates_by_object.get(obj.id, []) if bool(value.get('safe')) and (not bool(value.get('invalid')))]
        try:
            source_mesh = _load_candidate_collision_mesh(layout, obj)
            prepared_values = []
            for value in safe_values:
                transformed = _transform_candidate_mesh(source_mesh, value)
                prepared_value = dict(value)
                prepared_value['_mesh'] = transformed
                prepared_value['_intersector'] = trimesh.ray.ray_pyembree.RayMeshIntersector(transformed)
                prepared_values.append(prepared_value)
            prepared[obj.id] = sorted(prepared_values, key=lambda value: (float(value.get('stability_cost', float('inf'))), int(value['candidate_index'])))
        except Exception as error:
            prepared[obj.id] = []
            preparation_errors[obj.id] = str(error)
    ordered_objects = sorted(objects, key=lambda obj: (len(prepared[obj.id]), objects.index(obj)))
    conflict_cache = {}

    def conflicts(candidate: Dict[str, Any], selected: Sequence[Dict[str, Any]]) -> bool:
        for other in selected:
            key = tuple(sorted(((candidate['object_id'], int(candidate['candidate_index'])), (other['object_id'], int(other['candidate_index'])))))
            if key not in conflict_cache:
                conflict_cache[key] = _candidate_meshes_conflict(candidate, other, clearance)
            if conflict_cache[key]:
                return True
        return False
    beam_width = max(1, min(8192, int(beam_width)))
    states = [([], 0.0)]
    for obj in ordered_objects:
        next_states = []
        for (selected, stability_cost) in states:
            next_states.append((selected, stability_cost))
            for candidate in prepared[obj.id]:
                if not conflicts(candidate, selected):
                    next_states.append((selected + [candidate], stability_cost + float(candidate.get('stability_cost', float('inf')))))
        next_states.sort(key=lambda state: (-len(state[0]), state[1], tuple(((value['object_id'], int(value['candidate_index'])) for value in state[0]))))
        states = next_states[:beam_width]
    (best_selected, best_cost) = states[0]
    selected_by_object = {}
    for value in best_selected:
        selected_by_object[value['object_id']] = {key: item for (key, item) in value.items() if not key.startswith('_')}
    diagnostics = {'method': 'count_first_stability_beam', 'beam_width': beam_width, 'attempted_object_count': len(objects), 'objects_with_safe_candidates': sum((bool(prepared[obj.id]) for obj in objects)), 'safe_candidate_count': sum((len(values) for values in prepared.values())), 'selected_object_count': len(selected_by_object), 'selected_stability_cost': float(best_cost), 'candidate_conflict_checks': len(conflict_cache), 'candidate_conflict_count': sum(conflict_cache.values()), 'selection_order': [obj.id for obj in ordered_objects], 'mesh_preparation_errors': preparation_errors}
    return (selected_by_object, diagnostics)
