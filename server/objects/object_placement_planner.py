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
from models import Object, Room, FloorPlan, Point3D, Euler, Door, Window
from typing import List, Dict, Any, Optional, Tuple
import json
from key import ANTHROPIC_API_KEY
from vlm import call_vlm
import anthropic
from objects.get_objects import get_object_mesh
import copy
import re
import difflib
import random
import time
import math
import numpy as np
from shapely.geometry import Polygon, Point, box, LineString, MultiPoint
from rtree import index
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import datetime
from utils import extract_json_from_response
from dataclasses import asdict
import os
import base64
import io
from constants import RESULTS_DIR, SERVER_ROOT_DIR, PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED
from visualizer import RoomVisualizer
import sys
from isaacsim.isaac_mcp.server import create_single_room_layout_scene, simulate_the_scene, create_single_room_layout_scene_from_room, create_room_groups_layouts, simulate_the_scene_groups
from timing import timed_stage

def _feature_enabled(name: str, default: bool=False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(maximum, value))

def _room_physics_state_key(room: Room) -> str:
    """Return a stable key for the room state exported to Isaac."""
    return json.dumps(asdict(room), sort_keys=True, separators=(',', ':'))

def _validate_whole_room_after_placement(room: Room, current_layout: FloorPlan, stage: str, last_validated_state: Optional[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
    """Validate a room unless this exact state was already found stable."""
    state_key = _room_physics_state_key(room)
    reuse_enabled = _feature_enabled('SAGE_SKIP_REDUNDANT_WHOLE_ROOM_VALIDATION', True)
    if reuse_enabled and last_validated_state is not None and (last_validated_state[0] == state_key):
        reused_stage = last_validated_state[1]
        print(f'skipping unchanged whole-room validation {stage}; reusing stable verdict from {reused_stage}', file=sys.stderr)
        with timed_stage(stage, 'physics', details={'object_count': len(room.objects), 'skipped': True, 'reason': 'unchanged_since_stable_validation', 'reused_validation_stage': reused_stage}):
            pass
        return last_validated_state
    print('evaluating the stabillity after placement and remove objects that are not stable', file=sys.stderr)
    scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
    room_dict_save_path = os.path.join(scene_save_dir, f'{room.id}.json')
    with timed_stage(stage, 'physics', details={'object_count': len(room.objects), 'skipped': False}) as validation_timer:
        with open(room_dict_save_path, 'w') as f:
            json.dump(asdict(room), f)
        result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
        if not isinstance(result_create, dict) or result_create.get('status') != 'success':
            pass
        result_sim = simulate_the_scene()
        if not isinstance(result_sim, dict) or result_sim.get('status') != 'success':
            pass
        unstable_object_ids = result_sim['unstable_objects']
        validation_timer.add_details(unstable_count=len(unstable_object_ids))
    print(f'number of unstable objects: {len(unstable_object_ids)}', file=sys.stderr)
    print(f'room.objects: {len(room.objects)}', file=sys.stderr)
    if unstable_object_ids:
        print(f'unstable_object_ids: {unstable_object_ids}', file=sys.stderr)
        room.objects = [obj for obj in room.objects if obj.id not in unstable_object_ids]
        print(f'after removing unstable objects, room.objects: {len(room.objects)}', file=sys.stderr)
        return None
    return (state_key, stage)

def find_valid_place_id(object_to_place: Object, object_candidates: List[Object]) -> str:
    """
    Find a valid place id for the object to place from the object candidates.
    Possible options are "floor", "wall", and an existing object id from the object candidates.
    Args:
        object_to_place: The object to place
        object_candidates: The list of object candidates

    Returns:
        A valid place id
    """
    prompt = f"""\nYou are an interior design expert helping to determine the best placement location for an object that currently has an invalid placement assignment.\n\nTASK: Analyze the object that needs placement and the available placement options, then determine the most appropriate place_id.\n\nPLACEMENT OPTIONS:\n1. "floor" - Object sits on the floor (includes against walls, in corners, or anywhere on floor surface)\n2. "wall" - Object is mounted/attached directly on the wall (like wall shelves, paintings, wall-mounted TVs)  \n3. [existing_object_id] - Object is placed on top of a specific existing object (like lamp on table)\n\nEXISTING OBJECTS AVAILABLE FOR PLACEMENT:\n{(chr(10).join([f'  {i + 1}. {obj.type} (ID: {obj.id}) - {obj.dimensions.width:.1f}×{obj.dimensions.length:.1f}×{obj.dimensions.height:.1f}m placed on {obj.place_id}' for (i, obj) in enumerate(object_candidates)]) if object_candidates else 'None')}\n\nOBJECT TO PLACE:\n- Type: {object_to_place.type}\n- Dimensions: {object_to_place.dimensions.width:.1f}×{object_to_place.dimensions.length:.1f}×{object_to_place.dimensions.height:.1f}m\n- Description: {object_to_place.description}\n- Place Guidance: {object_to_place.place_guidance}\n\nPLACEMENT DECISION RULES:\n1. Consider object size, weight, and typical usage patterns\n2. Heavy/large furniture (sofas, tables, beds) → "floor"\n3. Wall-mounted items (paintings, shelves, TVs) → "wall" \n4. Small decorative/functional items (lamps, books, decorations) → existing object ID if suitable surface available\n5. Match object purpose with placement logic (desk lamp goes on desk, not floor)\n6. If placement guidance is placing on an object but no exact match of that place_id, try to find the most similar match of place_id from the existing objects.\n\nEXAMPLES:\n- Large sofa → "floor" (too big for other objects)\n- Wall painting → "wall" (designed for wall mounting)\n- Desk lamp + desk available → desk's object_id (functional pairing)\n- Coffee table → "floor" (standalone furniture)\n\nReturn JSON with format:\n```json\n{{\n    "place_id": "floor|wall|existing_object_id",\n    "reasoning": "Brief explanation of why this placement makes sense for this object type, considering its dimensions, typical usage, and available placement options"\n}}\n```\n\n"""
    response = call_vlm(vlm_type='openai', max_tokens=3000, temperature=0.2, messages=[{'role': 'user', 'content': prompt}])
    response_text = response.content[0].text.strip()
    response_text = extract_json_from_response(response_text)
    valid_placement = json.loads(response_text)
    place_id = valid_placement['place_id']
    return place_id

def place_objects(selected_objects: List[Object], room: Room, current_layout: FloorPlan) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Place selected objects in a room using Claude API for intelligent placement.
    
    Args:
        selected_objects: List of objects to place
        room: Target room for placement
        current_layout: Current floor plan layout
        
    Returns:
        Tuple of (placed_objects, updated_layout, claude_interactions)
    """
    claude_interactions = {'floor_placement': None, 'wall_placement': None, 'total_api_calls': 0, 'placement_method': []}
    room_id = room.id
    if not selected_objects:
        return (selected_objects, current_layout, claude_interactions)
    floor_objects = [obj for obj in selected_objects if obj.place_id == 'floor']
    wall_objects = [obj for obj in selected_objects if obj.place_id == 'wall']
    on_object_objects = [obj for obj in selected_objects if obj.place_id in [obj.id for obj in selected_objects]]
    invalid_objects = [obj for obj in selected_objects if obj.place_id == 'invalid']
    if len(invalid_objects) > 0:
        for invalid_obj in invalid_objects:
            invalid_obj_place_id = find_valid_place_id(invalid_obj, floor_objects + wall_objects + on_object_objects)
            invalid_obj.place_id = invalid_obj_place_id
            if invalid_obj_place_id == 'floor':
                floor_objects.append(invalid_obj)
            elif invalid_obj_place_id == 'wall':
                wall_objects.append(invalid_obj)
            elif invalid_obj_place_id in [obj.id for obj in floor_objects + wall_objects + on_object_objects]:
                on_object_objects.append(invalid_obj)
    placed_objects = []
    objects_need_to_be_placed = []
    last_validated_physics_state = None
    if floor_objects:
        floor_objects_need_to_be_placed = [obj for obj in floor_objects if obj.id not in [obj.id for obj in room.objects]]
        floor_objects_existing = [obj for obj in floor_objects if obj.id in [obj.id for obj in room.objects]]
        objects_need_to_be_placed.extend(floor_objects_need_to_be_placed)
        if len(floor_objects_need_to_be_placed) > 0:
            with timed_stage('placement.floor', 'placement', details={'attempted_count': len(floor_objects_need_to_be_placed)}) as floor_timer:
                (floor_objects, floor_interaction) = place_floor_objects(floor_objects, room, current_layout)
                floor_timer.add_details(succeeded_count=sum((obj.id in {placed.id for placed in floor_objects} for obj in floor_objects_need_to_be_placed)))
            claude_interactions['floor_placement'] = floor_interaction
            if floor_interaction and floor_interaction.get('api_called'):
                claude_interactions['total_api_calls'] += 1
                claude_interactions['placement_method'].append('claude_api_floor')
            else:
                claude_interactions['placement_method'].append('fallback_floor')
            placed_objects.extend(floor_objects)
        else:
            placed_objects.extend(floor_objects_existing)
    on_object_objects_existing = [obj for obj in on_object_objects if obj.id in [obj.id for obj in room.objects]]
    wall_objects_existing = [obj for obj in wall_objects if obj.id in [obj.id for obj in room.objects]]
    placed_objects.extend(on_object_objects_existing)
    placed_objects.extend(wall_objects_existing)
    for layout_room in current_layout.rooms:
        if layout_room.id == room_id:
            layout_room.objects = placed_objects
            break
    room = next((room for room in current_layout.rooms if room.id == room_id), None)
    if PHYSICS_CRITIC_ENABLED:
        last_validated_physics_state = _validate_whole_room_after_placement(room, current_layout, 'validation.whole_room.after_floor', last_validated_physics_state)
        placed_objects = list(room.objects)
    if wall_objects:
        wall_objects_need_to_be_placed = [obj for obj in wall_objects if obj.id not in [obj.id for obj in room.objects]]
        objects_need_to_be_placed.extend(wall_objects_need_to_be_placed)
        if len(wall_objects_need_to_be_placed) > 0:
            with timed_stage('placement.wall', 'placement', details={'attempted_count': len(wall_objects_need_to_be_placed)}) as wall_timer:
                (placed_wall_objects, wall_interaction) = place_wall_objects(wall_objects_need_to_be_placed, room, current_layout, placed_objects)
                wall_timer.add_details(succeeded_count=len(placed_wall_objects))
            placed_objects.extend(placed_wall_objects)
            claude_interactions['wall_placement'] = wall_interaction
            if wall_interaction and wall_interaction.get('api_called'):
                claude_interactions['total_api_calls'] += 1
                claude_interactions['placement_method'].append('claude_api_wall')
            else:
                claude_interactions['placement_method'].append('fallback_wall')
    for layout_room in current_layout.rooms:
        if layout_room.id == room_id:
            layout_room.objects = placed_objects
            break
    room = next((room for room in current_layout.rooms if room.id == room_id), None)
    if PHYSICS_CRITIC_ENABLED:
        last_validated_physics_state = _validate_whole_room_after_placement(room, current_layout, 'validation.whole_room.after_wall', last_validated_physics_state)
        placed_objects = list(room.objects)
    if on_object_objects:
        on_object_objects_need_to_be_placed = [obj for obj in on_object_objects if obj.id not in [obj.id for obj in room.objects]]
        objects_need_to_be_placed.extend(on_object_objects_need_to_be_placed)
        if len(on_object_objects_need_to_be_placed) > 0:
            with timed_stage('placement.on_object', 'placement', details={'attempted_count': len(on_object_objects_need_to_be_placed)}) as on_object_timer:
                (placed_on_object_objects, on_object_interaction) = place_on_object_objects(on_object_objects_need_to_be_placed, room, current_layout)
                on_object_timer.add_details(succeeded_count=len(placed_on_object_objects))
            placed_objects.extend(placed_on_object_objects)
            claude_interactions['on_object_placement'] = on_object_interaction
            if on_object_interaction and on_object_interaction.get('api_called'):
                claude_interactions['total_api_calls'] += 1
                claude_interactions['placement_method'].append('claude_api_on_object')
            else:
                claude_interactions['placement_method'].append('fallback_on_object')
    for layout_room in current_layout.rooms:
        if layout_room.id == room_id:
            layout_room.objects = placed_objects
            break
    room = next((room for room in current_layout.rooms if room.id == room_id), None)
    if PHYSICS_CRITIC_ENABLED:
        last_validated_physics_state = _validate_whole_room_after_placement(room, current_layout, 'validation.whole_room.after_on_object', last_validated_physics_state)
        placed_objects = list(room.objects)
    failed_to_be_placed_objects = [obj for obj in objects_need_to_be_placed if obj.id not in [obj.id for obj in room.objects]]
    print(f'failed_to_be_placed_objects: ', len(failed_to_be_placed_objects), file=sys.stderr)
    for obj in failed_to_be_placed_objects:
        print(f'  - {obj.id}', file=sys.stderr)
    claude_interactions['failed_to_be_placed_objects'] = failed_to_be_placed_objects
    successful_placed_objects = [obj for obj in objects_need_to_be_placed if obj.id in [obj.id for obj in room.objects]]
    print(f'successful_placed_objects: ', len(successful_placed_objects), file=sys.stderr)
    for obj in successful_placed_objects:
        print(f'  - {obj.id}', file=sys.stderr)
    claude_interactions['placed_objects'] = successful_placed_objects
    return (placed_objects, current_layout, claude_interactions)

def calculate_object_bounding_box(obj: Object, room: Room) -> str:
    """
    Calculate the bounding box of an object considering its rotation.
    Returns formatted string with x-y range relative to room coordinates.
    """
    obj_x_cm = (obj.position.x - room.position.x) * 100
    obj_y_cm = (obj.position.y - room.position.y) * 100
    obj_width_cm = obj.dimensions.width * 100
    obj_length_cm = obj.dimensions.length * 100
    object_rotation = obj.rotation.z
    if object_rotation == 0:
        obj_length_x_cm = obj_width_cm
        obj_length_y_cm = obj_length_cm
    elif object_rotation == 90:
        obj_length_x_cm = obj_length_cm
        obj_length_y_cm = obj_width_cm
    elif object_rotation == 180:
        obj_length_x_cm = obj_width_cm
        obj_length_y_cm = obj_length_cm
    elif object_rotation == 270:
        obj_length_x_cm = obj_length_cm
        obj_length_y_cm = obj_width_cm
    else:
        obj_length_x_cm = obj_width_cm
        obj_length_y_cm = obj_length_cm
    x_min = obj_x_cm - obj_length_x_cm / 2
    x_max = obj_x_cm + obj_length_x_cm / 2
    y_min = obj_y_cm - obj_length_y_cm / 2
    y_max = obj_y_cm + obj_length_y_cm / 2
    return f'{x_min:.0f}-{x_max:.0f} x {y_min:.0f}-{y_max:.0f} cm'

def get_object_facing_direction(obj: Object) -> str:
    """
    Get the facing direction of an object based on its rotation.
    Returns the direction as +x, -x, +y, or -y.
    """
    rotation = obj.rotation.z % 360
    if rotation == 0:
        return '+y'
    elif rotation == 90:
        return '-x'
    elif rotation == 180:
        return '-y'
    elif rotation == 270:
        return '+x'
    elif 0 <= rotation < 45 or 315 <= rotation < 360:
        return '+y'
    elif 45 <= rotation < 135:
        return '-x'
    elif 135 <= rotation < 225:
        return '-y'
    else:
        return '+x'

def generate_room_visualization_image(room: Room, current_layout: FloorPlan) -> str:
    """
    Generate a room visualization image using visualize_2d_render and return as base64 string.
    Returns base64 encoded PNG image, or None if visualization fails.
    """
    try:
        visualizer = RoomVisualizer(room, current_layout)
        temp_image_path = f'outputs/room_viz_{room.id}_{int(time.time())}.png'
        result_path = visualizer.visualize_2d_render(save_path=temp_image_path, show=False)
        if result_path and os.path.exists(temp_image_path):
            with open(temp_image_path, 'rb') as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            os.remove(temp_image_path)
            return base64_image
        else:
            print('Failed to generate room visualization', file=sys.stderr)
            return None
    except Exception as e:
        print(f'Error generating room visualization: {str(e)}', file=sys.stderr)
        return None

def get_room_layout_description(room: Room, current_layout: FloorPlan, floor_objects: List[Object]) -> str:
    """
    Generate a detailed description of the room layout for Claude API
    """
    description_parts = []
    description_parts.append(f'Room Type: {room.room_type}')
    description_parts.append(f'Room Dimensions: {room.dimensions.width * 100:.0f} cm (width) x {room.dimensions.length * 100:.0f} cm (length) x {room.ceiling_height * 100:.0f} cm (height)')
    if room.doors:
        description_parts.append(f'\nDoors ({len(room.doors)} total):')
        for (i, door) in enumerate(room.doors):
            door_pos_cm = door.position_on_wall * room.dimensions.width * 100
            description_parts.append(f'  - Door {i + 1}: {door.door_type} door, {door.width * 100:.0f} cm wide, {door.height * 100:.0f} cm tall')
            description_parts.append(f"    Position: {door_pos_cm:.0f} cm from left wall, opens {('inward' if door.opens_inward else 'outward')}")
    else:
        description_parts.append('\nDoors: None')
    if room.windows:
        description_parts.append(f'\nWindows ({len(room.windows)} total):')
        for (i, window) in enumerate(room.windows):
            window_pos_cm = window.position_on_wall * room.dimensions.width * 100
            description_parts.append(f'  - Window {i + 1}: {window.window_type} window, {window.width * 100:.0f} cm wide, {window.height * 100:.0f} cm tall')
            description_parts.append(f'    Position: {window_pos_cm:.0f} cm from left wall, sill height: {window.sill_height * 100:.0f} cm')
    else:
        description_parts.append('\nWindows: None')
    existing_objects = [obj for obj in room.objects if obj.id not in [floor_obj.id for floor_obj in floor_objects]]
    if existing_objects:
        description_parts.append(f'\nExisting Objects Already Placed ({len(existing_objects)} total):')
        for obj in existing_objects:
            obj_pos = f'({obj.position.x * 100:.0f}, {obj.position.y * 100:.0f}) cm'
            obj_dims = f'{obj.dimensions.width * 100:.0f} x {obj.dimensions.length * 100:.0f} x {obj.dimensions.height * 100:.0f} cm'
            obj_bbox = calculate_object_bounding_box(obj, room)
            obj_facing = get_object_facing_direction(obj)
            description_parts.append(f'  - ID: {obj.id} | Type: {obj.type} | Size: {obj_dims} | Position: {obj_pos}')
            description_parts.append(f'    Bounding Box: {obj_bbox}')
            description_parts.append(f'    Facing Direction: {obj_facing}')
            if hasattr(obj, 'description') and obj.description:
                description_parts.append(f'    Description: {obj.description}')
    else:
        description_parts.append('\nExisting Objects: None')
    description_parts.append(f'\nNew Objects to Place ({len(floor_objects)} total):')
    total_object_area = 0
    for obj in floor_objects:
        obj_dims = f'{obj.dimensions.width * 100:.0f} x {obj.dimensions.length * 100:.0f} x {obj.dimensions.height * 100:.0f} cm'
        obj_area = obj.dimensions.width * obj.dimensions.length * 10000
        total_object_area += obj_area
        description_parts.append(f'  - ID: {obj.id} | Type: {obj.type} | Size: {obj_dims} | Footprint: {obj_area:.0f} cm²')
        if hasattr(obj, 'description') and obj.description:
            description_parts.append(f'    Description: {obj.description}')
        if hasattr(obj, 'place_guidance') and obj.place_guidance:
            description_parts.append(f'    Placement Guidance: {obj.place_guidance}')
        if hasattr(obj, 'source') and obj.source:
            description_parts.append(f'    Source: {obj.source}')
    room_floor_area = room.dimensions.width * room.dimensions.length * 10000
    furniture_density = total_object_area / room_floor_area * 100 if room_floor_area > 0 else 0
    description_parts.append(f'\nLayout Considerations:')
    description_parts.append(f'- Total floor area: {room_floor_area:.0f} cm²')
    description_parts.append(f'- Total furniture footprint: {total_object_area:.0f} cm² ({furniture_density:.1f}% of floor area)')
    description_parts.append(f'- Recommended furniture density: 30-40% for comfortable living spaces')
    description_parts.append(f'- Door swing areas (approximately 90cm radius) need to be kept clear')
    description_parts.append(f'- Windows provide natural light - orient seating to take advantage')
    description_parts.append(f'- Maintain 60-90cm walkways between major furniture pieces')
    description_parts.append(f"- Consider the room's primary function when creating focal points")
    description_parts.append(f"- IMPORTANT: Use the full exact object IDs shown above (e.g., 'room_fd2c0c7a_chair_123abcde'), including any shared room prefix; never shorten, normalize, or omit ID prefixes when referencing objects in constraints")
    if room.room_type.lower() in ['living room', 'lounge', 'family room']:
        description_parts.append(f'- Living room: Create conversation areas, consider TV viewing angles, ensure traffic flow')
    elif room.room_type.lower() in ['bedroom']:
        description_parts.append(f'- Bedroom: Position bed away from door, ensure bedside access, consider natural light for dressing')
    elif room.room_type.lower() in ['dining room']:
        description_parts.append(f'- Dining room: Allow 60cm per person at table, 120cm clearance behind chairs for service')
    elif room.room_type.lower() in ['office', 'study']:
        description_parts.append(f'- Office: Position desk near window for natural light, ensure chair clearance, minimize glare on screens')
    return '\n'.join(description_parts)

def place_floor_objects(floor_objects: List[Object], room: Room, current_layout: FloorPlan) -> Tuple[List[Object], Dict[str, Any]]:
    """
    Place floor objects using Claude API for constraints and DFS solver for placement.
    Handles both existing positioned objects and new objects that need placement.
    """
    interaction_info = {'api_called': False, 'prompt': None, 'response': None, 'parsed_constraints': None, 'solver_result': None, 'retry_info': None, 'error': None, 'existing_objects_kept': 0, 'new_objects_placed': 0}
    if not floor_objects:
        return ([], interaction_info)
    try:
        existing_objects = []
        new_objects = []
        for obj in floor_objects:
            has_valid_position = obj.position.x > room.position.x and obj.position.x < room.position.x + room.dimensions.width and (obj.position.y > room.position.y) and (obj.position.y < room.position.y + room.dimensions.length) and (not (obj.position.x == 0 and obj.position.y == 0))
            if has_valid_position:
                existing_objects.append(obj)
            else:
                new_objects.append(obj)
        max_height_of_new_objects = 0
        for obj in new_objects:
            if obj.dimensions.height > max_height_of_new_objects:
                max_height_of_new_objects = obj.dimensions.height
        wall_objects_existing = [obj for obj in room.objects if obj.place_id == 'wall']
        wall_objects_existing_obstacles = []
        for wall_obj in wall_objects_existing:
            if wall_obj.position.z < max_height_of_new_objects:
                wall_objects_existing_obstacles.append(wall_obj)
        interaction_info['existing_objects_kept'] = len(existing_objects)
        interaction_info['new_objects_placed'] = len(new_objects)
        if not new_objects:
            return (existing_objects, interaction_info)
        room_layout_description = get_room_layout_description(room, current_layout, new_objects)
        if existing_objects:
            existing_objects_info = []
            for obj in existing_objects:
                obj_pos_cm = f'({(obj.position.x - room.position.x) * 100:.0f}, {(obj.position.y - room.position.y) * 100:.0f})'
                obj_dims_cm = f'{obj.dimensions.width * 100:.0f} x {obj.dimensions.length * 100:.0f} x {obj.dimensions.height * 100:.0f}'
                existing_objects_info.append(f'  - {obj.type} (ID: {obj.id}) at position {obj_pos_cm} cm, size {obj_dims_cm} cm')
            room_layout_description += f'\n\nEXISTING OBJECTS TO KEEP IN PLACE ({len(existing_objects)} total):\n'
            room_layout_description += '\n'.join(existing_objects_info)
            room_layout_description += '\n\nIMPORTANT: These existing objects cannot be moved and must be treated as fixed constraints when placing new objects.'
        room_visualization_base64 = generate_room_visualization_image(room, current_layout)
        object_constraints_prompt = 'You are an experienced interior designer with expertise in space planning and furniture arrangement.\n\nI need your help to create a beautiful and functional furniture layout for a room. I\'ll provide you with detailed information about the room\'s architecture, existing objects, and the new furniture pieces that need to be placed.\n\nROOM LAYOUT INFORMATION:\n{room_layout}\n\nCONSTRAINT SYSTEM:\nPlease assign constraints to each NEW object that needs placement. \nExisting objects are already positioned and cannot be moved.\n\nHere are the available constraints:\n\n1. GLOBAL CONSTRAINT (required for each new object):\n   - edge: Place at the edge of the room, close to walls\n   - middle: Place away from walls, in the central area of the room\n\n2. DISTANCE CONSTRAINTS:\n   - close to, [object_id]: Place really close to another object (as close as possible, typically within 30cm distance)\n   - near, [object_id]: Place near another object (middle range distance, typically 50cm to 150cm distance)\n   - far, [object_id]: Place far from another object (as far as possible, typically more than 150cm distance)\n\n3. POSITION CONSTRAINTS (relative to target object\'s facing direction):\n   - in front of, [object_id]: Position in front of another object (in the direction it faces)\n   - around, [object_id]: Position around another object (typically for chairs around tables)\n   - side of, [object_id]: Position to the left or right side of another object\n   - left of, [object_id]: Position to the LEFT of another object (relative to its facing direction)\n   - right of, [object_id]: Position to the RIGHT of another object (relative to its facing direction)\n   \n   IMPORTANT: "left", "right", and "front" are relative to the TARGET OBJECT\'S facing direction.\n\n4. ALIGNMENT CONSTRAINTS:\n   - center aligned, [object_id]: Align centers with another object\n\n5. ROTATION CONSTRAINTS:\n   - face to, [object_id]: Orient to face toward another object\'s center\n   - face same as, [object_id]: Orient to face the same as another object\'s facing direction\n\nDESIGN STRATEGY [IMPORTANT]:\n1. **Existing Objects**: Work around existing objects - they cannot be moved but can be referenced in constraints\n2. **Placement Guidance**: Consider each object\'s "Placement Guidance" for specific positioning instructions\n3. **Anchor Objects**: If no existing floor objects, start with larger anchor objects (only global constraint needed)\n4. **Object Dependencies**: Objects can only reference previously placed objects OR existing fixed objects\n5. **Object Sequence**: Object constraints should be generated in the order of the importance of the objects.\n- Higher importance objects (e.g. key, essential, and important functional objects) should be placed first. \n- Follow-up objects with direct constraints (e.g. close to, near, in front of, side of, left of, right of, center aligned, face to, face same as) with previous objects should get constraints with a sequence next to the previous objects.\ne.g. if the first object is "bed_001", we prefer the second object to be "nightstand_001" next to the bed since it has more direct relationship with the bed, instead of "desk_001" since it has less direct relationship with the bed.\n- Could be different from the sequence in the list of New Objects to Place, but should be reasonable and logical following the guidance above.\n6. **Room Function**: Consider door swing areas, traffic flow paths, and natural light from windows\n7. **Object Relationships**: Objects of the same type should typically be aligned; chairs should face tables/desks\n8. **Detailed Constraints Preferred**: We encourage you to provide as detailed constraints as possible (as many constraints as possible, e.g. 4-5 constraints for each object if applicable) for each object. It will greatly help the planner to find the optimal solution.\n9. **Avoid Cluttering and Ensure Circulation Space**: We encourage you to ensure the circulation space is sufficient for walking and moving around. Avoid placing objects too close to each other or too crowded.\n10. **Same type of constraints can be used for multiple times for the single object**: for example, you are allowed to describe relationship of object A is left of object B and right of object C at the same time (two relative position constraints at the same time) ["left of, object_B", "right of, object_C"].\n\nCONSTRAINT SELECTION RULES:\n1. **Distance + Position**: If using "close to", "near" or "far", add position/alignment/rotation constraints for specificity\n2. **Position + Distance**: If using position constraints ("in front of", "left of"), please add distance constraints (close to, near or far) for specificity (except for the grid alignment pattern)\n3. **Functional Pairing**: Chairs should use "close to" + "in front of" + "face to" relative to tables/desks. (choose "close to" instead of "near" because we need the chair to against the table/desk) You can generalize it to other meaningful pairs of objects.\n4. **Relationship Coupling**: When using "in front of" position relationship, it\'s better to consider using "face to" relationship as well to make the scene look more hormonic.\n5. **Distance Inference**: \n- If two objects in your reasoning are not placed in the same region of the room (e.g. [against different walls] or [in different corners] or [one against the wall and one in the middle of the room] or [in different functional areas]), you should add "far" constraint to them to ensure they are far from each other.\n- Only use "close to" constraint when you really need the object to be as close to the other object as possible, otherwise use near constraint to ensure the object is not too far from the other object.\n\nCONSTRAINT COMBINATION PATTERNS:\n\nBasic Furniture:\n• Anchor furniture (first object): ["edge"]\n• Secondary furniture: ["middle", "close to, target", "in front of, target", "center aligned, target", "face to, target"]\n\nIMPORTANT: Example IDs like table_001 or sofa_001 below are schematic placeholders only. In your final JSON, replace every placeholder with the full exact ID from New Objects to Place, including the complete room prefix.\nSome Examples:\n• Objects that with high possibility of being placed against wall with constraint ["edge", other constraints...] without explicitly mentioned in the placement guidance:\nincluding but not limited to bookcases, bookshelves, shelving units, armoires, wardrobes, TV stands, sideboards, buffets, sofa, bed, dressers, chests of drawers, plant, sculptures, etc.\n• Plants, sculptures or other decorated objects like or other aesthetic objects should be placed **at the edge of the room**: ["edge", other constraints...]\n• Chair close to dining table: ["middle", "close to, table_001", "in front of, table_001", "face to, table_001"]\n(choose "close to" instead of "near" because we need the chair to against the table)\n• Coffee table in front of sofa: ["middle", "near, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n(choose "near" instead of "close to" because we need the coffee table to be near the sofa, but not too close, we need to save space for walking.)\n• TV stand facing seating: ["edge", "far, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n• Four or more chairs around the dining table: each chair gets ["middle", "close to, table_001", "around, table_001", "face to, table_001"] \n("center aligned, table_001" is optional)  (choose "close to" instead of "near" because we need the chair to against the table)\n• Nightstand left of a bed which is against the wall (["edge"]) (both objects are against the wall): ["edge", "left of, bed_001", "close to, bed_001", "face same as, bed_001"] (Position + Distance combination, and it\'s not a grid alignment pattern, and "face same as" is to keep the nightstand facing the same direction as the bed)\n(choose "close to" instead of "near" because we need the nightstand to be as close to the bed as possible for convenience)\n• bookshelf and sofa against different walls (for bookshelf): ["edge", "far, sofa_001"] (use far constraint because they are against different walls)\n• dining table (in the middle of the room) and storage box (against the wall) should be placed far from each other: ["middle", "far, storage_box_001"] (use far constraint because they are in different functional areas)\n• Four, six, eight, or more tables in the dining room: following grid alignment pattern [4 = 2 rows x 2 columns, 6 = 2 rows x 3 columns, 8 = 2 rows x 4 columns, etc.], \nfirst table ("table_001") at location (0, 0) gets  ["middle"], \nsecond table ("table_002") at location (1, 0) gets ["middle", "in front of, table_001", "center aligned, table_001", "face same as, table_001"]\nthird table ("table_003") at location (0, 1) gets  ["middle", "right of, table_001", "center aligned, table_001", "face same as, table_001"]\nfourth table ("table_004") at location (1, 1) gets ["middle", "in front of, table_003", "center aligned, table_003", "right of, table_002", "center aligned, table_002", "face same as, table_001"]\n(if table number > 4:)\nfifth table ("table_005") at location (0, 2) gets  ["middle", "right of, table_003", "center aligned, table_003", "face same as, table_001"]\nsixth table ("table_006") at location (1, 2) gets  ["middle", "in front of, table_005", "center aligned, table_005", "right of, table_004", "center aligned, table_004", "face same as, table_001"]\n(if table number > 6:)\nthe seventh table at (0, 3) and eighth table at (1, 3) get similar constraints as the fifth and sixth tables.\nTips: for other number of tables, you can generalize the pattern to the number of tables:\nif the number of tables is n, you can set the number of rows to be sqrt(n) and the number of columns to be n / sqrt(n).\nthen the first table at (0, 0) gets ["middle"],\nfor the table at (i, j), it needs to be right of and center aligned with the table at (i, j-1), and in front of and center aligned with the table at (i-1, j). with the same face as the table at (0, 0).\n(Note: this is a grid alignment pattern, don\'t need to add distance constraints, don\'t add close to or near constraint between tables.)\n\n\nOUTPUT FORMAT:\nReturn a JSON object with the following structure: (Do not include any comments or annotations in the JSON)\n```json\n{{\n  "reasoning": {{\n    "design_strategy": "Overall approach and design principles guiding the layout",\n    "room_layout_analysis": "Analysis of the room\'s shape, dimensions, and spatial characteristics",\n    "existing_objects_consideration": "How existing objects in the room influence the placement decisions",\n    "placement_guidance_application": "How the provided placement guidance rules are being applied, and reflect your analysis with your constraints planning below",\n    "importance_ranking": "Priority order of objects to be placed and rationale for the ranking.",\n    "functional_relationships": "How objects relate to each other functionally (e.g., coffee table near sofa for use)",\n    "aesthetic_considerations": "Visual harmony, balance, and aesthetic relationships between objects",\n    "sequence_of_objects_to_place": "Sequence of objects to be placed in the constraints list based on the importance ranking. (Should be different from the sequence in the list of New Objects to Place)"\n  }},\n  "constraints": [\n    {{\n      "object_id": "room_fd2c0c7a_sofa_001",\n      "constraints": ["edge"]\n    }},\n    {{\n      "object_id": "room_fd2c0c7a_coffee_table_001",\n      "constraints": ["middle", "near, room_fd2c0c7a_sofa_001", "in front of, room_fd2c0c7a_sofa_001", "center aligned, room_fd2c0c7a_sofa_001", "face to, room_fd2c0c7a_sofa_001"]\n    }},\n    {{\n      "object_id": "room_fd2c0c7a_tv_stand_001",\n      "constraints": ["edge", "far, room_fd2c0c7a_sofa_001", "in front of, room_fd2c0c7a_sofa_001", "center aligned, room_fd2c0c7a_sofa_001", "face to, room_fd2c0c7a_sofa_001"]\n    }},\n    {{\n      "object_id": "room_fd2c0c7a_bookshelf_001",\n      "constraints": ["edge"]\n    }}\n  ]\n}}\n```\n\nIMPORTANT:\n- Include ALL NEW objects that need placement in the constraints list\n- Each object must have at least one global constraint ("edge" or "middle")\n- Use the full exact object IDs from the room layout information, including the complete room prefix (for example, use room_fd2c0c7a_table_2fd8b41b, not table_2fd8b41b)\n- Constraints are strings in the format shown above\n- Only include NEW objects (not existing objects) in the constraints list\n- Add as many meaningful constraints as possible for each object if applicable (4-6 constraints for each object if applicable).\n\nPlease design the layout now:'
        prompt_text = object_constraints_prompt.format(room_layout=room_layout_description)
        interaction_info['prompt'] = prompt_text
        interaction_info['visualization_included'] = room_visualization_base64 is not None
        message_content = []
        message_content.append({'type': 'text', 'text': prompt_text})
        with timed_stage('placement.floor.constraint_vlm', 'llm'):
            response = call_vlm(vlm_type='openai', max_tokens=8000 + len(new_objects) * 500, temperature=0.3, messages=[{'role': 'user', 'content': message_content}])
        interaction_info['api_called'] = True
        response_text = response.content[0].text.strip()
        interaction_info['response'] = response_text
        try:
            json_text = extract_json_from_response(response_text)
            response_data = json.loads(json_text)
            reasoning = response_data.get('reasoning', 'No reasoning provided')
            constraints_list = response_data.get('constraints', [])
            print(f'Design reasoning: {reasoning}', file=sys.stderr)
            interaction_info['reasoning'] = reasoning
        except Exception as e:
            interaction_info['error'] = f'Failed to parse JSON response: {str(e)}'
            print(f'Failed to parse JSON response: {e}', file=sys.stderr)
            print(f'Response text: {response_text}', file=sys.stderr)
            return (existing_objects, interaction_info)
        new_object_ids = [obj.id for obj in new_objects]
        existing_object_ids = [obj.id for obj in existing_objects]
        existing_id_mapping = {obj.id: f'existing-{obj.id}' for obj in existing_objects}
        constraints = parse_constraints_from_json(constraints_list, new_object_ids, existing_object_ids, existing_id_mapping)
        print('constraints: ', json.dumps(constraints, indent=4), file=sys.stderr)
        interaction_info['parsed_constraints'] = constraints
        if not constraints and new_objects:
            interaction_info['error'] = 'Failed to parse constraints from Claude response'
            return (existing_objects, interaction_info)
        max_wall_thickness_cm = 0
        if room.walls:
            max_wall_thickness_cm = max((wall.thickness * 0.5 * 100 for wall in room.walls))
        else:
            max_wall_thickness_cm = 5
        inner_width_cm = room.dimensions.width * 100 - 2 * max_wall_thickness_cm
        inner_length_cm = room.dimensions.length * 100 - 2 * max_wall_thickness_cm
        if inner_width_cm <= 0 or inner_length_cm <= 0:
            interaction_info['error'] = f'Room too small after accounting for wall thickness. Inner dimensions: {inner_width_cm}x{inner_length_cm} cm'
            return (existing_objects, interaction_info)
        room_vertices = [(max_wall_thickness_cm, max_wall_thickness_cm), (max_wall_thickness_cm, max_wall_thickness_cm + inner_length_cm), (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm + inner_length_cm), (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm)]
        room_poly = Polygon(room_vertices)
        new_objects_list = []
        object_id2dimension = {}
        for obj in new_objects:
            width_cm = obj.dimensions.width * 100 + 3.5
            length_cm = obj.dimensions.length * 100 + 3.5
            object_id2dimension[obj.id] = {'x': obj.dimensions.width, 'y': obj.dimensions.length, 'z': obj.dimensions.height}
            new_objects_list.append((obj.id, (width_cm, length_cm)))
        constraints_key_list = list(constraints.keys())
        constraints_order = {obj_id: idx for (idx, obj_id) in enumerate(constraints_key_list)}
        new_objects_list.sort(key=lambda obj_tuple: constraints_order.get(obj_tuple[0], len(constraints_key_list)))
        for obj_tuple in new_objects_list:
            print(f' new object to place: {obj_tuple}', file=sys.stderr)
        initial_state = get_door_window_placements(room)
        for obj in existing_objects:
            obj_x_cm = (obj.position.x - room.position.x) * 100
            obj_y_cm = (obj.position.y - room.position.y) * 100
            obj_width_cm = obj.dimensions.width * 100 + 3.5
            obj_length_cm = obj.dimensions.length * 100 + 3.5
            object_rotation = obj.rotation.z
            if object_rotation == 0:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 90:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            elif object_rotation == 180:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 270:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            else:
                raise ValueError(f'Invalid rotation: {object_rotation}')
            obj_vertices = [(obj_x_cm - obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2), (obj_x_cm - obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2)]
            initial_state[f'existing-{obj.id}'] = ((obj_x_cm, obj_y_cm), obj.rotation.z, obj_vertices, 1)
        for wall_obj in wall_objects_existing_obstacles:
            obj_x_cm = (wall_obj.position.x - room.position.x) * 100
            obj_y_cm = (wall_obj.position.y - room.position.y) * 100
            obj_width_cm = wall_obj.dimensions.width * 100 + 3.5
            obj_length_cm = wall_obj.dimensions.length * 100 + 3.5
            object_rotation = wall_obj.rotation.z
            if object_rotation == 0:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 90:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            elif object_rotation == 180:
                obj_length_x_cm = obj_width_cm
                obj_length_y_cm = obj_length_cm
            elif object_rotation == 270:
                obj_length_x_cm = obj_length_cm
                obj_length_y_cm = obj_width_cm
            else:
                raise ValueError(f'Invalid rotation: {object_rotation}')
            obj_vertices = [(obj_x_cm - obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2), (obj_x_cm - obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2)]
            initial_state[f'existing-{wall_obj.id}'] = ((obj_x_cm, obj_y_cm), wall_obj.rotation.z, obj_vertices, 1)
        solver = DFS_Solver_Floor(grid_size=20, max_duration=max(300, len(new_objects_list) * 60), constraint_bouns=1.0, enable_retry=True, room_id=room.id)
        with timed_stage('placement.floor.solver', 'placement', details={'attempted_count': len(new_objects_list)}) as solver_timer:
            (solution, solution_constraints) = solver.get_solution(room_poly, new_objects_list, constraints, initial_state, use_milp=False)
            solver_timer.add_details(succeeded_count=sum((not object_id.startswith(('door-', 'window-', 'existing-', 'open-')) for object_id in solution)))
        if solution:
            try:
                grid_points = solver.create_grids(room_poly)
                grid_points = solver.remove_points(grid_points, initial_state)
                combined_solution = {**initial_state, **solution}
                solver.visualize_grid(room_poly, grid_points, combined_solution, room.id)
            except Exception as e:
                pass
        placed_new_object_ids = [obj_id for obj_id in solution.keys() if not obj_id.startswith(('door-', 'window-', 'existing-', 'open-'))]
        all_new_object_ids = [obj.id for obj in new_objects]
        skipped_object_ids = [obj_id for obj_id in all_new_object_ids if obj_id not in placed_new_object_ids]
        interaction_info['solver_result'] = {'num_solutions': len(solver.solutions), 'solution_found': len(solution) > 0, 'solution': solution, 'initial_state': initial_state, 'total_new_objects': len(new_objects), 'placed_new_objects': len(placed_new_object_ids), 'skipped_new_objects': len(skipped_object_ids), 'placed_object_ids': placed_new_object_ids, 'skipped_object_ids': skipped_object_ids}
        interaction_info['retry_info'] = solver.retry_info
        placed_new_objects = [obj for obj in new_objects if obj.id in placed_new_object_ids]
        newly_positioned_objects = solution_to_objects(solution, solution_constraints, placed_new_objects, object_id2dimension, room)
        all_placed_objects = existing_objects + newly_positioned_objects
        return (all_placed_objects, interaction_info)
    except Exception as e:
        interaction_info['error'] = f'Error in place_floor_objects: {str(e)}'
        print(f'Error in place_floor_objects: {str(e)}', file=sys.stderr)
        return (existing_objects if 'existing_objects' in locals() else [], interaction_info)

def _place_on_object_objects_batched(on_object_objects: List[Object], room: Room, current_layout: FloorPlan):
    from objects.object_on_top_placement import filter_placements_batch_by_physics_critic, generate_on_object_candidate_batch, select_maximum_stability_placements
    interaction_info = {'api_called': False, 'prompt': None, 'response': None, 'parsed_constraints': None, 'solver_result': None, 'retry_info': None, 'error': None, 'existing_objects_kept': 0, 'new_objects_placed': 0, 'placement_results': [], 'mode': 'batched_isaac'}
    if not on_object_objects:
        return ([], interaction_info)
    base_room = copy.deepcopy(room)
    room_object_ids = {obj.id for obj in room.objects}
    valid_objects = []
    candidate_jobs = []
    invalid_errors = {}
    for obj in on_object_objects:
        if obj.place_id not in room_object_ids:
            invalid_errors[obj.id] = f'Target object {obj.place_id} not found in room'
            continue
        valid_objects.append(obj)
        candidate_jobs.append((obj, obj.place_id))
    candidate_workers = _bounded_env_int('SAGE_ONOBJECT_CANDIDATE_WORKERS', 8, 1, 32) if _feature_enabled('SAGE_PARALLEL_ONOBJECT_CANDIDATES', True) else 1
    candidate_seed = _bounded_env_int('SAGE_ONOBJECT_CANDIDATE_SEED', 0, 0, 2 ** 31 - 1)
    reuse_collision_context = _feature_enabled('SAGE_REUSE_ONOBJECT_COLLISION_CONTEXT')
    max_candidates_per_object = 10
    candidate_early_stop = _feature_enabled('SAGE_ONOBJECT_CANDIDATE_EARLY_STOP', True)
    with timed_stage('placement.on_object.candidate_selection', 'placement', details={'attempted_count': len(candidate_jobs), 'sample_count': 150, 'parallel': candidate_workers > 1, 'worker_count': min(candidate_workers, max(1, len(candidate_jobs))), 'base_seed': candidate_seed, 'reuse_collision_context': reuse_collision_context, 'shared_base_room': True, 'max_accepted_candidates_per_object': max_candidates_per_object if candidate_early_stop else None}) as candidate_timer:
        generated_results = generate_on_object_candidate_batch(current_layout, base_room, candidate_jobs, sample_count=150, regular_rotation=False, place_location='both', max_workers=candidate_workers, base_seed=candidate_seed, reuse_collision_context=reuse_collision_context, max_accepted_candidates=max_candidates_per_object if candidate_early_stop else None)
        candidate_timer.add_details(candidate_count=sum((len(result['placements']) for result in generated_results)), failed_count=sum((result['error'] is not None for result in generated_results)), task_inclusive_wall_sum_s=sum((result['duration_s'] for result in generated_results)))
    placements_by_object = {}
    generation_errors = dict(invalid_errors)
    objects_with_candidates = []
    for (obj, generated_result) in zip(valid_objects, generated_results):
        if generated_result['error'] is not None:
            generation_errors[obj.id] = generated_result['error']
            placements_by_object[obj.id] = []
            continue
        placements = generated_result['placements']
        placements_by_object[obj.id] = placements
        if placements:
            objects_with_candidates.append((obj, placements))
        else:
            generation_errors[obj.id] = f'No valid placements found on target object {obj.place_id}'
    interaction_info['candidate_generation'] = {'parallel': candidate_workers > 1, 'worker_count': min(candidate_workers, max(1, len(candidate_jobs))), 'base_seed': candidate_seed, 'task_count': len(candidate_jobs), 'reuse_collision_context': reuse_collision_context, 'shared_base_room': True, 'max_accepted_candidates_per_object': max_candidates_per_object if candidate_early_stop else None}
    candidates_by_object = {obj.id: [] for obj in on_object_objects}
    physics_metrics = {'scene_build_count': 0, 'candidate_count': 0, 'safe_count': 0}
    if objects_with_candidates:
        parallel_width = _bounded_env_int('SAGE_ISAAC_ONOBJECT_PARALLEL_WIDTH', 16, 1, 128)
        scheduling_policy = os.environ.get('SAGE_ISAAC_ONOBJECT_SCHEDULER', 'work_conserving').strip().lower()
        if scheduling_policy not in {'fixed_wave', 'work_conserving'}:
            raise ValueError(f'SAGE_ISAAC_ONOBJECT_SCHEDULER must be fixed_wave or work_conserving, got {scheduling_policy!r}')
        with timed_stage('placement.on_object.physics_filter', 'physics', details={'object_count': len(objects_with_candidates), 'candidate_count': sum((min(max_candidates_per_object, len(placements)) for (_, placements) in objects_with_candidates)), 'parallel_width': parallel_width, 'scheduling_policy': scheduling_policy, 'shared_scene_build': True}) as physics_timer:
            (candidates_by_object, physics_metrics) = filter_placements_batch_by_physics_critic(current_layout, base_room, objects_with_candidates, parallel_width=parallel_width, max_candidates_per_object=max_candidates_per_object, scheduling_policy=scheduling_policy)
            physics_timer.add_details(isaac_metrics=physics_metrics)
    interaction_info['batch_physics'] = physics_metrics
    beam_width = _bounded_env_int('SAGE_ONOBJECT_SELECTION_BEAM_WIDTH', 512, 1, 8192)
    with timed_stage('placement.on_object.stability_selection', 'placement', details={'object_count': len(valid_objects), 'beam_width': beam_width, 'objective': 'maximize_count_then_minimize_stability_cost'}) as selection_timer:
        (selected_by_object, selection_info) = select_maximum_stability_placements(current_layout, valid_objects, candidates_by_object, beam_width=beam_width)
        selection_timer.add_details(**selection_info)
    interaction_info['selection'] = selection_info
    placed_objects = []
    for obj in on_object_objects:
        selected = selected_by_object.get(obj.id)
        safe_candidates = [value for value in candidates_by_object.get(obj.id, []) if value.get('safe') and (not value.get('invalid'))]
        if selected is None:
            if obj.id in generation_errors:
                error = generation_errors[obj.id]
            elif safe_candidates:
                error = 'All safe candidates conflict with the selected placement set'
            else:
                error = 'No safe placements found after batched physics validation'
            interaction_info['placement_results'].append({'object_id': obj.id, 'success': False, 'error': error})
            continue
        position = selected['position']
        rotation = selected['rotation']
        placed_obj = Object(id=obj.id, room_id=room.id, type=obj.type, description=obj.description if hasattr(obj, 'description') else f'Placed {obj.type}', position=Point3D(x=position['x'], y=position['y'], z=position['z']), rotation=Euler(x=rotation['x'] * 180 / np.pi, y=rotation['y'] * 180 / np.pi, z=rotation['z'] * 180 / np.pi), dimensions=obj.dimensions, source=obj.source if hasattr(obj, 'source') else 'placement', source_id=obj.source_id if hasattr(obj, 'source_id') else obj.id, place_id=obj.place_id, mass=getattr(obj, 'mass', 1.0))
        placed_objects.append(placed_obj)
        interaction_info['placement_results'].append({'object_id': obj.id, 'success': True, 'target_object_id': obj.place_id, 'position': position, 'rotation': rotation, 'total_candidates': len(placements_by_object.get(obj.id, [])), 'physics_candidates': min(max_candidates_per_object, len(placements_by_object.get(obj.id, []))), 'safe_candidates': len(safe_candidates), 'selected_candidate_index': selected['candidate_index'], 'stability_cost': selected.get('stability_cost')})
    interaction_info['new_objects_placed'] = len(placed_objects)
    return (placed_objects, interaction_info)

def place_on_object_objects(on_object_objects: List[Object], room: Room, current_layout: FloorPlan):
    if _feature_enabled('SAGE_BATCHED_ISAAC_ONOBJECT', True):
        try:
            return _place_on_object_objects_batched(on_object_objects, room, current_layout)
        except Exception as error:
            if not _feature_enabled('SAGE_BATCHED_ISAAC_FALLBACK_SERIAL', True):
                raise
            print(f'Batched Isaac on-object placement failed; using serial fallback: {error}', file=sys.stderr)
            (placed, interaction) = _place_on_object_objects_serial(on_object_objects, room, current_layout)
            interaction['batch_fallback_error'] = str(error)
            interaction['mode'] = 'serial_fallback'
            return (placed, interaction)
    return _place_on_object_objects_serial(on_object_objects, room, current_layout)

def _place_on_object_objects_serial(on_object_objects: List[Object], room: Room, current_layout: FloorPlan):
    """
    Place objects on top of other objects using physics-based sampling and validation.
    
    Args:
        on_object_objects: List of objects to place on other objects
        room: Target room for placement
        current_layout: Current floor plan layout
        
    Returns:
        Tuple of (placed_objects, interaction_info)
    """
    from objects.object_on_top_placement import get_random_placements_on_target_object, filter_placements_by_physics_critic, generate_on_object_candidate_batch
    all_placed_objects = []
    interaction_info = {'api_called': False, 'prompt': None, 'response': None, 'parsed_constraints': None, 'solver_result': None, 'retry_info': None, 'error': None, 'existing_objects_kept': 0, 'new_objects_placed': 0, 'placement_results': []}
    room_copy_eval = copy.deepcopy(room)
    reuse_collision_context = _feature_enabled('SAGE_REUSE_ONOBJECT_COLLISION_CONTEXT')
    parallel_candidate_results = None
    parallel_candidates_enabled = _feature_enabled('SAGE_PARALLEL_ONOBJECT_CANDIDATES') and len(on_object_objects) > 1
    if parallel_candidates_enabled:
        candidate_jobs = []
        candidate_job_indices = []
        room_object_ids = {room_object.id for room_object in room.objects}
        for (object_index, obj) in enumerate(on_object_objects):
            if obj.place_id in room_object_ids:
                candidate_jobs.append((obj, obj.place_id))
                candidate_job_indices.append(object_index)
        candidate_workers = _bounded_env_int('SAGE_ONOBJECT_CANDIDATE_WORKERS', 8, 1, 32)
        candidate_seed = _bounded_env_int('SAGE_ONOBJECT_CANDIDATE_SEED', 0, 0, 2 ** 31 - 1)
        parallel_candidate_results = [None] * len(on_object_objects)
        with timed_stage('placement.on_object.candidate_selection', 'placement', details={'attempted_count': len(candidate_jobs), 'sample_count': 150, 'parallel': True, 'worker_count': min(candidate_workers, len(candidate_jobs)), 'base_seed': candidate_seed, 'reuse_collision_context': reuse_collision_context}) as candidate_timer:
            generated_results = generate_on_object_candidate_batch(current_layout, copy.deepcopy(room_copy_eval), candidate_jobs, sample_count=150, regular_rotation=False, place_location='both', max_workers=candidate_workers, base_seed=candidate_seed, reuse_collision_context=reuse_collision_context)
            for (object_index, generated_result) in zip(candidate_job_indices, generated_results):
                parallel_candidate_results[object_index] = generated_result
            candidate_timer.add_details(candidate_count=sum((len(result['placements']) for result in generated_results)), failed_count=sum((result['error'] is not None for result in generated_results)), task_inclusive_wall_sum_s=sum((result['duration_s'] for result in generated_results)))
        interaction_info['candidate_generation'] = {'parallel': True, 'worker_count': min(candidate_workers, len(candidate_jobs)), 'base_seed': candidate_seed, 'task_count': len(candidate_jobs), 'reuse_collision_context': reuse_collision_context}
    for (object_index, obj) in enumerate(on_object_objects):
        try:
            print(f'trying to place {obj.id} on {obj.place_id}, current room objects: {len(room_copy_eval.objects)}', file=sys.stderr)
            target_object_id = obj.place_id
            target_object = next((room_obj for room_obj in room.objects if room_obj.id == target_object_id), None)
            if target_object is None:
                interaction_info['placement_results'].append({'object_id': obj.id, 'success': False, 'error': f'Target object {target_object_id} not found in room'})
                continue
            if parallel_candidate_results is not None:
                generated_result = parallel_candidate_results[object_index]
                if generated_result is None:
                    raise RuntimeError(f'Candidate task for {obj.id} was not scheduled')
                if generated_result['error'] is not None:
                    raise RuntimeError(generated_result['error'])
                placements = generated_result['placements']
            else:
                with timed_stage('placement.on_object.candidate_selection', 'placement', details={'object_id': obj.id, 'sample_count': 150, 'reuse_collision_context': reuse_collision_context}) as candidate_timer:
                    placements = get_random_placements_on_target_object(current_layout, room_copy_eval, target_object_id, obj, sample_count=150, regular_rotation=False, place_location='both', reuse_collision_context=reuse_collision_context)
                    candidate_timer.add_details(candidate_count=len(placements))
            if not placements:
                interaction_info['placement_results'].append({'object_id': obj.id, 'success': False, 'error': f'No valid placements found on target object {target_object_id}'})
                continue
            with timed_stage('placement.on_object.physics_filter', 'physics', details={'object_id': obj.id, 'candidate_count': len(placements)}) as physics_filter_timer:
                safe_placements = filter_placements_by_physics_critic(current_layout, room_copy_eval, obj, placements)
                physics_filter_timer.add_details(safe_candidate_count=len(safe_placements))
            if not safe_placements:
                interaction_info['placement_results'].append({'object_id': obj.id, 'success': False, 'error': f'No safe placements found after physics validation'})
                continue
            best_placement = safe_placements[0]
            position_placed = best_placement['position']
            rotation_placed = best_placement['rotation']
            placed_obj = Object(id=obj.id, room_id=room.id, type=obj.type, description=obj.description if hasattr(obj, 'description') else f'Placed {obj.type}', position=Point3D(x=position_placed['x'], y=position_placed['y'], z=position_placed['z']), rotation=Euler(x=rotation_placed['x'] * 180 / np.pi, y=rotation_placed['y'] * 180 / np.pi, z=rotation_placed['z'] * 180 / np.pi), dimensions=obj.dimensions, source=obj.source if hasattr(obj, 'source') else 'placement', source_id=obj.source_id if hasattr(obj, 'source_id') else obj.id, place_id=obj.place_id, mass=getattr(obj, 'mass', 1.0))
            all_placed_objects.append(placed_obj)
            room_copy_eval.objects.append(placed_obj)
            print(f'final validation for placing {obj.id}', file=sys.stderr)
            if PHYSICS_CRITIC_ENABLED:
                print(f'evaluating the stabillity after placement and remove objects that are not stable', file=sys.stderr)
                scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
                room_dict_save_path = os.path.join(scene_save_dir, f'{room_copy_eval.id}.json')
                with open(room_dict_save_path, 'w') as f:
                    json.dump(asdict(room_copy_eval), f)
                with timed_stage('validation.on_object.final', 'physics', details={'object_id': obj.id, 'object_count': len(room_copy_eval.objects)}):
                    result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
                    if not isinstance(result_create, dict) or result_create.get('status') != 'success':
                        pass
                    result_sim = simulate_the_scene()
                    if not isinstance(result_sim, dict) or result_sim.get('status') != 'success':
                        pass
                unstable_object_ids = result_sim['unstable_objects']
                if len(unstable_object_ids) > 0:
                    room_copy_eval.objects = [obj for obj in room_copy_eval.objects if obj.id not in unstable_object_ids]
                    all_placed_objects = [obj for obj in all_placed_objects if obj.id not in unstable_object_ids]
            print(f'finished trial of placing {obj.id} on {obj.place_id}, current room objects: {len(room_copy_eval.objects)}', file=sys.stderr)
            if len(unstable_object_ids) == 0:
                interaction_info['placement_results'].append({'object_id': obj.id, 'success': True, 'target_object_id': target_object_id, 'position': {'x': position_placed['x'], 'y': position_placed['y'], 'z': position_placed['z']}, 'rotation': {'x': rotation_placed['x'], 'y': rotation_placed['y'], 'z': rotation_placed['z']}, 'total_candidates': len(placements), 'safe_candidates': len(safe_placements)})
                interaction_info['new_objects_placed'] += 1
        except Exception as e:
            interaction_info['placement_results'].append({'object_id': obj.id, 'success': False, 'error': f'Exception during placement: {str(e)}'})
            interaction_info['error'] = f'Error placing object {obj.id}: {str(e)}'
    return (all_placed_objects, interaction_info)

def parse_constraints(constraint_text: str, new_object_ids: List[str], existing_object_ids: List[str]=None, existing_id_mapping: Dict[str, str]=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse constraints text from Claude response into structured format.
    
    Args:
        constraint_text: The constraint text from Claude
        new_object_ids: List of new object IDs that need placement
        existing_object_ids: List of existing object IDs that can be referenced
        existing_id_mapping: Mapping from existing object IDs to their solver names (with prefix)
    """
    if existing_object_ids is None:
        existing_object_ids = []
    if existing_id_mapping is None:
        existing_id_mapping = {}
    constraint_name2type = {'edge': 'global', 'middle': 'global', 'in front of': 'relative', 'behind': 'relative', 'left of': 'relative', 'right of': 'relative', 'side of': 'relative', 'around': 'relative', 'face to': 'direction', 'face same as': 'direction', 'aligned': 'alignment', 'center alignment': 'alignment', 'center aligned': 'alignment', 'aligned center': 'alignment', 'edge alignment': 'alignment', 'near': 'distance', 'far': 'distance'}
    object2constraints = {}
    plans = [plan.lower() for plan in constraint_text.split('\n') if '|' in plan]
    for plan in plans:
        pattern = re.compile('^(\\d+[\\.\\)]\\s*|- )')
        plan = pattern.sub('', plan)
        if plan[-1] == '.':
            plan = plan[:-1]
        object_id = plan.split('|')[0].replace('*', '').strip()
        if object_id not in new_object_ids:
            continue
        object2constraints[object_id] = []
        constraints = plan.split('|')[1:]
        for constraint in constraints:
            constraint = constraint.strip()
            constraint_name = constraint.split(',')[0].strip()
            if constraint_name == 'n/a':
                continue
            try:
                constraint_type = constraint_name2type[constraint_name]
            except:
                close_matches = difflib.get_close_matches(constraint_name, constraint_name2type.keys(), n=1, cutoff=0.3)
                if close_matches:
                    new_constraint_name = close_matches[0]
                    constraint_name = new_constraint_name
                    constraint_type = constraint_name2type[constraint_name]
                else:
                    continue
            if constraint_type == 'global':
                object2constraints[object_id].append({'type': constraint_type, 'constraint': constraint_name})
            elif constraint_type in ['relative', 'direction', 'alignment', 'distance']:
                try:
                    target = constraint.split(',')[1].strip()
                except:
                    continue
                target_is_valid = False
                actual_target = target
                if target in object2constraints:
                    target_is_valid = True
                elif target in existing_object_ids:
                    actual_target = existing_id_mapping.get(target, target)
                    target_is_valid = True
                if target_is_valid:
                    if constraint_name == 'around':
                        object2constraints[object_id].append({'type': 'distance', 'constraint': 'near', 'target': actual_target})
                        object2constraints[object_id].append({'type': 'direction', 'constraint': 'face to', 'target': actual_target})
                    elif constraint_name == 'in front of':
                        object2constraints[object_id].append({'type': 'relative', 'constraint': 'in front of', 'target': actual_target})
                        object2constraints[object_id].append({'type': 'alignment', 'constraint': 'center aligned', 'target': actual_target})
                    else:
                        object2constraints[object_id].append({'type': constraint_type, 'constraint': constraint_name, 'target': actual_target})
                else:
                    continue
            else:
                continue
    object2constraints_cleaned = {}
    for (object_id, constraints) in object2constraints.items():
        constraints_cleaned = []
        constraint_types = []
        for constraint in constraints:
            if constraint['type'] not in constraint_types:
                constraint_types.append(constraint['type'])
                constraints_cleaned.append(constraint)
        object2constraints_cleaned[object_id] = constraints_cleaned
    return object2constraints_cleaned

def parse_constraints_from_json(constraints_list: List[Dict[str, Any]], new_object_ids: List[str], existing_object_ids: List[str]=None, existing_id_mapping: Dict[str, str]=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse constraints from JSON format returned by the VLM.
    
    Args:
        constraints_list: List of constraint dicts from VLM response, each containing:
                         {"object_id": "obj_id", "constraints": ["constraint1", "constraint2", ...]}
        new_object_ids: List of new object IDs that need placement
        existing_object_ids: List of existing object IDs that can be referenced
        existing_id_mapping: Mapping from existing object IDs to their solver names (with prefix)
    
    Returns:
        Dictionary mapping object_id to list of parsed constraint dictionaries
    """
    if existing_object_ids is None:
        existing_object_ids = []
    if existing_id_mapping is None:
        existing_id_mapping = {}
    constraint_name2type = {'edge': 'global', 'middle': 'global', 'in front of': 'relative', 'behind': 'relative', 'left of': 'relative', 'right of': 'relative', 'side of': 'relative', 'around': 'relative', 'face to': 'direction', 'face same as': 'direction', 'aligned': 'alignment', 'center alignment': 'alignment', 'center aligned': 'alignment', 'aligned center': 'alignment', 'edge alignment': 'alignment', 'close to': 'distance', 'near': 'distance', 'far': 'distance'}
    object2constraints = {}
    for constraint_entry in constraints_list:
        object_id = constraint_entry.get('object_id', '').strip()
        constraint_strings = constraint_entry.get('constraints', [])
        if object_id not in new_object_ids:
            continue
        object2constraints[object_id] = []
        for constraint_str in constraint_strings:
            constraint_str = constraint_str.strip().lower()
            parts = [p.strip() for p in constraint_str.split(',')]
            constraint_name = parts[0]
            if constraint_name == 'n/a':
                continue
            try:
                constraint_type = constraint_name2type[constraint_name]
            except KeyError:
                close_matches = difflib.get_close_matches(constraint_name, constraint_name2type.keys(), n=1, cutoff=0.3)
                if close_matches:
                    new_constraint_name = close_matches[0]
                    constraint_name = new_constraint_name
                    constraint_type = constraint_name2type[constraint_name]
                else:
                    print(f"Constraint type not found for '{constraint_name}', skipping", file=sys.stderr)
                    continue
            if constraint_type == 'global':
                object2constraints[object_id].append({'type': constraint_type, 'constraint': constraint_name})
            elif constraint_type in ['relative', 'direction', 'alignment', 'distance']:
                if len(parts) < 2:
                    print(f"Constraint '{constraint_str}' missing target, skipping", file=sys.stderr)
                    continue
                target = parts[1].strip()
                target_is_valid = False
                actual_target = target
                if target in object2constraints:
                    target_is_valid = True
                elif target in existing_object_ids:
                    actual_target = existing_id_mapping.get(target, target)
                    target_is_valid = True
                else:
                    for existing_object_id in existing_object_ids:
                        if target in existing_object_id:
                            actual_target = existing_id_mapping.get(existing_object_id, existing_object_id)
                            target_is_valid = True
                            break
                if target_is_valid:
                    if constraint_name == 'around':
                        object2constraints[object_id].append({'type': 'distance', 'constraint': 'close to', 'target': actual_target})
                        object2constraints[object_id].append({'type': 'direction', 'constraint': 'face to', 'target': actual_target})
                    elif constraint_name == 'in front of':
                        object2constraints[object_id].append({'type': 'relative', 'constraint': 'in front of', 'target': actual_target})
                        object2constraints[object_id].append({'type': 'alignment', 'constraint': 'center aligned', 'target': actual_target})
                    else:
                        object2constraints[object_id].append({'type': constraint_type, 'constraint': constraint_name, 'target': actual_target})
                else:
                    print(f"Target object '{target}' not found, skipping constraint", file=sys.stderr)
                    continue
    object2constraints_cleaned = {}
    for (object_id, constraints) in object2constraints.items():
        constraints_cleaned = []
        seen_constraints = set()
        for constraint in constraints:
            constraint_key = (constraint['type'], constraint['constraint'], constraint.get('target', None))
            if constraint_key not in seen_constraints:
                seen_constraints.add(constraint_key)
                constraints_cleaned.append(constraint)
        object2constraints_cleaned[object_id] = constraints_cleaned
    return object2constraints_cleaned

def get_door_window_placements(room: Room) -> Dict[str, Tuple]:
    """
    Get door and window placements as initial state for the solver.
    Uses actual wall thickness from the Wall model and properly maps doors/windows to walls.
    Positions doors/windows at the inner room boundary (wall-room interface).
    """
    door_window_placements = {}
    i = 0
    wall_map = {wall.id: wall for wall in room.walls}
    for door in room.doors:
        wall = wall_map.get(door.wall_id)
        assert wall is not None, f'Wall {door.wall_id} not found'
        start_point = wall.start_point
        end_point = wall.end_point
        position_on_wall = door.position_on_wall
        door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall - room.position.x
        door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall - room.position.y
        door_center_x_cm = door_center_x * 100
        door_center_y_cm = door_center_y * 100
        door_width_cm = door.width * 100
        room_center_x = room.position.x + room.dimensions.width / 2
        room_center_y = room.position.y + room.dimensions.length / 2
        mid_point_x = (start_point.x + end_point.x) / 2
        mid_point_y = (start_point.y + end_point.y) / 2
        wall_offset_x = room_center_x - mid_point_x
        wall_offset_y = room_center_y - mid_point_y
        if door.opening:
            opening_space_cm = 50
            if abs(wall_offset_x) > abs(wall_offset_y):
                door_open_length_x_cm = opening_space_cm
                door_open_length_y_cm = door_width_cm
                if wall_offset_x > 0:
                    door_open_center_offset = (opening_space_cm / 2, 0)
                else:
                    door_open_center_offset = (-opening_space_cm / 2, 0)
            else:
                door_open_length_x_cm = door_width_cm
                door_open_length_y_cm = opening_space_cm
                if wall_offset_y > 0:
                    door_open_center_offset = (0, opening_space_cm / 2)
                else:
                    door_open_center_offset = (0, -opening_space_cm / 2)
            door_open_center_x_cm = door_center_x_cm + door_open_center_offset[0]
            door_open_center_y_cm = door_center_y_cm + door_open_center_offset[1]
            door_vertices = [(door_open_center_x_cm - door_open_length_x_cm / 2, door_open_center_y_cm - door_open_length_y_cm / 2), (door_open_center_x_cm + door_open_length_x_cm / 2, door_open_center_y_cm - door_open_length_y_cm / 2), (door_open_center_x_cm + door_open_length_x_cm / 2, door_open_center_y_cm + door_open_length_y_cm / 2), (door_open_center_x_cm - door_open_length_x_cm / 2, door_open_center_y_cm + door_open_length_y_cm / 2)]
        else:
            if abs(wall_offset_x) > abs(wall_offset_y):
                if wall_offset_x > 0:
                    door_open_center_offset = (door_width_cm / 2, 0)
                else:
                    door_open_center_offset = (-door_width_cm / 2, 0)
            elif wall_offset_y > 0:
                door_open_center_offset = (0, door_width_cm / 2)
            else:
                door_open_center_offset = (0, -door_width_cm / 2)
            door_open_center_x_cm = door_center_x_cm + door_open_center_offset[0]
            door_open_center_y_cm = door_center_y_cm + door_open_center_offset[1]
            door_vertices = [(door_open_center_x_cm - door_width_cm / 2, door_open_center_y_cm - door_width_cm / 2), (door_open_center_x_cm + door_width_cm / 2, door_open_center_y_cm - door_width_cm / 2), (door_open_center_x_cm + door_width_cm / 2, door_open_center_y_cm + door_width_cm / 2), (door_open_center_x_cm - door_width_cm / 2, door_open_center_y_cm + door_width_cm / 2)]
        door_window_placements[f'door-{i}'] = ((door_open_center_x_cm, door_open_center_y_cm), 0, door_vertices, 1)
        i += 1
    for window in room.windows:
        wall = wall_map.get(window.wall_id)
        assert wall is not None, f'Wall {window.wall_id} not found'
        start_point = wall.start_point
        end_point = wall.end_point
        position_on_wall = window.position_on_wall
        window_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall - room.position.x
        window_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall - room.position.y
        window_center_x_cm = window_center_x * 100
        window_center_y_cm = window_center_y * 100
        window_width_cm = window.width * 100
        room_center_x = room.position.x + room.dimensions.width / 2
        room_center_y = room.position.y + room.dimensions.length / 2
        mid_point_x = (start_point.x + end_point.x) / 2
        mid_point_y = (start_point.y + end_point.y) / 2
        wall_offset_x = room_center_x - mid_point_x
        wall_offset_y = room_center_y - mid_point_y
        if abs(wall_offset_x) > abs(wall_offset_y):
            if wall_offset_x > 0:
                window_open_center_offset = (wall.thickness * 0.5 * 100 / 2, 0)
                window_length_x = wall.thickness * 0.45 * 100
                window_length_y = window_width_cm
            else:
                window_open_center_offset = (-wall.thickness * 0.5 * 100 / 2, 0)
                window_length_x = wall.thickness * 0.45 * 100
                window_length_y = window_width_cm
        elif wall_offset_y > 0:
            window_open_center_offset = (0, wall.thickness * 0.5 * 100 / 2)
            window_length_x = window_width_cm
            window_length_y = wall.thickness * 0.45 * 100
        else:
            window_open_center_offset = (0, -wall.thickness * 0.5 * 100 / 2)
            window_length_x = window_width_cm
            window_length_y = wall.thickness * 0.45 * 100
        window_open_center_x_cm = window_center_x_cm + window_open_center_offset[0]
        window_open_center_y_cm = window_center_y_cm + window_open_center_offset[1]
        window_vertices = [(window_open_center_x_cm - window_length_x / 2, window_open_center_y_cm - window_length_y / 2), (window_open_center_x_cm + window_length_x / 2, window_open_center_y_cm - window_length_y / 2), (window_open_center_x_cm + window_length_x / 2, window_open_center_y_cm + window_length_y / 2), (window_open_center_x_cm - window_length_x / 2, window_open_center_y_cm + window_length_y / 2)]
        door_window_placements[f'window-{i}'] = ((window_open_center_x_cm, window_open_center_y_cm), 0, window_vertices, 0.3)
        i += 1
    return door_window_placements

def solution_to_objects(solution: Dict[str, Any], solution_constraints: Dict[str, Any], floor_objects: List[Object], object_id2dimension: Dict[str, Dict], room: Room) -> List[Object]:
    """
    Convert DFS solver solution to placed Object instances
    """
    placed_objects = []
    for (object_id, solution_data) in solution.items():
        constraints = solution_constraints.get(object_id, [])
        if 'door' in object_id or 'window' in object_id or 'open' in object_id:
            continue
        original_obj = next((obj for obj in floor_objects if obj.id == object_id), None)
        if not original_obj:
            continue
        center_pos = solution_data[0]
        rotation_degrees = solution_data[1]
        x_meters = center_pos[0] / 100 + room.position.x
        y_meters = center_pos[1] / 100 + room.position.y
        z_meters = room.position.z
        placed_obj = Object(id=f'{original_obj.id}', room_id=room.id, type=original_obj.type, description=original_obj.description if hasattr(original_obj, 'description') else f'Placed {original_obj.type}', position=Point3D(x=x_meters, y=y_meters, z=z_meters), rotation=Euler(x=0, y=0, z=rotation_degrees), dimensions=original_obj.dimensions, source=original_obj.source if hasattr(original_obj, 'source') else 'placement', source_id=original_obj.source_id if hasattr(original_obj, 'source_id') else original_obj.id, place_id=original_obj.place_id, placement_constraints=constraints, mass=getattr(original_obj, 'mass', 1.0))
        placed_objects.append(placed_obj)
    return placed_objects

def place_wall_objects(wall_objects: List[Object], room: Room, current_layout: FloorPlan, placed_objects: List[Object]) -> Tuple[List[Object], Dict[str, Any]]:
    """
    Place wall objects using geometric placement and optional Claude API constraints.
    """
    interaction_info = {'api_called': False, 'prompt': None, 'response': None, 'parsed_constraints': None, 'solver_result': None, 'error': None, 'placement_results': []}
    if not wall_objects:
        return ([], interaction_info)
    placed_wall_objects = []
    try:
        wall_systems = create_wall_coordinate_systems(room)
        wall_grids = create_wall_grid_points(wall_systems, grid_density=20)
        for wall_obj in wall_objects:
            try:
                impossible_regions = calculate_impossible_wall_regions(room, placed_objects + placed_wall_objects, wall_systems, wall_obj)
                valid_points = filter_valid_wall_points(wall_obj, wall_grids, impossible_regions, wall_systems)
                if not valid_points:
                    interaction_info['placement_results'].append({'object_id': wall_obj.id, 'success': False, 'error': 'No valid placement points found'})
                    continue
                best_placement = select_best_wall_placement(wall_obj, valid_points, placed_objects + placed_wall_objects, wall_systems)
                visualize_wall_placement(wall_obj, wall_systems, wall_grids, impossible_regions, valid_points, best_placement, room.id)
                if best_placement:
                    adjusted_placement = adjust_wall_object_position(best_placement, wall_obj, wall_systems)
                    placed_obj = create_wall_placed_object(wall_obj, adjusted_placement, room, best_placement['constraints'])
                    placed_wall_objects.append(placed_obj)
                    interaction_info['placement_results'].append({'object_id': wall_obj.id, 'success': True, 'wall_id': best_placement['wall_id'], 'position': best_placement['position_3d'], 'rotation': best_placement['rotation']})
                elif best_placement is None:
                    interaction_info['placement_results'].append({'object_id': wall_obj.id, 'success': False, 'error': 'Failed to generate VLM constraints for wall object placement'})
                else:
                    interaction_info['placement_results'].append({'object_id': wall_obj.id, 'success': False, 'error': 'Failed to find suitable placement'})
            except Exception as e:
                interaction_info['placement_results'].append({'object_id': wall_obj.id, 'success': False, 'error': f'Exception during placement: {str(e)}'})
    except Exception as e:
        interaction_info['error'] = f'Error in place_wall_objects: {str(e)}'
    return (placed_wall_objects, interaction_info)

def create_wall_coordinate_systems(room: Room) -> Dict[str, Dict]:
    """
    Create 2D coordinate systems for each wall of the room.
    Returns a dictionary with wall info including 2D coordinate system mappings.
    """
    wall_systems = {}
    max_wall_thickness = max((wall.thickness for wall in room.walls)) if room.walls else 0.1
    for wall in room.walls:
        wall_vector = np.array([wall.end_point.x - wall.start_point.x, wall.end_point.y - wall.start_point.y, 0])
        wall_length = np.linalg.norm(wall_vector[:2])
        wall_direction = wall_vector / np.linalg.norm(wall_vector) if np.linalg.norm(wall_vector) > 0 else np.array([1, 0, 0])
        wall_normal_2d = np.array([-wall_direction[1], wall_direction[0]])
        wall_center = np.array([(wall.start_point.x + wall.end_point.x) / 2, (wall.start_point.y + wall.end_point.y) / 2])
        room_center = np.array([room.position.x + room.dimensions.width / 2, room.position.y + room.dimensions.length / 2])
        to_room_center = room_center - wall_center
        if np.dot(wall_normal_2d, to_room_center) < 0:
            wall_normal_2d = -wall_normal_2d
        wall_normal = np.array([wall_normal_2d[0], wall_normal_2d[1], 0])
        rect_width = wall_length - wall.thickness
        rect_height = room.ceiling_height
        wall_systems[wall.id] = {'wall': wall, 'wall_direction': wall_direction, 'wall_normal': wall_normal, 'wall_center_3d': np.array([wall_center[0], wall_center[1], room.position.z + rect_height / 2]), 'rect_width': rect_width, 'rect_height': rect_height, 'wall_length': wall_length, 'thickness': wall.thickness}
    return wall_systems

def create_wall_grid_points(wall_systems: Dict[str, Dict], grid_density: int=20) -> Dict[str, List]:
    """
    Create grid sampling points on each wall rectangle.
    Returns dictionary mapping wall_id to list of 2D and 3D grid points.
    """
    wall_grids = {}
    for (wall_id, wall_info) in wall_systems.items():
        grid_points = []
        rect_width = wall_info['rect_width']
        rect_height = wall_info['rect_height']
        margin = 0.1
        effective_width = rect_width - 2 * margin
        effective_height = rect_height - 2 * margin
        if effective_width <= 0 or effective_height <= 0:
            continue
        x_step = effective_width / grid_density
        y_step = effective_height / grid_density
        for i in range(grid_density):
            for j in range(grid_density):
                x_2d = -effective_width / 2 + i * x_step + x_step / 2
                y_2d = -effective_height / 2 + j * y_step + y_step / 2
                pos_3d = wall_2d_to_3d(np.array([x_2d, y_2d]), wall_info)
                grid_points.append({'pos_2d': np.array([x_2d, y_2d]), 'pos_3d': pos_3d, 'wall_id': wall_id})
        wall_grids[wall_id] = grid_points
    return wall_grids

def wall_2d_to_3d(pos_2d: np.ndarray, wall_info: Dict) -> np.ndarray:
    """
    Convert 2D wall coordinates to 3D world coordinates.
    """
    (x_2d, y_2d) = pos_2d
    wall_direction = wall_info['wall_direction']
    wall_normal = wall_info['wall_normal']
    wall_center_3d = wall_info['wall_center_3d']
    pos_3d = wall_center_3d + x_2d * wall_direction + y_2d * np.array([0, 0, 1]) + wall_info['thickness'] * 0.4 * wall_normal
    return pos_3d

def calculate_impossible_wall_regions(room: Room, placed_objects: List[Object], wall_systems: Dict[str, Dict], wall_object_to_place: Object=None) -> Dict[str, List]:
    """
    Calculate regions on walls where objects cannot be placed due to doors, windows, and existing objects.
    
    Args:
        room: The room containing the walls
        placed_objects: Existing objects that may block wall placement
        wall_systems: Wall coordinate system information
        wall_object_to_place: The wall object that will be placed (used to determine clearance depth)
    """
    impossible_regions = {wall_id: [] for wall_id in wall_systems.keys()}
    for (wall_id, wall_info) in wall_systems.items():
        wall = wall_info['wall']
        for door in room.doors:
            if door.wall_id == wall.id:
                door_region = calculate_door_wall_region(door, wall_info)
                if door_region:
                    impossible_regions[wall_id].append(door_region)
        for window in room.windows:
            if window.wall_id == wall.id:
                window_region = calculate_window_wall_region(window, wall_info)
                if window_region:
                    impossible_regions[wall_id].append(window_region)
    wall_object_depth = 0.5
    print('wall_object_depth default: ', wall_object_depth, file=sys.stderr)
    if wall_object_to_place:
        wall_object_depth = wall_object_to_place.dimensions.length + 0.2
        print('updating wall_object_depth to ', wall_object_depth, file=sys.stderr)
    for obj in placed_objects:
        obj_regions = calculate_object_wall_regions(obj, wall_systems, wall_object_depth=wall_object_depth)
        for (wall_id, regions) in obj_regions.items():
            impossible_regions[wall_id].extend(regions)
    return impossible_regions

def calculate_door_wall_region(door: Door, wall_info: Dict) -> Dict:
    """
    Calculate the 2D region on wall occupied by a door.
    """
    wall = wall_info['wall']
    door_center_along_wall = door.position_on_wall * wall_info['wall_length']
    door_center_x_2d = door_center_along_wall - wall_info['wall_length'] / 2
    door_bottom_y_2d = -wall_info['rect_height'] / 2
    door_top_y_2d = door_bottom_y_2d + door.height
    return {'type': 'door', 'x_min': door_center_x_2d - door.width / 2, 'x_max': door_center_x_2d + door.width / 2, 'y_min': door_bottom_y_2d, 'y_max': door_top_y_2d}

def calculate_window_wall_region(window: Window, wall_info: Dict) -> Dict:
    """
    Calculate the 2D region on wall occupied by a window.
    """
    wall = wall_info['wall']
    window_center_along_wall = window.position_on_wall * wall_info['wall_length']
    window_center_x_2d = window_center_along_wall - wall_info['wall_length'] / 2
    window_bottom_y_2d = -wall_info['rect_height'] / 2 + window.sill_height
    window_top_y_2d = window_bottom_y_2d + window.height
    return {'type': 'window', 'x_min': window_center_x_2d - window.width / 2, 'x_max': window_center_x_2d + window.width / 2, 'y_min': window_bottom_y_2d, 'y_max': window_top_y_2d}

def calculate_object_wall_regions(obj: Object, wall_systems: Dict[str, Dict], wall_object_depth: float=None) -> Dict[str, List]:
    """
    Calculate regions on walls blocked by existing objects.
    Returns dictionary mapping wall_id to list of blocked regions.
    
    Args:
        obj: The existing object to check for wall conflicts
        wall_systems: Wall coordinate system information
        wall_object_depth: Depth that wall objects will occupy (uses object's length dimension if None)
    """
    regions = {wall_id: [] for wall_id in wall_systems.keys()}
    obj_bbox = get_object_3d_bbox(obj)
    if wall_object_depth is None:
        assert False, 'wall_object_depth is None'
    for (wall_id, wall_info) in wall_systems.items():
        wall_distance = calculate_distance_to_wall(obj_bbox, wall_info)
        if wall_distance < wall_object_depth:
            wall_region = project_bbox_to_wall(obj_bbox, wall_info)
            if wall_region:
                regions[wall_id].append(wall_region)
    return regions

def get_object_3d_bbox(obj: Object) -> Dict:
    """
    Get 3D axis-aligned bounding box of an object.
    """
    rotation_z = obj.rotation.z
    if rotation_z in [0, 180]:
        width_x = obj.dimensions.width
        length_y = obj.dimensions.length
    elif rotation_z in [90, 270]:
        width_x = obj.dimensions.length
        length_y = obj.dimensions.width
    else:
        width_x = max(obj.dimensions.width, obj.dimensions.length)
        length_y = max(obj.dimensions.width, obj.dimensions.length)
    return {'center': np.array([obj.position.x, obj.position.y, obj.position.z + obj.dimensions.height / 2]), 'half_extents': np.array([width_x / 2, length_y / 2, obj.dimensions.height / 2]), 'min': np.array([obj.position.x - width_x / 2, obj.position.y - length_y / 2, obj.position.z]), 'max': np.array([obj.position.x + width_x / 2, obj.position.y + length_y / 2, obj.position.z + obj.dimensions.height])}

def calculate_distance_to_wall(obj_bbox: Dict, wall_info: Dict) -> float:
    """
    Calculate minimum distance from object bounding box to wall interior surface.
    Accounts for wall thickness to calculate distance to the actual wall surface facing the room.
    """
    wall = wall_info['wall']
    wall_thickness = wall_info['thickness']
    wall_normal = wall_info['wall_normal']
    wall_centerline_point = np.array([(wall.start_point.x + wall.end_point.x) / 2, (wall.start_point.y + wall.end_point.y) / 2, 0])
    wall_interior_point = wall_centerline_point + wall_thickness / 2 * wall_normal
    obj_center = obj_bbox['center']
    obj_half_extents = obj_bbox['half_extents']
    center_to_wall_surface = obj_center - wall_interior_point
    distance_to_surface = abs(np.dot(center_to_wall_surface, wall_normal))
    obj_extent_toward_wall = abs(np.dot(obj_half_extents, np.abs(wall_normal)))
    return max(0, distance_to_surface - obj_extent_toward_wall)

def project_bbox_to_wall(obj_bbox: Dict, wall_info: Dict) -> Dict:
    """
    Project 3D object bounding box onto wall 2D coordinate system.
    """
    wall_direction = wall_info['wall_direction']
    wall_center_3d = wall_info['wall_center_3d']
    obj_min = obj_bbox['min']
    obj_max = obj_bbox['max']
    corners_3d = [np.array([obj_min[0], obj_min[1], obj_min[2]]), np.array([obj_max[0], obj_min[1], obj_min[2]]), np.array([obj_min[0], obj_max[1], obj_min[2]]), np.array([obj_max[0], obj_max[1], obj_min[2]]), np.array([obj_min[0], obj_min[1], obj_max[2]]), np.array([obj_max[0], obj_min[1], obj_max[2]]), np.array([obj_min[0], obj_max[1], obj_max[2]]), np.array([obj_max[0], obj_max[1], obj_max[2]])]
    projected_2d = []
    for corner in corners_3d:
        pos_2d = world_3d_to_wall_2d(corner, wall_info)
        projected_2d.append(pos_2d)
    if not projected_2d:
        return None
    x_coords = [p[0] for p in projected_2d]
    y_coords = [p[1] for p in projected_2d]
    return {'type': 'object', 'object_id': obj_bbox.get('object_id', 'unknown'), 'x_min': min(x_coords), 'x_max': max(x_coords), 'y_min': min(y_coords), 'y_max': max(y_coords)}

def world_3d_to_wall_2d(pos_3d: np.ndarray, wall_info: Dict) -> np.ndarray:
    """
    Convert 3D world coordinates to 2D wall coordinates.
    """
    wall_direction = wall_info['wall_direction']
    wall_center_3d = wall_info['wall_center_3d']
    relative_pos = pos_3d - wall_center_3d
    x_2d = np.dot(relative_pos, wall_direction)
    y_2d = relative_pos[2]
    return np.array([x_2d, y_2d])

def filter_valid_wall_points(wall_obj: Object, wall_grids: Dict[str, List], impossible_regions: Dict[str, List], wall_systems: Dict[str, Dict]) -> List:
    """
    Filter grid points to find valid placement locations for wall object.
    """
    valid_points = []
    obj_width_2d = wall_obj.dimensions.width
    obj_height_2d = wall_obj.dimensions.height
    minimum_gaps = 0.2
    for (wall_id, grid_points) in wall_grids.items():
        wall_regions = impossible_regions[wall_id]
        for point in grid_points:
            pos_2d = point['pos_2d']
            obj_bbox_2d = {'x_min': pos_2d[0] - obj_width_2d / 2 - minimum_gaps, 'x_max': pos_2d[0] + obj_width_2d / 2 + minimum_gaps, 'y_min': pos_2d[1] - obj_height_2d / 2 - minimum_gaps, 'y_max': pos_2d[1] + obj_height_2d / 2 + minimum_gaps}
            is_valid = True
            for region in wall_regions:
                if rectangles_overlap(obj_bbox_2d, region):
                    is_valid = False
                    break
            wall_info = wall_systems[wall_id]
            if obj_bbox_2d['x_min'] < -wall_info['rect_width'] / 2 or obj_bbox_2d['x_max'] > wall_info['rect_width'] / 2 or obj_bbox_2d['y_min'] < -wall_info['rect_height'] / 2 or (obj_bbox_2d['y_max'] > wall_info['rect_height'] / 2):
                is_valid = False
            if is_valid:
                point['bbox_2d'] = obj_bbox_2d
                valid_points.append(point)
    return valid_points

def rectangles_overlap(rect1: Dict, rect2: Dict) -> bool:
    """
    Check if two rectangles overlap.
    """
    return not (rect1['x_max'] <= rect2['x_min'] or rect2['x_max'] <= rect1['x_min'] or rect1['y_max'] <= rect2['y_min'] or (rect2['y_max'] <= rect1['y_min']))

def select_best_wall_placement(wall_obj: Object, valid_points: List, placed_objects: List[Object], wall_systems: Dict[str, Dict]) -> Dict:
    """
    Select the best placement point for wall object using VLM constraints when available.
    """
    if not valid_points:
        return None
    wall_height = wall_systems[valid_points[0]['wall_id']]['rect_height']
    HEIGHT_POSITION_RATIOS = {'top': 0.9, 'upper': 0.75, 'middle': 0.55, 'lower': 0.4, 'low': 0.25}
    constraints_result = get_wall_object_constraints_from_vlm(wall_obj, placed_objects, wall_height)
    if constraints_result is None:
        return None
    constraints = constraints_result
    formatted_constraints = [[]]
    if constraints:
        for constraint in constraints:
            formatted_constraints[-1].append({'type': 'alignment', 'constraint': constraint['type'], 'target': constraint['target']})
    scored_points = []
    for point in valid_points:
        score = 0.0
        pos_2d = point['pos_2d']
        wall_info = wall_systems[point['wall_id']]
        if constraints:
            for constraint in constraints:
                if constraint['type'] == 'on_top_of':
                    target_obj_id = constraint['target']
                    height_position = constraint['height_position']
                    height_ratio_from_ground = HEIGHT_POSITION_RATIOS.get(height_position, 0.55)
                    estimated_height = wall_info['rect_height'] * height_ratio_from_ground
                    height_ratio = 1.0 - abs(pos_2d[1] + wall_info['rect_height'] / 2 - estimated_height) / wall_info['rect_height']
                    score += height_ratio * 0.1
                    target_obj = next((obj for obj in placed_objects if obj.id == target_obj_id), None)
                    if target_obj:
                        target_wall_distance = float('inf')
                        target_wall_id = None
                        target_bbox = get_object_3d_bbox(target_obj)
                        for (wall_id, wall_info_check) in wall_systems.items():
                            wall_dist = calculate_distance_to_wall(target_bbox, wall_info_check)
                            if wall_dist < target_wall_distance:
                                target_wall_distance = wall_dist
                                target_wall_id = wall_id
                        if point['wall_id'] == target_wall_id:
                            target_pos_3d = np.array([target_obj.position.x, target_obj.position.y, target_obj.position.z + target_obj.dimensions.height])
                            target_pos_2d = world_3d_to_wall_2d(target_pos_3d, wall_info)
                            if pos_2d[1] > target_pos_2d[1]:
                                horizontal_distance = abs(pos_2d[0] - target_pos_2d[0])
                                if horizontal_distance < 1.5:
                                    score += 1.5 - horizontal_distance
        scored_points.append((score, point))
    scored_points.sort(key=lambda x: x[0], reverse=True)
    best_point = scored_points[0][1]
    wall_info = wall_systems[best_point['wall_id']]
    wall_normal = wall_info['wall_normal']
    target_direction = wall_normal[:2]
    current_direction = np.array([0, 1])
    angle = math.atan2(target_direction[1], target_direction[0]) - math.atan2(current_direction[1], current_direction[0])
    rotation_z = math.degrees(angle)
    rotation_z = round(rotation_z / 90) * 90
    rotation_z = rotation_z % 360
    return {'wall_id': best_point['wall_id'], 'position_2d': best_point['pos_2d'], 'position_3d': best_point['pos_3d'], 'rotation': rotation_z, 'bbox_2d': best_point['bbox_2d'], 'scored_points': scored_points, 'constraints': formatted_constraints}

def get_wall_object_constraints_from_vlm(wall_obj: Object, placed_objects: List[Object], wall_height: float) -> List[Dict]:
    """
    Get constraints for wall object placement using VLM.
    Returns list of constraints like [{"type": "on_top_of", "target": "desk_123"}], or None if VLM call fails.
    """
    try:
        if not hasattr(sys.modules[__name__], 'call_vlm'):
            print('VLM function not available', file=sys.stderr)
            return None
        placed_objects_info = []
        for obj in placed_objects:
            obj_info = f'- {obj.type} (ID: {obj.id}) at position ({obj.position.x:.1f}, {obj.position.y:.1f}, {obj.position.z:.1f})'
            if hasattr(obj, 'description') and obj.description:
                obj_info += f' - {obj.description}'
            placed_objects_info.append(obj_info)
        placed_objects_text = '\n'.join(placed_objects_info) if placed_objects_info else 'No objects currently placed.'
        wall_obj_info = f'{wall_obj.type} (ID: {wall_obj.id})'
        if hasattr(wall_obj, 'description') and wall_obj.description:
            wall_obj_info += f' - {wall_obj.description}'
        if hasattr(wall_obj, 'place_guidance') and wall_obj.place_guidance:
            wall_obj_info += f' - Guidance: {wall_obj.place_guidance}'
        prompt = f"""You are an interior designer helping to place a wall-mounted object in a room.\n\nOBJECT TO PLACE:\n{wall_obj_info}\n\nEXISTING OBJECTS IN ROOM:\n{placed_objects_text}\n\nTASK:\nDetermine if this wall object should be placed "on top of" (above) any existing object, and determine the RELATIVE HEIGHT POSITION on the wall. \nWall objects are typically placed above related furniture items.\n\nWall Height: {wall_height}m\n\nIMPORTANT: Instead of specifying absolute heights in meters, use relative position descriptors that indicate where on the wall the object should be placed.\n\nHEIGHT POSITION DESCRIPTORS (choose one):\n- "top": Upper portion of the wall (e.g., wall-mounted lights, high shelves, ceiling-mounted fixtures)\n- "upper": Upper-middle portion (e.g., wall clocks, high artwork, sconces in tall spaces)\n- "middle": Middle portion at eye level (e.g., most wall art, mirrors, typical decorative items)\n- "lower": Lower-middle portion (e.g., light switches, wall outlets, low artwork above furniture)\n- "low": Lower portion of the wall (e.g., items just above low furniture, baseboard decorations)\n\nExamples with typical relative positions:\n- Wall lights/sconces: "top" or "upper" - positioned high to provide ambient lighting\n- Wall art above sofas: "middle" - at eye level when seated\n- Wall art above beds: "lower" or "middle" - hung lower for viewing from lying/sitting position\n- Wall shelves above desks: "middle" or "upper" - accessible while seated or standing\n- Mirrors above dressers: "middle" - at typical eye level for standing adults\n- Mirrors above bathroom vanities: "middle" - positioned for face viewing while standing\n- Wall-mounted TV above media console: "lower" or "middle" - at comfortable viewing height from seated position\n- Floating shelves above toilets: "upper" - high enough to clear head space\n- Cabinet storage: "upper" or "middle" - depending on accessibility needs\n- Decorative wall plates/art: "middle" - standard gallery height\n\nGeneral guidelines:\n- "middle" (~50-60% of wall height): Standard for most decorative items, mirrors, and art\n- "upper" (~70-80% of wall height): For items that should be elevated but still accessible\n- "top" (~85-95% of wall height): For lighting fixtures and items that don't need frequent access\n- "lower" (~35-45% of wall height): For items above furniture or in lower viewing contexts\n- "low" (~20-30% of wall height): For items very close to furniture surfaces\n\nRESPONSE FORMAT:\nReturn a JSON object with the following format:\n\n```json\n{{\n    "given_placement_guidance": "the placement guidance of the wall object",\n    "reasoning": "detailed explanation of why this placement makes sense (analyze the object type, placement guidance, and choose the appropriate height position)",\n    "object_id": "the_object_id_to_place_above",\n    "height_position": "one of: top, upper, middle, lower, low"\n}}\n```\n\nIf there are multiple objects that the wall object could be placed above, choose the one with highest priority.\nIf no specific placement relationship exists, you still need to come up with your own judgment and select the most appropriate object.\n\nConsider the object type and placement guidance when making your decision."""
        response = call_vlm(vlm_type='openai', max_tokens=8000, temperature=0.3, messages=[{'role': 'user', 'content': prompt}])
        if not response or not hasattr(response, 'content') or (not response.content):
            print('Invalid VLM response received', file=sys.stderr)
            return None
        response_text = response.content[0].text.strip()
        try:
            json_text = extract_json_from_response(response_text)
            response_data = json.loads(json_text)
        except Exception as e:
            print(f'Failed to parse JSON from VLM response: {e}', file=sys.stderr)
            print(f'Response text: {response_text}', file=sys.stderr)
            return None
        if 'object_id' not in response_data:
            print(f"Missing 'object_id' in VLM response: {response_data}", file=sys.stderr)
            return None
        target_id = response_data['object_id']
        reasoning = response_data.get('reasoning', 'No reasoning provided')
        height_position = response_data.get('height_position', 'middle')
        valid_positions = ['top', 'upper', 'middle', 'lower', 'low']
        if height_position not in valid_positions:
            print(f"Invalid height_position '{height_position}' in VLM response, defaulting to 'middle'", file=sys.stderr)
            height_position = 'middle'
        print(f'Wall object placement reasoning: {reasoning}', file=sys.stderr)
        print(f'Wall object height position: {height_position}', file=sys.stderr)
        constraints = []
        if any((obj.id == target_id for obj in placed_objects)):
            constraints.append({'type': 'on_top_of', 'target': target_id, 'height_position': height_position})
        else:
            print(f"Target object '{target_id}' not found in placed_objects", file=sys.stderr)
            constraints = []
        return constraints
    except Exception as e:
        print(f'Error getting VLM constraints for wall object: {e}', file=sys.stderr)
        return None

def adjust_wall_object_position(placement: Dict, wall_obj: Object, wall_systems: Dict[str, Dict]) -> Dict:
    """
    Adjust wall object position to account for object dimensions, wall facing direction, and wall thickness.
    The placement position is currently slightly inward from the wall surface (by thickness * 0.4),
    but the object center should be positioned properly considering both wall thickness and object depth.
    
    Args:
        placement: Original placement dictionary with position_3d at wall surface
        wall_obj: The wall object being placed
        wall_systems: Wall coordinate system information
        
    Returns:
        Adjusted placement dictionary with corrected 3D position
    """
    wall_id = placement['wall_id']
    wall_info = wall_systems[wall_id]
    wall_normal = wall_info['wall_normal']
    wall_thickness = wall_info['thickness']
    object_depth = wall_obj.dimensions.length
    current_inward_offset = wall_thickness * 0.4
    desired_outward_offset = object_depth / 2
    total_adjustment = current_inward_offset + desired_outward_offset
    adjustment_vector = wall_normal * total_adjustment
    original_pos_3d = placement['position_3d']
    adjusted_pos_3d = original_pos_3d + adjustment_vector
    adjusted_placement = placement.copy()
    adjusted_placement['position_3d'] = adjusted_pos_3d
    return adjusted_placement

def create_wall_placed_object(wall_obj: Object, placement: Dict, room: Room, constraints: List[Dict]) -> Object:
    """
    Create a placed wall object with proper 3D position and rotation.
    """
    pos_3d = placement['position_3d']
    rotation_z = placement['rotation']
    adjusted_z = pos_3d[2] - wall_obj.dimensions.height / 2
    return Object(id=wall_obj.id, room_id=room.id, type=wall_obj.type, description=wall_obj.description if hasattr(wall_obj, 'description') else f'Wall-mounted {wall_obj.type}', position=Point3D(x=pos_3d[0], y=pos_3d[1], z=adjusted_z), rotation=Euler(x=0, y=0, z=rotation_z), dimensions=wall_obj.dimensions, source=wall_obj.source if hasattr(wall_obj, 'source') else 'placement', source_id=wall_obj.source_id if hasattr(wall_obj, 'source_id') else wall_obj.id, place_id='wall', place_guidance=getattr(wall_obj, 'place_guidance', 'Wall-mounted placement'), mass=getattr(wall_obj, 'mass', 1.0), placement_constraints=constraints)

def visualize_wall_placement(wall_obj: Object, wall_systems: Dict[str, Dict], wall_grids: Dict[str, List], impossible_regions: Dict[str, List], valid_points: List, best_placement: Dict, room_id: str):
    """
    Visualize wall object placement process showing all 4 walls as 2D rectangles.
    Shows grid points, impossible regions, valid points, and final placement.
    """
    try:
        vis_dir = f'{SERVER_ROOT_DIR}/vis'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        (fig, axes) = plt.subplots(1, 4, figsize=(22, 6))
        fig.suptitle(f'Wall Object Placement: {wall_obj.type} (ID: {wall_obj.id})', fontsize=16, fontweight='bold')
        axes_flat = axes if isinstance(axes, np.ndarray) else [axes]
        wall_ids = list(wall_systems.keys())
        global_scores = []
        global_scatter = None
        if best_placement and 'scored_points' in best_placement:
            scored_points = best_placement['scored_points']
            global_scores = [score for (score, _) in scored_points]
        if global_scores:
            global_vmin = min(global_scores)
            global_vmax = max(global_scores)
            if global_vmax == global_vmin:
                global_vmax = global_vmin + 0.001
        else:
            global_vmin = 0
            global_vmax = 1
        for (i, wall_id) in enumerate(wall_ids[:4]):
            ax = axes_flat[i]
            wall_info = wall_systems[wall_id]
            rect_width = wall_info['rect_width']
            rect_height = wall_info['rect_height']
            wall_rect = plt.Rectangle((-rect_width / 2, -rect_height / 2), rect_width, rect_height, fill=False, edgecolor='black', linewidth=3, label='Wall Boundary')
            ax.add_patch(wall_rect)
            if wall_id in wall_grids:
                grid_points = wall_grids[wall_id]
                if grid_points:
                    grid_x = [point['pos_2d'][0] for point in grid_points]
                    grid_y = [point['pos_2d'][1] for point in grid_points]
                    ax.scatter(grid_x, grid_y, c='lightgray', s=8, alpha=0.5, label='Grid Points')
            impossible_rects = impossible_regions.get(wall_id, [])
            for region in impossible_rects:
                region_width = region['x_max'] - region['x_min']
                region_height = region['y_max'] - region['y_min']
                if region['type'] == 'door':
                    color = 'brown'
                    alpha = 0.7
                elif region['type'] == 'window':
                    color = 'cyan'
                    alpha = 0.6
                elif region['type'] == 'object':
                    color = 'red'
                    alpha = 0.5
                else:
                    color = 'gray'
                    alpha = 0.5
                impossible_rect = plt.Rectangle((region['x_min'], region['y_min']), region_width, region_height, facecolor=color, alpha=alpha, edgecolor='black', linewidth=1)
                ax.add_patch(impossible_rect)
                center_x = region['x_min'] + region_width / 2
                center_y = region['y_min'] + region_height / 2
                ax.text(center_x, center_y, region['type'].capitalize(), ha='center', va='center', fontsize=8, fontweight='bold')
            wall_valid_points = [p for p in valid_points if p['wall_id'] == wall_id]
            if wall_valid_points:
                if best_placement and 'scored_points' in best_placement:
                    scored_points = best_placement['scored_points']
                    point_to_score = {}
                    for (score, point) in scored_points:
                        if point['wall_id'] == wall_id:
                            point_key = (point['pos_2d'][0], point['pos_2d'][1])
                            point_to_score[point_key] = score
                    valid_x = []
                    valid_y = []
                    scores = []
                    for point in wall_valid_points:
                        (x, y) = (point['pos_2d'][0], point['pos_2d'][1])
                        valid_x.append(x)
                        valid_y.append(y)
                        point_key = (x, y)
                        scores.append(point_to_score.get(point_key, 0.0))
                    if scores and global_scores:
                        scatter = ax.scatter(valid_x, valid_y, c=scores, s=20, alpha=0.8, cmap='viridis', marker='o', label='Valid Points (scored)', vmin=global_vmin, vmax=global_vmax)
                        if global_scatter is None:
                            global_scatter = scatter
                    else:
                        ax.scatter(valid_x, valid_y, c='green', s=20, alpha=0.8, label='Valid Points', marker='o')
                else:
                    valid_x = [point['pos_2d'][0] for point in wall_valid_points]
                    valid_y = [point['pos_2d'][1] for point in wall_valid_points]
                    ax.scatter(valid_x, valid_y, c='green', s=20, alpha=0.8, label='Valid Points', marker='o')
            if best_placement and best_placement['wall_id'] == wall_id:
                pos_2d = best_placement['position_2d']
                bbox_2d = best_placement['bbox_2d']
                obj_width = bbox_2d['x_max'] - bbox_2d['x_min']
                obj_height = bbox_2d['y_max'] - bbox_2d['y_min']
                placement_rect = plt.Rectangle((bbox_2d['x_min'], bbox_2d['y_min']), obj_width, obj_height, facecolor='gold', alpha=0.8, edgecolor='orange', linewidth=3, label='Final Placement')
                ax.add_patch(placement_rect)
                ax.scatter([pos_2d[0]], [pos_2d[1]], c='red', s=100, marker='x', linewidth=3, label='Placement Center')
                ax.text(pos_2d[0], pos_2d[1] + obj_height / 2 + 0.1, f'{wall_obj.type}\n{wall_obj.id}', ha='center', va='bottom', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            ax.set_xlim(-rect_width / 2 - 0.2, rect_width / 2 + 0.2)
            ax.set_ylim(-rect_height / 2 - 0.2, rect_height / 2 + 0.2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Wall Width (m)', fontsize=10)
            ax.set_ylabel('Wall Height (m)', fontsize=10)
            wall = wall_info['wall']
            wall_title = f'Wall {wall_id}\n({rect_width:.1f}m × {rect_height:.1f}m)'
            if wall_id in wall_grids:
                wall_title += f'\nGrid: {len(wall_grids[wall_id])} points'
            if wall_valid_points:
                wall_title += f'\nValid: {len(wall_valid_points)} points'
            ax.set_title(wall_title, fontsize=11, fontweight='bold')
            if i == 0:
                ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), fontsize=8)
        for i in range(len(wall_ids), 4):
            axes_flat[i].set_visible(False)
        if global_scatter is not None:
            cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
            cbar = plt.colorbar(global_scatter, cax=cbar_ax)
            cbar.set_label('Placement Score', fontsize=10, rotation=270, labelpad=15)
        summary_text = f"Wall Object Placement Summary:\nObject: {wall_obj.type} (ID: {wall_obj.id})\nDimensions: {wall_obj.dimensions.width:.2f}m × {wall_obj.dimensions.length:.2f}m × {wall_obj.dimensions.height:.2f}m\nTotal Valid Points: {len(valid_points)}\nPlacement Result: {('SUCCESS' if best_placement else 'FAILED')}"
        if best_placement:
            summary_text += f"\nPlaced on Wall: {best_placement['wall_id']}\nPosition 2D: ({best_placement['position_2d'][0]:.2f}, {best_placement['position_2d'][1]:.2f})\nPosition 3D: ({best_placement['position_3d'][0]:.2f}, {best_placement['position_3d'][1]:.2f}, {best_placement['position_3d'][2]:.2f})\nRotation: {best_placement['rotation']:.0f}°"
            if 'scored_points' in best_placement:
                scored_points = best_placement['scored_points']
                scores = [score for (score, _) in scored_points]
                if scores:
                    summary_text += f'\nScoring Info:\n  Best Score: {max(scores):.3f}\n  Avg Score: {sum(scores) / len(scores):.3f}\n  Score Range: {min(scores):.3f} - {max(scores):.3f}'
        fig.text(0.02, 0.02, summary_text, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8), verticalalignment='bottom')
        if global_scatter is not None:
            plt.tight_layout(rect=[0, 0.25, 0.85, 0.96])
        else:
            plt.tight_layout(rect=[0, 0.25, 1, 0.96])
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{vis_dir}/wall_placement_{room_id}_{wall_obj.id}_{timestamp}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        print(f'Wall placement visualization saved: {filename}', file=sys.stderr)
        plt.close(fig)
    except Exception as e:
        print(f'Error creating wall placement visualization: {e}', file=sys.stderr)
        pass

class SolutionFound(Exception):

    def __init__(self, solution):
        self.solution = solution

class DFS_Solver_Floor:

    def __init__(self, grid_size, random_seed=0, max_duration=5, constraint_bouns=0.2, enable_retry=False, room_id=None):
        self.grid_size = grid_size
        self.random_seed = random_seed
        self.max_duration = max_duration
        self.constraint_bouns = constraint_bouns
        self.enable_retry = enable_retry
        self.start_time = None
        self.solutions = []
        self.constraints_dict_list = []
        self.retry_info = {'retries': 0, 'max_retries': 3, 'objects_removed': []}
        self.func_dict = {'global': {'edge': self.place_edge}, 'relative': self.place_relative, 'direction': self.place_face, 'alignment': self.place_alignment_center, 'distance': self.place_distance}
        self.constraint_type2weight = {'global': 2.0, 'relative': 1.0, 'direction': 2.0, 'alignment': 0.8, 'distance': 1.0}
        self.edge_bouns = 0.0
        self.room_id = room_id

    def get_solution(self, bounds, objects_list, constraints, initial_state, use_milp=False):
        self.start_time = time.time()
        original_objects_list = objects_list.copy()
        original_constraints = constraints.copy()
        constraints_dict = {}
        (result, result_constraints) = self._attempt_solution(bounds, objects_list, constraints, initial_state, constraints_dict, use_milp)
        if not result:
            result = {}
            result_constraints = {}
        return (result, result_constraints)

    def _attempt_solution(self, bounds, objects_list, constraints, initial_state, constraints_dict, use_milp=False):
        self.solutions = []
        self.constraints_dict_list = []
        if use_milp:
            pass
        else:
            grid_points = self.create_grids(bounds)
            grid_points = self.remove_points(grid_points, initial_state)
            self.dfs(bounds, objects_list, constraints, grid_points, initial_state, constraints_dict, 15 if len(objects_list) < 20 else 5)
        if self.solutions:
            max_solution = self.get_max_solution(self.solutions)
            max_solution_constraints = self.get_max_solution_constraints(self.solutions)
            return (max_solution, max_solution_constraints)
        return (None, None)

    def get_max_solution(self, solutions):
        path_weights = []
        for solution in solutions:
            path_weights.append(sum([obj[-1] for obj in solution.values()]))
        max_index = np.argmax(path_weights)
        return solutions[max_index]

    def get_max_solution_constraints(self, solutions):
        path_weights = []
        for solution in solutions:
            path_weights.append(sum([obj[-1] for obj in solution.values()]))
        max_index = np.argmax(path_weights)
        return self.constraints_dict_list[max_index]

    def dfs(self, room_poly, objects_list, constraints, grid_points, placed_objects, constraints_dict, branch_factor):
        if len(objects_list) == 0:
            self.solutions.append(placed_objects)
            self.constraints_dict_list.append(constraints_dict)
            return placed_objects
        if time.time() - self.start_time > self.max_duration:
            raise Exception(f'Time limit reached.')
            raise SolutionFound(self.solutions)
        (object_id, object_dim) = objects_list[0]
        print(f'dfs object_id: {object_id}, placed_objects: {len(placed_objects)}, object_dim: {object_dim}', file=sys.stderr)
        object_constraints = constraints.get(object_id, [{'type': 'global', 'constraint': 'edge'}])
        (placements, placements_constraints) = self.get_possible_placements(room_poly, object_dim, object_constraints, grid_points, placed_objects)
        if len(placements) == 0 and len(placed_objects) != 0:
            self.solutions.append(placed_objects)
            self.constraints_dict_list.append(constraints_dict)
        paths = []
        if branch_factor > 1 and len(placements) > branch_factor:
            scores = np.array([placement[-1] for placement in placements])
            temperature = 0.4
            scores_scaled = scores / temperature
            exp_scores = np.exp(scores_scaled - np.max(scores_scaled))
            probabilities = exp_scores / np.sum(exp_scores)
            candidate_indices = np.flatnonzero(probabilities > 0)
            candidate_probabilities = probabilities[candidate_indices]
            candidate_probabilities /= candidate_probabilities.sum()
            selected_indices = np.random.choice(candidate_indices, size=min(branch_factor, len(candidate_indices)), replace=False, p=candidate_probabilities)
            selected_placements = [placements[i] for i in selected_indices]
            selected_placements_constraints = [placements_constraints[i] for i in selected_indices]
        else:
            selected_placements = placements[:branch_factor]
            selected_placements_constraints = placements_constraints[:branch_factor]
        for (placement, placement_constraints) in zip(selected_placements, selected_placements_constraints):
            placed_objects_updated = copy.deepcopy(placed_objects)
            placed_objects_updated[object_id] = placement
            grid_points_updated = self.remove_points(grid_points, placed_objects_updated)
            constraints_dict_updated = copy.deepcopy(constraints_dict)
            constraints_dict_updated[object_id] = placement_constraints
            sub_paths = self.dfs(room_poly, objects_list[1:], constraints, grid_points_updated, placed_objects_updated, constraints_dict_updated, 1)
            paths.extend(sub_paths)
        return paths

    def get_possible_placements(self, room_poly, object_dim, constraints, grid_points, placed_objects):
        solutions = self.filter_collision(placed_objects, self.get_all_solutions(room_poly, grid_points, object_dim))
        solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
        edge_solutions = self.place_edge(room_poly, copy.deepcopy(solutions), object_dim, placed_objects=None)
        global_constraint = next((constraint for constraint in constraints if constraint['type'] == 'global'), None)
        if global_constraint is None:
            global_constraint = {'type': 'global', 'constraint': 'edge'}
        if global_constraint['constraint'] == 'edge':
            candidate_solutions = copy.deepcopy(edge_solutions)
        else:
            candidate_solutions = copy.deepcopy(solutions)
        candidate_solutions = self.filter_collision(placed_objects, candidate_solutions)
        if candidate_solutions == []:
            return (candidate_solutions, [])
        random.shuffle(candidate_solutions)
        placement2score = {tuple(solution[:3]): float(solution[-1]) for solution in candidate_solutions}
        placement2constraints = {tuple(solution[:3]): [] for solution in candidate_solutions}
        for solution in candidate_solutions:
            if len(edge_solutions) > 0 and solution in edge_solutions and (len(constraints) >= 1):
                placement2score[tuple(solution[:3])] += self.edge_bouns
                placement2constraints[tuple(solution[:3])].append({'type': 'global', 'constraint': 'edge'})
        for constraint in constraints:
            if constraint['type'] == 'global' and constraint['constraint'] == 'middle':
                valid_solutions = self.place_middle(candidate_solutions, placed_objects, room_poly)
                for solution in valid_solutions:
                    bouns = solution[-1]
                    placement2score[tuple(solution[:3])] += bouns * 1.0
                    placement2constraints[tuple(solution[:3])].append(constraint)
                continue
            if 'target' not in constraint:
                continue
            if constraint['target'] not in placed_objects:
                print(f'skipping constraint because target not in placed_objects: {constraint}', file=sys.stderr)
                continue
            func = self.func_dict.get(constraint['type'])
            valid_solutions = func(constraint['constraint'], placed_objects[constraint['target']], candidate_solutions)
            weight = self.constraint_type2weight[constraint['type']]
            if constraint['type'] == 'distance':
                for solution in valid_solutions:
                    bouns = solution[-1]
                    placement2score[tuple(solution[:3])] += bouns * weight
                    placement2constraints[tuple(solution[:3])].append(constraint)
            elif constraint['type'] == 'relative':
                for solution in valid_solutions:
                    bouns = solution[-1]
                    placement2score[tuple(solution[:3])] += bouns * weight
                    placement2constraints[tuple(solution[:3])].append(constraint)
            else:
                for solution in valid_solutions:
                    if constraint['constraint'] == 'face to':
                        face_to_score = solution[-1]
                        placement2score[tuple(solution[:3])] += face_to_score * self.constraint_bouns * weight
                        placement2constraints[tuple(solution[:3])].append(constraint)
                    else:
                        placement2score[tuple(solution[:3])] += self.constraint_bouns * weight
                        placement2constraints[tuple(solution[:3])].append(constraint)
        sorted_by_score = sorted(placement2score.keys(), key=lambda p: -placement2score[p])
        top_k = max(5, int(len(sorted_by_score) * 0.03))
        top_placements = sorted_by_score[:top_k]
        if top_placements:
            mean_x = sum((p[0][0] for p in top_placements)) / len(top_placements)
            mean_y = sum((p[0][1] for p in top_placements)) / len(top_placements)
        else:
            (mean_x, mean_y) = (0, 0)
        sorted_placements = sorted(placement2score.keys(), key=lambda p: (-placement2score[p], ((p[0][0] - mean_x) ** 2 + (p[0][1] - mean_y) ** 2) ** 0.5))
        sorted_solutions = [list(placement) + [placement2score[placement]] for placement in sorted_placements]
        sorted_solutions_constraints = [list(placement) + [placement2constraints[placement]] for placement in sorted_placements]
        if sorted_solutions[0][-1] < 0:
            return ([], [])
        return (sorted_solutions, sorted_solutions_constraints)

    def create_grids(self, room_poly):
        (min_x, min_y, max_x, max_y) = room_poly.bounds
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_y), int(max_y), self.grid_size):
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))
        return grid_points

    def remove_points(self, grid_points, objects_dict):
        idx = index.Index()
        for (i, (_, _, obj, _)) in enumerate(objects_dict.values()):
            idx.insert(i, Polygon(obj).bounds)
        polygons = [Polygon(obj) for (_, _, obj, _) in objects_dict.values()]
        valid_points = []
        for point in grid_points:
            p = Point(point)
            candidates = [polygons[i] for i in idx.intersection(p.bounds)]
            if not any((candidate.contains(p) for candidate in candidates)):
                valid_points.append(point)
        return valid_points

    def get_all_solutions(self, room_poly, grid_points, object_dim):
        (obj_length, obj_width) = object_dim
        (obj_half_length, obj_half_width) = (obj_length / 2, obj_width / 2)
        rotation_adjustments = {0: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)), 90: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length)), 180: ((-obj_half_length, -obj_half_width), (obj_half_length, obj_half_width)), 270: ((-obj_half_width, -obj_half_length), (obj_half_width, obj_half_length))}
        solutions = []
        for rotation in [0, 90, 180, 270]:
            for point in grid_points:
                (center_x, center_y) = point
                (lower_left_adjustment, upper_right_adjustment) = rotation_adjustments[rotation]
                lower_left = (center_x + lower_left_adjustment[0], center_y + lower_left_adjustment[1])
                upper_right = (center_x + upper_right_adjustment[0], center_y + upper_right_adjustment[1])
                obj_box = box(*lower_left, *upper_right)
                if room_poly.contains(obj_box):
                    solutions.append([point, rotation, tuple(obj_box.exterior.coords[:]), 1])
        return solutions

    def filter_collision(self, objects_dict, solutions):
        valid_solutions = []
        object_polygons = [Polygon(obj_coords) for (_, _, obj_coords, _) in list(objects_dict.values())]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any((sol_obj.intersects(obj) for obj in object_polygons)):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2
        front_center_adjustments = {0: (0, obj_half_width), 90: (-obj_half_width, 0), 180: (0, -obj_half_width), 270: (obj_half_width, 0)}
        valid_solutions = []
        for solution in solutions:
            (center_x, center_y) = solution[0]
            rotation = solution[1]
            front_center_adjustment = front_center_adjustments[rotation]
            (front_center_x, front_center_y) = (center_x + front_center_adjustment[0], center_y + front_center_adjustment[1])
            front_center_distance = room_poly.boundary.distance(Point(front_center_x, front_center_y))
            if front_center_distance >= 10:
                valid_solutions.append(solution)
        return valid_solutions

    def place_edge(self, room_poly, solutions, obj_dim, placed_objects=None):
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2
        back_center_adjustments = {0: (0, -obj_half_width), 90: (obj_half_width, 0), 180: (0, obj_half_width), 270: (-obj_half_width, 0)}
        two_side_vectors = {0: [(1, 0), (-1, 0)], 90: [(0, 1), (0, -1)], 180: [(1, 0), (-1, 0)], 270: [(0, 1), (0, -1)]}
        for solution in solutions:
            (center_x, center_y) = solution[0]
            rotation = solution[1]
            back_center_adjustment = back_center_adjustments[rotation]
            (back_center_x, back_center_y) = (center_x + back_center_adjustment[0], center_y + back_center_adjustment[1])
            back_center_distance = room_poly.boundary.distance(Point(back_center_x, back_center_y))
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))
            if back_center_distance <= self.grid_size and back_center_distance < center_distance:
                solution[-1] += self.constraint_bouns
                center2back_vector = np.array([back_center_x - center_x, back_center_y - center_y])
                center2back_vector /= np.linalg.norm(center2back_vector)
                offset = center2back_vector * back_center_distance
                solution[0] = (center_x + offset[0], center_y + offset[1])
                solution[2] = ((solution[2][0][0] + offset[0], solution[2][0][1] + offset[1]), (solution[2][1][0] + offset[0], solution[2][1][1] + offset[1]), (solution[2][2][0] + offset[0], solution[2][2][1] + offset[1]), (solution[2][3][0] + offset[0], solution[2][3][1] + offset[1]))
                if placed_objects is not None:
                    sol_center = solution[0]
                    sol_rotation = solution[1]
                    sol_center_np = np.array(sol_center)
                    left_vector = np.array(two_side_vectors[sol_rotation][0])
                    right_vector = np.array(two_side_vectors[sol_rotation][1])
                    far_point_left = sol_center_np + 1000000.0 * left_vector
                    far_point_right = sol_center_np + 1000000.0 * right_vector
                    left_line = LineString([sol_center, tuple(far_point_left)])
                    right_line = LineString([sol_center, tuple(far_point_right)])
                    min_distances = []
                    for line in [left_line, right_line]:
                        room_intersection = line.intersection(room_poly.boundary)
                        room_points = []
                        if not room_intersection.is_empty:
                            if isinstance(room_intersection, Point):
                                room_points.append((room_intersection.x, room_intersection.y))
                            elif isinstance(room_intersection, MultiPoint):
                                room_points = [(point.x, point.y) for point in room_intersection.geoms]
                            elif isinstance(room_intersection, LineString):
                                room_points = list(room_intersection.coords)
                            elif hasattr(room_intersection, 'geoms'):
                                for geom in room_intersection.geoms:
                                    if isinstance(geom, Point):
                                        room_points.append((geom.x, geom.y))
                                    elif isinstance(geom, LineString):
                                        room_points.extend(list(geom.coords))
                        for point in room_points:
                            dist = np.linalg.norm(sol_center_np - np.array(point))
                            min_distances.append(dist)
                        for obj_data in placed_objects.values():
                            obj_coords = obj_data[2]
                            obj_poly = Polygon(obj_coords)
                            obj_intersection = line.intersection(obj_poly)
                            obj_points = []
                            if not obj_intersection.is_empty:
                                if isinstance(obj_intersection, Point):
                                    obj_points.append((obj_intersection.x, obj_intersection.y))
                                elif isinstance(obj_intersection, MultiPoint):
                                    obj_points = [(point.x, point.y) for point in obj_intersection.geoms]
                                elif isinstance(obj_intersection, LineString):
                                    obj_points = list(obj_intersection.coords)
                                elif hasattr(obj_intersection, 'geoms'):
                                    for geom in obj_intersection.geoms:
                                        if isinstance(geom, Point):
                                            obj_points.append((geom.x, geom.y))
                                        elif isinstance(geom, LineString):
                                            obj_points.extend(list(geom.coords))
                            for point in obj_points:
                                dist = np.linalg.norm(sol_center_np - np.array(point))
                                min_distances.append(dist)
                    min_distance_to_obstacle = min(min_distances) if min_distances else 0.0
                    solution[-1] += min(min_distance_to_obstacle, 1000.0) / 1000.0 * 0.01
                valid_solutions.append(solution)
        return valid_solutions

    def place_relative(self, place_type, target_object, solutions):
        valid_solutions = []
        (_, target_rotation, target_coords, _) = target_object
        target_polygon = Polygon(target_coords)
        (min_x, min_y, max_x, max_y) = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2
        comparison_dict = {'left of': {0: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y, 90: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x, 180: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y, 270: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x}, 'right of': {0: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y, 90: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x, 180: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y, 270: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x}, 'in front of': {0: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x, 90: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y, 180: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x, 270: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y}, 'behind': {0: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x, 90: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y, 180: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x, 270: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y}, 'side of': {0: lambda sol_center: min_y <= sol_center[1] <= max_y, 90: lambda sol_center: min_x <= sol_center[0] <= max_x, 180: lambda sol_center: min_y <= sol_center[1] <= max_y, 270: lambda sol_center: min_x <= sol_center[0] <= max_x}}
        compare_func = comparison_dict.get(place_type).get(target_rotation)
        comparison_dict_loose = {'left of': {0: lambda sol_center: sol_center[0] < min_x, 90: lambda sol_center: sol_center[1] < min_y, 180: lambda sol_center: sol_center[0] > max_x, 270: lambda sol_center: sol_center[1] > max_y}, 'right of': {0: lambda sol_center: sol_center[0] > max_x, 90: lambda sol_center: sol_center[1] > max_y, 180: lambda sol_center: sol_center[0] < min_x, 270: lambda sol_center: sol_center[1] < min_y}, 'in front of': {0: lambda sol_center: sol_center[1] > max_y, 90: lambda sol_center: sol_center[0] < min_x, 180: lambda sol_center: sol_center[1] < min_y, 270: lambda sol_center: sol_center[0] > max_x}, 'behind': {0: lambda sol_center: sol_center[1] < min_y, 90: lambda sol_center: sol_center[0] > max_x, 180: lambda sol_center: sol_center[1] > max_y, 270: lambda sol_center: sol_center[0] < min_x}, 'side of': {0: lambda sol_center: min_y <= sol_center[1] <= max_y, 90: lambda sol_center: min_x <= sol_center[0] <= max_x, 180: lambda sol_center: min_y <= sol_center[1] <= max_y, 270: lambda sol_center: min_x <= sol_center[0] <= max_x}}
        compare_func_loose = comparison_dict_loose.get(place_type).get(target_rotation)
        comparison_dict_looser = {'left of': {0: lambda sol_center: sol_center[0] < mean_x, 90: lambda sol_center: sol_center[1] < mean_y, 180: lambda sol_center: sol_center[0] > mean_x, 270: lambda sol_center: sol_center[1] > mean_y}, 'right of': {0: lambda sol_center: sol_center[0] > mean_x, 90: lambda sol_center: sol_center[1] > mean_y, 180: lambda sol_center: sol_center[0] < mean_x, 270: lambda sol_center: sol_center[1] < mean_y}, 'in front of': {0: lambda sol_center: sol_center[1] > mean_y, 90: lambda sol_center: sol_center[0] < mean_x, 180: lambda sol_center: sol_center[1] < mean_y, 270: lambda sol_center: sol_center[0] > mean_x}, 'behind': {0: lambda sol_center: sol_center[1] < mean_y, 90: lambda sol_center: sol_center[0] > mean_x, 180: lambda sol_center: sol_center[1] > mean_y, 270: lambda sol_center: sol_center[0] < mean_x}, 'side of': {0: lambda sol_center: min_y <= sol_center[1] <= max_y, 90: lambda sol_center: min_x <= sol_center[0] <= max_x, 180: lambda sol_center: min_y <= sol_center[1] <= max_y, 270: lambda sol_center: min_x <= sol_center[0] <= max_x}}
        compare_func_looser = comparison_dict_looser.get(place_type).get(target_rotation)
        for solution in solutions:
            sol_center = solution[0]
            if compare_func(sol_center):
                solution[-1] = self.constraint_bouns
                valid_solutions.append(solution)
            elif compare_func_loose(sol_center):
                solution[-1] = self.constraint_bouns * 0.2
                valid_solutions.append(solution)
            elif compare_func_looser(sol_center):
                solution[-1] = self.constraint_bouns * 0.01
                valid_solutions.append(solution)
        return valid_solutions

    def place_distance(self, distance_type, target_object, solutions):
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        distances = []
        valid_solutions = []
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)
            distances.append(distance)
            solution[-1] = distance
            valid_solutions.append(solution)
        if not distances:
            return valid_solutions
        min_distance = min(distances)
        max_distance = max(distances)
        if distance_type == 'close to':
            if min_distance > 50:
                points = [(min_distance, -1000000.0), (max_distance, -1000000.0)]
            else:
                points = [(min_distance, 1), (min(max_distance, 50), 0), (max(max_distance, 50), -1000000.0)]
        elif distance_type == 'near':
            if min_distance > 150:
                points = [(min_distance, -1000000.0), (max_distance, -1000000.0)]
            else:
                points = [(0, 0.2), (30, 1), (60, 1), (150, 0), (10000, 0)]
        elif distance_type == 'far':
            points = [(min_distance, 0), (max_distance, 1)]
        x = [point[0] for point in points]
        y = [point[1] for point in points]
        if len(x) > 1:
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            for solution in valid_solutions:
                distance = solution[-1]
                solution[-1] = float(f(distance))
        else:
            for solution in valid_solutions:
                solution[-1] = y[0] if y else 0
        return valid_solutions

    def place_face(self, face_type, target_object, solutions):
        if face_type == 'face to':
            return self.place_face_to(target_object, solutions)
        elif face_type == 'face same as':
            return self.place_face_same(target_object, solutions)

    def place_face_to(self, target_object, solutions):
        unit_vectors = {0: np.array([0.0, 1.0]), 90: np.array([-1.0, 0.0]), 180: np.array([0.0, -1.0]), 270: np.array([1.0, 0.0])}
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        target_coords_center = target_object[0]
        (target_obj_x_cm, target_obj_y_cm) = (target_coords_center[0], target_coords_center[1])
        target_coords_np = np.array(target_coords).reshape(-1, 2)
        (coords_x_max, coords_y_max) = target_coords_np.max(axis=0)
        (coords_x_min, coords_y_min) = target_coords_np.min(axis=0)
        target_obj_x_cm = (coords_x_max + coords_x_min) / 2
        target_obj_y_cm = (coords_y_max + coords_y_min) / 2
        target_coords_np_x_inf = np.copy(target_coords_np)
        target_coords_np_x_inf[target_coords_np_x_inf[:, 0] > target_obj_x_cm, 0] = coords_x_max + 1000000.0
        target_coords_np_x_inf[target_coords_np_x_inf[:, 0] < target_obj_x_cm, 0] = coords_x_min - 1000000.0
        target_coords_np_y_inf = np.copy(target_coords_np)
        target_coords_np_y_inf[target_coords_np_y_inf[:, 1] > target_obj_y_cm, 1] = coords_y_max + 1000000.0
        target_coords_np_y_inf[target_coords_np_y_inf[:, 1] < target_obj_y_cm, 1] = coords_y_min - 1000000.0
        target_coords_x_inf = target_coords_np_x_inf.tolist()
        target_coords_y_inf = target_coords_np_y_inf.tolist()
        target_poly_x_inf = Polygon(target_coords_x_inf)
        target_poly_y_inf = Polygon(target_coords_y_inf)
        valid_solutions = []
        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]
            far_point = sol_center + 1000000.0 * unit_vectors[sol_rotation]
            half_line = LineString([sol_center, far_point])
            sol_center_point = Point(sol_center[0], sol_center[1])
            if half_line.intersects(target_poly):
                solution[-1] = 1.0
                valid_solutions.append(solution)
            else:
                pass
                if not target_poly_x_inf.contains(sol_center_point) and half_line.intersects(target_poly_x_inf):
                    solution[-1] = 0.3
                    valid_solutions.append(solution)
                elif not target_poly_y_inf.contains(sol_center_point) and half_line.intersects(target_poly_y_inf):
                    solution[-1] = 0.3
                    valid_solutions.append(solution)
        return valid_solutions

    def place_face_same(self, target_object, solutions):
        target_rotation = target_object[1]
        valid_solutions = []
        for solution in solutions:
            sol_rotation = solution[1]
            if abs(sol_rotation - target_rotation) < 10:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)
        return valid_solutions

    def place_middle(self, candidate_solutions, placed_objects, room_poly):
        """
        Add a small score bonus to placements that are farther from room edges and other objects.
        This encourages placing objects toward the middle of the room.
        """
        valid_solutions = []
        if not candidate_solutions:
            return valid_solutions
        middle_scores = []
        for solution in candidate_solutions:
            sol_center = solution[0]
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            dist_to_room_edge = room_poly.exterior.distance(sol_poly)
            min_dist_to_objects = float('inf')
            for obj_data in placed_objects.values():
                obj_coords = obj_data[2]
                obj_poly = Polygon(obj_coords)
                dist = sol_poly.distance(obj_poly)
                min_dist_to_objects = min(min_dist_to_objects, dist)
            if min_dist_to_objects == float('inf'):
                middle_score = dist_to_room_edge
            else:
                middle_score = min(dist_to_room_edge, min_dist_to_objects)
            middle_scores.append(middle_score)
        if middle_scores:
            min_score = min(middle_scores)
            max_score = max(middle_scores)
            for (i, solution) in enumerate(candidate_solutions):
                if max_score > min_score:
                    normalized_score = (middle_scores[i] - min_score) / (max_score - min_score)
                else:
                    normalized_score = 0.5
                solution[-1] = normalized_score * 0.02
                valid_solutions.append(solution)
        return valid_solutions

    def place_alignment_center(self, alignment_type, target_object, solutions):
        target_center = target_object[0]
        valid_solutions = []
        eps = self.grid_size / 2
        for solution in solutions:
            sol_center = solution[0]
            if abs(sol_center[0] - target_center[0]) < eps or abs(sol_center[1] - target_center[1]) < eps:
                solution[-1] += self.constraint_bouns
                valid_solutions.append(solution)
        return valid_solutions

    def visualize_grid(self, room_poly, grid_points, solutions, room_id='unknown'):
        """
        Visualize the room layout, grid points, and object placements for debugging
        """
        vis_dir = f'{SERVER_ROOT_DIR}/vis'
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        plt.rcParams['font.size'] = 12
        (fig, ax) = plt.subplots(figsize=(12, 10))
        (x, y) = room_poly.exterior.xy
        ax.plot(x, y, '-', label='Room', color='black', linewidth=2)
        if grid_points:
            grid_x = [point[0] for point in grid_points]
            grid_y = [point[1] for point in grid_points]
            ax.plot(grid_x, grid_y, 'o', markersize=1, color='lightgrey', alpha=0.5)
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        color_idx = 0
        for (object_id, solution) in solutions.items():
            if object_id.startswith(('door-', 'window-', 'open-')):
                (center, rotation, box_coords) = solution[:3]
                (center_x, center_y) = center
                obj_poly = Polygon(box_coords)
                (x_coords, y_coords) = obj_poly.exterior.xy
                if object_id.startswith('door-'):
                    ax.plot(x_coords, y_coords, '-', linewidth=3, color='brown', alpha=0.7)
                    ax.fill(x_coords, y_coords, color='brown', alpha=0.3)
                    ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                elif object_id.startswith('window-'):
                    ax.plot(x_coords, y_coords, '-', linewidth=2, color='cyan', alpha=0.7)
                    ax.fill(x_coords, y_coords, color='cyan', alpha=0.3)
                    ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                else:
                    ax.plot(x_coords, y_coords, '-', linewidth=2, color='orange', alpha=0.7)
                    ax.fill(x_coords, y_coords, color='orange', alpha=0.3)
                    ax.text(center_x, center_y, 'Open', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            else:
                (center, rotation, box_coords) = solution[:3]
                (center_x, center_y) = center
                obj_poly = Polygon(box_coords)
                (x_coords, y_coords) = obj_poly.exterior.xy
                current_color = colors[color_idx % len(colors)]
                ax.plot(x_coords, y_coords, '-', linewidth=2, color=current_color)
                ax.fill(x_coords, y_coords, color=current_color, alpha=0.3)
                ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                arrow_length = 25
                if rotation == 0:
                    ax.arrow(center_x, center_y, 0, arrow_length, head_width=8, head_length=5, fc=current_color, ec=current_color)
                elif rotation == 90:
                    ax.arrow(center_x, center_y, -arrow_length, 0, head_width=8, head_length=5, fc=current_color, ec=current_color)
                elif rotation == 180:
                    ax.arrow(center_x, center_y, 0, -arrow_length, head_width=8, head_length=5, fc=current_color, ec=current_color)
                elif rotation == 270:
                    ax.arrow(center_x, center_y, arrow_length, 0, head_width=8, head_length=5, fc=current_color, ec=current_color)
                color_idx += 1
        ax.set_title(f'Room Layout Visualization - {room_id}', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position (cm)', fontsize=10)
        ax.set_ylabel('Y Position (cm)', fontsize=10)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.3)
        create_time = str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')
        filename = f'{vis_dir}/room_{room_id}_{create_time}.png'
        plt.savefig(filename, bbox_inches='tight', dpi=150, facecolor='white')
        print(f'Visualization saved to: {filename}', file=sys.stderr)
        plt.close(fig)

    def visualize_dfs_placements(self, room_poly, grid_points, placed_objects, placements, object_id, vis_fig_name, room_id):
        """
        Visualize DFS placement process showing grid points with crosses and colored dots for rotation scores.
        Each grid point shows a cross with colored dots at the ends representing scores for different rotations.
        
        Args:
            room_poly: Room polygon boundary
            grid_points: List of available grid points
            placed_objects: Dictionary of already placed objects
            placements: List of possible placements with scores [point, rotation, box_coords, score]
            object_id: ID of the object being placed
            vis_fig_name: Name for the visualization file
        """
        try:
            vis_dir = f'{SERVER_ROOT_DIR}/vis/dfs_{room_id}'
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            (fig, ax) = plt.subplots(figsize=(16, 12))
            (x, y) = room_poly.exterior.xy
            ax.plot(x, y, '-', label='Room Boundary', color='black', linewidth=3)
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
            color_idx = 0
            for (obj_id, solution) in placed_objects.items():
                if obj_id.startswith(('door-', 'window-', 'open-')):
                    (center, rotation, box_coords) = solution[:3]
                    (center_x, center_y) = center
                    obj_poly = Polygon(box_coords)
                    (x_coords, y_coords) = obj_poly.exterior.xy
                    if obj_id.startswith('door-'):
                        ax.plot(x_coords, y_coords, '-', linewidth=3, color='brown', alpha=0.7)
                        ax.fill(x_coords, y_coords, color='brown', alpha=0.3)
                        ax.text(center_x, center_y, 'Door', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    elif obj_id.startswith('window-'):
                        ax.plot(x_coords, y_coords, '-', linewidth=2, color='cyan', alpha=0.7)
                        ax.fill(x_coords, y_coords, color='cyan', alpha=0.3)
                        ax.text(center_x, center_y, 'Window', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                else:
                    (center, rotation, box_coords) = solution[:3]
                    (center_x, center_y) = center
                    obj_poly = Polygon(box_coords)
                    (x_coords, y_coords) = obj_poly.exterior.xy
                    current_color = colors[color_idx % len(colors)]
                    ax.plot(x_coords, y_coords, '-', linewidth=2, color=current_color)
                    ax.fill(x_coords, y_coords, color=current_color, alpha=0.3)
                    ax.text(center_x, center_y, obj_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                    arrow_length = 25
                    angle_radians = math.radians(rotation + 90)
                    arrow_dx = arrow_length * math.cos(angle_radians)
                    arrow_dy = arrow_length * math.sin(angle_radians)
                    ax.arrow(center_x, center_y, arrow_dx, arrow_dy, head_width=8, head_length=5, fc=current_color, ec=current_color)
                    color_idx += 1
            grid_rotation_scores = {}
            all_scores = []
            for placement in placements:
                (point, rotation, box_coords, score) = placement
                grid_key = tuple(point)
                if grid_key not in grid_rotation_scores:
                    grid_rotation_scores[grid_key] = {}
                grid_rotation_scores[grid_key][rotation] = score
                all_scores.append(score)
            if all_scores:
                min_score = min(all_scores)
                max_score = max(all_scores)
                if max_score == min_score:
                    max_score = min_score + 0.001
            else:
                (min_score, max_score) = (0, 1)
            if grid_points:
                grid_x = [point[0] for point in grid_points]
                grid_y = [point[1] for point in grid_points]
                ax.scatter(grid_x, grid_y, c='lightgray', s=4, alpha=0.3, label='Grid Points')
            cross_length = 4
            dot_size = 20
            rotation_directions = {0: (0, 1), 90: (-1, 0), 180: (0, -1), 270: (1, 0)}
            from matplotlib import cm
            import matplotlib.colors as mcolors
            viridis = cm.get_cmap('viridis')
            scatter_points = []
            scatter_colors = []
            for (grid_point, rotation_scores) in grid_rotation_scores.items():
                (center_x, center_y) = grid_point
                ax.plot([center_x - cross_length, center_x + cross_length], [center_y, center_y], '-', color='black', linewidth=0.8, alpha=0.6)
                ax.plot([center_x, center_x], [center_y - cross_length, center_y + cross_length], '-', color='black', linewidth=0.8, alpha=0.6)
                for (rotation, score) in rotation_scores.items():
                    (dx, dy) = rotation_directions[rotation]
                    dot_x = center_x + dx * cross_length
                    dot_y = center_y + dy * cross_length
                    if max_score > min_score:
                        normalized_score = (score - min_score) / (max_score - min_score)
                    else:
                        normalized_score = 0.0
                    color = viridis(normalized_score)
                    ax.scatter([dot_x], [dot_y], c=[color], s=dot_size, alpha=0.9, edgecolors='white', linewidth=1)
                    scatter_points.append([dot_x, dot_y])
                    scatter_colors.append(score)
            scatter = None
            if scatter_points and scatter_colors:
                scatter_x = [p[0] for p in scatter_points]
                scatter_y = [p[1] for p in scatter_points]
                scatter = ax.scatter(scatter_x, scatter_y, c=scatter_colors, s=0, cmap='viridis', alpha=0, vmin=min_score, vmax=max_score)
            ax.set_title(f'DFS Placement Visualization - Object: {object_id}\nCrosses show rotation options, colored dots show scores', fontsize=14, fontweight='bold')
            ax.set_xlabel('X Position (cm)', fontsize=12)
            ax.set_ylabel('Y Position (cm)', fontsize=12)
            legend_text = 'Legend:\n• Light gray dots: Available grid points\n• Black crosses: Possible rotations at each grid point\n• Colored dots: Placement scores for rotations\n  - Top dot: 0° rotation (+Y direction)\n  - Left dot: 90° rotation (-X direction)\n  - Bottom dot: 180° rotation (-Y direction)  \n  - Right dot: 270° rotation (+X direction)'
            ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            ax.set_aspect('equal', 'box')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')
            if scatter is not None:
                cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
                cbar = plt.colorbar(scatter, cax=cbar_ax)
                cbar.set_label('Placement Score', fontsize=12, rotation=270, labelpad=20)
            stats_text = f'Statistics:\nTotal placements: {len(placements)}\nScore range: {min_score:.3f} - {max_score:.3f}\nGrid points with options: {len(grid_rotation_scores)}'
            ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))
            if scatter is not None:
                plt.tight_layout(rect=[0, 0, 0.83, 1])
            else:
                plt.tight_layout()
            filename = f'{vis_dir}/{vis_fig_name}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
            print(f'DFS placement visualization saved: {filename}', file=sys.stderr)
            plt.close(fig)
        except Exception as e:
            print(f'Error creating DFS placement visualization: {e}', file=sys.stderr)
            pass
