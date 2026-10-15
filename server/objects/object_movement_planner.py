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
from models import Object, Room, FloorPlan, Point3D, Euler, Dimensions
from typing import List, Dict, Any, Tuple, Optional
import json
import re
from key import ANTHROPIC_API_KEY
from vlm import call_vlm
import anthropic
from utils import extract_wall_side_from_id
from .object_placement_planner import get_door_window_placements, solution_to_objects, get_room_layout_description, create_wall_coordinate_systems, create_wall_grid_points, calculate_impossible_wall_regions, filter_valid_wall_points, select_best_wall_placement, adjust_wall_object_position, create_wall_placed_object, visualize_wall_placement, get_wall_object_constraints_from_vlm
import uuid
import os
import numpy as np
from shapely.geometry import Polygon, Point, box, LineString
from rtree import index
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import datetime
import shutil
import base64
import io
import copy
import random
import time
import math
import sys
from constants import SERVER_ROOT_DIR, RESULTS_DIR, PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED
from utils import extract_json_from_response
from dataclasses import asdict
from isaacsim.isaac_mcp.server import create_single_room_layout_scene_from_room, simulate_the_scene

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
    Generate a room visualization image using RoomVisualizer.visualize_2d_render and return as base64 string.
    Returns base64 encoded PNG image, or None if visualization fails.
    """
    try:
        from visualizer import RoomVisualizer
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
            print('Failed to generate room visualization')
            return None
    except Exception as e:
        print(f'Error generating room visualization: {str(e)}')
        return None

async def analyze_object_to_move_from_condition(room: Room, current_layout: FloorPlan, condition: str) -> Dict[str, Any]:
    """
    Use Claude API to analyze the condition and identify which object to move and where.
    
    Args:
        room: The target room
        current_layout: Current floor plan layout
        condition: Text condition describing what object to move and where
        
    Returns:
        Dictionary containing object identification information or error
    """
    if not room.objects:
        return {'success': False, 'error': 'No objects in room to move'}
    try:
        doors_info = []
        if room.doors:
            for door in room.doors:
                wall_side = extract_wall_side_from_id(door.wall_id) if door.wall_id else 'unknown wall'
                doors_info.append(f'- Door (ID: {door.id}): {door.door_type} door on {wall_side}, {door.width:.1f}m wide')
        doors_text = '\n'.join(doors_info) if doors_info else '- No doors'
        windows_info = []
        if room.windows:
            for window in room.windows:
                wall_side = extract_wall_side_from_id(window.wall_id) if window.wall_id else 'unknown wall'
                windows_info.append(f'- Window (ID: {window.id}): {window.window_type} window on {wall_side}, {window.width:.1f}m wide')
        windows_text = '\n'.join(windows_info) if windows_info else '- No windows'
        existing_objects_info = []
        if room.objects:
            for obj in room.objects:
                obj_x_rel = obj.position.x - room.position.x
                obj_y_rel = obj.position.y - room.position.y
                existing_objects_info.append(f'- {obj.type} (ID: {obj.id}): {obj.description}, positioned at ({obj_x_rel:.1f}m, {obj_y_rel:.1f}m) relative to room corner, dimensions {obj.dimensions.width:.1f}×{obj.dimensions.length:.1f}×{obj.dimensions.height:.1f}m, rotation {obj.rotation.z}°')
        existing_objects_text = '\n'.join(existing_objects_info)
        initial_analysis_prompt = f"""You are an interior design expert analyzing a request to move an object within a room.\n\nROOM: {room.room_type} | {room.dimensions.width:.1f}×{room.dimensions.length:.1f}×{room.dimensions.height:.1f}m\nDOORS ({len(room.doors)} total): {doors_text}\nWINDOWS ({len(room.windows)} total): {windows_text}\nEXISTING OBJECTS ({len(room.objects)} total): {existing_objects_text}\n\nUSER REQUEST: {condition.strip()}\n\nTASK: Identify which object to move, its type, and target location type.\n\nOUTPUT FORMAT (JSON only):\n```json\n{{\n    "success": true,\n    "object_id": "exact_object_id_from_existing_objects_list",\n    "object_type": "object_type_from_existing_objects",\n    "movement_target_location": "floor|wall|exact_existing_object_id_from_list",\n    "justification": "Brief explanation of object identified and target location"\n}}\n```\n\nMOVEMENT TARGET RULES:\n- "floor": repositioning on the ground/floor (includes placing against walls, in corners, or anywhere on the floor surface)\n- "wall": ONLY for objects that are attached/mounted directly on the wall (like wall shelves, paintings, wall-mounted TVs)\n- "object_id": placing onto/on top of a specific existing object\n\nEXAMPLES:\n- "move table against the wall" → "floor" (table sits on floor, just positioned near wall)\n- "move sofa to corner" → "floor" (sofa sits on floor in corner)\n- "hang picture on wall" → "wall" (picture is mounted on wall)\n- "put book on table" → use table's object_id (book goes on top of table)\n- "move picture on wall region that on top of the bed" → "wall" (picture is mounted on wall region that on top of the bed)\n\nREQUIREMENTS:\n- Choose exactly ONE object ID from existing objects list\n- Use exact ID match from EXISTING OBJECTS above\n- Target location: "floor", "wall", or exact object ID\n- If ambiguous, choose most likely candidate\n\nAnalyze the movement request now:"""
        initial_response = call_vlm(vlm_type='openai', max_tokens=1000, temperature=0.1, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': initial_analysis_prompt}]}])
        initial_response_text = initial_response.content[0].text.strip()
        try:
            initial_json_content = extract_json_from_response(initial_response_text)
            if not initial_json_content:
                return {'success': False, 'error': 'Could not extract JSON from initial Claude response'}
            initial_analysis = json.loads(initial_json_content)
            required_fields = ['object_id', 'object_type', 'movement_target_location']
            for field in required_fields:
                if field not in initial_analysis:
                    return {'success': False, 'error': f'Missing required field in initial analysis: {field}'}
            object_id = initial_analysis['object_id']
            target_object = next((obj for obj in room.objects if obj.id == object_id), None)
            if not target_object:
                return {'success': False, 'error': f'Identified object {object_id} not found in room'}
            movement_target_location = initial_analysis['movement_target_location']
            if movement_target_location == 'wall':
                return {'success': True, 'object_to_move': {'id': object_id, 'type': initial_analysis['object_type'], 'current_position': {'x': target_object.position.x, 'y': target_object.position.y, 'z': target_object.position.z}, 'current_rotation': target_object.rotation.z}, 'movement_intent': {'target_description': f'Move to {movement_target_location}: {condition.strip()}', 'spatial_relationships': [], 'movement_type': 'repositioning and rotation'}, 'movement_guidance': f'Object should be moved to {movement_target_location}. Detailed floor placement analysis skipped for non-floor targets.', 'justification': initial_analysis.get('justification', 'Object identified from initial analysis'), 'movement_target_location': movement_target_location, 'initial_analysis': initial_analysis, 'floor_analysis_skipped': True}
            elif movement_target_location != 'floor':
                if object_id == movement_target_location:
                    return {'success': False, 'error': f'Object {object_id} can not be moved to itself'}
                if movement_target_location in [obj.id for obj in room.objects]:
                    return {'success': True, 'object_to_move': {'id': object_id, 'type': initial_analysis['object_type'], 'current_position': {'x': target_object.position.x, 'y': target_object.position.y, 'z': target_object.position.z}, 'current_rotation': target_object.rotation.z}, 'movement_intent': {'target_description': f'Move to {movement_target_location}', 'spatial_relationships': [], 'movement_type': 'repositioning and rotation'}, 'movement_guidance': f'Object should be moved to {movement_target_location}. Detailed floor placement analysis skipped for non-floor targets.', 'justification': initial_analysis.get('justification', 'Object identified from initial analysis'), 'movement_target_location': movement_target_location, 'initial_analysis': initial_analysis, 'floor_analysis_skipped': True}
                else:
                    return {'success': False, 'error': f'Object {object_id} not found in room'}
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f'Failed to parse initial Claude response as JSON: {e}'}
    except Exception as e:
        return {'success': False, 'error': f'Error in initial movement analysis: {e}'}
    try:
        existing_objects_info = []
        if room.objects:
            for obj in room.objects:
                obj_x_rel = obj.position.x - room.position.x
                obj_y_rel = obj.position.y - room.position.y
                obj_bbox = calculate_object_bounding_box(obj, room)
                obj_facing = get_object_facing_direction(obj)
                existing_objects_info.append(f'- ID: {obj.id} | Type: {obj.type} | Description: {obj.description}')
                existing_objects_info.append(f'  Position: ({obj_x_rel:.1f}m, {obj_y_rel:.1f}m) relative to room corner')
                existing_objects_info.append(f'  Dimensions: {obj.dimensions.width:.1f}×{obj.dimensions.length:.1f}×{obj.dimensions.height:.1f}m')
                existing_objects_info.append(f'  Bounding Box: {obj_bbox}')
                existing_objects_info.append(f'  Facing Direction: {obj_facing}')
                existing_objects_info.append(f'  Rotation: {obj.rotation.z}°')
        existing_objects_text = '\n'.join(existing_objects_info)
        doors_info = []
        if room.doors:
            for door in room.doors:
                wall_side = extract_wall_side_from_id(door.wall_id) if door.wall_id else 'unknown wall'
                doors_info.append(f'- Door (ID: {door.id}): {door.door_type} door on {wall_side}, {door.width:.1f}m wide')
        doors_text = '\n'.join(doors_info) if doors_info else '- No doors'
        windows_info = []
        if room.windows:
            for window in room.windows:
                wall_side = extract_wall_side_from_id(window.wall_id) if window.wall_id else 'unknown wall'
                windows_info.append(f'- Window (ID: {window.id}): {window.window_type} window on {wall_side}, {window.width:.1f}m wide')
        windows_text = '\n'.join(windows_info) if windows_info else '- No windows'
        analysis_prompt = f'''You are an experienced interior designer with expertise in space planning and furniture arrangement.\n\nTASK: Analyze the movement request for {object_id} to determine:\n1. Target location description with design strategy\n2. Applicable spatial relationship constraints\n3. Movement guidance with functional considerations\n\n═══════════════════════════════════════════════════════════════════════════════\n\nINPUT INFORMATION:\n\nUSER REQUEST: "{condition.strip()}"\n\nOBJECT TO MOVE:\n- Type: {initial_analysis['object_type']} (ID: {object_id})\n- Current Position: ({target_object.position.x:.1f}m, {target_object.position.y:.1f}m, {target_object.position.z:.1f}m)\n- Current Rotation: {target_object.rotation.z}° | System: 0°=+Y(up), 90°=-X(left), 180°=-Y(down), 270°=+X(right)\n\nROOM CONTEXT:\n- Type: {room.room_type}\n- Dimensions: {room.dimensions.width:.1f}×{room.dimensions.length:.1f}×{room.dimensions.height:.1f}m (Area: {room.dimensions.width * room.dimensions.length:.1f}m²)\n- Floor: {room.floor_material} | Ceiling Height: {room.ceiling_height:.1f}m\n- Doors ({len(room.doors)} total): {doors_text}\n- Windows ({len(room.windows)} total): {windows_text}\n- Current Objects ({len(room.objects)} total): {existing_objects_text}\n\n═══════════════════════════════════════════════════════════════════════════════\n\nCONSTRAINT SYSTEM:\n\nHere are the available constraints:\n\n1. GLOBAL CONSTRAINT (required):\n   - edge: Place at the edge of the room, close to walls\n   - middle: Place away from walls, in the central area of the room\n\n2. DISTANCE CONSTRAINTS:\n   - close to, [object_id]: Place really close to another object (as close as possible, typically within 30cm distance)\n   - near, [object_id]: Place near another object (middle range distance, typically 50cm to 150cm distance)\n   - far, [object_id]: Place far from another object (as far as possible, typically more than 150cm distance)\n\n3. POSITION CONSTRAINTS (relative to target object's facing direction):\n   - in front of, [object_id]: Position in front of another object (in the direction it faces)\n   - around, [object_id]: Position around another object (typically for chairs around tables)\n   - side of, [object_id]: Position to the left or right side of another object\n   - left of, [object_id]: Position to the LEFT of another object (relative to its facing direction)\n   - right of, [object_id]: Position to the RIGHT of another object (relative to its facing direction)\n   - behind, [object_id]: Position behind another object\n   \n   IMPORTANT: "left", "right", "front", and "behind" are relative to the TARGET OBJECT'S facing direction.\n\n4. ALIGNMENT CONSTRAINTS:\n   - center aligned, [object_id]: Align centers with another object\n\n5. ROTATION CONSTRAINTS:\n   - face to, [object_id]: Orient to face toward another object's center\n   - face same as, [object_id]: Orient to face the same as another object's facing direction\n\n═══════════════════════════════════════════════════════════════════════════════\n\nDESIGN STRATEGY [IMPORTANT]:\n\n1. **Room Function**: Consider door swing areas, traffic flow paths, and natural light from windows\n2. **Object Relationships**: Objects of the same type should typically be aligned; chairs should face tables/desks\n3. **Detailed Constraints Preferred**: Provide as detailed constraints as possible (4-5 constraints if applicable) for optimal placement\n4. **Circulation Space**: Ensure sufficient space for walking and moving around. Avoid placing objects too close to each other or too crowded\n5. **Same Type Constraints**: You can use multiple constraints of the same type for a single object (e.g., "left of, object_B" and "right of, object_C")\n6. **Existing Objects**: Consider the spatial relationships with all existing objects in the room\n7. **Functional Areas**: Group related objects together (dining area, seating area, work area, etc.)\n\n═══════════════════════════════════════════════════════════════════════════════\n\nCONSTRAINT SELECTION RULES:\n\n1. **Distance + Position**: If using "close to", "near" or "far", add position/alignment/rotation constraints for specificity\n2. **Position + Distance**: If using position constraints ("in front of", "left of"), add distance constraints (close to, near or far) for specificity (except for grid alignment pattern)\n3. **Functional Pairing**: Chairs should use "close to" + "in front of" + "face to" relative to tables/desks. (choose "close to" instead of "near" because we need the chair to against the table/desk) Generalize to other meaningful pairs\n4. **Relationship Coupling**: When using "in front of" position relationship, consider using "face to" relationship as well to make the scene look more harmonic\n5. **Distance Inference**: \n   - If two objects are not placed in the same region (e.g., [against different walls] or [in different corners] or [one against wall and one in middle] or [in different functional areas]), use "far" constraint\n   - Only use "close to" when you really need the object to be as close as possible, otherwise use "near" constraint\n\n═══════════════════════════════════════════════════════════════════════════════\n\nCONSTRAINT COMBINATION PATTERNS:\n\nBasic Patterns:\n• Edge furniture (first object): ["edge"]\n• Middle furniture: ["middle", "close to, target", "in front of, target", "center aligned, target", "face to, target"]\n\nObject-Specific Examples:\n• Objects typically placed against wall: bookcases, bookshelves, shelving units, armoires, wardrobes, TV stands, sideboards, buffets, sofa, bed, dressers, chests of drawers → ["edge", other constraints...]\n• Plants, sculptures, or aesthetic objects → ["edge", other constraints...]\n• Chair close to dining table: ["middle", "close to, table_001", "in front of, table_001", "face to, table_001"]\n  (choose "close to" instead of "near" because chair needs to be against the table)\n• Coffee table in front of sofa: ["middle", "near, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n  (choose "near" instead of "close to" because coffee table needs space, not too close for walking)\n• TV stand facing seating: ["edge", "far, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n• Four or more chairs around dining table: each chair gets ["middle", "close to, table_001", "around, table_001", "face to, table_001"]\n  ("center aligned, table_001" is optional) (choose "close to" instead of "near" because chair needs to be against table)\n• Nightstand left of bed (both against wall): ["edge", "left of, bed_001", "close to, bed_001", "face same as, bed_001"]\n  (Position + Distance combination, "face same as" keeps nightstand facing same direction as bed)\n  (choose "close to" instead of "near" because nightstand needs to be as close as possible for convenience)\n• Bookshelf and sofa against different walls (for bookshelf): ["edge", "far, sofa_001"]\n  (use "far" constraint because they are against different walls)\n• Dining table (middle of room) and storage box (against wall) far apart: ["middle", "far, storage_box_001"]\n  (use "far" constraint because they are in different functional areas)\n\nGrid Alignment Pattern (for multiple similar objects like tables):\n• First table ("table_001") at location (0, 0): ["middle"]\n• Second table ("table_002") at location (1, 0): ["middle", "in front of, table_001", "center aligned, table_001", "face same as, table_001"]\n• Third table ("table_003") at location (0, 1): ["middle", "right of, table_001", "center aligned, table_001", "face same as, table_001"]\n• Fourth table ("table_004") at location (1, 1): ["middle", "in front of, table_003", "center aligned, table_003", "right of, table_002", "center aligned, table_002", "face same as, table_001"]\n(Note: this is a grid alignment pattern - don't need to add distance constraints, don't add "close to" or "near" between tables)\n\n═══════════════════════════════════════════════════════════════════════════════\n\nOUTPUT SPECIFICATION:\n\nFormat (JSON only):\n```json\n{{\n    "success": true,\n    "movement_intent": {{\n        "target_description": "Where user wants to move the object",\n        "spatial_relationships": ["constraint_1", "constraint_2", ...],\n        "movement_guidance": "Detailed guidance with movement requirements, target location, alignment preferences, distance relationships, and functional considerations. Include brief constraint descriptions and reasoning.",\n    }},\n    "reasoning": {{\n        "design_strategy": "Overall approach and design principles guiding the movement",\n        "room_layout_analysis": "Analysis of the room's shape, dimensions, and spatial characteristics",\n        "existing_objects_consideration": "How existing objects influence the movement decision",\n        "functional_relationships": "How the moved object relates to other objects functionally",\n        "aesthetic_considerations": "Visual harmony, balance, and aesthetic relationships"\n    }},\n    "justification": "Brief explanation of movement intent interpretation"\n}}\n```\n\nRequirements:\n• First constraint in spatial_relationships MUST be: "edge" or "middle"\n• Use exact object IDs from room objects list above\n• Include detailed descriptions in movement_guidance\n• For rotation: prioritize "face to" constraint over direct angles\n• Apply constraint selection rules and patterns above\n• Provide 4-5 constraints if applicable for best results\n• Include reasoning section with detailed analysis\n\n═══════════════════════════════════════════════════════════════════════════════\n\nEXAMPLES:\n\n1. "move chair to corner" → \n   spatial_relationships: ["edge"]\n   movement_guidance: "Edge placement for corner positioning"\n\n2. "center coffee table with sofa" → \n   spatial_relationships: ["middle", "near, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n   movement_guidance: "Balanced arrangement in front of sofa with proper spacing for circulation" \n   (choose "near" instead of "close to" because coffee table needs space for walking)\n\n3. "move chair to dining table" → \n   spatial_relationships: ["middle", "close to, table_001", "in front of, table_001", "face to, table_001"]\n   movement_guidance: "Functional pairing for dining with chair against table for proper seating"\n   (choose "close to" because chair needs to be against table)\n\n4. "reposition TV stand facing seating area" → \n   spatial_relationships: ["edge", "far, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n   movement_guidance: "Optimal viewing alignment with TV stand against wall facing the seating area"\n\n5. "place side table to the left of sofa facing it" → \n   spatial_relationships: ["middle", "near, sofa_001", "left of, sofa_001", "face to, sofa_001"]\n   movement_guidance: "Side positioning with orientation toward sofa for functional access"\n\n6. "move lamp to right of desk and face it" → \n   spatial_relationships: ["middle", "close to, desk_001", "right of, desk_001", "face to, desk_001"]\n   movement_guidance: "Lateral placement with facing for optimal task lighting"\n   (choose "close to" for functional lamp placement)\n\n7. "move nightstand next to bed" → \n   spatial_relationships: ["edge", "left of, bed_001", "close to, bed_001", "face same as, bed_001"]\n   movement_guidance: "Bedside placement for convenience with matching orientation"\n   (choose "close to" for easy reach)\n\n8. "move bookshelf against opposite wall from sofa" → \n   spatial_relationships: ["edge", "far, sofa_001", "in front of, sofa_001", "center aligned, sofa_001", "face to, sofa_001"]\n   movement_guidance: "Separate functional areas with bookshelf on different wall"\n   (use "far" because they are against different walls)\n\n═══════════════════════════════════════════════════════════════════════════════\n\nAnalyze the movement request now:'''
        response = call_vlm(vlm_type='openai', max_tokens=8000, temperature=0.1, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': analysis_prompt}]}])
        response_text = response.content[0].text.strip()
        try:
            json_content = None
            json_content = extract_json_from_response(response_text)
            if not json_content:
                raise ValueError('Could not extract JSON from Claude response')
            analysis_result = json.loads(json_content)
            required_fields = ['movement_intent']
            for field in required_fields:
                if field not in analysis_result:
                    return {'success': False, 'error': f'Missing required field: {field}'}
            analysis_result['object_to_move'] = {'id': object_id, 'type': initial_analysis['object_type'], 'current_position': {'x': target_object.position.x, 'y': target_object.position.y, 'z': target_object.position.z}, 'current_rotation': target_object.rotation.z}
            analysis_result['success'] = True
            analysis_result['claude_prompt'] = analysis_prompt
            analysis_result['claude_response'] = response_text
            analysis_result['visualization_used'] = True
            analysis_result['movement_target_location'] = movement_target_location
            analysis_result['initial_analysis'] = initial_analysis
            analysis_result['floor_analysis_completed'] = True
            return analysis_result
        except json.JSONDecodeError as e:
            return {'success': False, 'error': f'Failed to parse Claude response as JSON: {e}'}
    except Exception as e:
        return {'success': False, 'error': f'Error in analyze_object_to_move_from_condition: {e}'}

async def get_movement_location_from_claude_floor(room: Room, current_layout: FloorPlan, object_to_move: Object, movement_intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use movement intent from analyze_object_to_move_from_condition to prepare movement constraints.
    No additional VLM call needed - constraints are already determined.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout
        object_to_move: Object that needs to be moved
        movement_intent: Intent from analyze_object_to_move_from_condition (contains spatial_relationships)
        
    Returns:
        Dictionary containing movement constraints and information
    """
    try:
        spatial_relationships = movement_intent.get('spatial_relationships', [])
        if not spatial_relationships:
            return {'success': False, 'error': 'No spatial relationships found in movement_intent'}
        has_rotation_constraints = any(('face to' in rel.lower() for rel in spatial_relationships))
        print(f'spatial_relationships: {spatial_relationships}', file=sys.stderr)
        print(f'movement_intent: {movement_intent}', file=sys.stderr)
        movement_result = {'success': True, 'constraints': spatial_relationships, 'movement_reasoning': movement_intent.get('movement_guidance', 'Movement based on spatial relationships'), 'design_strategy': movement_intent.get('target_description', 'Optimal placement'), 'new_suggested_rotation': None if has_rotation_constraints else object_to_move.rotation.z, 'movement_intent_used': True, 'skipped_vlm_call': True}
        return movement_result
    except Exception as e:
        return {'success': False, 'error': f'Error in get_movement_location_from_claude_floor: {e}'}

async def get_movement_location_from_claude_wall(room: Room, current_layout: FloorPlan, object_to_move: Object, movement_intent: Dict[str, Any], movement_target_location: str) -> Dict[str, Any]:
    return {'success': True, 'movement_target_location': movement_target_location, 'movement_intent': movement_intent}


def create_movement_room_visualization(room_poly: Polygon, grid_points: List[Tuple[float, float]], initial_state: Dict[str, Any], object_to_move: Object, room_id: str) -> str:
    """
    Create a visualization of the room layout highlighting the object to be moved.
    Enhanced version with better visual elements and consistency.
    
    Args:
        room_poly: Room polygon
        grid_points: Available grid points for placement
        initial_state: Existing objects, doors, windows
        object_to_move: Object that needs to be moved (for highlighting)
        room_id: Room identifier
        
    Returns:
        Path to the saved visualization image
    """
    vis_dir = f'{SERVER_ROOT_DIR}/vis'
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    plt.rcParams['font.size'] = 10
    (fig, ax) = plt.subplots(figsize=(14, 12))
    (x, y) = room_poly.exterior.xy
    ax.plot(x, y, '-', label='Room Boundary', color='black', linewidth=4)
    ax.fill(x, y, color='lightblue', alpha=0.1)
    if grid_points:
        grid_x = [point[0] for point in grid_points]
        grid_y = [point[1] for point in grid_points]
        ax.scatter(grid_x, grid_y, s=12, color='lightgray', alpha=0.7, label='Available Movement Points', marker='.')
    colors = ['blue', 'green', 'orange', 'purple', 'pink', 'gray', 'olive', 'navy']
    color_idx = 0
    arrows_to_draw = []
    for (object_id, (center, rotation, vertices, _)) in initial_state.items():
        (center_x, center_y) = center
        obj_poly = Polygon(vertices)
        (x_coords, y_coords) = obj_poly.exterior.xy
        if object_id.startswith('door-') or object_id.startswith('opening-'):
            ax.plot(x_coords, y_coords, '-', linewidth=4, color='brown')
            ax.fill(x_coords, y_coords, color='brown', alpha=0.6)
            ax.text(center_x, center_y, 'DOOR', fontsize=9, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        elif object_id.startswith('window-'):
            ax.plot(x_coords, y_coords, '-', linewidth=3, color='cyan')
            ax.fill(x_coords, y_coords, color='cyan', alpha=0.6)
            ax.text(center_x, center_y, 'WINDOW', fontsize=9, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        else:
            is_object_to_move = object_id.endswith(object_to_move.id)
            if is_object_to_move:
                current_color = 'red'
                alpha = 0.8
                linewidth = 4
                label_text = f'TO MOVE:\n{object_to_move.type}'
                label_bg_color = 'yellow'
            else:
                current_color = colors[color_idx % len(colors)]
                alpha = 0.5
                linewidth = 3
                label_text = object_id.replace('existing-', '')
                label_bg_color = 'white'
                color_idx += 1
            ax.plot(x_coords, y_coords, '-', linewidth=linewidth, color=current_color)
            ax.fill(x_coords, y_coords, color=current_color, alpha=alpha)
            ax.text(center_x, center_y, label_text, fontsize=8, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor=label_bg_color, alpha=0.9, edgecolor='black'))
            arrows_to_draw.append((center_x, center_y, rotation, current_color, is_object_to_move))
    for (center_x, center_y, rotation, current_color, is_object_to_move) in arrows_to_draw:
        arrow_length = 25
        head_width = 8
        head_length = 6
        arrow_face_color = 'black'
        arrow_edge_color = 'black'
        arrow_alpha = 0.9
        arrow_linewidth = 2
        shadow_offset = 2
        if rotation == 0:
            ax.arrow(center_x + shadow_offset, center_y + shadow_offset, 0, arrow_length, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.3, linewidth=arrow_linewidth + 1, zorder=10)
            ax.arrow(center_x, center_y, 0, arrow_length, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=arrow_alpha, linewidth=arrow_linewidth, zorder=11)
        elif rotation == 90:
            ax.arrow(center_x + shadow_offset, center_y + shadow_offset, -arrow_length, 0, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.3, linewidth=arrow_linewidth + 1, zorder=10)
            ax.arrow(center_x, center_y, -arrow_length, 0, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=arrow_alpha, linewidth=arrow_linewidth, zorder=11)
        elif rotation == 180:
            ax.arrow(center_x + shadow_offset, center_y + shadow_offset, 0, -arrow_length, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.3, linewidth=arrow_linewidth + 1, zorder=10)
            ax.arrow(center_x, center_y, 0, -arrow_length, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=arrow_alpha, linewidth=arrow_linewidth, zorder=11)
        elif rotation == 270:
            ax.arrow(center_x + shadow_offset, center_y + shadow_offset, arrow_length, 0, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.3, linewidth=arrow_linewidth + 1, zorder=10)
            ax.arrow(center_x, center_y, arrow_length, 0, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=arrow_alpha, linewidth=arrow_linewidth, zorder=11)
    title_text = f'Room Layout - Moving {object_to_move.type}\nRoom ID: {room_id}'
    title_text += '\n\n→ White arrows indicate object facing direction'
    ax.set_title(title_text, fontsize=16, fontweight='bold', pad=25)
    ax.set_xlabel('X Position (cm)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Y Position (cm)', fontsize=14, fontweight='bold')
    ax.set_aspect('equal', 'box')
    ax.grid(True, alpha=0.4, linewidth=1, color='gray', linestyle='-')
    ax.tick_params(axis='both', which='major', labelsize=12, width=2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    legend = ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'{vis_dir}/movement_enhanced_{room_id}_{object_to_move.type}_{timestamp}.png'
    plt.savefig(filename, bbox_inches='tight', dpi=200, facecolor='white', edgecolor='none')
    plt.close(fig)
    return filename

def create_visualization_grid_points(room_poly: Polygon, initial_state: Dict[str, Any], grid_size: int=20) -> List[Tuple[float, float]]:
    """
    Create grid points for visualization purposes. This is a simplified version of the DFS solver's
    grid creation that only creates points for visualization (no complex placement logic).
    
    Args:
        room_poly: Room polygon
        initial_state: Existing objects/doors/windows
        grid_size: Grid spacing in cm
        
    Returns:
        List of (x, y) grid points that are inside the room and not occupied
    """
    try:
        (minx, miny, maxx, maxy) = room_poly.bounds
        grid_points = []
        x = minx
        while x <= maxx:
            y = miny
            while y <= maxy:
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))
                y += grid_size
            x += grid_size
        filtered_points = []
        for (x, y) in grid_points:
            point = Point(x, y)
            is_occupied = False
            for (obj_id, (center, rotation, vertices, _)) in initial_state.items():
                obj_poly = Polygon(vertices)
                if obj_poly.contains(point) or obj_poly.distance(point) < 10:
                    is_occupied = True
                    break
            if not is_occupied:
                filtered_points.append((x, y))
        return filtered_points
    except Exception as e:
        print(f'Warning: Error creating visualization grid points: {e}')
        return []

def get_detailed_room_description_for_movement(room: Room, current_layout: FloorPlan, object_to_move: Object) -> str:
    """
    Generate a detailed room description for movement context.
    """
    description_parts = []
    description_parts.append(f'Room Type: {room.room_type}')
    description_parts.append(f'Dimensions: {room.dimensions.width * 100:.0f} × {room.dimensions.length * 100:.0f} cm')
    description_parts.append(f'Area: {room.dimensions.width * room.dimensions.length:.1f} m²')
    description_parts.append(f'Ceiling Height: {room.ceiling_height * 100:.0f} cm')
    description_parts.append(f'Floor Material: {room.floor_material}')
    if room.doors:
        description_parts.append(f'\nDoors ({len(room.doors)}):')
        for (i, door) in enumerate(room.doors):
            pos_cm = door.position_on_wall * room.dimensions.width * 100
            description_parts.append(f'  - Door {i + 1}: {door.width * 100:.0f}cm wide at {pos_cm:.0f}cm from left wall')
    if room.windows:
        description_parts.append(f'\nWindows ({len(room.windows)}):')
        for (i, window) in enumerate(room.windows):
            pos_cm = window.position_on_wall * room.dimensions.width * 100
            description_parts.append(f'  - Window {i + 1}: {window.width * 100:.0f}cm wide at {pos_cm:.0f}cm from left wall')
    other_objects = [obj for obj in room.objects if obj.id != object_to_move.id]
    if other_objects:
        description_parts.append(f'\nOther Objects ({len(other_objects)}):')
        for obj in other_objects:
            pos_x_cm = (obj.position.x - room.position.x) * 100
            pos_y_cm = (obj.position.y - room.position.y) * 100
            dims_cm = f'{obj.dimensions.width * 100:.0f}×{obj.dimensions.length * 100:.0f}×{obj.dimensions.height * 100:.0f}'
            obj_bbox = calculate_object_bounding_box(obj, room)
            obj_facing = get_object_facing_direction(obj)
            description_parts.append(f'  - ID: {obj.id} | Type: {obj.type} | Size: {dims_cm}cm | Position: ({pos_x_cm:.0f}, {pos_y_cm:.0f})')
            description_parts.append(f'    Bounding Box: {obj_bbox}')
            description_parts.append(f'    Facing Direction: {obj_facing}')
            if hasattr(obj, 'description') and obj.description:
                description_parts.append(f'    Description: {obj.description}')
    else:
        description_parts.append('\nOther Objects: None')
    return '\n'.join(description_parts)

class MovementFloorSolver:
    """
    Custom floor solver for moving objects that combines Claude's suggested position 
    with constraint-based ranking to find optimal new placement.
    Based on AdditionFloorSolver but adapted for object movement.
    """

    def __init__(self, grid_size=50, constraint_bouns=0.2, position_tolerance=100):
        self.grid_size = grid_size
        self.constraint_bouns = constraint_bouns
        self.position_tolerance = position_tolerance
        self.constraint_type2weight = {'global': 2.0, 'relative': 1.0, 'direction': 2.0, 'alignment': 0.8, 'distance': 1.0}
        self.edge_bouns = 0.0

    def get_best_movement_placement(self, room_poly, object_dim, suggested_position, suggested_rotation, constraints, initial_state, current_object_id):
        """
        Find the best new placement for moving an object.
        
        Args:
            room_poly: Room polygon
            object_dim: Object dimensions (width_cm, length_cm)
            suggested_position: Claude's suggested position {"x": x_cm, "y": y_cm}
            suggested_rotation: Claude's suggested rotation in degrees
            constraints: List of constraint dictionaries
            initial_state: Existing objects/doors/windows (excluding the object being moved)
            current_object_id: ID of object being moved (to exclude from collision detection)
            
        Returns:
            Best placement tuple: (center, rotation, vertices, score) or None
        """
        try:
            grid_points = self.create_grids(room_poly)
            filtered_initial_state = {k: v for (k, v) in initial_state.items() if not k.endswith(current_object_id)}
            grid_points = self.remove_points(grid_points, filtered_initial_state)
            if not grid_points:
                return None
            all_solutions = self.get_all_solutions(room_poly, grid_points, object_dim)
            if not all_solutions:
                return None
            solutions = self.filter_collision(filtered_initial_state, all_solutions)
            solutions = self.filter_facing_wall(room_poly, solutions, object_dim)
            if not solutions:
                return None
            filtered_solutions = solutions
            candidate_solutions = self.apply_global_constraints(room_poly, filtered_solutions, object_dim, constraints)
            if not candidate_solutions:
                return None
            best_solution = self.rank_solutions_by_constraints(candidate_solutions, constraints, filtered_initial_state)
            return best_solution
        except Exception as e:
            print(f'Error in get_best_movement_placement: {e}')
            return None

    def prefer_suggested_rotation(self, solutions, suggested_rotation):
        """Give preference to solutions with the suggested rotation."""
        valid_rotations = [0, 90, 180, 270]
        if suggested_rotation not in valid_rotations:
            suggested_rotation = min(valid_rotations, key=lambda x: abs(x - suggested_rotation))
        preferred_solutions = []
        other_solutions = []
        for solution in solutions:
            if solution[1] == suggested_rotation:
                solution[3] += 0.3
                preferred_solutions.append(solution)
            else:
                other_solutions.append(solution)
        return preferred_solutions + other_solutions

    def create_grids(self, room_poly):
        """Create grid points within the room polygon."""
        (min_x, min_y, max_x, max_y) = room_poly.bounds
        grid_points = []
        for x in range(int(min_x), int(max_x), self.grid_size):
            for y in range(int(min_y), int(max_y), self.grid_size):
                point = Point(x, y)
                if room_poly.contains(point):
                    grid_points.append((x, y))
        return grid_points

    def remove_points(self, grid_points, objects_dict):
        """Remove grid points that are occupied by existing objects."""
        if not objects_dict:
            return grid_points
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
        """Generate all possible object placements (position + rotation)."""
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
                    solutions.append([point, rotation, tuple(obj_box.exterior.coords[:]), 1.0])
        return solutions

    def filter_collision(self, objects_dict, solutions):
        """Filter out solutions that collide with existing objects."""
        if not objects_dict:
            return solutions
        valid_solutions = []
        object_polygons = [Polygon(obj_coords) for (_, _, obj_coords, _) in objects_dict.values()]
        for solution in solutions:
            sol_obj_coords = solution[2]
            sol_obj = Polygon(sol_obj_coords)
            if not any((sol_obj.intersects(obj) for obj in object_polygons)):
                valid_solutions.append(solution)
        return valid_solutions

    def filter_facing_wall(self, room_poly, solutions, obj_dim):
        """Filter out solutions where front of object is too close to wall."""
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

    def filter_by_suggested_position(self, solutions, target_x, target_y, tolerance):
        """Filter solutions to those near Claude's suggested position."""
        filtered_solutions = []
        for solution in solutions:
            (center_x, center_y) = solution[0]
            distance = math.sqrt((center_x - target_x) ** 2 + (center_y - target_y) ** 2)
            if distance <= tolerance:
                proximity_score = max(0, 1.0 - distance / tolerance)
                solution[3] += proximity_score * 0.5
                filtered_solutions.append(solution)
        return filtered_solutions

    def apply_global_constraints(self, room_poly, solutions, obj_dim, constraints):
        """Apply global constraints (edge/middle) to filter solutions."""
        global_constraint = None
        for constraint in constraints:
            if constraint.get('type') == 'global' and constraint.get('constraint') in ['edge', 'middle']:
                global_constraint = constraint
                break
        if not global_constraint:
            global_constraint = {'type': 'global', 'constraint': 'edge'}
        if global_constraint['constraint'] == 'edge':
            edge_solutions = self.place_edge(room_poly, copy.deepcopy(solutions), obj_dim)
            return edge_solutions if edge_solutions else solutions
        else:
            return solutions

    def place_edge(self, room_poly, solutions, obj_dim):
        """Apply edge placement constraint."""
        valid_solutions = []
        obj_width = obj_dim[1]
        obj_half_width = obj_width / 2
        back_center_adjustments = {0: (0, -obj_half_width), 90: (obj_half_width, 0), 180: (0, obj_half_width), 270: (-obj_half_width, 0)}
        for solution in solutions:
            (center_x, center_y) = solution[0]
            rotation = solution[1]
            back_center_adjustment = back_center_adjustments[rotation]
            (back_center_x, back_center_y) = (center_x + back_center_adjustment[0], center_y + back_center_adjustment[1])
            back_center_distance = room_poly.boundary.distance(Point(back_center_x, back_center_y))
            center_distance = room_poly.boundary.distance(Point(center_x, center_y))
            if back_center_distance <= self.grid_size and back_center_distance < center_distance:
                solution[3] += self.constraint_bouns
                center2back_vector = np.array([back_center_x - center_x, back_center_y - center_y])
                if np.linalg.norm(center2back_vector) > 0:
                    center2back_vector /= np.linalg.norm(center2back_vector)
                    offset = center2back_vector * back_center_distance
                    solution[0] = (center_x + offset[0], center_y + offset[1])
                    solution[2] = tuple([(coord[0] + offset[0], coord[1] + offset[1]) for coord in solution[2]])
                valid_solutions.append(solution)
        return valid_solutions

    def rank_solutions_by_constraints(self, candidate_solutions, constraints, initial_state):
        """Rank solutions by how well they satisfy constraints."""
        if not candidate_solutions:
            return None
        placement2score = {tuple(solution[:3]): solution[3] for solution in candidate_solutions}
        for constraint in constraints:
            constraint_type = constraint.get('type')
            constraint_value = constraint.get('constraint')
            print(f'pass a constraint: {constraint_type} {constraint_value}', file=sys.stderr)
            if constraint_type == 'global' and constraint_value == 'middle':
                room_poly_coords = []
                for obj_data in initial_state.values():
                    if isinstance(obj_data[2], (list, tuple)):
                        room_poly_coords.extend(obj_data[2])
                if not room_poly_coords:
                    continue
                try:
                    for solution in candidate_solutions:
                        sol_coords = solution[2]
                        sol_poly = Polygon(sol_coords)
                        min_dist = float('inf')
                        for obj_data in initial_state.values():
                            obj_coords = obj_data[2]
                            obj_poly = Polygon(obj_coords)
                            dist = sol_poly.distance(obj_poly)
                            if dist < min_dist:
                                min_dist = dist
                        if min_dist != float('inf') and min_dist > 0:
                            middle_bonus = min(1.0, min_dist / 100.0) * 0.02
                            placement_key = tuple(solution[:3])
                            if placement_key in placement2score:
                                placement2score[placement_key] += middle_bonus * 1.0
                except Exception as e:
                    print(f'Warning: Failed to apply middle constraint: {e}', file=sys.stderr)
                continue
            if constraint_type == 'global':
                continue
            target_id = constraint.get('target')
            if not constraint_type or not constraint_value:
                continue
            if target_id and target_id not in initial_state:
                print(f'skipping constraint because target {target_id} not in initial_state: {constraint}', file=sys.stderr)
                continue
            print(f'pass a constraint: {constraint_type} {constraint_value} target {target_id}', file=sys.stderr)
            if constraint_type == 'distance' and constraint_value in ['close to', 'near', 'far']:
                self.apply_distance_constraint(candidate_solutions, constraint_value, initial_state[target_id], placement2score)
            elif constraint_type == 'relative' and constraint_value in ['in front of', 'behind', 'left of', 'right of', 'side of']:
                self.apply_relative_constraint(candidate_solutions, constraint_value, initial_state[target_id], placement2score)
            elif constraint_type == 'direction' and constraint_value == 'face to':
                self.apply_face_constraint(candidate_solutions, initial_state[target_id], placement2score)
            elif constraint_type == 'direction' and constraint_value == 'face same as':
                self.apply_face_same_constraint(candidate_solutions, initial_state[target_id], placement2score)
            elif constraint_type == 'alignment' and constraint_value == 'center aligned':
                self.apply_alignment_constraint(candidate_solutions, initial_state[target_id], placement2score)
        if not placement2score:
            return candidate_solutions[0] if candidate_solutions else None
        sorted_by_score = sorted(placement2score.keys(), key=lambda p: -placement2score[p])
        top_k = max(5, int(len(sorted_by_score) * 0.03))
        top_placements = sorted_by_score[:top_k]
        if top_placements:
            mean_x = sum((p[0][0] for p in top_placements)) / len(top_placements)
            mean_y = sum((p[0][1] for p in top_placements)) / len(top_placements)
        else:
            (mean_x, mean_y) = (0, 0)
        sorted_placements = sorted(placement2score.keys(), key=lambda p: (-placement2score[p], ((p[0][0] - mean_x) ** 2 + (p[0][1] - mean_y) ** 2) ** 0.5))
        if sorted_placements:
            best_placement = sorted_placements[0]
            best_solution = next((sol for sol in candidate_solutions if tuple(sol[:3]) == best_placement), None)
            if best_solution:
                best_solution[3] = placement2score[best_placement]
            return best_solution
        return None

    def apply_distance_constraint(self, solutions, distance_type, target_object, placement2score):
        """Apply distance constraints (close to/near/far)."""
        target_coords = target_object[2]
        target_poly = Polygon(target_coords)
        distances = []
        for solution in solutions:
            sol_coords = solution[2]
            sol_poly = Polygon(sol_coords)
            distance = target_poly.distance(sol_poly)
            distances.append(distance)
        if not distances:
            return
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
        else:
            return
        if len(points) > 1:
            x = [point[0] for point in points]
            y = [point[1] for point in points]
            f = interp1d(x, y, kind='linear', fill_value='extrapolate')
            for (i, solution) in enumerate(solutions):
                distance = distances[i]
                constraint_score = float(f(distance)) * self.constraint_type2weight['distance']
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += constraint_score

    def apply_relative_constraint(self, solutions, place_type, target_object, placement2score):
        """Apply relative position constraints."""
        (_, target_rotation, target_coords, _) = target_object
        target_polygon = Polygon(target_coords)
        (min_x, min_y, max_x, max_y) = target_polygon.bounds
        mean_x = (min_x + max_x) / 2
        mean_y = (min_y + max_y) / 2
        comparison_dict = {'left of': {0: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y, 90: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x, 180: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y, 270: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x}, 'right of': {0: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y, 90: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x, 180: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y, 270: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x}, 'in front of': {0: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x, 90: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y, 180: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x, 270: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y}, 'behind': {0: lambda sol_center: sol_center[1] < min_y and min_x <= sol_center[0] <= max_x, 90: lambda sol_center: sol_center[0] > max_x and min_y <= sol_center[1] <= max_y, 180: lambda sol_center: sol_center[1] > max_y and min_x <= sol_center[0] <= max_x, 270: lambda sol_center: sol_center[0] < min_x and min_y <= sol_center[1] <= max_y}, 'side of': {0: lambda sol_center: min_y <= sol_center[1] <= max_y, 90: lambda sol_center: min_x <= sol_center[0] <= max_x, 180: lambda sol_center: min_y <= sol_center[1] <= max_y, 270: lambda sol_center: min_x <= sol_center[0] <= max_x}}
        comparison_dict_loose = {'left of': {0: lambda sol_center: sol_center[0] < min_x, 90: lambda sol_center: sol_center[1] < min_y, 180: lambda sol_center: sol_center[0] > max_x, 270: lambda sol_center: sol_center[1] > max_y}, 'right of': {0: lambda sol_center: sol_center[0] > max_x, 90: lambda sol_center: sol_center[1] > max_y, 180: lambda sol_center: sol_center[0] < min_x, 270: lambda sol_center: sol_center[1] < min_y}, 'in front of': {0: lambda sol_center: sol_center[1] > max_y, 90: lambda sol_center: sol_center[0] < min_x, 180: lambda sol_center: sol_center[1] < min_y, 270: lambda sol_center: sol_center[0] > max_x}, 'behind': {0: lambda sol_center: sol_center[1] < min_y, 90: lambda sol_center: sol_center[0] > max_x, 180: lambda sol_center: sol_center[1] > max_y, 270: lambda sol_center: sol_center[0] < min_x}, 'side of': {0: lambda sol_center: min_y <= sol_center[1] <= max_y, 90: lambda sol_center: min_x <= sol_center[0] <= max_x, 180: lambda sol_center: min_y <= sol_center[1] <= max_y, 270: lambda sol_center: min_x <= sol_center[0] <= max_x}}
        comparison_dict_looser = {'left of': {0: lambda sol_center: sol_center[0] < mean_x, 90: lambda sol_center: sol_center[1] < mean_y, 180: lambda sol_center: sol_center[0] > mean_x, 270: lambda sol_center: sol_center[1] > mean_y}, 'right of': {0: lambda sol_center: sol_center[0] > mean_x, 90: lambda sol_center: sol_center[1] > mean_y, 180: lambda sol_center: sol_center[0] < mean_x, 270: lambda sol_center: sol_center[1] < mean_y}, 'in front of': {0: lambda sol_center: sol_center[1] > mean_y, 90: lambda sol_center: sol_center[0] < mean_x, 180: lambda sol_center: sol_center[1] < mean_y, 270: lambda sol_center: sol_center[0] > mean_x}, 'behind': {0: lambda sol_center: sol_center[1] < mean_y, 90: lambda sol_center: sol_center[0] > mean_x, 180: lambda sol_center: sol_center[1] > mean_y, 270: lambda sol_center: sol_center[0] < mean_x}, 'side of': {0: lambda sol_center: min_y <= sol_center[1] <= max_y, 90: lambda sol_center: min_x <= sol_center[0] <= max_x, 180: lambda sol_center: min_y <= sol_center[1] <= max_y, 270: lambda sol_center: min_x <= sol_center[0] <= max_x}}
        compare_func = comparison_dict.get(place_type, {}).get(target_rotation)
        compare_func_loose = comparison_dict_loose.get(place_type, {}).get(target_rotation)
        compare_func_looser = comparison_dict_looser.get(place_type, {}).get(target_rotation)
        if not compare_func:
            return
        for solution in solutions:
            sol_center = solution[0]
            placement_key = tuple(solution[:3])
            if placement_key not in placement2score:
                continue
            if compare_func(sol_center):
                placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight['relative']
            elif compare_func_loose and compare_func_loose(sol_center):
                placement2score[placement_key] += self.constraint_bouns * 0.2 * self.constraint_type2weight['relative']
            elif compare_func_looser and compare_func_looser(sol_center):
                placement2score[placement_key] += self.constraint_bouns * 0.01 * self.constraint_type2weight['relative']

    def apply_face_constraint(self, solutions, target_object, placement2score):
        """Apply face-to constraint."""
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
        for solution in solutions:
            sol_center = solution[0]
            sol_rotation = solution[1]
            far_point = (sol_center[0] + 1000000.0 * unit_vectors[sol_rotation][0], sol_center[1] + 1000000.0 * unit_vectors[sol_rotation][1])
            half_line = LineString([sol_center, far_point])
            sol_center_point = Point(sol_center[0], sol_center[1])
            if half_line.intersects(target_poly):
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight['direction']
            elif not target_poly_x_inf.contains(sol_center_point) and half_line.intersects(target_poly_x_inf):
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += 0.3 * self.constraint_bouns * self.constraint_type2weight['direction']
            elif not target_poly_y_inf.contains(sol_center_point) and half_line.intersects(target_poly_y_inf):
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += 0.3 * self.constraint_bouns * self.constraint_type2weight['direction']

    def apply_face_same_constraint(self, solutions, target_object, placement2score):
        """Apply face-same-as constraint (same facing direction as target)."""
        target_rotation = target_object[1]
        for solution in solutions:
            sol_rotation = solution[1]
            if abs(sol_rotation - target_rotation) < 10:
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight['direction']

    def apply_alignment_constraint(self, solutions, target_object, placement2score):
        """Apply center alignment constraint."""
        target_center = target_object[0]
        eps = self.grid_size / 2
        for solution in solutions:
            sol_center = solution[0]
            if abs(sol_center[0] - target_center[0]) < eps or abs(sol_center[1] - target_center[1]) < eps:
                placement_key = tuple(solution[:3])
                if placement_key in placement2score:
                    placement2score[placement_key] += self.constraint_bouns * self.constraint_type2weight['alignment']

    def visualize_object_movement(self, room_poly, grid_points, initial_state, object_to_move, best_solution, suggested_position, suggested_rotation, room_obj, room_id='unknown'):
        """
        Visualize object movement showing room, grid points, existing objects, current position, 
        suggested position and final placement.
        """
        try:
            vis_dir = f'{SERVER_ROOT_DIR}/vis'
            if not os.path.exists(vis_dir):
                os.makedirs(vis_dir)
            plt.rcParams['font.size'] = 12
            (fig, ax) = plt.subplots(figsize=(16, 14))
            (x, y) = room_poly.exterior.xy
            ax.plot(x, y, '-', label='Room Boundary', color='black', linewidth=3)
            if grid_points:
                grid_x = [point[0] for point in grid_points]
                grid_y = [point[1] for point in grid_points]
                ax.scatter(grid_x, grid_y, s=8, color='lightgray', alpha=0.6, label='Available Movement Points')
            colors = ['brown', 'cyan', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
            color_idx = 0
            for (object_id, solution) in initial_state.items():
                (center, rotation, box_coords) = solution[:3]
                (center_x, center_y) = center
                obj_poly = Polygon(box_coords)
                (x_coords, y_coords) = obj_poly.exterior.xy
                if object_id.startswith('door-'):
                    ax.plot(x_coords, y_coords, '-', linewidth=3, color='brown', alpha=0.8)
                    ax.fill(x_coords, y_coords, color='brown', alpha=0.5)
                    ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                elif object_id.startswith('window-'):
                    ax.plot(x_coords, y_coords, '-', linewidth=2, color='cyan', alpha=0.8)
                    ax.fill(x_coords, y_coords, color='cyan', alpha=0.5)
                    ax.text(center_x, center_y, object_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
                elif object_id.startswith('existing-'):
                    display_id = object_id.replace('existing-', '')
                    is_object_being_moved = display_id == object_to_move.id
                    if not is_object_being_moved:
                        current_color = colors[color_idx % len(colors)]
                        ax.plot(x_coords, y_coords, '-', linewidth=2, color=current_color, alpha=0.8)
                        ax.fill(x_coords, y_coords, color=current_color, alpha=0.4)
                        ax.text(center_x, center_y, display_id, fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
                        arrow_length = 20
                        if rotation == 0:
                            ax.arrow(center_x, center_y, 0, arrow_length, head_width=6, head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                        elif rotation == 90:
                            ax.arrow(center_x, center_y, -arrow_length, 0, head_width=6, head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                        elif rotation == 180:
                            ax.arrow(center_x, center_y, 0, -arrow_length, head_width=6, head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                        elif rotation == 270:
                            ax.arrow(center_x, center_y, arrow_length, 0, head_width=6, head_length=4, fc=current_color, ec=current_color, alpha=0.7)
                        color_idx += 1
            current_obj_width_cm = object_to_move.dimensions.width * 100
            current_obj_length_cm = object_to_move.dimensions.length * 100
            current_rotation = object_to_move.rotation.z
            if current_rotation == 0:
                curr_obj_length_x_cm = current_obj_width_cm
                curr_obj_length_y_cm = current_obj_length_cm
            elif current_rotation == 90:
                curr_obj_length_x_cm = current_obj_length_cm
                curr_obj_length_y_cm = current_obj_width_cm
            elif current_rotation == 180:
                curr_obj_length_x_cm = current_obj_width_cm
                curr_obj_length_y_cm = current_obj_length_cm
            elif current_rotation == 270:
                curr_obj_length_x_cm = current_obj_length_cm
                curr_obj_length_y_cm = current_obj_width_cm
            else:
                curr_obj_length_x_cm = current_obj_width_cm
                curr_obj_length_y_cm = current_obj_length_cm
            try:
                current_obj_x_cm = (object_to_move.position.x - room_obj.position.x) * 100
                current_obj_y_cm = (object_to_move.position.y - room_obj.position.y) * 100
                current_obj_vertices = [(current_obj_x_cm - curr_obj_length_x_cm / 2, current_obj_y_cm - curr_obj_length_y_cm / 2), (current_obj_x_cm + curr_obj_length_x_cm / 2, current_obj_y_cm - curr_obj_length_y_cm / 2), (current_obj_x_cm + curr_obj_length_x_cm / 2, current_obj_y_cm + curr_obj_length_y_cm / 2), (current_obj_x_cm - curr_obj_length_x_cm / 2, current_obj_y_cm + curr_obj_length_y_cm / 2)]
                current_obj_poly = Polygon(current_obj_vertices)
                (x_coords, y_coords) = current_obj_poly.exterior.xy
                ax.plot(x_coords, y_coords, '-', linewidth=3, color='red', alpha=0.8, linestyle='--')
                ax.fill(x_coords, y_coords, color='red', alpha=0.3)
                ax.text(current_obj_x_cm, current_obj_y_cm, f'CURRENT:\n{object_to_move.type}\n{object_to_move.id}', fontsize=9, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='pink', alpha=0.9))
                arrow_length = 25
                if current_rotation == 0:
                    ax.arrow(current_obj_x_cm, current_obj_y_cm, 0, arrow_length, head_width=7, head_length=5, fc='darkred', ec='darkred', alpha=0.8, linestyle='--')
                elif current_rotation == 90:
                    ax.arrow(current_obj_x_cm, current_obj_y_cm, -arrow_length, 0, head_width=7, head_length=5, fc='darkred', ec='darkred', alpha=0.8, linestyle='--')
                elif current_rotation == 180:
                    ax.arrow(current_obj_x_cm, current_obj_y_cm, 0, -arrow_length, head_width=7, head_length=5, fc='darkred', ec='darkred', alpha=0.8, linestyle='--')
                elif current_rotation == 270:
                    ax.arrow(current_obj_x_cm, current_obj_y_cm, arrow_length, 0, head_width=7, head_length=5, fc='darkred', ec='darkred', alpha=0.8, linestyle='--')
            except Exception as e:
                print(f'Warning: Could not draw current object position: {e}')
            if suggested_position:
                suggested_x = suggested_position['x']
                suggested_y = suggested_position['y']
                ax.scatter([suggested_x], [suggested_y], c='magenta', s=200, marker='*', label='Claude Suggested Position', alpha=0.9, edgecolors='black', linewidth=2)
                tolerance_circle = plt.Circle((suggested_x, suggested_y), self.position_tolerance, color='magenta', fill=False, linestyle='--', alpha=0.5, linewidth=2)
                ax.add_patch(tolerance_circle)
                if suggested_rotation is not None:
                    arrow_length = 35
                    if suggested_rotation == 0:
                        ax.arrow(suggested_x, suggested_y, 0, arrow_length, head_width=8, head_length=6, fc='magenta', ec='magenta', alpha=0.7, linewidth=2)
                    elif suggested_rotation == 90:
                        ax.arrow(suggested_x, suggested_y, -arrow_length, 0, head_width=8, head_length=6, fc='magenta', ec='magenta', alpha=0.7, linewidth=2)
                    elif suggested_rotation == 180:
                        ax.arrow(suggested_x, suggested_y, 0, -arrow_length, head_width=8, head_length=6, fc='magenta', ec='magenta', alpha=0.7, linewidth=2)
                    elif suggested_rotation == 270:
                        ax.arrow(suggested_x, suggested_y, arrow_length, 0, head_width=8, head_length=6, fc='magenta', ec='magenta', alpha=0.7, linewidth=2)
            if best_solution:
                (center, rotation, polygon_coords, score) = best_solution
                (center_x, center_y) = center
                obj_poly = Polygon(polygon_coords)
                (x_coords, y_coords) = obj_poly.exterior.xy
                ax.plot(x_coords, y_coords, '-', linewidth=4, color='gold', alpha=0.9)
                ax.fill(x_coords, y_coords, color='gold', alpha=0.6)
                ax.text(center_x, center_y, f'NEW POSITION:\n{object_to_move.type}\n{object_to_move.id}\nScore: {score:.2f}', fontsize=10, ha='center', va='center', fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.9))
                arrow_length = 30
                if rotation == 0:
                    ax.arrow(center_x, center_y, 0, arrow_length, head_width=8, head_length=6, fc='darkgoldenrod', ec='darkgoldenrod', linewidth=2)
                elif rotation == 90:
                    ax.arrow(center_x, center_y, -arrow_length, 0, head_width=8, head_length=6, fc='darkgoldenrod', ec='darkgoldenrod', linewidth=2)
                elif rotation == 180:
                    ax.arrow(center_x, center_y, 0, -arrow_length, head_width=8, head_length=6, fc='darkgoldenrod', ec='darkgoldenrod', linewidth=2)
                elif rotation == 270:
                    ax.arrow(center_x, center_y, arrow_length, 0, head_width=8, head_length=6, fc='darkgoldenrod', ec='darkgoldenrod', linewidth=2)
                if 'current_obj_x_cm' in locals() and 'current_obj_y_cm' in locals():
                    ax.annotate('', xy=(center_x, center_y), xytext=(current_obj_x_cm, current_obj_y_cm), arrowprops=dict(arrowstyle='->', color='blue', lw=3, alpha=0.7, linestyle=':'), zorder=5)
                    distance = math.sqrt((center_x - current_obj_x_cm) ** 2 + (center_y - current_obj_y_cm) ** 2)
                    mid_x = (center_x + current_obj_x_cm) / 2
                    mid_y = (center_y + current_obj_y_cm) / 2
                    ax.text(mid_x, mid_y, f'Move: {distance:.0f}cm', fontsize=8, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightblue', alpha=0.8))
            title = f'Object Movement: {object_to_move.type} (ID: {object_to_move.id})'
            if best_solution:
                title += f'\nRoom: {room_id} | Movement: SUCCESS'
            else:
                title += f'\nRoom: {room_id} | Movement: FAILED'
            ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('X Position (cm)', fontsize=12)
            ax.set_ylabel('Y Position (cm)', fontsize=12)
            ax.set_aspect('equal', 'box')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='upper right')
            summary_text = f'Movement Details:\nObject: {object_to_move.type} (ID: {object_to_move.id})\nDimensions: {object_to_move.dimensions.width * 100:.0f} × {object_to_move.dimensions.length * 100:.0f} × {object_to_move.dimensions.height * 100:.0f} cm\nGrid Size: {self.grid_size} cm\nPosition Tolerance: {self.position_tolerance} cm'
            if suggested_rotation is not None:
                summary_text += f'\nSuggested Rotation: {suggested_rotation}°'
            if best_solution:
                summary_text += f'\nFinal Position: ({center[0]:.0f}, {center[1]:.0f}) cm\nFinal Rotation: {rotation}°\nFinal Score: {score:.3f}'
                if 'current_obj_x_cm' in locals() and 'current_obj_y_cm' in locals():
                    distance = math.sqrt((center[0] - current_obj_x_cm) ** 2 + (center[1] - current_obj_y_cm) ** 2)
                    rotation_change = abs(rotation - current_rotation)
                    if rotation_change > 180:
                        rotation_change = 360 - rotation_change
                    summary_text += f'\nMovement Distance: {distance:.0f} cm\nRotation Change: {rotation_change:.0f}°'
            ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8), verticalalignment='top')
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'{vis_dir}/object_movement_{room_id}_{object_to_move.id}_{timestamp}.png'
            plt.savefig(filename, bbox_inches='tight', dpi=150, facecolor='white')
            print(f'Object movement visualization saved: {filename}')
            plt.close(fig)
            return filename
        except Exception as e:
            print(f'Error creating object movement visualization: {e}')
            return None

def parse_claude_constraints_for_movement(claude_constraints: List[str], object_id: str, existing_objects: List[Object]=None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Parse Claude's constraint strings for movement into the format expected by the MovementFloorSolver.
    Similar to parse_claude_constraints but adapted for movement context.
    """
    parsed_constraints = {object_id: []}
    if existing_objects is None:
        existing_objects = []
    existing_object_ids = {f'existing-{obj.id}' for obj in existing_objects}
    for constraint_str in claude_constraints:
        constraint_str = constraint_str.strip()
        if constraint_str.lower() in ['edge', 'middle']:
            parsed_constraints[object_id].append({'type': 'global', 'constraint': constraint_str.lower(), 'target': None})
        elif ',' in constraint_str:
            parts = [part.strip() for part in constraint_str.split(',', 1)]
            if len(parts) == 2:
                (constraint_type, target_object_id) = parts
                constraint_mapping = {'close to': ('distance', 'close to'), 'near': ('distance', 'near'), 'far': ('distance', 'far'), 'in front of': ('relative', 'in front of'), 'behind': ('relative', 'behind'), 'left of': ('relative', 'left of'), 'right of': ('relative', 'right of'), 'side of': ('relative', 'side of'), 'around': ('relative', 'around'), 'face to': ('direction', 'face to'), 'face same as': ('direction', 'face same as'), 'center aligned': ('alignment', 'center aligned')}
                if constraint_type.lower() in constraint_mapping:
                    (solver_type, solver_constraint) = constraint_mapping[constraint_type.lower()]
                    if not target_object_id.startswith('existing-'):
                        target_object_id = f'existing-{target_object_id}'
                    if target_object_id not in existing_object_ids:
                        short_id = target_object_id.removeprefix('existing-')
                        suffix_matches = [existing_id for existing_id in existing_object_ids if existing_id.removeprefix('existing-').endswith(short_id)]
                        if len(suffix_matches) == 1:
                            target_object_id = suffix_matches[0]
                    if target_object_id in existing_object_ids:
                        if constraint_type.lower() == 'around':
                            parsed_constraints[object_id].append({'type': 'distance', 'constraint': 'close to', 'target': target_object_id})
                            parsed_constraints[object_id].append({'type': 'direction', 'constraint': 'face to', 'target': target_object_id})
                        elif constraint_type.lower() == 'in front of':
                            parsed_constraints[object_id].append({'type': 'relative', 'constraint': 'in front of', 'target': target_object_id})
                            parsed_constraints[object_id].append({'type': 'alignment', 'constraint': 'center aligned', 'target': target_object_id})
                        else:
                            parsed_constraints[object_id].append({'type': solver_type, 'constraint': solver_constraint, 'target': target_object_id})
    has_global = any((c['type'] == 'global' for c in parsed_constraints[object_id]))
    if not has_global:
        parsed_constraints[object_id].insert(0, {'type': 'global', 'constraint': 'edge', 'target': None})
    return parsed_constraints

async def move_object_floor(room: Room, current_layout: FloorPlan, object_to_move: Object, claude_movement_result: Dict[str, Any]) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Move the object on the room floor using MovementFloorSolver based on Claude's suggested position and constraints.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_move: Object to move on floor
        claude_movement_result: Result from get_movement_location_from_claude_floor
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, movement_info)
    """
    movement_info = {'success': False, 'error': None, 'object_moved': None, 'claude_recommendation': claude_movement_result, 'solver_result': None, 'child_objects_removed': [], 'original_room_backup': None}
    try:
        if not claude_movement_result.get('success'):
            movement_info['error'] = f"Claude movement analysis failed: {claude_movement_result.get('error', 'Unknown error')}"
            return (room.objects, current_layout, movement_info)
        original_room = copy.deepcopy(room)
        movement_info['original_room_backup'] = original_room
        child_objects_to_remove = []
        removed_indices = set()
        object_to_move_index = None
        for (i, obj) in enumerate(room.objects):
            if obj.id == object_to_move.id:
                object_to_move_index = i
                break
        if object_to_move_index is None:
            movement_info['error'] = f'Object to move {object_to_move.id} not found in room'
            return (room.objects, current_layout, movement_info)
        for (i, obj_child) in enumerate(room.objects):
            if i == object_to_move_index:
                continue
            current_obj = obj_child
            while True:
                if current_obj.place_id == 'floor' or current_obj.place_id == 'wall':
                    break
                if current_obj.place_id == object_to_move.id:
                    removed_indices.add(i)
                    child_objects_to_remove.append(obj_child)
                    break
                parent_obj = next((obj for obj in room.objects if obj.id == current_obj.place_id), None)
                if parent_obj is None:
                    break
                current_obj = parent_obj
        movement_info['child_objects_removed'] = [{'id': obj.id, 'type': obj.type} for obj in child_objects_to_remove]
        suggested_position = claude_movement_result.get('new_suggested_position', {'x': 0, 'y': 0})
        suggested_rotation = claude_movement_result.get('new_suggested_rotation')
        claude_constraints = claude_movement_result.get('constraints', [])
        movement_info['claude_constraints'] = claude_constraints
        movement_info['suggested_position'] = suggested_position
        movement_info['suggested_rotation'] = suggested_rotation
        try:
            other_objects = [obj for (i, obj) in enumerate(room.objects) if obj.id != object_to_move.id and i not in removed_indices]
            parsed_constraints = parse_claude_constraints_for_movement(claude_constraints, object_to_move.id, other_objects)
            movement_info['parsed_constraints'] = parsed_constraints
        except Exception as e:
            movement_info['error'] = f'Failed to parse constraints: {e}'
            return (room.objects, current_layout, movement_info)
        print(f'parsed_constraints: {parsed_constraints}', file=sys.stderr)
        max_wall_thickness_cm = 5
        if room.walls:
            max_wall_thickness_cm = max((wall.thickness * 0.5 * 100 for wall in room.walls))
        inner_width_cm = room.dimensions.width * 100 - 2 * max_wall_thickness_cm
        inner_length_cm = room.dimensions.length * 100 - 2 * max_wall_thickness_cm
        room_vertices = [(max_wall_thickness_cm, max_wall_thickness_cm), (max_wall_thickness_cm, max_wall_thickness_cm + inner_length_cm), (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm + inner_length_cm), (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm)]
        room_poly = Polygon(room_vertices)
        initial_state = get_door_window_placements(room)
        for (i, obj) in enumerate(room.objects):
            if obj.id == object_to_move.id or i in removed_indices:
                continue
            if obj.place_id == 'wall':
                if obj.position.z > object_to_move.dimensions.height:
                    continue
            obj_x_cm = (obj.position.x - room.position.x) * 100
            obj_y_cm = (obj.position.y - room.position.y) * 100
            obj_width_cm = obj.dimensions.width * 100 + 3.5
            obj_length_cm = obj.dimensions.length * 100 + 3.5
            object_rotation = int(obj.rotation.z) % 360
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
            obj_vertices = [(obj_x_cm - obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2), (obj_x_cm - obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2)]
            initial_state[f'existing-{obj.id}'] = [(obj_x_cm, obj_y_cm), object_rotation, obj_vertices, 1.0]
        solver = MovementFloorSolver(grid_size=20, constraint_bouns=1.0, position_tolerance=100)
        object_dim = (object_to_move.dimensions.width * 100 + 3.5, object_to_move.dimensions.length * 100 + 3.5)
        best_solution = solver.get_best_movement_placement(room_poly=room_poly, object_dim=object_dim, suggested_position=suggested_position, suggested_rotation=suggested_rotation, constraints=parsed_constraints.get(object_to_move.id, []), initial_state=initial_state, current_object_id=object_to_move.id)
        movement_info['solver_result'] = {'solution_found': best_solution is not None, 'best_solution': best_solution, 'constraints_used': parsed_constraints, 'used_suggested_position': True, 'used_suggested_rotation': suggested_rotation is not None}
        if not best_solution:
            movement_info['error'] = 'MovementFloorSolver could not find valid new placement for object'
            return (room.objects, current_layout, movement_info)
        (center, rotation, polygon_coords, score) = best_solution
        old_position = {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}
        old_rotation = object_to_move.rotation.z
        object_to_move.position = Point3D(x=room.position.x + center[0] / 100, y=room.position.y + center[1] / 100, z=room.position.z)
        object_to_move.rotation = Euler(x=0, z=rotation, y=0)
        object_to_move.place_id = 'floor'
        updated_room_objects = [obj for (i, obj) in enumerate(room.objects) if i not in removed_indices]
        for layout_room in current_layout.rooms:
            if layout_room.id == room.id:
                layout_room.objects = updated_room_objects
                break
        room.objects = updated_room_objects
        if PHYSICS_CRITIC_ENABLED:
            print(f'Evaluating the stability after movement and checking if object is stable', file=sys.stderr)
            scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
            room_dict_save_path = os.path.join(scene_save_dir, f'{room.id}.json')
            with open(room_dict_save_path, 'w') as f:
                json.dump(asdict(room), f)
            result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
            if not isinstance(result_create, dict) or result_create.get('status') != 'success':
                movement_info['error'] = f'Failed to create scene for physics validation: {result_create}'
                movement_info['success'] = False
                if movement_info['original_room_backup']:
                    original_objects = movement_info['original_room_backup'].objects
                    room.objects = original_objects
                    for layout_room in current_layout.rooms:
                        if layout_room.id == room.id:
                            layout_room.objects = original_objects
                            break
                    return (original_objects, current_layout, movement_info)
                return (room.objects, current_layout, movement_info)
            result_sim = simulate_the_scene()
            if not isinstance(result_sim, dict) or result_sim.get('status') != 'success':
                movement_info['error'] = f'Failed to simulate scene for physics validation: {result_sim}'
                movement_info['success'] = False
                if movement_info['original_room_backup']:
                    original_objects = movement_info['original_room_backup'].objects
                    room.objects = original_objects
                    for layout_room in current_layout.rooms:
                        if layout_room.id == room.id:
                            layout_room.objects = original_objects
                            break
                    return (original_objects, current_layout, movement_info)
                return (room.objects, current_layout, movement_info)
            unstable_object_ids = result_sim['unstable_objects']
            print(f'Number of unstable objects: {len(unstable_object_ids)}', file=sys.stderr)
            if object_to_move.id in unstable_object_ids:
                movement_info['error'] = f'Object {object_to_move.id} is unstable after movement. Movement rejected.'
                movement_info['success'] = False
                print(f'Object {object_to_move.id} is unstable after movement. Restoring original state.', file=sys.stderr)
                if movement_info['original_room_backup']:
                    original_objects = movement_info['original_room_backup'].objects
                    room.objects = original_objects
                    for layout_room in current_layout.rooms:
                        if layout_room.id == room.id:
                            layout_room.objects = original_objects
                            break
                    return (original_objects, current_layout, movement_info)
                return (room.objects, current_layout, movement_info)
            print(f'Object {object_to_move.id} is stable after movement.', file=sys.stderr)
        movement_info['success'] = True
        movement_info['object_moved'] = {'id': object_to_move.id, 'type': object_to_move.type, 'old_position': old_position, 'new_position': {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}, 'old_rotation': old_rotation, 'new_rotation': rotation, 'score': score, 'polygon_coords_cm': polygon_coords}
        return (updated_room_objects, current_layout, movement_info)
    except Exception as e:
        movement_info['error'] = f'Error in move_object_floor: {e}'
        movement_info['success'] = False
        if movement_info['original_room_backup']:
            original_objects = movement_info['original_room_backup'].objects
            room.objects = original_objects
            for layout_room in current_layout.rooms:
                if layout_room.id == room.id:
                    layout_room.objects = original_objects
                    break
            return (original_objects, current_layout, movement_info)
        return (room.objects, current_layout, movement_info)

async def move_object_wall(room: Room, current_layout: FloorPlan, object_to_move: Object, claude_movement_result: Dict[str, Any], movement_target_location: str) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Move an object on a wall using physics-based sampling and validation.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_move: Object to move on top of another object
        claude_movement_result: Result from get_movement_location_from_claude_object
        movement_target_location: ID of target object to place on
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, movement_info)
    """
    movement_info = {'success': False, 'error': None, 'object_moved': None, 'claude_recommendation': claude_movement_result, 'child_objects_removed': [], 'original_room_backup': None, 'placement_results': []}
    try:
        original_room = copy.deepcopy(room)
        movement_info['original_room_backup'] = original_room
        child_objects_to_remove = []
        removed_indices = set()
        object_to_move_index = None
        for (i, obj) in enumerate(room.objects):
            if obj.id == object_to_move.id:
                object_to_move_index = i
                break
        if object_to_move_index is None:
            movement_info['error'] = f'Object to move {object_to_move.id} not found in room'
            return (room.objects, current_layout, movement_info)
        for (i, obj_child) in enumerate(room.objects):
            if i == object_to_move_index:
                continue
            current_obj = obj_child
            while True:
                if current_obj.place_id == 'floor' or current_obj.place_id == 'wall':
                    break
                if current_obj.place_id == object_to_move.id:
                    removed_indices.add(i)
                    child_objects_to_remove.append(obj_child)
                    break
                parent_obj = next((obj for obj in room.objects if obj.id == current_obj.place_id), None)
                if parent_obj is None:
                    break
                current_obj = parent_obj
        movement_info['child_objects_removed'] = [{'id': obj.id, 'type': obj.type} for obj in child_objects_to_remove]
        object_to_move.place_guidance = claude_movement_result['movement_intent']
        room_objects_without_moved = [obj for (i, obj) in enumerate(room.objects) if i != object_to_move_index and i not in removed_indices]
        room_copy_eval = copy.deepcopy(room)
        room_copy_eval.objects = room_objects_without_moved
        try:
            wall_systems = create_wall_coordinate_systems(room_copy_eval)
            wall_grids = create_wall_grid_points(wall_systems, grid_density=20)
            impossible_regions = calculate_impossible_wall_regions(room_copy_eval, room_objects_without_moved, wall_systems, object_to_move)
            valid_points = filter_valid_wall_points(object_to_move, wall_grids, impossible_regions, wall_systems)
            if not valid_points:
                movement_info['error'] = 'No valid wall placement points found'
                return (room.objects, current_layout, movement_info)
            best_placement = select_best_wall_placement(object_to_move, valid_points, room_objects_without_moved, wall_systems)
            try:
                visualize_wall_placement(object_to_move, wall_systems, wall_grids, impossible_regions, valid_points, best_placement, room.id)
            except Exception as e:
                print(f'Warning: Wall placement visualization failed: {e}')
                pass
            if best_placement:
                adjusted_placement = adjust_wall_object_position(best_placement, object_to_move, wall_systems)
                placed_obj = create_wall_placed_object(object_to_move, adjusted_placement, room_copy_eval, best_placement['constraints'])
                old_position = {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}
                old_rotation = {'x': object_to_move.rotation.x, 'y': object_to_move.rotation.y, 'z': object_to_move.rotation.z}
                old_place_id = object_to_move.place_id
                object_to_move.position = placed_obj.position
                object_to_move.rotation = placed_obj.rotation
                object_to_move.place_id = 'wall'
                object_to_move.placement_constraints = placed_obj.placement_constraints
                updated_room_objects = room_objects_without_moved + [object_to_move]
                for layout_room in current_layout.rooms:
                    if layout_room.id == room.id:
                        layout_room.objects = updated_room_objects
                        break
                room.objects = updated_room_objects
                if PHYSICS_CRITIC_ENABLED:
                    print(f'Evaluating the stability after wall movement and checking if object is stable', file=sys.stderr)
                    scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
                    room_dict_save_path = os.path.join(scene_save_dir, f'{room.id}.json')
                    with open(room_dict_save_path, 'w') as f:
                        json.dump(asdict(room), f)
                    result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
                    if not isinstance(result_create, dict) or result_create.get('status') != 'success':
                        movement_info['error'] = f'Failed to create scene for physics validation: {result_create}'
                        movement_info['success'] = False
                        if movement_info['original_room_backup']:
                            original_objects = movement_info['original_room_backup'].objects
                            room.objects = original_objects
                            for layout_room in current_layout.rooms:
                                if layout_room.id == room.id:
                                    layout_room.objects = original_objects
                                    break
                            return (original_objects, current_layout, movement_info)
                        return (room.objects, current_layout, movement_info)
                    result_sim = simulate_the_scene()
                    if not isinstance(result_sim, dict) or result_sim.get('status') != 'success':
                        movement_info['error'] = f'Failed to simulate scene for physics validation: {result_sim}'
                        movement_info['success'] = False
                        if movement_info['original_room_backup']:
                            original_objects = movement_info['original_room_backup'].objects
                            room.objects = original_objects
                            for layout_room in current_layout.rooms:
                                if layout_room.id == room.id:
                                    layout_room.objects = original_objects
                                    break
                            return (original_objects, current_layout, movement_info)
                        return (room.objects, current_layout, movement_info)
                    unstable_object_ids = result_sim['unstable_objects']
                    print(f'Number of unstable objects: {len(unstable_object_ids)}', file=sys.stderr)
                    if object_to_move.id in unstable_object_ids:
                        movement_info['error'] = f'Object {object_to_move.id} is unstable after wall movement. Movement rejected.'
                        movement_info['success'] = False
                        print(f'Object {object_to_move.id} is unstable after wall movement. Restoring original state.', file=sys.stderr)
                        if movement_info['original_room_backup']:
                            original_objects = movement_info['original_room_backup'].objects
                            room.objects = original_objects
                            for layout_room in current_layout.rooms:
                                if layout_room.id == room.id:
                                    layout_room.objects = original_objects
                                    break
                            return (original_objects, current_layout, movement_info)
                        return (room.objects, current_layout, movement_info)
                    print(f'Object {object_to_move.id} is stable after wall movement.', file=sys.stderr)
                movement_info['success'] = True
                movement_info['object_moved'] = {'id': object_to_move.id, 'type': object_to_move.type, 'old_position': old_position, 'new_position': {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}, 'old_rotation': old_rotation, 'new_rotation': {'x': object_to_move.rotation.x, 'y': object_to_move.rotation.y, 'z': object_to_move.rotation.z}, 'old_place_id': old_place_id, 'new_place_id': 'wall', 'wall_id': best_placement['wall_id']}
                movement_info['placement_results'].append({'object_id': object_to_move.id, 'success': True, 'wall_id': best_placement['wall_id'], 'position': best_placement['position_3d'], 'rotation': best_placement['rotation']})
                return (updated_room_objects, current_layout, movement_info)
            else:
                movement_info['error'] = 'Failed to find suitable wall placement'
                movement_info['placement_results'].append({'object_id': object_to_move.id, 'success': False, 'error': 'Failed to find suitable wall placement'})
        except Exception as e:
            movement_info['error'] = f'Error in wall placement process: {str(e)}'
            movement_info['placement_results'].append({'object_id': object_to_move.id, 'success': False, 'error': f'Exception during wall placement: {str(e)}'})
    except Exception as e:
        movement_info['error'] = f'Error in move_object_wall: {str(e)}'
        movement_info['success'] = False
    if movement_info['original_room_backup']:
        original_objects = movement_info['original_room_backup'].objects
        room.objects = original_objects
        for layout_room in current_layout.rooms:
            if layout_room.id == room.id:
                layout_room.objects = original_objects
                break
        return (original_objects, current_layout, movement_info)
    return (room.objects, current_layout, movement_info)

async def get_movement_location_from_claude_object(room: Room, current_layout: FloorPlan, object_to_move: Object, movement_intent: Dict[str, Any], movement_target_location: str) -> Dict[str, Any]:
    return {'success': True, 'movement_target_location': movement_target_location}

async def move_object_object(room: Room, current_layout: FloorPlan, object_to_move: Object, claude_movement_result: Dict[str, Any], movement_target_location: str) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Move an object on top of another object using physics-based sampling and validation.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_move: Object to move on top of another object
        claude_movement_result: Result from get_movement_location_from_claude_object
        movement_target_location: ID of target object to place on
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, movement_info)
    """
    from objects.object_on_top_placement import get_random_placements_on_target_object, filter_placements_by_physics_critic
    movement_info = {'success': False, 'error': None, 'object_moved': None, 'claude_recommendation': claude_movement_result, 'child_objects_removed': [], 'original_room_backup': None, 'placement_results': []}
    try:
        original_room = copy.deepcopy(room)
        movement_info['original_room_backup'] = original_room
        child_objects_to_remove = []
        removed_indices = set()
        object_to_move_index = None
        for (i, obj) in enumerate(room.objects):
            if obj.id == object_to_move.id:
                object_to_move_index = i
                break
        if object_to_move_index is None:
            movement_info['error'] = f'Object to move {object_to_move.id} not found in room'
            return (room.objects, current_layout, movement_info)
        for (i, obj_child) in enumerate(room.objects):
            if i == object_to_move_index:
                continue
            current_obj = obj_child
            while True:
                if current_obj.place_id == 'floor' or current_obj.place_id == 'wall':
                    break
                if current_obj.place_id == object_to_move.id:
                    removed_indices.add(i)
                    child_objects_to_remove.append(obj_child)
                    break
                parent_obj = next((obj for obj in room.objects if obj.id == current_obj.place_id), None)
                if parent_obj is None:
                    break
                current_obj = parent_obj
        movement_info['child_objects_removed'] = [{'id': obj.id, 'type': obj.type} for obj in child_objects_to_remove]
        room_objects_without_moved = [obj for (i, obj) in enumerate(room.objects) if i != object_to_move_index and i not in removed_indices]
        target_object = next((obj for obj in room_objects_without_moved if obj.id == movement_target_location), None)
        if target_object is None:
            movement_info['error'] = f'Target object {movement_target_location} not found in room or was removed'
            return (room.objects, current_layout, movement_info)
        room_copy_eval = copy.deepcopy(room)
        room_copy_eval.objects = room_objects_without_moved
        placements = get_random_placements_on_target_object(current_layout, room_copy_eval, movement_target_location, object_to_move, sample_count=150, regular_rotation=True, place_location='both')
        if not placements:
            movement_info['error'] = f'No valid placements found on target object {movement_target_location}'
            return (room.objects, current_layout, movement_info)
        safe_placements = filter_placements_by_physics_critic(current_layout, room_copy_eval, object_to_move, placements)
        if not safe_placements:
            movement_info['error'] = f'No safe placements found after physics validation'
            return (room.objects, current_layout, movement_info)
        best_placement = safe_placements[0]
        position_placed = best_placement['position']
        rotation_placed = best_placement['rotation']
        old_position = {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}
        old_rotation = {'x': object_to_move.rotation.x, 'y': object_to_move.rotation.y, 'z': object_to_move.rotation.z}
        old_place_id = object_to_move.place_id
        object_to_move.position = Point3D(x=position_placed['x'], y=position_placed['y'], z=position_placed['z'])
        object_to_move.rotation = Euler(x=rotation_placed['x'] * 180 / np.pi, y=rotation_placed['y'] * 180 / np.pi, z=rotation_placed['z'] * 180 / np.pi)
        object_to_move.place_id = movement_target_location
        updated_room_objects = room_objects_without_moved + [object_to_move]
        for layout_room in current_layout.rooms:
            if layout_room.id == room.id:
                layout_room.objects = updated_room_objects
                break
        room.objects = updated_room_objects
        if PHYSICS_CRITIC_ENABLED:
            print(f'Final validation for moving {object_to_move.id} on {movement_target_location}', file=sys.stderr)
            print(f'Evaluating the stability after placement and remove objects that are not stable', file=sys.stderr)
            scene_save_dir = os.path.join(RESULTS_DIR, current_layout.id)
            room_dict_save_path = os.path.join(scene_save_dir, f'{room.id}.json')
            with open(room_dict_save_path, 'w') as f:
                json.dump(asdict(room), f)
            result_create = create_single_room_layout_scene_from_room(scene_save_dir, room_dict_save_path)
            if not isinstance(result_create, dict) or result_create.get('status') != 'success':
                movement_info['error'] = f'Failed to create scene for physics validation: {result_create}'
                movement_info['success'] = False
                if movement_info['original_room_backup']:
                    original_objects = movement_info['original_room_backup'].objects
                    room.objects = original_objects
                    for layout_room in current_layout.rooms:
                        if layout_room.id == room.id:
                            layout_room.objects = original_objects
                            break
                    return (original_objects, current_layout, movement_info)
                return (room.objects, current_layout, movement_info)
            result_sim = simulate_the_scene()
            if not isinstance(result_sim, dict) or result_sim.get('status') != 'success':
                movement_info['error'] = f'Failed to simulate scene for physics validation: {result_sim}'
                movement_info['success'] = False
                if movement_info['original_room_backup']:
                    original_objects = movement_info['original_room_backup'].objects
                    room.objects = original_objects
                    for layout_room in current_layout.rooms:
                        if layout_room.id == room.id:
                            layout_room.objects = original_objects
                            break
                    return (original_objects, current_layout, movement_info)
                return (room.objects, current_layout, movement_info)
            unstable_object_ids = result_sim['unstable_objects']
            print(f'Number of unstable objects: {len(unstable_object_ids)}', file=sys.stderr)
            print(f'room.objects: {len(room.objects)}', file=sys.stderr)
            if len(unstable_object_ids) > 0:
                print(f'unstable_object_ids: {unstable_object_ids}', file=sys.stderr)
                if object_to_move.id in unstable_object_ids:
                    movement_info['error'] = f'Object {object_to_move.id} is unstable after being placed on {movement_target_location}. Movement rejected.'
                    movement_info['success'] = False
                    print(f'Object {object_to_move.id} is unstable after movement. Restoring original state.', file=sys.stderr)
                    if movement_info['original_room_backup']:
                        original_objects = movement_info['original_room_backup'].objects
                        room.objects = original_objects
                        for layout_room in current_layout.rooms:
                            if layout_room.id == room.id:
                                layout_room.objects = original_objects
                                break
                        return (original_objects, current_layout, movement_info)
                    return (room.objects, current_layout, movement_info)
                else:
                    print(f'Warning: Other objects became unstable: {unstable_object_ids}', file=sys.stderr)
        print(f'Finished trial of moving {object_to_move.id} on {movement_target_location}, current room objects: {len(room.objects)}', file=sys.stderr)
        if object_to_move.id not in unstable_object_ids:
            movement_info['success'] = True
        movement_info['object_moved'] = {'id': object_to_move.id, 'type': object_to_move.type, 'old_position': old_position, 'new_position': {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}, 'old_rotation': old_rotation, 'new_rotation': {'x': object_to_move.rotation.x, 'y': object_to_move.rotation.y, 'z': object_to_move.rotation.z}, 'old_place_id': old_place_id, 'new_place_id': movement_target_location, 'total_placement_candidates': len(placements), 'safe_placement_candidates': len(safe_placements)}
        movement_info['placement_results'].append({'object_id': object_to_move.id, 'success': True, 'target_object_id': movement_target_location, 'position': {'x': position_placed['x'], 'y': position_placed['y'], 'z': position_placed['z']}, 'rotation': {'x': rotation_placed['x'], 'y': rotation_placed['y'], 'z': rotation_placed['z']}, 'total_candidates': len(placements), 'safe_candidates': len(safe_placements), 'child_objects_removed_count': len(child_objects_to_remove)})
        return (updated_room_objects, current_layout, movement_info)
    except Exception as e:
        movement_info['error'] = f'Error in move_object_object: {str(e)}'
        movement_info['success'] = False
        if movement_info['original_room_backup']:
            original_objects = movement_info['original_room_backup'].objects
            room.objects = original_objects
            for layout_room in current_layout.rooms:
                if layout_room.id == room.id:
                    layout_room.objects = original_objects
                    break
            return (original_objects, current_layout, movement_info)
        return (room.objects, current_layout, movement_info)

async def get_movement_location_from_claude(room: Room, current_layout: FloorPlan, object_to_move: Object, movement_intent: Dict[str, Any], movement_target_location: str) -> Dict[str, Any]:
    """
    Main dispatcher function that determines movement location type and calls appropriate handler.
    Currently only supports floor movement.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout
        object_to_move: Object that needs to be moved
        movement_intent: Intent from analyze_object_to_move_from_condition
        
    Returns:
        Dictionary containing movement placement information
    """
    try:
        if movement_target_location == 'floor':
            return await get_movement_location_from_claude_floor(room, current_layout, object_to_move, movement_intent)
        elif movement_target_location == 'wall':
            return await get_movement_location_from_claude_wall(room, current_layout, object_to_move, movement_intent, movement_target_location)
        elif movement_target_location in [obj.id for obj in room.objects]:
            return await get_movement_location_from_claude_object(room, current_layout, object_to_move, movement_intent, movement_target_location)
        else:
            raise ValueError(f'Invalid movement target location: {movement_target_location}')
    except Exception as e:
        return {'success': False, 'error': f'Error in get_movement_location_from_claude: {e}'}

async def move_object(room: Room, current_layout: FloorPlan, object_to_move: Object, claude_movement_result: Dict[str, Any], movement_target_location: str) -> Tuple[List[Object], FloorPlan, Dict[str, Any]]:
    """
    Main dispatcher function for moving objects.
    Currently only supports floor movement.
    
    Args:
        room: Target room
        current_layout: Current floor plan layout  
        object_to_move: Object to move
        claude_movement_result: Result from get_movement_location_from_claude
        
    Returns:
        Tuple of (updated_room_objects, updated_layout, movement_info)
    """
    try:
        if movement_target_location == 'floor':
            return await move_object_floor(room, current_layout, object_to_move, claude_movement_result)
        elif movement_target_location in [obj.id for obj in room.objects]:
            return await move_object_object(room, current_layout, object_to_move, claude_movement_result, movement_target_location)
        elif movement_target_location == 'wall':
            return await move_object_wall(room, current_layout, object_to_move, claude_movement_result, movement_target_location)
        else:
            raise ValueError(f'Invalid movement target location: {movement_target_location}')
    except Exception as e:
        movement_info = {'success': False, 'error': f'Error in move_object dispatcher: {e}', 'claude_recommendation': claude_movement_result}
        return (room.objects, current_layout, movement_info)
