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
import sys
sys.path.append('objects')
import uuid
from datetime import datetime
mcp_init_id = str(datetime.now().strftime('%Y%m%d_%H%M%S')) + '_' + str(uuid.uuid4())[:8]

def get_mcp_init_id():
    global mcp_init_id
    return mcp_init_id
from vlm import call_vlm
from typing import Optional, List, Dict, Any
import json
import os
import anthropic
from dataclasses import asdict
from mcp.server.fastmcp import FastMCP, Context, Image
from key import ANTHROPIC_API_KEY
from constants import RESULTS_DIR
from pathlib import Path
from models import FloorPlan, Room, Object, Point3D, Dimensions, Euler
from llm_client import call_llm_for_rooms_with_validation, call_llm_for_doors_windows, call_llm_for_doors_windows_graph_based
from validation import validate_room_layout, validate_llm_response_structure, validate_door_window_placement, validate_room_structure_issues, validate_door_window_issues, check_door_window_integrity, check_room_overlap
from layout_parser import parse_llm_response_to_floor_plan, parse_rooms_only_to_floor_plan, add_doors_windows_to_floor_plan
from correction import generate_detailed_correction_prompt, correct_door_window_integrity, correct_layout_issues, aggressive_layout_correction
from utils import extract_wall_side_from_id, get_room_priorities_from_claude, calculate_room_reduction, export_layout_to_json, export_layout_to_mesh, dict_to_floor_plan, generate_unique_id, export_layout_to_mesh_dict_list
from objects.object_selection_planner import select_objects
from objects.object_placement_planner import place_objects
from isaacsim.isaac_mcp.server import get_isaac_connection, create_room_layout_scene, simulate_the_scene, validate_room_stability, create_physics_scene, get_room_layout_scene_usd, create_single_room_layout_scene, get_room_layout_scene_usd_separate_from_layout
import copy
import shutil
import glob
import numpy as np
from constants import SERVER_ROOT_DIR, PHYSICS_CRITIC_ENABLED, SEMANTIC_CRITIC_ENABLED
from utils import extract_json_from_response
from timing import set_operation_counts, timed_stage, timed_tool
from floor_plan_materials.material_generator import material_generate_from_prompt
from concurrent.futures import ThreadPoolExecutor
from run_artifacts import persist_raw_vlm_response
current_layout: Optional[FloorPlan] = None
room_num_calls: Dict = {}
policy_analysis: Dict = {}
occupancy_ratio: float = 45

def _feature_enabled(name: str, default: bool=True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {'0', 'false', 'no', 'off'}
mcp = FastMCP(f"layout_{os.environ.get('SLURM_JOB_ID')}")

def get_room_num_calls_description(room_num_calls: Dict) -> str:
    """
    Get room calls description
    """
    global current_layout
    max_room_num_calls = 4
    if not room_num_calls:
        return 'No room decoration calls recorded yet.'
    description_parts = []
    description_parts.append('Room Decoration Call Summary:')
    room_types = {}
    if current_layout and current_layout.rooms:
        room_types = {room.id: room.room_type for room in current_layout.rooms}
    sorted_rooms = sorted(room_num_calls.items(), key=lambda x: x[1])
    for (room_id, call_count) in sorted_rooms:
        room_type = room_types.get(room_id, 'Unknown')
        description_parts.append(f'  • {room_type} ({room_id}): {call_count} iterations')
    under_decorated_rooms = [(room_id, call_count, room_types.get(room_id, 'Unknown')) for (room_id, call_count) in sorted_rooms if call_count <= max_room_num_calls - 1]
    if under_decorated_rooms:
        description_parts.append('\nSuggested Actions for Under-Decorated Rooms:')
        for (room_id, call_count, room_type) in under_decorated_rooms:
            if call_count == 0:
                description_parts.append(f"  • {room_type} ({room_id}): No furniture yet - use tools with room_id '{room_id}' to add essential furniture")
            elif call_count <= max_room_num_calls - 1:
                description_parts.append(f"  • {room_type} ({room_id}): Basic furniture only -  use tools with room_id '{room_id}' more for enhancements")
    else:
        description_parts.append('\nAll rooms have been decorated with multiple iterations.')
    return '\n'.join(description_parts)

def get_descendants_description(target_object: Object, all_objects: List[Object]) -> str:
    """
    Find all descendants of target_object in all_objects and return a description.
    Descendants are objects that are placed on target_object directly or indirectly.
    
    Returns a string with the count and object IDs of descendants.
    """
    descendants = []
    for obj in all_objects:
        if obj.id == target_object.id:
            continue
        current_obj = obj
        while True:
            if current_obj.place_id == 'floor' or current_obj.place_id == 'wall':
                break
            if current_obj.place_id == target_object.id:
                descendants.append(obj.id)
                break
            parent_obj = next((o for o in all_objects if o.id == current_obj.place_id), None)
            if parent_obj is None:
                break
            current_obj = parent_obj
    if not descendants:
        return 'No objects on top of this object.'
    count = len(descendants)
    ids_str = ', '.join(descendants)
    if count == 0:
        return 'No objects on top of this object.'
    return f"Object on top: Totally {count} object{('s' if count > 1 else '')} ({ids_str})"

def annotate_top_down_view_with_objects(rgb_image, room):
    """
    Annotate the top-down view image with bounding boxes and arrows for objects.
    
    Args:
        rgb_image: numpy array of the rendered top-down view (uint8, shape HxWx3)
        room: Room object containing the objects to annotate
    
    Returns:
        Annotated numpy array image
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    img = PILImage.fromarray(rgb_image)
    draw = ImageDraw.Draw(img)
    (img_height, img_width) = rgb_image.shape[:2]
    room_width_m = room.dimensions.width
    room_length_m = room.dimensions.length
    scale_x = img_width / room_width_m
    scale_y = img_height / room_length_m
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 16)
        small_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()
    axis_length = 50
    axis_offset = 5
    draw.line([(axis_offset, img_height - axis_offset), (axis_offset + axis_length, img_height - axis_offset)], fill='red', width=3)
    draw.text((axis_offset + axis_length + 5, img_height - axis_offset - 8), 'X', fill='red', font=font)
    draw.line([(axis_offset, img_height - axis_offset), (axis_offset, img_height - axis_offset - axis_length)], fill='green', width=3)
    draw.text((axis_offset + 5, img_height - axis_offset - axis_length - 5), 'Y', fill='green', font=font)
    draw.text((axis_offset + 10, img_height - axis_offset - 20), '[0,0]', fill='white', font=small_font)
    for obj in room.objects:
        obj_x = obj.position.x - room.position.x
        obj_y = obj.position.y - room.position.y
        pixel_x = obj_x * scale_x
        pixel_y = img_height - obj_y * scale_y
        obj_width = obj.dimensions.width
        obj_length = obj.dimensions.length
        rotation = obj.rotation.z
        if rotation == 0 or rotation == 180:
            bbox_width_px = obj_width * scale_x
            bbox_length_px = obj_length * scale_y
        elif rotation == 90 or rotation == 270:
            bbox_width_px = obj_length * scale_x
            bbox_length_px = obj_width * scale_y
        else:
            bbox_width_px = obj_width * scale_x
            bbox_length_px = obj_length * scale_y
        x1 = pixel_x - bbox_width_px / 2
        y1 = pixel_y - bbox_length_px / 2
        x2 = pixel_x + bbox_width_px / 2
        y2 = pixel_y + bbox_length_px / 2
        label = obj.type
        if obj.place_id == 'floor' or obj.place_id == 'wall':
            draw.rectangle([x1, y1, x2, y2], outline='blue', width=5)
        arrow_length = min(45, bbox_width_px * 1.8, bbox_length_px * 1.8)
        arrow_width = 8
        head_length = 20
        head_width = 10
        if rotation == 0:
            arrow_end_x = pixel_x
            arrow_end_y = pixel_y - (arrow_length - head_length)
            arrow_end_x_stick = pixel_x
            arrow_end_y_stick = pixel_y - arrow_length
        elif abs(rotation - 90) < 1:
            arrow_end_x = pixel_x - (arrow_length - head_length)
            arrow_end_y = pixel_y
            arrow_end_x_stick = pixel_x - arrow_length
            arrow_end_y_stick = pixel_y
        elif abs(rotation - 180) < 1:
            arrow_end_x = pixel_x
            arrow_end_y = pixel_y + (arrow_length - head_length)
            arrow_end_x_stick = pixel_x
            arrow_end_y_stick = pixel_y + arrow_length
        elif abs(rotation - 270) < 1:
            arrow_end_x = pixel_x + (arrow_length - head_length)
            arrow_end_y = pixel_y
            arrow_end_x_stick = pixel_x + arrow_length
            arrow_end_y_stick = pixel_y
        else:
            rotation_rad = np.radians(rotation)
            arrow_end_x = pixel_x + (arrow_length - head_length) * np.sin(rotation_rad)
            arrow_end_y = pixel_y - (arrow_length - head_length) * np.cos(rotation_rad)
            arrow_end_x_stick = pixel_x + arrow_length * np.sin(rotation_rad)
            arrow_end_y_stick = pixel_y - arrow_length * np.cos(rotation_rad)
        if obj.place_id == 'floor':
            draw.line([(pixel_x, pixel_y), (arrow_end_x, arrow_end_y)], fill='red', width=arrow_width)
            dx = arrow_end_x_stick - pixel_x
            dy = arrow_end_y_stick - pixel_y
            length = np.sqrt(dx ** 2 + dy ** 2)
            if length > 0:
                dx /= length
                dy /= length
                base_x = arrow_end_x_stick - dx * head_length
                base_y = arrow_end_y_stick - dy * head_length
                perp_x = -dy * head_width
                perp_y = dx * head_width
                p1 = (arrow_end_x_stick, arrow_end_y_stick)
                p2 = (base_x + perp_x, base_y + perp_y)
                p3 = (base_x - perp_x, base_y - perp_y)
                draw.polygon([p1, p2, p3], fill='red', outline='black')
        if obj.place_id == 'floor' or obj.place_id == 'wall':
            text_bbox = draw.textbbox((pixel_x, pixel_y), label, font=small_font, anchor='mb')
            padding = 2
            text_bg_bbox = [text_bbox[0] - padding, text_bbox[1] - padding, text_bbox[2] + padding, text_bbox[3] + padding]
            draw.rectangle(text_bg_bbox, fill='blue')
            draw.text((pixel_x, pixel_y), label, fill='white', font=small_font, anchor='mb')
    grid_color = (200, 200, 200)
    label_color = (255, 255, 255)
    bg_color = (0, 0, 0)
    for x_m in range(0, int(room_width_m) + 1):
        pixel_x = x_m * scale_x
        draw.line([(pixel_x, 0), (pixel_x, img_height)], fill=grid_color, width=2)
    for y_m in range(0, int(room_length_m) + 1):
        pixel_y = img_height - y_m * scale_y
        draw.line([(0, pixel_y), (img_width, pixel_y)], fill=grid_color, width=2)
    for x_m in range(0, int(room_width_m) + 1):
        for y_m in range(0, int(room_length_m) + 1):
            if x_m == 0 and y_m == 0:
                continue
            pixel_x = x_m * scale_x + 10
            pixel_y = max(10, img_height - y_m * scale_y - 10)
            text = f'({x_m},{y_m})'
            bbox = draw.textbbox((pixel_x, pixel_y), text, font=small_font, anchor='mm')
            draw.rectangle(bbox, fill=bg_color)
            draw.text((pixel_x, pixel_y), text, fill=label_color, font=small_font, anchor='mm')
    return np.array(img)

def get_object_description_list(objects: List[Object], add_object_descendants=False) -> List[str]:
    """
    Get object description list
    """
    return [f"{obj.type} (ID: {obj.id}; Description: {obj.description}); Placed on {obj.place_id};             {(f'{get_descendants_description(obj, objects)}' if add_object_descendants and (obj.place_id == 'floor' or obj.place_id == 'wall') else '')};\n" for obj in objects]

def dict2str(d, indent=0):
    """
    Convert a dictionary into a formatted string.

    Parameters:
    - d: dict, the dictionary to convert.
    - indent: int, the current indentation level (used for nested structures).

    Returns:
    - str: The string representation of the dictionary.
    """
    if not isinstance(d, dict):
        raise ValueError('Input must be a dictionary')
    result = []
    indent_str = ' ' * (indent * 4)
    for (key, value) in d.items():
        if isinstance(value, dict):
            result.append(f'{indent_str}{key}: {{\n{dict2str(value, indent + 1)}\n{indent_str}}}')
        elif isinstance(value, list):
            list_str = ', '.join((dict2str(item, indent + 1) if isinstance(item, dict) else f'{item:.2f}' if isinstance(item, float) else str(item) for item in value))
            result.append(f'{indent_str}{key}: [{list_str}]')
        else:
            result.append(f'{indent_str}{key}: {repr(value)}')
    return '{' + ',\n'.join(result) + '}'

def get_object_description_list_with_relation(objects):
    """
    Get object description list
    """
    object_dict = {}

    def map_constraint_name_to_description(constraint_name):
        if constraint_name == 'edge':
            return 'against wall'
        if constraint_name == 'middle':
            return 'placed in'
        else:
            return constraint_name
    for obj in objects:
        other_relationships = [[map_constraint_name_to_description(constraint['constraint']), constraint.get('target', obj.id[:len('room_xxxxxxxx')])] for constraint in obj.placement_constraints[-1]] if obj.placement_constraints else []
        object_dict[obj.id] = {'location': [f'{obj.position.x:.2f}', f'{obj.position.y:.2f}', f'{obj.position.z:.2f}'], 'rotation': [f'{np.radians(obj.rotation.x):.2f}', f'{np.radians(obj.rotation.y):.2f}', f'{np.radians(obj.rotation.z + 90) % 360:.2f}'], 'size': [f'{obj.dimensions.width:.2f}', f'{obj.dimensions.length:.2f}', f'{obj.dimensions.height:.2f}'], 'relation': [['on top of', obj.place_id]] + other_relationships}
    return object_dict

def get_object_description_list_detailed(objects, add_object_descendants=False):
    """
    Get object description list
    """
    return [f"Object {obj_i + 1}: {obj.id[len('room_xxxxxxxx_'):]}; ({obj.description}); placed on {obj.place_id};             {(f'{get_descendants_description(obj, objects)}' if add_object_descendants and (obj.place_id == 'floor' or obj.place_id == 'wall') else '')};\n" for (obj_i, obj) in enumerate(objects)]

def get_room_description(room):
    """
    Get room description
    """
    objects_description = dict2str(get_object_description_list_with_relation(room.objects))
    room_description = f'Room Type: {room.room_type}\nRoom Dimensions: {room.dimensions.width}m x {room.dimensions.length}m x {room.dimensions.height}m\nObjects: {objects_description}'
    return room_description

def get_failed_placements_description(failed_to_be_placed_objects: List[Object]) -> str:
    """
    Get failed placements description
    """
    return [f"{obj.type} (ID: {obj.id}; Description: {obj.description}) failed to be placed on {obj.place_id}; Try to change the placement location and place it on a different location on {obj.place_id} if possible; If it's not the first time of failure, no retry of placement is needed." for obj in failed_to_be_placed_objects]

def get_child_objects_removed_reminder(child_objects_removed: List[Dict[str, str]]) -> str:
    """
    Get reminder message for child objects that were removed during movement.
    These objects may need to be placed back if necessary.
    
    Args:
        child_objects_removed: List of dicts with 'id' and 'type' keys
        
    Returns:
        A string reminder message about removed objects
    """
    if not child_objects_removed:
        return 'No child objects were removed during this movement operation.'
    object_descriptions = [f"{obj['type']} (ID: {obj['id']})" for obj in child_objects_removed]
    if len(object_descriptions) == 1:
        return f'⚠️ Child object removed: {object_descriptions[0]}. This object was placed on the moved object and has been removed. Consider placing it back in the room if needed.'
    else:
        objects_list = ', '.join(object_descriptions)
        return f'⚠️ Child objects removed: {objects_list}. These objects were placed on the moved object and have been removed. Consider placing them back in the room if needed.'

async def select_materials_for_rooms(floor_plan: FloorPlan) -> FloorPlan:
    """
    Select appropriate materials for floors and walls in all rooms of the floor plan.
    
    Args:
        floor_plan: The floor plan with rooms to select materials for
        
    Returns:
        Updated floor plan with materials assigned to rooms and walls
    """
    try:
        material_descriptions = await get_material_descriptions_from_claude(floor_plan)
        layout_id = current_layout.id
        material_save_dir = os.path.join(RESULTS_DIR, layout_id, 'materials')
        os.makedirs(material_save_dir, exist_ok=True)
        for room in floor_plan.rooms:
            room_id = room.id
            room_type = room.room_type
            if room_id in material_descriptions:
                floor_data = material_descriptions[room_id]['floor']
                wall_data = material_descriptions[room_id]['wall']
                floor_description = floor_data.get('description', floor_data) if isinstance(floor_data, dict) else floor_data
                wall_description = wall_data.get('description', wall_data) if isinstance(wall_data, dict) else wall_data
                from PIL import Image as PILImage

                def repeat_texture(texture_map_pil, times):
                    for _ in range(times):
                        texture_map_pil_np = np.array(texture_map_pil).astype(np.float32) / 255.0
                        texture_map_pil_np = texture_map_pil_np[int(0.1 * texture_map_pil_np.shape[0]):int(0.9 * texture_map_pil_np.shape[0]), int(0.1 * texture_map_pil_np.shape[1]):int(0.9 * texture_map_pil_np.shape[1])]
                        texture_map_pil_np = np.concatenate([texture_map_pil_np, texture_map_pil_np], axis=0)
                        texture_map_pil_np = np.concatenate([texture_map_pil_np, texture_map_pil_np], axis=1)
                        texture_map_pil = PILImage.fromarray(np.clip(texture_map_pil_np * 255, 0, 255).astype(np.uint8))
                    return texture_map_pil
                floor_texture_map_pil = material_generate_from_prompt([floor_description])[0]
                floor_texture_map_pil = repeat_texture(floor_texture_map_pil, 2)
                room.floor_material = room_id + '_floor'
                floor_material_save_path = os.path.join(material_save_dir, f'{room.floor_material}.png')
                floor_texture_map_pil.save(floor_material_save_path)
                wall_texture_map_pil = material_generate_from_prompt([wall_description])[0]
                wall_texture_map_pil = repeat_texture(wall_texture_map_pil, 2)
                wall_material = room_id + '_wall'
                wall_material_save_path = os.path.join(material_save_dir, f'{wall_material}.png')
                wall_texture_map_pil.save(wall_material_save_path)
                for wall in room.walls:
                    wall.material = wall_material
        return floor_plan
    except Exception as e:
        print(f'Warning: Material selection failed: {e}. Using default materials.', file=sys.stderr)
        for room in floor_plan.rooms:
            room.floor_material = 'hardwood'
            for wall in room.walls:
                wall.material = 'drywall'
        return floor_plan

async def get_material_descriptions_from_claude(floor_plan: FloorPlan) -> Dict[str, Dict[str, Dict[str, any]]]:
    """
    Use Claude API to get material descriptions for each room based on room type and context.
    
    Args:
        floor_plan: The floor plan containing rooms to get material descriptions for
        
    Returns:
        Dictionary mapping room_id to {
            'floor': {'description': str, 'color_palette': [[R, G, B], ...]},
            'wall': {'description': str, 'color_palette': [[R, G, B], ...]}
        }
    """
    try:
        room_info_list = []
        for room in floor_plan.rooms:
            room_info = {'id': room.id, 'type': room.room_type, 'area': room.dimensions.width * room.dimensions.length, 'dimensions': f'{room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m'}
            room_info_list.append(room_info)
        prompt = f"""You are an interior design expert creating material descriptions for AI texture generation.\n\nBUILDING: {floor_plan.building_style} | {floor_plan.description} | {floor_plan.total_area:.1f} sq m\n\nROOMS:\n{chr(10).join([f"- {room['type']} (ID: {room['id']}): {room['dimensions']}, {room['area']:.1f} sq m" for room in room_info_list])}\n\nTASK: Provide detailed visual descriptions for UV texture generation. Focus purely on visual appearance. Include:\n- Specific colors, patterns, textures, surface finish (matte/glossy/textured)\n- Material characteristics, grain patterns, surface details, lighting properties\n- Color palette with 3-5 representative RGB values from the texture\n- Consider room function and building style for appropriate material selection\n\nIMPORTANT: Do NOT mention "floor" or "wall" in descriptions. Focus only on the material and visual properties (e.g., "white ceramic tiles" not "white ceramic tile floor", "light cream surface" not "light cream painted wall").\n\nOUTPUT FORMAT:\n```json\n{{\n    "room_id": {{\n        "floor": {{\n            "description": "Pure visual description: texture type, color, pattern, surface finish, material characteristics",\n            "color_palette": [[240, 235, 220], [200, 185, 160], [120, 105, 90]]\n        }},\n        "wall": {{\n            "description": "Pure visual description: texture type, color, pattern, surface finish, material characteristics",\n            "color_palette": [[240, 235, 220], [200, 185, 160], [120, 105, 90]]\n        }}\n    }}\n}}\n```\n\nEXAMPLES:\n- Bathroom: \n```json\n{{\n    "floor": {{\n        "description": "White ceramic tiles with subtle gray veining, glossy reflective finish, clean grout lines creating regular grid pattern",\n        "color_palette": [[245, 245, 245], [220, 220, 220], [192, 192, 192], [180, 180, 180]]\n    }},\n    "wall": {{\n        "description": "Crisp white surface, semi-gloss finish with smooth texture and moisture-resistant appearance",\n        "color_palette": [[252, 252, 252], [248, 248, 248], [240, 240, 240]]\n    }}\n}}\n```\n- Kitchen: \n```json\n{{\n    "floor": {{\n        "description": "Light oak hardwood planks with natural grain patterns, semi-gloss finish, honey tones with darker grain lines running lengthwise",\n        "color_palette": [[218, 187, 134], [200, 168, 115], [168, 140, 95], [135, 110, 75], [95, 75, 50]]\n    }},\n    "wall": {{\n        "description": "Soft off-white surface, eggshell finish with smooth texture and gentle light reflection, subtle warmth",\n        "color_palette": [[250, 245, 235], [245, 240, 230], [240, 235, 225]]\n    }}\n}}\n```\n- Bedroom: \n```json\n{{\n    "floor": {{\n        "description": "Medium brown oak with rich visible grain, satin finish, chocolate brown base with darker wood grain accents creating depth",\n        "color_palette": [[139, 105, 70], [120, 90, 60], [100, 75, 50], [80, 60, 40]]\n    }},\n    "wall": {{\n        "description": "Warm ivory surface, flat matte finish with fine texture, calming neutral tone with subtle undertones",\n        "color_palette": [[245, 240, 220], [240, 235, 215], [235, 230, 210]]\n    }}\n}}\n```\n- Living Room: \n```json\n{{\n    "floor": {{\n        "description": "Dark walnut hardwood with prominent grain patterns, semi-gloss finish, deep brown with black grain accents creating dramatic contrast",\n        "color_palette": [[75, 55, 40], [60, 45, 35], [50, 38, 30], [40, 30, 22], [30, 22, 15]]\n    }},\n    "wall": {{\n        "description": "Light cream surface, eggshell finish with smooth texture and subtle sheen, warm undertones throughout",\n        "color_palette": [[248, 238, 215], [245, 235, 210], [240, 230, 205]]\n    }}\n}}\n```\n\nFocus on visual details for realistic AI texture generation."""
        response = call_vlm(vlm_type='openai', max_tokens=8000, temperature=0.5, messages=[{'role': 'user', 'content': prompt}])
        response_text = response.content[0].text.strip()
        try:
            response_text = extract_json_from_response(response_text)
            if not response_text:
                raise ValueError('Could not extract JSON content from Claude response')
            material_descriptions = json.loads(response_text)
            validated_descriptions = {}
            for room in floor_plan.rooms:
                room_id = room.id
                if room_id in material_descriptions:
                    room_materials = material_descriptions[room_id]
                    if isinstance(room_materials, dict) and 'floor' in room_materials and ('wall' in room_materials):
                        floor_data = room_materials['floor']
                        wall_data = room_materials['wall']
                        if isinstance(floor_data, dict) and isinstance(wall_data, dict):
                            validated_descriptions[room_id] = {'floor': {'description': str(floor_data.get('description', '')), 'color_palette': floor_data.get('color_palette', [])}, 'wall': {'description': str(wall_data.get('description', '')), 'color_palette': wall_data.get('color_palette', [])}}
                        else:
                            validated_descriptions[room_id] = {'floor': {'description': str(floor_data), 'color_palette': []}, 'wall': {'description': str(wall_data), 'color_palette': []}}
                    else:
                        assert False, 'Room not found in material description response from Claude'
                else:
                    assert False, 'Room not found in material description response from Claude'
            return validated_descriptions
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse Claude's material response as JSON: {e}")
            return {}
    except anthropic.APIError as e:
        print(f'Warning: Claude API error during material selection: {e}')
        return {}
    except Exception as e:
        print(f'Warning: Error during material description generation: {e}')
        return {}

async def generate_room_structure(input_text: str) -> str:
    """
    Generate room structure only (walls, no doors/windows) from text description using Claude Sonnet 4.
    This is the first stage of the new staged layout generation process.
    
    Args:
        input_text: Natural language description of the desired room layout
        
    Returns:
        JSON string containing the generated room structure information
    """
    global current_layout
    try:
        if not input_text or not input_text.strip():
            return json.dumps({'success': False, 'error': 'Input text cannot be empty'})
        llm_response = await call_llm_for_rooms_with_validation(input_text, max_attempts=1)
        if not isinstance(llm_response, dict) or 'rooms' not in llm_response:
            return json.dumps({'success': False, 'error': 'Invalid response structure from Claude API'})
        if not llm_response['rooms']:
            return json.dumps({'success': False, 'error': 'No rooms generated in the layout'})
        try:
            current_layout = parse_rooms_only_to_floor_plan(llm_response, input_text)
        except ValueError as parse_error:
            return json.dumps({'success': False, 'error': f'Failed to generate room structure. Parse error: {str(parse_error)}'})
        validation_warning = llm_response.get('validation_warning')
        summary = {'success': True, 'stage': 'room_structure_only', 'layout_id': current_layout.id, 'num_rooms': len(current_layout.rooms), 'layout_type': 'single_room' if len(current_layout.rooms) == 1 else 'multi_room', 'total_area': current_layout.total_area, 'building_style': current_layout.building_style, 'description': current_layout.description, 'rooms': [{'id': room.id, 'type': room.room_type, 'area': room.dimensions.width * room.dimensions.length, 'dimensions': f'{room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m', 'position': f'({room.position.x:.1f}, {room.position.y:.1f}, {room.position.z:.1f})', 'doors': 0, 'windows': 0} for room in current_layout.rooms], 'api_info': {'model_used': 'claude-sonnet-4-20250514', 'generated_from': input_text}, 'next_steps': ["Run 'validate_current_layout()' to check for room overlaps and connectivity", "If valid, run 'add_doors_windows()' to complete the layout", "If issues found, run 'correct_layout_issues()' to fix room positioning"]}
        if len(current_layout.rooms) == 1:
            summary['message'] = f'🏗️ Room structure generated: {current_layout.rooms[0].room_type} (doors/windows will be added next)'
        else:
            summary['message'] = f'🏗️ Multi-room structure generated with {len(current_layout.rooms)} rooms (doors/windows will be added next)'
        if validation_warning:
            summary['validation_warning'] = validation_warning
        return json.dumps(summary, indent=2)
    except ValueError as e:
        if 'ANTHROPIC_API_KEY' in str(e):
            return json.dumps({'success': False, 'error': 'Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.'})
        else:
            return json.dumps({'success': False, 'error': f'Configuration error: {str(e)}'})
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Room structure generation failed: {str(e)}'})

async def add_doors_windows() -> str:
    """
    Add doors and windows to the current room structure using Claude Sonnet 4.
    This is the second stage of the staged layout generation process.
    
    Returns:
        JSON string containing the updated layout information with doors and windows
    """
    global current_layout
    if current_layout is None:
        return json.dumps({'error': "No room structure has been generated yet. Use 'generate_room_structure()' first."})
    try:
        has_doors_or_windows = any((room.doors or room.windows for room in current_layout.rooms))
        if has_doors_or_windows:
            return json.dumps({'success': False, 'error': "Current layout already has doors or windows. Use 'clear_layout()' and regenerate to modify door/window placement."})
        rooms_data = []
        for room in current_layout.rooms:
            room_data = {'room_type': room.room_type, 'dimensions': {'width': room.dimensions.width, 'length': room.dimensions.length, 'height': room.dimensions.height}, 'position': {'x': room.position.x, 'y': room.position.y, 'z': room.position.z}}
            rooms_data.append(room_data)
        print('🔍 Stage 4: Adding doors and windows with traffic flow analysis...', file=sys.stderr)
        doors_windows_response = await call_llm_for_doors_windows_graph_based(rooms_data, current_layout.created_from_text)
        debug_info = doors_windows_response.get('debug_info', {})
        if not isinstance(doors_windows_response, dict) or 'rooms' not in doors_windows_response:
            return json.dumps({'success': False, 'error': 'Invalid doors/windows response structure from Claude API', 'debug_info': debug_info})
        print('🔍 Stage 4: Adding doors and windows to the current floor plan...', file=sys.stderr)
        try:
            current_layout = add_doors_windows_to_floor_plan(current_layout, doors_windows_response)
        except ValueError as e:
            return json.dumps({'success': False, 'error': f'Failed to add doors and windows: {str(e)}', 'debug_info': debug_info})
        summary = {'success': True, 'stage': 'complete_layout', 'layout_id': current_layout.id, 'num_rooms': len(current_layout.rooms), 'layout_type': 'single_room' if len(current_layout.rooms) == 1 else 'multi_room', 'total_area': current_layout.total_area, 'building_style': current_layout.building_style, 'description': current_layout.description, 'rooms': [{'id': room.id, 'type': room.room_type, 'area': room.dimensions.width * room.dimensions.length, 'dimensions': f'{room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m', 'position': f'({room.position.x:.1f}, {room.position.y:.1f}, {room.position.z:.1f})', 'doors': len(room.doors), 'windows': len(room.windows)} for room in current_layout.rooms], 'door_window_summary': {'total_doors': sum((len(room.doors) for room in current_layout.rooms)), 'total_windows': sum((len(room.windows) for room in current_layout.rooms)), 'rooms_with_doors': sum((1 for room in current_layout.rooms if room.doors)), 'rooms_with_windows': sum((1 for room in current_layout.rooms if room.windows))}, 'api_info': {'model_used': 'claude-sonnet-4-20250514', 'generated_from': current_layout.created_from_text}, 'debug_info': debug_info, 'next_steps': ["Run 'validate_current_layout()' to ensure doors/windows are properly placed", "Use 'visualize_current_layout()' to see the complete layout", "Use 'generate_room_layout()' to regenerate if any issues are found"]}
        door_count = sum((len(room.doors) for room in current_layout.rooms))
        window_count = sum((len(room.windows) for room in current_layout.rooms))
        if len(current_layout.rooms) == 1:
            summary['message'] = f'✅ Complete layout: {current_layout.rooms[0].room_type} with {door_count} doors and {window_count} windows'
        else:
            summary['message'] = f'✅ Complete multi-room layout with {door_count} doors and {window_count} windows across {len(current_layout.rooms)} rooms'
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Adding doors and windows failed: {str(e)}', 'debug_info': debug_info if 'debug_info' in locals() else {}})

def generate_scene_requirements(input_text: str) -> str:
    """
    Generate task-aligned design suggestions using the configured local model.
    """
    global current_layout
    global policy_analysis
    print(f'Generating scene requirements for the layout input {input_text}...', file=sys.stderr)
    scene_requirements = ''
    ar_total_iterations = 1
    all_object_proposals = []
    room_size = ''
    room_type = ''
    style = ''
    overall_reasoning = []
    prompt = f"""You are an interior design expert. Based on the room description, generate comprehensive scene requirements.\n\nROOM DESCRIPTION: {input_text}\n\nTASK: Generate initial scene requirements following the room description and including:\n1. Room size description (e.g., large-sized, medium-sized, compact)\n2. Explicit decoration style (should be style similar to modern and bright)\n3. Object placement proposals listed by the importance from high to low consistent with the user's requested quantities, restrictions, room size and purpose, covering:\n   - Essential furniture (placed on floor)\n   - Background objects (placed on floor or wall)\n   - Decorative/functional objects (placed on top of furniture surfaces)\n   - Preserve requested support relations and explicitly empty surfaces.\n   - Scale the selection and spacing to the room size and activity. A larger room may accommodate more useful items, but size alone does not require extra objects or filling empty space.\n   - Object type proposals should not be odd or unrealistic or uncommon or weird. should be believable, and common daily objects make the room feel lived-in. Rich of daily furniture and objects.\n   - Object proposals should include the necessary furniture and setup for the specified function.\n   - Include required objects first. Unless the user explicitly restricts unrelated or decorative objects, suggest a moderate selection of decorative objects or small everyday items to improve beauty and diversity. Do not require filling every shelf or surface. Place small accessories on suitable tables, desks, shelves or cabinet tops; use wall-mounted decoration where appropriate, and put floor-standing decoration only in suitable free floor areas. Respect the specified supports and empty surfaces, keep working surfaces and circulation usable, and do not make the room look cluttered.\n\nObject proposals should add the realism, functionality, completeness and diversity to the room.\n\nOBJECT RESTRICTIONS:\nDO NOT propose: doors, windows (does not belong to the objects to be proposed)\nDo NOT propose: rugs, mats, carpets, curtains, blankets, ceiling-hanging objects, ceiling objects. (already installed)\n\nObject Quantity Rule:\nIf need to place multiple objects of the same type, the quantity should be the number of objects to place.\nThis is critical when you need to place multiple objects of the same type.\nE.g. A large-sized room typically needs multiple objects with the same type.\nE.g. You may also want to place multiple objects of the same type to add diversity to the room.\nFor example:\ni. if you want to place 12 chairs, the quantity should be 12.\nii. if you want to place 6 tables, the quantity should be 6.\n\nObject Type Rule:\n- Use a stable, specific lowercase category name. Use underscores for multiword categories, e.g. coffee_table, floor_lamp, pencil_holder. Preserve the requested identity instead of collapsing distinct objects to a generic table or lamp.\n\nObject Placement Rule:\nFormat: a string of "floor|wall|estimated_object_name".\n- "floor": Objects resting on ground (furniture against walls, in corners, beside other floor objects, or on open floor)\n- "wall": Objects physically mounted/attached to wall above floor level (paintings, shelves, wall-mounted TVs)\n- "estimated_object_name": Objects placed on top of another object's surface (lamp on table, vase on cabinet, book on shelf)\n"estimated_object_name" means the object type/name that will support the object to be placed. (e.g., "lamp on table" → placement: "table")\n\nREQUIRED JSON RESPONSE:\n```json\n{{\n    "holistic_reasoning": "Brief reasoning for the scene requirements and design choices",\n    "room_type": "room type (e.g., living room, bedroom, office, kitchen, dining room, etc.)",\n    "room_size": "size description, expressed in meters (e.g., 3.6m x 4.0m), the size should be reasonable and believable for placing the following objects, ensuring the room won't be too sparse or too crowded.",\n    "style": "explicit decoration style (e.g., modern and bright; for common household room, the style should be modern and bright always)",\n    "object_proposals_reasoning": {{\n        "object_types_reasoning": "Reasoning for the selected object types and their relevance to the room function and style; formatted as a list of object types and their relevance to the room function and style",\n        "object_quantity_reasoning": "Analysis of object quantities considering room size, possible object sizes, and balance between sparsity and cluttering. Evaluate whether the proposed quantities would result in an overly sparse or overly cluttered room; formatted as firstly recall the room size you just estimated, and then analyze the possible object sizes (give actual sizes of every proposed object type in meters) and the number of objects to place, and then analyze the balance between sparsity and cluttering, and then propose a list of object quantities and their relevance to the room size and possible object sizes",\n        "object_importance_reasoning": "Ranking of objects from high to low importance. This ranking will be used as the sequence for the following object proposals. Explain the priority order based on functional necessity and design impact; formatted as a list of object types and their importance ranking"\n    }},\n    "object_proposals": [\n        {{\n            "object_type": "specific lowercase category, using underscores for multiple words",\n            "object_short_description": "A wooden office chair",\n            "object_placement": "floor|wall|estimated_object_name",\n            "object_quantity": 1\n        }}\n    ]\n}}\n```\n\n[IMPORTANT FOR THE RESPONSE]\n- object_proposals should be sorted by the sequence in object_importance_reasoning, from high to low importance.\n\nThere is no minimum number of proposals. Explicit user counts and limits on extras take precedence. Additional proposals are optional suggestions, not new user requirements. Return valid JSON only: numeric quantities, no comments or trailing commas."""
    try:
        response = call_vlm(vlm_type='openai', max_tokens=20000, temperature=0.3, messages=[{'role': 'user', 'content': prompt}])
        response_text = response.content[0].text.strip()
        try:
            artifact_path = persist_raw_vlm_response('scene_requirements', response_text, metadata={'input_text': input_text})
            print(f'Saved raw scene-requirements response to {artifact_path}', file=sys.stderr)
        except Exception as artifact_error:
            print(f'Warning: could not persist raw scene-requirements response: {artifact_error}', file=sys.stderr)
        response_text = extract_json_from_response(response_text)
        if not response_text:
            print(f'⚠️ Warning: Could not extract JSON from response', file=sys.stderr)
            assert False, 'Could not extract JSON from response'
        result = json.loads(response_text)
        if 'holistic_reasoning' in result:
            overall_reasoning.append(result['holistic_reasoning'])
        room_size = result.get('room_size', 'medium-sized')
        room_type = result.get('room_type', 'room')
        style = result.get('style', 'modern')
        proposals = result.get('object_proposals', [])
        all_object_proposals.extend(proposals)
        print(f'{len(proposals)} object proposals generated', file=sys.stderr)
    except Exception as e:
        print(f'⚠️ Warning: Failed to generate scene requirements: {e}', file=sys.stderr)
    if not room_type:
        room_type = current_layout.rooms[0].room_type if current_layout and current_layout.rooms else 'room'
    if not room_size:
        room_size = 'medium-sized'
    if not style:
        style = 'modern'
    if not all_object_proposals:
        scene_requirements = f'The scene requires a {room_size} {room_type} with basic furniture and decorations in {style} style.'
    else:
        floor_objects = [p for p in all_object_proposals if p.get('object_placement') == 'floor']
        wall_objects = [p for p in all_object_proposals if p.get('object_placement') == 'wall']
        ontop_objects = [p for p in all_object_proposals if p.get('object_placement') not in ['floor', 'wall']]
        scene_requirements = f'The scene requires a {room_size} {room_type} designed in {style} style. '
        if floor_objects:
            scene_requirements += f'The room should include {len(floor_objects)} floor-placed furniture and background objects: '
            for (idx, obj) in enumerate(floor_objects):
                obj_type = obj.get('object_type', 'object')
                obj_short_description = obj.get('object_short_description', 'object')
                obj_qty = obj.get('object_quantity', 1)
                if obj_qty > 1:
                    obj_desc = f'{obj_qty} {obj_short_description}s'
                else:
                    obj_desc = f'{obj_short_description}'
                scene_requirements += f'{obj_desc} placed on floor'
                if idx < len(floor_objects) - 1:
                    scene_requirements += '; '
                else:
                    scene_requirements += '. '
        if wall_objects:
            scene_requirements += f'Wall decorations consist of {len(wall_objects)} items: '
            for (idx, obj) in enumerate(wall_objects):
                obj_type = obj.get('object_type', 'object')
                obj_short_description = obj.get('object_short_description', 'object')
                obj_qty = obj.get('object_quantity', 1)
                if obj_qty > 1:
                    obj_desc = f'{obj_qty} {obj_short_description}s'
                else:
                    obj_desc = f'{obj_short_description}'
                scene_requirements += f'{obj_desc} mounted on wall'
                if idx < len(wall_objects) - 1:
                    scene_requirements += '; '
                else:
                    scene_requirements += '. '
        if ontop_objects:
            scene_requirements += f'Furniture surfaces should be decorated with {len(ontop_objects)} functional and decorative objects: '
            for (idx, obj) in enumerate(ontop_objects):
                obj_type = obj.get('object_type', 'object')
                obj_short_description = obj.get('object_short_description', 'object')
                obj_qty = obj.get('object_quantity', 1)
                obj_placement = obj.get('object_placement', 'surface')
                if obj_qty > 1:
                    obj_desc = f'{obj_qty} {obj_short_description}s'
                else:
                    obj_desc = f'{obj_short_description}'
                scene_requirements += f'{obj_desc} placed on {obj_placement}'
                if idx < len(ontop_objects) - 1:
                    scene_requirements += '; '
                else:
                    scene_requirements += '. '
        scene_requirements += 'These are design suggestions, not additional mandatory objects. Explicit user counts, support relations, empty surfaces and limits on extras take precedence. Unless the user explicitly restricts unrelated or decorative objects, add suitable accessories and varied decoration to improve the room, without imposing surface-filling quotas, duplicating objects unnecessarily or blocking circulation. Place small accessories on suitable tables, desks, shelves or cabinet tops; use wall-mounted decoration where appropriate, and put floor-standing decoration only in suitable free floor areas. Respect the specified supports and empty surfaces, keep working surfaces and circulation usable, and do not make the room look cluttered. Keep object sizes, support, spacing and orientation physically reasonable.'
    print(f'✅ Scene requirements generated: {len(scene_requirements)} characters, {len(all_object_proposals)} object proposals', file=sys.stderr)
    return scene_requirements

@mcp.tool()
@timed_tool()
async def generate_room_layout(input_text: str) -> str:
    """

    Start generating room layout with this tool! 
    It can generate a room floor plan with walls, doors, and windows.
    After that, you can start placing objects.
    Generate a complete room layout from text description.
    This combines room structure generation, validation, correction, door/window placement with integrity checks.
    
    Args:
        input_text: Natural language description of the desired room scene, this should be as detailed as possible, 
        including the room size, room type, room style, and all the objects with detailed description to be placed in the room, etc.
        
    Returns:
        JSON string containing the generated layout information and the scene object proposals
    """
    global current_layout
    try:
        with timed_stage('layout.scene_requirements', 'llm'):
            scene_requirements = generate_scene_requirements(input_text)
        input_text = f"{input_text}; {scene_requirements[:scene_requirements.find('style.') + len('style.')]}"
        processing_text = input_text
        with timed_stage('layout.room_structure', 'llm'):
            room_structure_result = await generate_room_structure(processing_text)
        room_structure_data = json.loads(room_structure_result)
        if not room_structure_data.get('success'):
            return room_structure_result
        print('🏗️ Stage 3.5: Selecting materials for floors and walls...', file=sys.stderr)
        try:
            with timed_stage('layout.material_selection', 'generation'):
                current_layout = await select_materials_for_rooms(current_layout)
            print(f'✅ Materials selected for {len(current_layout.rooms)} rooms', file=sys.stderr)
        except Exception as e:
            print(f'⚠️ Material selection failed: {e}. Using default materials.', file=sys.stderr)
        doors_windows_removed = []
        for room in current_layout.rooms:
            if room.doors or room.windows:
                doors_windows_removed.append({'room': room.room_type, 'doors_removed': len(room.doors), 'windows_removed': len(room.windows)})
                room.doors = []
                room.windows = []
        current_layout_copy = copy.deepcopy(current_layout)
        total_attempts = 0
        while True:
            current_layout = copy.deepcopy(current_layout_copy)
            print('🏗️ Stage 4: Adding doors and windows...', file=sys.stderr)
            with timed_stage('layout.doors_windows', 'llm', details={'attempt': total_attempts + 1}):
                doors_windows_result = await add_doors_windows()
            doors_windows_data = json.loads(doors_windows_result)
            doors_windows_debug_info = doors_windows_data.get('debug_info', {})
            if not doors_windows_data.get('success'):
                return doors_windows_result
            print('🔍 Post-Stage 4 Check: Verifying door/window integrity...', file=sys.stderr)
            rooms_data_for_check = []
            for room in current_layout.rooms:
                room_data = {'room_type': room.room_type, 'position': {'x': room.position.x, 'y': room.position.y, 'z': room.position.z}, 'dimensions': {'width': room.dimensions.width, 'length': room.dimensions.length, 'height': room.dimensions.height}, 'doors': [], 'windows': []}
                for door in room.doors:
                    door_data = {'width': door.width, 'height': door.height, 'position_on_wall': door.position_on_wall, 'wall_side': extract_wall_side_from_id(door.wall_id), 'door_type': door.door_type}
                    room_data['doors'].append(door_data)
                for window in room.windows:
                    window_data = {'width': window.width, 'height': window.height, 'position_on_wall': window.position_on_wall, 'wall_side': extract_wall_side_from_id(window.wall_id), 'sill_height': window.sill_height, 'window_type': window.window_type}
                    room_data['windows'].append(window_data)
                rooms_data_for_check.append(room_data)
            print('🔍 Post-Stage 4 Check: Checking door/window integrity...', file=sys.stderr)
            with timed_stage('layout.doors_windows_validation', 'validation', details={'attempt': total_attempts + 1}):
                door_window_integrity = check_door_window_integrity(rooms_data_for_check)
            if door_window_integrity['valid']:
                break
            print(f"Door/window integrity issues found: {door_window_integrity['total_issues']} issues. Attempting regeneration...", file=sys.stderr)
            print('Debug info ofdoor_window_integrity: ', door_window_integrity, file=sys.stderr)
            total_attempts += 1
            if total_attempts > 3:
                return json.dumps({'success': False, 'error': 'Failed to add doors and windows after 3 attempts', 'suggestions': "Please try to regenerate the layout with a different description using 'clear_layout()' and 'generate_room_layout()'"})
        output_path = os.path.join(RESULTS_DIR, current_layout.id)
        os.makedirs(output_path, exist_ok=True)
        with timed_stage('layout.save', 'io'):
            export_layout_to_json(current_layout, os.path.join(output_path, f'{current_layout.id}.json'))
        for room in current_layout.rooms:
            room_num_calls[room.id] = 0
        policy_analysis['scene_requirements'] = scene_requirements
        current_layout.policy_analysis = policy_analysis
        summary = {'success': True, 'layout_id': current_layout.id, 'num_rooms': len(current_layout.rooms), 'layout_type': 'single_room' if len(current_layout.rooms) == 1 else 'multi_room', 'total_area': current_layout.total_area, 'rooms': [{'id': room.id, 'type': room.room_type, 'area': room.dimensions.width * room.dimensions.length, 'dimensions': f'{room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m'} for room in current_layout.rooms], 'scene_recommendations': policy_analysis['scene_requirements']}
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Enhanced staged layout generation failed: {str(e)}'})

@mcp.tool()
@timed_tool()
async def get_current_layout() -> str:
    """
    Get the complete current room layout data structure.
    
    Returns:
        JSON string containing the full layout data
    """
    global current_layout
    if current_layout is None:
        return json.dumps({'error': 'No layout has been generated yet'})
    layout_dict = asdict(current_layout)
    return json.dumps(layout_dict, indent=2)

async def clear_layout() -> str:
    """
    Clear the current room layout.
    
    Returns:
        Confirmation message
    """
    global current_layout
    current_layout = None
    return json.dumps({'success': True, 'message': 'Layout cleared'})

@mcp.tool()
@timed_tool()
async def get_room_details(room_id: str) -> str:
    """
    Get detailed information about a specific room.
    
    Args:
        room_id: The ID of the room to get details for
        
    Returns:
        JSON string containing room details
    """
    global current_layout
    if current_layout is None:
        return json.dumps({'error': 'No layout has been generated yet'})
    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        return json.dumps({'error': f'Room with ID {room_id} not found'})
    room_dict = {'id': room.id, 'type': room.room_type, 'dimensions': asdict(room.dimensions), 'position': asdict(room.position), 'area': room.dimensions.width * room.dimensions.length, 'num_doors': len(room.doors), 'num_windows': len(room.windows), 'objects': get_object_description_list(room.objects)}
    return json.dumps(room_dict, indent=2)

@mcp.tool()
@timed_tool()
async def list_rooms() -> str:
    """
    List all rooms in the current layout.
    
    Returns:
        JSON string containing room summary list
    """
    global current_layout
    if current_layout is None:
        return json.dumps({'error': 'No layout has been generated yet'})
    rooms_list = []
    for room in current_layout.rooms:
        room_summary = {'id': room.id, 'type': room.room_type, 'dimensions': asdict(room.dimensions), 'position': asdict(room.position), 'area': room.dimensions.width * room.dimensions.length, 'num_doors': len(room.doors), 'num_windows': len(room.windows)}
        rooms_list.append(room_summary)
    return json.dumps({'rooms': rooms_list}, indent=2)

async def visualize_current_layout(output_directory: str=RESULTS_DIR):
    """
    Generate visualizations of the current layout.
    
    Args:
        output_directory: Directory to save visualization files
        
    Returns:
        JSON string with information about generated files and the 2d visualization image
    """
    global current_layout
    output_directory = RESULTS_DIR
    if current_layout is None:
        return json.dumps({'error': 'No layout has been generated yet'})
    try:
        base_name = f'{current_layout.id}'
        output_path = Path(output_directory) / base_name
        os.makedirs(str(output_path), exist_ok=True)
        try:
            from visualizer import LayoutVisualizer
        except ImportError:
            return json.dumps({'error': 'Visualization module not available. Make sure matplotlib and numpy are installed.'})
        visualizer = LayoutVisualizer(current_layout)
        generated_files = []
        report_path = output_path / f'{base_name}_summary.txt'
        with open(report_path, 'w') as f:
            f.write(visualizer.generate_summary_report())
        generated_files.append(str(report_path))
        export_layout_to_json(current_layout, output_path / f'{base_name}.json')
        export_layout_to_mesh(current_layout, output_path / f'{base_name}.ply')
        return json.dumps({'success': True, 'message': 'Visualization successful'})
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Visualization failed: {str(e)}'})

@mcp.tool()
@timed_tool()
async def get_layout_from_json(ctx: Context, json_file_path: str='') -> str:
    """
    Load a room layout from JSON data and set it as the current layout.
    
    Args:
        json_file_path: Path to a JSON file containing layout data (optional)
        
    Note: json_file_path must be provided.
        
    Returns:
        JSON string containing information about the loaded layout
    """
    global current_layout
    print(f'🔧 Starting get_layout_from_json for room: {json_file_path}', file=sys.stderr)
    try:
        if not json_file_path:
            return json.dumps({'success': False, 'error': 'json_file_path must be provided'})
        if not json_file_path.endswith('.json'):
            layout_id = os.path.basename(json_file_path)
            json_file_path = os.path.join(json_file_path, f'{layout_id}.json')
        try:
            with open(json_file_path, 'r') as f:
                layout_data = json.load(f)
        except FileNotFoundError:
            return json.dumps({'success': False, 'error': f'JSON file not found: {json_file_path}'})
        except json.JSONDecodeError as e:
            return json.dumps({'success': False, 'error': f'Invalid JSON format in file: {str(e)}'})
        except Exception as e:
            return json.dumps({'success': False, 'error': f'Error reading file: {str(e)}'})
        required_fields = ['id', 'rooms', 'total_area', 'building_style', 'description', 'created_from_text']
        missing_fields = [field for field in required_fields if field not in layout_data]
        if missing_fields:
            return json.dumps({'success': False, 'error': f'Missing required fields in JSON data: {missing_fields}'})
        if not isinstance(layout_data['rooms'], list) or len(layout_data['rooms']) == 0:
            return json.dumps({'success': False, 'error': "JSON data must contain a non-empty 'rooms' list"})
        try:
            floor_plan = dict_to_floor_plan(layout_data)
            current_layout = floor_plan
            original_layout_id = current_layout.id
            current_layout.id = generate_unique_id('layout')
        except ValueError as e:
            return json.dumps({'success': False, 'error': f'Failed to convert JSON data to FloorPlan: {str(e)}'})
        except Exception as e:
            return json.dumps({'success': False, 'error': f'Unexpected error during conversion: {str(e)}'})
        json_file_dir = os.path.dirname(json_file_path)
        new_json_file_dir = json_file_dir.replace(original_layout_id, current_layout.id)
        os.makedirs(new_json_file_dir, exist_ok=True)
        for object_src in ['materials', 'generation']:
            object_src_dir = os.path.join(json_file_dir, object_src)
            if os.path.exists(object_src_dir):
                shutil.copytree(object_src_dir, os.path.join(new_json_file_dir, object_src))
        export_layout_to_json(current_layout, os.path.join(new_json_file_dir, f'{current_layout.id}.json'))
        for room in current_layout.rooms:
            room_num_calls[room.id] = 0
        await visualize_current_layout()
        summary = {'success': True, 'source': 'json_file', 'source_path': json_file_path if json_file_path else None, 'layout_id': current_layout.id, 'num_rooms': len(current_layout.rooms), 'layout_type': 'single_room' if len(current_layout.rooms) == 1 else 'multi_room', 'total_area': current_layout.total_area}
        if len(current_layout.rooms) == 1:
            summary['message'] = f'✅ Successfully loaded single room layout: {current_layout.rooms[0].room_type}'
        else:
            summary['message'] = f'✅ Successfully loaded multi-room layout with {len(current_layout.rooms)} rooms'
        summary['next_steps'] = ["Use 'validate_current_layout()' to check for any issues", "Use 'visualize_current_layout()' to generate visual representations", "Use 'list_rooms()' to see all rooms in the layout"]
        return json.dumps(summary, indent=2)
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Unexpected error loading layout from JSON: {str(e)}'})

async def analyze_placement_operation(room: Room, placement_conditions: str) -> Dict[str, Any]:
    """
    Analyze placement conditions to determine the type of operation (add/remove/replace)
    """
    existing_objects_info = []
    if room.objects:
        for obj in room.objects:
            existing_objects_info.append(f'- {obj.type} (ID: {obj.id})')
    existing_objects_text = '\n'.join(existing_objects_info) if existing_objects_info else '- No existing objects'
    analysis_prompt = f"""You are an interior design assistant analyzing placement instructions for a {room.room_type}.\n\nCURRENT ROOM STATE:\n- Room Type: {room.room_type}\n- Room Dimensions: {room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m\n- Existing Objects in Room:\n{existing_objects_text}\n\nPLACEMENT CONDITIONS:\n{placement_conditions.strip()}\n\nTASK:\nAnalyze the placement conditions to determine what type of operation is requested:\n\n1. "add" - Adding new objects while keeping existing ones (keywords: "add", "place additional", "also add", "include", "plus")\n2. "remove" - Removing specific objects (keywords: "remove", "delete", "take away", "get rid of")\n\nIf it's a "remove" operation, identify which specific objects should be removed based on the conditions.\n\nOUTPUT FORMAT:\nPlease respond with a JSON object in this exact format:\n\n```json\n{{\n    "operation_type": "add|remove",\n    "analysis": "Brief explanation of your analysis",\n    "objects_to_remove": ["object_id_1", "object_id_2"] // Only if operation_type is "remove"\n}}\n```\n\nExamples:\n- "add a desk and chair" → \n```json\n{{"operation_type": "add", "analysis": "Adding furniture while keeping existing objects"}}\n```\n- "remove the old sofa" → \n```json\n{{"operation_type": "remove", "analysis": "Removing specific furniture", "objects_to_remove": ["sofa_id"]}}\n```\n\nAnalyze the conditions now:"""
    try:
        response = call_vlm(vlm_type='openai', max_tokens=3000, temperature=0.1, messages=[{'role': 'user', 'content': analysis_prompt}])
        response_text = response.content[0].text.strip()
        try:
            response_text = extract_json_from_response(response_text)
            if not response_text:
                raise ValueError('Could not extract JSON content from Claude response')
            analysis_result = json.loads(response_text)
            if 'operation_type' not in analysis_result:
                analysis_result['operation_type'] = 'add'
            if 'analysis' not in analysis_result:
                analysis_result['analysis'] = 'Analysis completed'
            if 'objects_to_remove' not in analysis_result:
                analysis_result['objects_to_remove'] = []
            return analysis_result
        except json.JSONDecodeError:
            if any((keyword in placement_conditions.lower() for keyword in ['add', 'place additional', 'also add', 'include'])):
                return {'operation_type': 'add', 'analysis': 'Detected add operation from keywords', 'objects_to_remove': []}
            else:
                return {'operation_type': 'remove', 'analysis': 'Detected remove operation from keywords', 'objects_to_remove': []}
    except Exception as e:
        print(f'Error in analyze_placement_operation: {e}')
        return {'operation_type': 'add', 'analysis': f'Error during analysis: {e}', 'objects_to_remove': []}

async def handle_object_removal(room: Room, current_layout: FloorPlan, operation_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle object removal from the room based on operation analysis
    """
    api_key = ANTHROPIC_API_KEY
    if not room.objects:
        return {'success': True, 'removed_objects': [], 'remaining_objects': [], 'requested_count': 0, 'attempted_count': 0, 'message': 'No objects to remove - room is already empty'}
    objects_to_remove_ids = operation_analysis.get('objects_to_remove', [])
    if not objects_to_remove_ids and operation_analysis.get('operation_type') == 'remove':
        removal_analysis = await analyze_objects_for_removal(room, operation_analysis.get('analysis', ''))
        objects_to_remove_ids = removal_analysis.get('objects_to_remove', [])
    removed_indices = set()
    remaining_indices = set()
    for (i, obj) in enumerate(room.objects):
        if obj.id in objects_to_remove_ids or obj.type in objects_to_remove_ids:
            removed_indices.add(i)
    for removed_idx in list(removed_indices):
        obj_id_parent = room.objects[removed_idx].id
        for (i, obj_child) in enumerate(room.objects):
            if i in removed_indices:
                continue
            current_obj = obj_child
            while True:
                if current_obj.place_id == 'floor' or current_obj.place_id == 'wall':
                    break
                if current_obj.place_id == obj_id_parent:
                    removed_indices.add(i)
                    break
                parent_obj = next((obj for obj in room.objects if obj.id == current_obj.place_id), None)
                if parent_obj is None:
                    break
                current_obj = parent_obj
    for i in range(len(room.objects)):
        if i not in removed_indices:
            remaining_indices.add(i)
    removed_objects = [room.objects[i] for i in removed_indices]
    remaining_objects = [room.objects[i] for i in remaining_indices]
    return {'success': True, 'removed_objects': removed_objects, 'remaining_objects': remaining_objects, 'requested_count': len(objects_to_remove_ids), 'attempted_count': len(removed_indices), 'message': f'Removed {len(removed_objects)} objects, {len(remaining_objects)} objects remaining'}

def _remove_objects_and_descendants(objects: List[Object], object_ids: List[str]) -> tuple[List[Object], List[Object]]:
    """Return removed and remaining objects while preserving their input order."""
    existing_ids = {obj.id for obj in objects}
    removed_ids = set(object_ids) & existing_ids
    changed = True
    while changed:
        changed = False
        for obj in objects:
            if obj.id not in removed_ids and obj.place_id in removed_ids:
                removed_ids.add(obj.id)
                changed = True
    removed = [obj for obj in objects if obj.id in removed_ids]
    remaining = [obj for obj in objects if obj.id not in removed_ids]
    return (removed, remaining)

async def analyze_objects_for_removal(room: Room, removal_context: str) -> Dict[str, Any]:
    """
    Use Claude to determine which specific objects should be removed
    """
    api_key = ANTHROPIC_API_KEY
    existing_objects_info = [f"- {obj.type} (ID: {obj.id}) - {(obj.description if hasattr(obj, 'description') and obj.description else 'No description')}" for obj in room.objects]
    removal_prompt = f'You are helping to remove objects from a {room.room_type}. \n\nEXISTING OBJECTS:\n{chr(10).join(existing_objects_info)}\n\nREMOVAL CONTEXT:\n{removal_context}\n\nTASK:\nBased on the removal context, determine which specific objects should be removed from the room. \nConsider object types, functionality, and any specific mentions in the context.\n\nOUTPUT FORMAT:\nRespond with a JSON object:\n\n```json\n{{\n    "objects_to_remove": ["object_id_1", "object_type_1"],\n    "analysis": "Explanation of why these objects were selected for removal"\n}}\n```\n\nYou can specify either exact object IDs or object types. If removing by type, all objects of that type will be removed.\n\nAnalyze now:'
    try:
        response = call_vlm(vlm_type='openai', max_tokens=1500, temperature=0.1, messages=[{'role': 'user', 'content': removal_prompt}])
        response_text = response.content[0].text.strip()
        try:
            response_text = extract_json_from_response(response_text)
            if not response_text:
                raise ValueError('Could not extract JSON content from Claude response')
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError:
            return {'objects_to_remove': [], 'analysis': 'Failed to parse removal analysis'}
    except Exception as e:
        print(f'Error in analyze_objects_for_removal: {e}')
        return {'objects_to_remove': [], 'analysis': f'Error: {e}'}

@mcp.tool()
@timed_tool()
async def place_objects_in_room(ctx: Context, room_id: str='', placement_conditions: str=''):
    """
    OPERATION MODES:
    1. ADD MODE: Add multiple new objects while keeping existing ones
       - Triggered by: "add furniture", "place additional items", "include more objects"
       - Combines existing objects with new recommendations
    
    2. REMOVE MODE: Remove specific objects from the room
       - Triggered by: "remove the sofa", "delete old furniture", "get rid of clutter"
       - Uses AI analysis to identify which objects to remove based on conditions
       - If there is an object id, then you could use the object id to remove the object
    
    3. REPLACE MODE: Replace all objects with a completely new furniture set
       - Triggered by: "design the room", "furnish with modern style", general room descriptions
       - Clears existing objects and places an entirely new set

    WHEN NOT TO USE:
    - Adding/moving just one specific object (use add_one_object_with_condition_in_room)
    - Moving existing objects (use move_one_object_with_condition_in_room)
    - Precise single-object placement requirements
    - wanted to reorganize the existing furniture, not replace it with completely new items

    After the operation, it uses semantic and physics critic to analyze the room design quality and get improvement suggestions.
    
    Args:
        room_id: The ID of the room to furnish/modify
        placement_conditions: Text describing the operation and requirements:
                             
        ADD EXAMPLES:
        - "add essential furniture for a teenager"
        - "place additional seating and storage"
        - "include modern workspace items"
        Usage Guidelines:
        1. [Important]: Please include the object type, quantity, location (i. placed on floor, wall, or other object; ii. other relative relationship between the objects)
        Preferred relative relationship with other objects: near/far, in front/side/left/right/around, center aligned, face to/face same as, on top of, above. (other relationships are hard to parse in the tool)
        2. [Important]: Please include the object shape, size, color, material, style, finish, etc. in the description if applicable.
        3. [Important]: When adding multiple objects, please briefly mention the spatial relative relationship between the objects in the description.
        4. [Important]: When refer to existing objects, mention exact object id if applicable. e.g. "add lamp on table_001", "add sofa on the floor near the couch_002", etc.

        REMOVE EXAMPLES: 
        - "remove all the old furniture"
        - "delete the sofa and coffee table"
        - "get rid of clutter and unnecessary items"
        
        REPLACE EXAMPLES:
        - "design a modern minimalist bedroom"
        - "furnish as a cozy living room"
        - "create a professional home office"
        - "budget-friendly student apartment setup"
        
    Returns:
        JSON string containing:
        - List of objects placed/removed/replaced
        - Operation type performed (add/remove/replace)
        - Summary of changes made to the room
        - Details about object selection and placement reasoning
    """
    global current_layout
    global policy_analysis
    set_operation_counts('place', attempted=0, succeeded=0)
    print(f'🔧 Starting place_objects_in_room for room: {room_id}', file=sys.stderr)
    print(f'📝 Placement conditions: {placement_conditions}', file=sys.stderr)
    if current_layout is None:
        print('❌ No layout found - layout generation required', file=sys.stderr)
        return json.dumps({'success': False, 'error': "No layout has been generated yet. Use 'generate_room_layout()' first."})
    if not room_id:
        print('❌ Room ID parameter missing', file=sys.stderr)
        return json.dumps({'success': False, 'error': 'room_id parameter is required'})
    print(f'🔍 Finding room in the current layout: {room_id}', file=sys.stderr)
    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        available_rooms = [{'id': r.id, 'type': r.room_type} for r in current_layout.rooms]
        print(f"❌ Room {room_id} not found. Available rooms: {[r['id'] for r in available_rooms]}", file=sys.stderr)
        return json.dumps({'success': False, 'error': f"Room with ID '{room_id}' not found", 'available_rooms': available_rooms})
    try:
        print(f'✅ Room found: {room.room_type} with {len(room.objects)} existing objects', file=sys.stderr)
        print('🔑 VLM backend configured - preparing room information', file=sys.stderr)
        print('🔍 Analyzing placement conditions to determine operation type...', file=sys.stderr)
        with timed_stage('operation.analysis', 'llm'):
            operation_analysis = await analyze_placement_operation(room, placement_conditions)
        print(f"📊 Operation analysis complete - Type: {operation_analysis['operation_type']}", file=sys.stderr)
        operation_type = operation_analysis['operation_type']
        set_operation_counts('delete' if operation_type == 'remove' else 'place', attempted=0, succeeded=0)
        room_info = {'room_type': room.room_type, 'dimensions': {'width': room.dimensions.width, 'length': room.dimensions.length, 'height': room.dimensions.height}, 'area': room.dimensions.width * room.dimensions.length, 'doors': [{'position_on_wall': door.position_on_wall, 'width': door.width, 'height': door.height, 'wall_side': extract_wall_side_from_id(door.wall_id), 'door_type': door.door_type, 'opens_inward': door.opens_inward} for door in room.doors], 'windows': [{'position_on_wall': window.position_on_wall, 'width': window.width, 'height': window.height, 'wall_side': extract_wall_side_from_id(window.wall_id), 'sill_height': window.sill_height, 'window_type': window.window_type} for window in room.windows], 'existing_objects': [{'id': obj.id, 'type': obj.type, 'position': {'x': obj.position.x, 'y': obj.position.y, 'z': obj.position.z}, 'dimensions': {'width': obj.dimensions.width, 'length': obj.dimensions.length, 'height': obj.dimensions.height}} for obj in room.objects], 'floor_material': room.floor_material, 'ceiling_height': room.ceiling_height}
        if operation_analysis['operation_type'] == 'remove':
            print('🗑️ Skipping Claude API call for removal operation - no object recommendations needed', file=sys.stderr)
            claude_recommendations = {}
        else:
            if placement_conditions and placement_conditions.strip():
                prompt = f""" Based ONLY on these conditions:\n\n{placement_conditions.strip()}\n\nPlace objects based on the conditions for {room.room_type} ({room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m × {room.dimensions.height:.1f}m)\n\nRoom Context:\n- Existing Objects: {(chr(10).join([f"  {i + 1}. {obj['type']} (ID: {obj['id']}) - {obj['dimensions']['width']:.1f}×{obj['dimensions']['length']:.1f}×{obj['dimensions']['height']:.1f}m at ({obj['position']['x']:.1f},{obj['position']['y']:.1f},{obj['position']['z']:.1f})" for (i, obj) in enumerate(room_info['existing_objects'])]) if room.objects else 'None')}\n- Doors/Windows: {len(room.doors)} doors, {len(room.windows)} windows\n\nCRITICAL: Only recommend objects mentioned in the conditions above. Do NOT add extra objects that are not mentioned in the conditions.\n\nPLACEMENT TARGET RULES:\n- "floor": placing on the ground/floor (includes against walls, in corners, or anywhere on floor surface)\n- "wall": ONLY for objects that are attached/mounted directly on the wall, above the floor (like wall shelves, paintings, wall-mounted TVs, all above the floor)\n- "object_id": placing onto/on top of a specific existing object\n\nEXAMPLES:\n- "add table against the wall" → "floor" (table sits on floor, positioned near wall)\n- "add sofa to corner" → "floor" (sofa sits on floor in corner)\n- "add chair next to / beside / left of / right of / in front of / other positions near the table" → "floor" (chair sits on floor next to table)\n- "add picture on wall" → "wall" (picture is mounted on wall and above the floor)\n- "add lamp on table" → use table's object_id (lamp goes on top of table)\n- "add picture on wall region that on top of the bed" → "wall" (picture is mounted on wall region that on top of the bed, and above the floor)\n\nNote: You need to ensure that the placement location is reasonable and logical. e.g. chair is not placed on top of the table, but next to the table **on floor**.\n\nOBJECT RESTRICTIONS:\nDO NOT recommend: rugs, mats, carpets (or other floor coverings), windows, doors, curtains, hanging-on-ceiling objects (like hanging top lights), ceiling objects (already installed)\n\nObject Name Rule:\n- Use a stable, specific lowercase category name. Use underscores for multiword categories, e.g. coffee_table, floor_lamp, pencil_holder. Preserve the requested identity instead of collapsing distinct objects to a generic table or lamp.\n\nObject Description Rule:\n- Start with the full name of the object (e.g. "A wooden office chair", "A glass coffee table", "A ceramic table lamp")\n- Focus on the physical characteristics of the object, including object shape \n(important since it will be used to generate the 3D model, so please include the shape of the object in the description clearly,{chr(32)}\ne.g. rectangular, square, circular, oval, triangular, cubic, cylindrical, spherical, conical, curved, angular, straight-edged,\nand elongated, oblong, squat, slender, chunky, low-slung, compact, sprawling, towering, leggy, bulky, streamlined, narrow, wide, deep, shallow, stocky, lanky, flat, oversized), \ncolor, finish, style, material, etc.\n- If you are describing a wall mounted **thin** object including but not limited to paintings, posters, tv screens, clocks, mirrors, artworks, etc., you must add adjective "single piece of", "thin" and "upright" in the description.\ne.g. "A single piece of thin and upright painting with ...", "A single piece of thin and upright tv screen with ...", "A single piece of thin and upright clock with ...", etc.\n- Use your imagination to come up with at least one unique physical feature of the object. This should distinguish the object from other object with the same type.\n- Do not include the size of the object in the description. e.g. width is xx cm, height is xx cm, etc. are all not allowed in object description.\n- Do not include the usage of the object in the description, including what is it for, where to place the object, etc.\nThe presence of other object names in the description would lead to the failure of 3D model generation.{chr(32)}\n(e.g. if you describe a "cushion" on the sofa, you can't say "cushion on the sofa", or "cushion colored with sofa's color", etc. in the description. Focus only on the object itself and its physical characteristics.)\n\nObject Size Rule:\nYou need to specify the size of the object in a list of three numbers, \nsequence as length, width, height, in the format of [length_cm, width_cm, height_cm].\nYou should consider the object size relative to the room size and the existing objects in the room.\nE.g. if the object is surrounded by a lot of objects, maybe the size should be larger than normal accordingly.\n\nObject Quantity Rule:\nIf need to place multiple objects, the quantity should be the number of objects to place.\n\nObject Variance Type Rule:\nif object quantity is larger than 1, you need to specify the variance type.\n- "same": multiple objects share the **same** geometric shape and size and **the same** texture, color, style, material, etc.\nThis usually happens in the design of large and repetitive and symmetric objects like restaurant tables, chairs, and office tables, etc.\n- "varied": For the remaining situations, the objects have **different** geometric shapes and sizes, and **different** texture, color, style, material, etc.\nThis usually happens in the design of other diverse objects like decorative objects, paintings, plants, vases, small objects including but not limited to books, cups, plates, etc.\n\nFor example:\ni. if you want to place 6 same size chairs, the quantity should be 6 and the variance type should be "same".\nii. if you want to place 6 different chairs, the quantity should be 6 and the variance type should be "varied".\n\nObject Location Rule:\nThe object location can be one of the following:\n- "floor": placing on the ground/floor (includes against walls, in corners, beside / next to other floor objects, or anywhere on floor surface)\n- "wall": ONLY for objects that are attached/mounted directly on the wall, and above the floor (like wall shelves, paintings, wall-mounted TVs, all above the floor)\n- "object_id": placing **only onto/on top of** a specific existing object (the object_id is the id of the existing object in the room) (if it's not on top of, you can't use this location)\n- "estimated_object_name": placing **only onto/on top of** some specific object without knowing the exact object id, you can use an estimated the object name here (e.g. table, sofa, bed, etc.). (only used when placing onto/on top of the estimated object name, if it's not onto/on top of, but next to or near or etc., you can only use floor or wall, can't use this location either)\nDon't use "floor" or "wall" as fallback, This is not accurate and not allowed.\n\nObject Place Guidance Rule:\nYou need to ensure:\n- Each objects is neatly placed, \n- Objects of the same category are well aglined.\n- Relationships are reasonable (e.g., chairs face desks)\n- Sufficient space exists for walking when considering object placement on floor.\n- Orientations must be correct. \n\nReturn JSON with format:\n```json\n{{\n    "object_category": {{\n        "description": "A wooden office chair with a curved back and matte finish",\n        "location": "floor|wall|existing_object_id|estimated_object_name", \n        "size": [50, 50, 90],\n        "quantity": 1,\n        "variance_type": "same",\n        "place_guidance": "Placement instructions"\n    }}\n}}\n```\n\nThe numeric values above are examples; use the requested quantity and suitable dimensions in centimeters. Return valid JSON only, with numeric size and quantity values and no comments or trailing commas. Focus on physical details for 3D generation."""
            else:
                prompt = f"""Recommend essential objects for {room.room_type} ({room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m × {room.dimensions.height:.1f}m).\n\nRoom Context:\n- Existing Objects: {(chr(10).join([f"  {i + 1}. {obj['type']} (ID: {obj['id']}) - {obj['dimensions']['width']:.1f}×{obj['dimensions']['length']:.1f}×{obj['dimensions']['height']:.1f}m at ({obj['position']['x']:.1f},{obj['position']['y']:.1f},{obj['position']['z']:.1f})" for (i, obj) in enumerate(room_info['existing_objects'])]) if room.objects else 'None')}\n- Doors/Windows: {len(room.doors)} doors, {len(room.windows)} windows\n\nPLACEMENT TARGET RULES:\n- "floor": placing on the ground/floor (includes against walls, in corners, or anywhere on floor surface)\n- "wall": ONLY for objects that are attached/mounted directly on the wall (like wall shelves, paintings, wall-mounted TVs)\n- "object_id": placing onto/on top of a specific existing object\n\nEXAMPLES:\n- "add table against the wall" → "floor" (table sits on floor, positioned near wall)\n- "add sofa to corner" → "floor" (sofa sits on floor in corner)\n- "add chair/stool near/beside/in front of the table" → "floor" (chair sits on floor near the table)\n- "add picture on wall" → "wall" (picture is mounted on wall)\n- "add lamp on table" → use table's object_id (lamp goes on top of table)\n\nOBJECT RESTRICTIONS:\nDO NOT recommend: rugs, mats, windows, doors, curtains, ceiling objects (already installed)\n\nSINGLE OBJECT RULE:\nEach recommendation must be for exactly ONE individual object, not sets or groups of objects.\nUse basic object names like "chair", "table", "lamp", "desk", "sofa" - NOT composite names like "readingset", "workstation", "dinette", "seatinggroup".\n\nReturn JSON with format:\n```json\n{{\n    "objectname (simple basic object name)": {{\n        "description": "Physical characteristics only",\n        "location": "floor|wall|existing_object_id (must be exactly ONE existing object ID string; not a list; cannot be empty)", \n        "size": [50, 50, 90],\n        "quantity": 1,\n        "variance_type": "same",\n        "place_guidance": "Placement instructions"\n    }}\n}}\n```\n\nThe numeric values above are examples; use the requested quantity and suitable dimensions in centimeters. Return valid JSON only, with numeric size and quantity values and no comments or trailing commas. Focus on physical details for 3D generation."""
            print('🤖 Calling Claude API for object recommendations...', file=sys.stderr)
            with timed_stage('object.recommendation', 'llm'):
                response = call_vlm(vlm_type='openai', max_tokens=8000, temperature=0.1, messages=[{'role': 'user', 'content': prompt}])
            response_text = response.content[0].text.strip()
            await ctx.info(f'✅ Claude API response received ({len(response_text)} characters)')
            response_text = extract_json_from_response(response_text)
            if not response_text:
                raise ValueError('Could not extract JSON content from Claude response')
            try:
                claude_recommendations = json.loads(response_text)
                if not isinstance(claude_recommendations, dict):
                    claude_recommendations = {}
                else:
                    validated_recommendations = {}
                    for (object_name, object_data) in claude_recommendations.items():
                        if isinstance(object_data, dict):
                            quantity = int(object_data.get('quantity', 1))
                            variance_type = str(object_data.get('variance_type', 'same')).lower()
                            if variance_type not in ['same', 'varied']:
                                variance_type = 'varied'
                            if quantity < 1:
                                quantity = 1
                            location = str(object_data.get('location', 'floor')).lower()
                            if location not in ['floor', 'wall']:
                                existing_object_ids = [obj.id for obj in room.objects]
                                if location not in existing_object_ids:
                                    location = 'invalid'
                            if quantity > 1 and variance_type == 'varied':
                                for object_cnt_i in range(quantity):
                                    validated_object = {'description': str(object_data.get('description', f'A {object_name} for the room')), 'location': location, 'size': object_data.get('size', [100, 50, 75]), 'quantity': 1, 'variance_type': 'same', 'place_guidance': str(object_data.get('place_guidance', f'Standard placement for {object_name}'))}
                                    validated_recommendations[str(object_name).lower().replace(' ', '_') + f'_{object_cnt_i + 1}'] = validated_object
                            else:
                                validated_object = {'description': str(object_data.get('description', f'A {object_name} for the room')), 'location': location, 'size': object_data.get('size', [100, 50, 75]), 'quantity': quantity, 'variance_type': 'same', 'place_guidance': str(object_data.get('place_guidance', f'Standard placement for {object_name}'))}
                                validated_recommendations[str(object_name).lower().replace(' ', '_')] = validated_object
                        else:
                            continue
                    claude_recommendations = validated_recommendations
            except json.JSONDecodeError as e:
                return json.dumps({'success': False, 'error': f'Could not parse JSON content from Claude response: {str(e)}', 'room_id': room_id})
            except json.JSONDecodeError as e:
                return json.dumps({'success': False, 'error': f'Could not parse JSON content from Claude response: {str(e)}', 'room_id': room_id})
        selected_objects = []
        updated_recommendation_list = []
        removal_result = None
        requested_new_objects = sum((int(object_data.get('quantity', 1)) for object_data in claude_recommendations.values() if isinstance(object_data, dict)))
        if operation_analysis['operation_type'] == 'remove':
            print('🗑️ Executing REMOVE operation...', file=sys.stderr)
            with timed_stage('object.removal', 'operation'):
                removal_result = await handle_object_removal(room, current_layout, operation_analysis)
            print(f"✅ Remove operation complete - {len(removal_result['removed_objects'])} objects removed", file=sys.stderr)
            set_operation_counts('delete', requested=removal_result.get('requested_count'), attempted=removal_result.get('attempted_count', len(removal_result['removed_objects'])), succeeded=len(removal_result['removed_objects']), removed_with_descendants=len(removal_result['removed_objects']))
            for layout_room in current_layout.rooms:
                if layout_room.id == room.id:
                    layout_room.objects = removal_result['remaining_objects']
                    break
            room.objects = removal_result['remaining_objects']
            all_objects_to_place = room.objects.copy()
        if operation_analysis['operation_type'] in ['add', 'replace']:
            print(f"🔧 Executing {operation_analysis['operation_type'].upper()} operation...", file=sys.stderr)
            objects_to_keep = []
            if operation_analysis['operation_type'] == 'add':
                objects_to_keep = room.objects.copy()
                print(f'📦 ADD mode - keeping {len(objects_to_keep)} existing objects', file=sys.stderr)
            print('🎯 Selecting new objects from Claude recommendations...', file=sys.stderr)
            asset_stage = 'asset.generation'
            with timed_stage(asset_stage, 'asset', details={'requested_count': requested_new_objects}) as asset_timer:
                (selected_objects, updated_recommendation_list) = select_objects(claude_recommendations, room, room.objects, current_layout)
                asset_timer.add_details(selected_count=len(selected_objects))
            print(f'✅ Object selection complete - {len(selected_objects)} new objects selected', file=sys.stderr)
            selected_floor_count = sum((obj.place_id == 'floor' for obj in selected_objects))
            selected_wall_count = sum((obj.place_id == 'wall' for obj in selected_objects))
            selected_on_object_count = len(selected_objects) - selected_floor_count - selected_wall_count
            set_operation_counts('place', requested=requested_new_objects, attempted=len(selected_objects), succeeded=0, floor_attempted=selected_floor_count, wall_attempted=selected_wall_count, on_object_attempted=selected_on_object_count)
            all_objects_to_place = objects_to_keep + selected_objects
        elif operation_analysis['operation_type'] == 'remove' and (not room.objects):
            all_objects_to_place = []
        total_objects = sum((obj_data['quantity'] for obj_data in updated_recommendation_list))
        floor_objects = sum((obj_data['quantity'] for obj_data in updated_recommendation_list if obj_data['location'] == 'floor'))
        wall_objects = sum((obj_data['quantity'] for obj_data in updated_recommendation_list if obj_data['location'] == 'wall'))
        existing_object_ids = [obj.id for obj in room.objects]
        on_object_objects = sum((obj_data['quantity'] for obj_data in updated_recommendation_list if obj_data['location'] in existing_object_ids))
        if all_objects_to_place:
            print(f'🎯 Placing {len(all_objects_to_place)} objects using DFS solver...', file=sys.stderr)
            with timed_stage('placement.total', 'placement', details={'all_object_count': len(all_objects_to_place), 'new_object_count': len(selected_objects)}):
                (placed_objects, current_layout, claude_placement_interactions) = place_objects(all_objects_to_place, room, current_layout)
            print(f'✅ Object placement complete - {len(placed_objects)} objects successfully placed', file=sys.stderr)
        else:
            print('⚠️ No objects to place', file=sys.stderr)
            placed_objects = []
            claude_placement_interactions = {'no_objects_to_place': True}
        final_room_objects = []
        for layout_room in current_layout.rooms:
            if layout_room.id == room.id:
                final_room_objects = layout_room.objects
                break
        print(f'💾 Saving updated layout with {len(final_room_objects)} objects in room', file=sys.stderr)
        output_path = Path(RESULTS_DIR) / f'{current_layout.id}'
        os.makedirs(str(output_path), exist_ok=True)
        with timed_stage('layout.save', 'io'):
            export_layout_to_json(current_layout, output_path / f'{current_layout.id}.json')
        final_floor_objects = [obj for obj in final_room_objects if obj.place_id == 'floor']
        floor_area_occupied = sum((obj.dimensions.width * obj.dimensions.length for obj in final_floor_objects))
        floor_occupied_ratio = floor_area_occupied / (room.dimensions.width * room.dimensions.length)
        room_num_calls[room_id] += 1
        result = {'success': True, 'room_id': room_id}
        if 'placed_objects' in claude_placement_interactions:
            result['placed_objects'] = get_object_description_list(claude_placement_interactions['placed_objects'])
        if 'failed_to_be_placed_objects' in claude_placement_interactions:
            result['failed_to_be_placed_objects'] = get_failed_placements_description(claude_placement_interactions['failed_to_be_placed_objects'])
        result['all_objects'] = get_object_description_list(final_room_objects)
        if operation_analysis['operation_type'] == 'remove' and removal_result:
            result['removal_info'] = {'objects_removed': len(removal_result['removed_objects']), 'objects_remaining': len(removal_result['remaining_objects']), 'removal_message': removal_result['message'], 'removed_object_details': [{'id': obj.id, 'type': obj.type, 'description': getattr(obj, 'description', 'N/A')} for obj in removal_result['removed_objects']]}
        if PHYSICS_CRITIC_ENABLED:
            try:
                print('🔍 Running physics critic analysis...', file=sys.stderr)
                with timed_stage('critic.physics.final', 'critic'):
                    physics_critic_result = await room_physics_critic(room_id)
                physics_critic_result = json.loads(physics_critic_result)
                result['physics_critic_info'] = physics_critic_result
                print('✅ Physics critic analysis complete', file=sys.stderr)
            except Exception as e:
                print(f'⚠️ Physics critic analysis failed: {str(e)}', file=sys.stderr)
                result['physics_critic_info'] = 'Failed to do physics critic analysis'
        if SEMANTIC_CRITIC_ENABLED:
            try:
                print('🔍 Running semantic critic analysis...', file=sys.stderr)
                with timed_stage('critic.semantic.final', 'critic'):
                    semantic_critic_result = await room_semantic_critic(room_id, current_action_condition=placement_conditions.strip(), propose_modifications=True)
                semantic_critic_result = json.loads(semantic_critic_result)
                result['semantic_critic_info'] = semantic_critic_result
                print('✅ Semantic critic analysis complete', file=sys.stderr)
            except Exception as e:
                print(f'⚠️ Semantic critic analysis failed: {str(e)}', file=sys.stderr)
                result['semantic_critic_info'] = 'Failed to do semantic critic analysis'
        counted_room = next((r for r in current_layout.rooms if r.id == room_id), None)
        counted_objects = counted_room.objects if counted_room else []
        if operation_type == 'remove' and removal_result:
            removed_ids = {obj.id for obj in removal_result['removed_objects']}
            succeeded = sum((obj_id not in {obj.id for obj in counted_objects} for obj_id in removed_ids))
            set_operation_counts('delete', requested=removal_result.get('requested_count'), attempted=removal_result.get('attempted_count', len(removed_ids)), succeeded=succeeded, removed_with_descendants=len(removed_ids))
        else:
            final_ids = {obj.id for obj in counted_objects}
            succeeded = sum((obj.id in final_ids for obj in selected_objects))
            set_operation_counts('place', requested=requested_new_objects, attempted=len(selected_objects), succeeded=succeeded, floor_attempted=sum((obj.place_id == 'floor' for obj in selected_objects)), floor_succeeded=sum((obj.place_id == 'floor' and obj.id in final_ids for obj in selected_objects)), wall_attempted=sum((obj.place_id == 'wall' for obj in selected_objects)), wall_succeeded=sum((obj.place_id == 'wall' and obj.id in final_ids for obj in selected_objects)), on_object_attempted=sum((obj.place_id not in {'floor', 'wall'} for obj in selected_objects)), on_object_succeeded=sum((obj.place_id not in {'floor', 'wall'} and obj.id in final_ids for obj in selected_objects)))
        try:
            print('🎨 Creating room visualization...', file=sys.stderr)
            from visualizer import RoomVisualizer
            import tempfile
            updated_room = next((r for r in current_layout.rooms if r.id == room_id), None)
            if updated_room:
                visualizer = RoomVisualizer(updated_room)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                with timed_stage('visualization.room_2d', 'visualization'):
                    visualizer.visualize_2d(save_path=temp_path, show=False)
                os.unlink(temp_path)
                print('✅ Room visualization created successfully', file=sys.stderr)
                return json.dumps(result, indent=2)
            else:
                print('⚠️ Could not find updated room for visualization', file=sys.stderr)
                return json.dumps(result, indent=2)
        except Exception as viz_error:
            print(f'⚠️ Room visualization failed: {str(viz_error)}', file=sys.stderr)
            result['visualization_error'] = f'Failed to create room visualization: {str(viz_error)}'
            return json.dumps(result, indent=2)
    except anthropic.APIError as e:
        print(f'❌ Anthropic API error: {str(e)}', file=sys.stderr)
        return json.dumps({'success': False, 'error': f'Anthropic API error: {str(e)}', 'room_id': room_id})
    except Exception as e:
        print(f'❌ Object placement recommendation failed: {str(e)}', file=sys.stderr)
        return json.dumps({'success': False, 'error': f'Object placement recommendation failed: {str(e)}', 'room_id': room_id})

@mcp.tool()
@timed_tool()
async def get_layout_save_dir() -> str:
    """
    Get the directory where the layout is saved.

    Returns:
        A dictionary containing the layout save directory.
    """
    global current_layout
    if current_layout is None:
        return json.dumps({'success': False, 'error': "No layout has been generated yet. Use 'generate_room_layout()' first."})
    return json.dumps({'success': True, 'layout_save_dir': os.path.abspath(str(Path(RESULTS_DIR) / f'{current_layout.id}'))})

@mcp.tool()
@timed_tool()
async def get_room_information(room_id: str):
    """
    Attain the information about the room, including the room type, walls, doors, windows, objects.

    The function will also return the visualization of the room, including the walls, doors, windows, objects.

    It is strongly recommended to use this tool to get the information about the room before and after any other tools are used.

    Args:
        room_id: The id of the room to get information about.

    Returns:
        the room information as a dict.
    """
    global current_layout
    if current_layout is None:
        return (json.dumps({'success': False, 'error': "No layout has been generated yet. Use 'generate_room_layout()' first."}), None)
    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        available_rooms = [{'id': r.id, 'type': r.room_type} for r in current_layout.rooms]
        return (json.dumps({'success': False, 'error': f"Room with ID '{room_id}' not found", 'available_rooms': available_rooms}), None)
    try:
        room_info = {'success': True, 'room_id': room.id, 'room_type': room.room_type, 'dimensions_cm': {'width': room.dimensions.width * 100, 'length': room.dimensions.length * 100, 'height': room.dimensions.height * 100}, 'area_sq_m': room.dimensions.width * room.dimensions.length, 'floor_material': room.floor_material, 'ceiling_height_cm': room.ceiling_height * 100, 'walls': [], 'doors': [], 'windows': [], 'objects': []}
        for wall in room.walls:
            wall_info = {'id': wall.id, 'start_point_cm': {'x': (wall.start_point.x - room.position.x) * 100, 'y': (wall.start_point.y - room.position.y) * 100, 'z': (wall.start_point.z - room.position.z) * 100}, 'end_point_cm': {'x': (wall.end_point.x - room.position.x) * 100, 'y': (wall.end_point.y - room.position.y) * 100, 'z': (wall.end_point.z - room.position.z) * 100}, 'thickness_cm': wall.thickness * 100, 'height_cm': wall.height * 100, 'material': wall.material}
            room_info['walls'].append(wall_info)
        for door in room.doors:
            door_info = {'id': door.id, 'wall_id': door.wall_id, 'width_cm': door.width * 100, 'height_cm': door.height * 100, 'position_on_wall': door.position_on_wall, 'door_type': door.door_type, 'opens_inward': door.opens_inward}
            room_info['doors'].append(door_info)
        for window in room.windows:
            window_info = {'id': window.id, 'wall_id': window.wall_id, 'width_cm': window.width * 100, 'height_cm': window.height * 100, 'position_on_wall': window.position_on_wall, 'sill_height_cm': window.sill_height * 100, 'window_type': window.window_type}
            room_info['windows'].append(window_info)
        for obj in room.objects:
            obj_x_cm = (obj.position.x - room.position.x) * 100
            obj_y_cm = (obj.position.y - room.position.y) * 100
            obj_z_cm = (obj.position.z - room.position.z) * 100
            obj_info = {'id': obj.id, 'type': obj.type, 'description': obj.description, 'position_cm': {'x': obj_x_cm, 'y': obj_y_cm, 'z': obj_z_cm}, 'dimensions_cm': {'width': obj.dimensions.width * 100, 'length': obj.dimensions.length * 100, 'height': obj.dimensions.height * 100}, 'rotation': {'x': obj.rotation.x, 'y': obj.rotation.y, 'z': obj.rotation.z}, 'source': obj.source, 'source_id': obj.source_id, 'place_guidance': obj.place_guidance, 'place_id': obj.place_id}
            room_info['objects'].append(obj_info)
        room_info['summary'] = {'total_walls': len(room.walls), 'total_doors': len(room.doors), 'total_windows': len(room.windows), 'total_objects': len(room.objects)}
        from visualizer import RoomVisualizer
        visualizer = RoomVisualizer(room)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            temp_path = tmp_file.name
        visualizer.visualize_2d(save_path=temp_path, show=False)
        with open(temp_path, 'rb') as img_file:
            image_data = img_file.read()
        os.unlink(temp_path)
        return json.dumps(room_info, indent=2)
    except Exception as e:
        return (json.dumps({'success': False, 'error': f'Failed to get room information: {str(e)}', 'room_id': room_id}), None)

@mcp.tool()
@timed_tool()
async def move_one_object_with_condition_in_room(ctx: Context, room_id: str='', condition: str=''):
    """
    Intelligently move exactly ONE object in a room based on a targeted condition or instruction.
    After the movement, it uses semantic and physics critic to analyze the room design quality and get improvement suggestions.
    
    USE THIS WHEN:
    - Moving a specific object to a new location (floor, wall, or on top of another object)
    - Repositioning existing furniture with spatial constraints
    - Reorienting objects with specific facing directions
    - Adjusting object placement relative to other objects or room features
    
    Note: Condition must follow the format specified below with object_id and location
    
    DO NOT USE for:
    - Adding new objects (use add_one_object_with_condition_in_room instead)
    - Removing objects (use place_objects_in_room instead)
    - Complete room reorganization (use other layout tools)
    
    This tool uses intelligent condition analysis to:
    1. Identify which object needs to be moved from the user's description
    2. Understand the target location and spatial requirements
    3. Find optimal new placement using Claude's visual analysis and constraint system
    4. Move the object with collision avoidance and architectural correctness

    [Important] Format and Examples of Condition:

    Format:
    Move [object_type] (object_id: [object_id]) to [floor|wall|on top of object_id], [spatial_relationships with other objects or wall (window and door not supported)]
    
    Components:
    - object_type: Type of object to move (e.g., chair, desk, sofa)
    - object_id: Unique identifier of the object
    - location: Must be one of: 'floor', 'wall', or an existing object_id
    - spatial_relationships: Positioning relative to other objects or wall (window and door not supported)
    Preferred relative relationship with other objects: near/far, in front/side/left/right/around, center aligned, face to/face same as, on top of, above. (other relationships are hard to parse in the tool)
    
    Examples:
    1. Move chair (object_id: chair_001) to floor, facing and aligned with the desk.
    2. Move desk (object_id: desk_002) to floor, against the wall.
    3. Move lamp (object_id: lamp_003) to on top of desk_002.
    4. Move picture (object_id: picture_004) to wall, above the sofa.
    5. Move sofa (object_id: sofa_005) to floor, facing the coffee table.
    
    Args:
        room_id: The ID of the room containing the object to move
        condition: Movement instruction that MUST follow the format above.
                  Format: Move [object_type] (object_id: [object_id]) to [location] with [spatial_relationships]
        
    Returns:
        JSON string containing the movement result, object details, and movement reasoning
    """
    global current_layout
    global policy_analysis
    set_operation_counts('move', attempted=0, succeeded=0, requested=1)
    print(f'🔧 Starting move_one_object_with_condition_in_room for room: {room_id}', file=sys.stderr)
    print(f'📝 Movement condition: {condition}', file=sys.stderr)
    if current_layout is None:
        print('❌ No layout found - layout generation required', file=sys.stderr)
        return json.dumps({'success': False, 'error': "No layout has been generated yet. Use 'generate_room_layout()' first."})
    if not room_id:
        print('❌ Room ID parameter missing', file=sys.stderr)
        return json.dumps({'success': False, 'error': 'room_id parameter is required'})
    if not condition or not condition.strip():
        print('❌ Condition parameter missing or empty', file=sys.stderr)
        return json.dumps({'success': False, 'error': 'condition parameter is required and cannot be empty'})
    print(f'🔍 Finding room in the current layout: {room_id}', file=sys.stderr)
    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        available_rooms = [{'id': r.id, 'type': r.room_type} for r in current_layout.rooms]
        print(f"❌ Room {room_id} not found. Available rooms: {[r['id'] for r in available_rooms]}", file=sys.stderr)
        return json.dumps({'success': False, 'error': f"Room with ID '{room_id}' not found", 'available_rooms': available_rooms})
    if not room.objects:
        print(f'❌ No objects found in {room.room_type} to move', file=sys.stderr)
        return json.dumps({'success': False, 'error': f'No objects found in {room.room_type} to move', 'room_id': room_id})
    try:
        print(f'✅ Room found: {room.room_type} with {len(room.objects)} objects', file=sys.stderr)
        from objects.object_movement_planner import analyze_object_to_move_from_condition, get_movement_location_from_claude, move_object
        print('🔍 Step 1: Analyzing condition to identify object to move...', file=sys.stderr)
        with timed_stage('move.operation_analysis', 'llm'):
            analysis_result = await analyze_object_to_move_from_condition(room, current_layout, condition)
        if not analysis_result.get('success'):
            print(f"❌ Movement condition analysis failed: {analysis_result.get('error', 'Unknown error')}", file=sys.stderr)
            return json.dumps({'success': False, 'error': f"Failed to analyze movement condition: {analysis_result.get('error', 'Unknown error')}", 'room_id': room_id, 'condition': condition.strip(), 'available_objects': [{'id': obj.id, 'type': obj.type} for obj in room.objects]})
        object_to_move_id = analysis_result['object_to_move']['id']
        print(f'🔍 Step 2: Finding object to move - ID: {object_to_move_id}', file=sys.stderr)
        object_to_move = next((obj for obj in room.objects if obj.id == object_to_move_id), None)
        if object_to_move is None:
            print(f'❌ Object {object_to_move_id} not found in room', file=sys.stderr)
            return json.dumps({'success': False, 'error': f'Object {object_to_move_id} not found in room', 'room_id': room_id, 'condition': condition.strip(), 'analysis_result': analysis_result})
        print(f'✅ Object found: {object_to_move.type} ({object_to_move.id})', file=sys.stderr)
        set_operation_counts('move', requested=1, attempted=1, succeeded=0, object_id=object_to_move.id)
        print('🎯 Step 3: Getting movement location from Claude...', file=sys.stderr)
        movement_intent = analysis_result['movement_intent']
        movement_target_location = analysis_result['movement_target_location']
        with timed_stage('move.target_selection', 'llm', details={'target_location': movement_target_location}):
            movement_result = await get_movement_location_from_claude(room, current_layout, object_to_move, movement_intent, movement_target_location)
        if not movement_result.get('success'):
            print(f"❌ Movement location determination failed: {movement_result.get('error', 'Unknown error')}", file=sys.stderr)
            return json.dumps({'success': False, 'error': f"Failed to determine movement location: {movement_result.get('error', 'Unknown error')}", 'movement_error_details': movement_result, 'room_id': room_id, 'condition': condition.strip(), 'object_info': {'id': object_to_move.id, 'type': object_to_move.type, 'current_position': {'x': object_to_move.position.x, 'y': object_to_move.position.y, 'z': object_to_move.position.z}}, 'analysis_result': analysis_result})
        print('✅ Movement location determined successfully', file=sys.stderr)
        print('🎯 Step 4: Moving object using MovementFloorSolver...', file=sys.stderr)
        with timed_stage('move.execution', 'placement', details={'target_location': movement_target_location}):
            (updated_room_objects, updated_layout, movement_info) = await move_object(room, current_layout, object_to_move, movement_result, movement_target_location)
        if not movement_info.get('success'):
            print(f"❌ Object movement failed: {movement_info.get('error', 'Unknown error')}", file=sys.stderr)
            return json.dumps({'success': False, 'error': f"Failed to move object: {movement_info.get('error', 'Unknown error')}"})
        print(f'✅ Object movement successful - {object_to_move.type} moved in room', file=sys.stderr)
        set_operation_counts('move', requested=1, attempted=1, succeeded=1, object_id=object_to_move.id, target_location=movement_target_location)
        current_layout = updated_layout
        print('💾 Saving updated layout to JSON...', file=sys.stderr)
        output_path = Path(RESULTS_DIR) / f'{current_layout.id}'
        os.makedirs(str(output_path), exist_ok=True)
        with timed_stage('layout.save', 'io'):
            export_layout_to_json(current_layout, output_path / f'{current_layout.id}.json')
        room_num_calls[room_id] += 1
        result = {'success': True, 'message': f'✅ Successfully moved {object_to_move.type} in {room.room_type}', 'room_id': room_id, 'room_type': room.room_type, 'object_moved': {'id': object_to_move.id, 'type': object_to_move.type, 'description': object_to_move.description, 'dimensions': {'width': object_to_move.dimensions.width, 'length': object_to_move.dimensions.length, 'height': object_to_move.dimensions.height}, 'old_position': movement_info['object_moved']['old_position'], 'new_position': movement_info['object_moved']['new_position'], 'old_rotation': movement_info['object_moved']['old_rotation'], 'new_rotation': movement_info['object_moved']['new_rotation'], 'source': object_to_move.source, 'source_id': object_to_move.source_id}, 'child_objects_removed_infomation': get_child_objects_removed_reminder(movement_info['child_objects_removed']), 'condition_analyzed': condition.strip()}
        if PHYSICS_CRITIC_ENABLED:
            try:
                print('🔍 Running physics critic analysis...', file=sys.stderr)
                with timed_stage('critic.physics.final', 'critic'):
                    physics_critic_result = await room_physics_critic(room_id)
                physics_critic_result = json.loads(physics_critic_result)
                result['physics_critic_info'] = physics_critic_result
                print('✅ Physics critic analysis complete', file=sys.stderr)
            except Exception as e:
                print(f'⚠️ Physics critic analysis failed: {str(e)}', file=sys.stderr)
                result['physics_critic_info'] = 'Failed to do physics critic analysis'
        if SEMANTIC_CRITIC_ENABLED:
            try:
                print('🔍 Running semantic critic analysis...', file=sys.stderr)
                with timed_stage('critic.semantic.final', 'critic'):
                    semantic_critic_result = await room_semantic_critic(room_id, current_action_condition=condition.strip(), propose_modifications=True)
                semantic_critic_result = json.loads(semantic_critic_result)
                result['semantic_critic_info'] = semantic_critic_result
                print('✅ Semantic critic analysis complete', file=sys.stderr)
            except Exception as e:
                print(f'⚠️ Semantic critic analysis failed: {str(e)}', file=sys.stderr)
                result['semantic_critic_info'] = 'Failed to do semantic critic analysis'
        try:
            print('🎨 Creating room visualization...', file=sys.stderr)
            from visualizer import RoomVisualizer
            import tempfile
            updated_room = next((r for r in current_layout.rooms if r.id == room_id), None)
            if updated_room:
                visualizer = RoomVisualizer(updated_room)
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    temp_path = tmp_file.name
                with timed_stage('visualization.room_2d', 'visualization'):
                    visualizer.visualize_2d(save_path=temp_path, show=False)
                with open(temp_path, 'rb') as img_file:
                    image_data = img_file.read()
                os.unlink(temp_path)
                print('✅ Room visualization created successfully', file=sys.stderr)
                return json.dumps(result, indent=2)
            else:
                print('⚠️ Could not find updated room for visualization', file=sys.stderr)
                return json.dumps(result, indent=2)
        except Exception as viz_error:
            print(f'⚠️ Room visualization failed: {str(viz_error)}', file=sys.stderr)
            result['visualization_error'] = f'Failed to create room visualization: {str(viz_error)}'
            return json.dumps(result, indent=2)
    except ImportError as e:
        print(f'❌ ImportError: {str(e)}', file=sys.stderr)
        return json.dumps({'success': False, 'error': f'Missing required modules for object movement: {str(e)}', 'room_id': room_id})
    except Exception as e:
        print(f'❌ Unexpected error in move_one_object_with_condition_in_room: {str(e)}', file=sys.stderr)
        return json.dumps({'success': False, 'error': f'Unexpected error in move_one_object_with_condition_in_room: {str(e)}', 'room_id': room_id, 'condition': condition.strip()})

async def room_semantic_critic(room_id: str='', current_action_condition: str='', propose_modifications: bool=True, max_recommendations: int=1) -> str:
    """
    Analyze whether the room is designed well semantically.

    The tool will first render the room from four top views.

    Then it will use Claude to analyze the room from:

    1. Whether each object is placed in a reasonable location (relative to the room and other objects).

    2. Whether each object is oriented in a reasonable direction (relative to the room and other objects).

    3. Whether the room is too crowded or too empty (remove unnecessary objects if the room is too crowded, add objects if the room is too empty).

    After the analysis, it will return recommendations based on:
    - All object addition analysis (minimum required objects, object combos, and background objects) - no limit applied
    - Top modification operations (REMOVE, REPLACE, MOVE) - limited by max_recommendations
    
    The function returns all object addition suggestions and the highest-scoring max_recommendations 
    modification actions, then sorts them all by priority/score for final output.

    Args:
        room_id: The ID of the room to analyze
        current_action_condition: Optional context about the current object placement action being performed
        propose_modifications: If True, return modification suggestions (REMOVE/REPLACE/MOVE). If False, exclude them
        max_recommendations: Maximum number of modification recommendations to return (default: 2)
                           For example, if max_recommendations=2, you'll get:
                           - All object addition analysis items (minimum required + object combos + background objects)
                           - Top 2 REMOVE/REPLACE/MOVE operations (only if propose_modifications=True)
                           - All items sorted by priority/score

    Returns:
        JSON string containing the analysis result with recommendations sorted by score

    """
    global current_layout
    if current_layout is None:
        return json.dumps({'success': False, 'error': "No layout has been generated yet. Use 'generate_room_layout()' first."})
    if not room_id:
        return json.dumps({'success': False, 'error': 'room_id parameter is required'})
    room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if room is None:
        available_rooms = [{'id': r.id, 'type': r.room_type} for r in current_layout.rooms]
        return json.dumps({'success': False, 'error': f"Room with ID '{room_id}' not found", 'available_rooms': available_rooms})
    scene_requirements = ''
    try:
        if hasattr(current_layout, 'policy_analysis') and current_layout.policy_analysis:
            scene_requirements = current_layout.policy_analysis.get('scene_requirements', '')
    except Exception as e:
        print(f'⚠️ Could not extract scene requirements: {e}', file=sys.stderr)
        scene_requirements = ''
    try:
        from room_render import render_room_five_views
        from visualizer import RoomVisualizer
        import base64
        import io
        import tempfile
        from PIL import Image as PILImage
        try:
            with timed_stage('critic.semantic.render_five_views', 'render'):
                (all_rgb, top_down_rgb) = render_room_five_views(current_layout, room_id, edges_resolution=1920, top_resolution=1024)
        except Exception as e:
            return json.dumps({'success': False, 'error': f'Failed to render room views: {str(e)}', 'room_id': room_id})
        vis_dir = f'{SERVER_ROOT_DIR}/vis'
        os.makedirs(vis_dir, exist_ok=True)
        try:
            with timed_stage('critic.semantic.annotate_top', 'render'):
                top_down_rgb = np.clip(top_down_rgb * 255, 0, 255).astype(np.uint8)
                top_down_rgb = annotate_top_down_view_with_objects(top_down_rgb, room)
                top_down_debug_path = os.path.join(vis_dir, f'{room_id}_top_down_annotated.png')
                PILImage.fromarray(top_down_rgb).save(top_down_debug_path)
                buffer = io.BytesIO()
                PILImage.fromarray(top_down_rgb).save(buffer, format='PNG')
                top_down_image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            print(f'✅ Annotated top-down view created: {top_down_debug_path}', file=sys.stderr)
        except Exception as e:
            return json.dumps({'success': False, 'error': f'Failed to create annotated top-down view: {str(e)}', 'room_id': room_id})
        image_data_list = []
        saved_image_paths = [top_down_debug_path]
        with timed_stage('critic.semantic.encode_images', 'io'):
            for (i, rgb_array) in enumerate(all_rgb):
                try:
                    rgb_uint8 = (rgb_array * 255).astype(np.uint8)
                    pil_image = PILImage.fromarray(rgb_uint8, 'RGB')
                    debug_path = os.path.join(vis_dir, f'{room_id}_rendered_view_{i + 1}.png')
                    pil_image.save(debug_path)
                    saved_image_paths.append(debug_path)
                    buffer = io.BytesIO()
                    pil_image.save(buffer, format='PNG')
                    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    image_data_list.append(image_base64)
                except Exception as e:
                    return json.dumps({'success': False, 'error': f'Failed to process rendered view {i}: {str(e)}', 'room_id': room_id})
        room_description = get_room_description(room)
        final_floor_objects = [obj for obj in room.objects if obj.place_id == 'floor']
        floor_area_occupied = sum((obj.dimensions.width * obj.dimensions.length for obj in final_floor_objects))
        floor_occupied_ratio = floor_area_occupied / (room.dimensions.width * room.dimensions.length)
        if floor_occupied_ratio < occupancy_ratio / 100:
            background_objects_floor_and_wall_description = 'useful furniture or varied decoration on floor and wall, unless explicitly restricted by the user and only where useful for this room, and '
        else:
            background_objects_floor_and_wall_description = ''
        user_demand_section = f'User Preferences: {current_layout.created_from_text};\n'
        scene_requirements_section = ''
        if scene_requirements:
            scene_requirements_section = f'\nSCENE GENERATION RECOMMENDATIONS:\n{scene_requirements}\n\nThese are optional design recommendations. They do not override the original user requirements or make suggested extra objects mandatory.\n'
        current_action_section = ''
        if current_action_condition:
            current_action_section = f'\nCURRENT ACTION CONTEXT:\n{current_action_condition}\n\nThis describes the current object placement action being performed. Consider this context when evaluating the room layout.\n'
        prompt = f"""You are an expert interior designer. Analyze this room design for semantic correctness and provide actionable improvement suggestions.\n\nIMAGES PROVIDED:\n1. First image: Annotated top-down orthogonal view with object bounding boxes (blue), facing directions (yellow arrows), and coordinate axes\n\n2. Next 4 images: Four perspective rendered views from different angles\n   - Use these for visual realism, aesthetics, and overall room atmosphere assessment\n\n{user_demand_section}\n{scene_requirements_section}\n{current_action_section}\n\nThe explicit user requirements and current action constraints take precedence over\nall generic room-completeness or decoration recommendations below. Do not recommend\nadditional objects that exceed a stated count or extra-object limit. A deliberately\nrestrained room is not incomplete just because it is sparse. Unless the user explicitly restricts unrelated or decorative objects,\nrecommend a moderate selection of decoration or small everyday items for beauty and diversity.\nPlace small accessories on suitable tables, desks, shelves or cabinet tops; use wall-mounted decoration where appropriate, and put floor-standing decoration only in suitable free floor areas. Respect the specified supports and empty surfaces, keep working surfaces and circulation usable, and do not make the room look cluttered.\n\nROOM INFORMATION:\n{room_description}\n\nStep 1:\nROOM Analysis Summary: \nFirstly, based on the given images and room information, provide detailed reasoning on how to improve the room quality.\nYou need to analyze from the following aspects: realism, functionality, layout, and completion.\nProvide the detailed problem causes, and your suggestions for each aspect.\n\nStep 2:\nObject Addition Analysis (At most 5-8 object addition analysis recommendations; Propose the most important;): \nSecond, you need to think about whether we need to add more objects to the current room based on the analysis.\nYou need to first provide your detailed reasoning on whether we need to add more objects to the current room.\nYou need to ensure the completeness and diversity of the room, but also avoid being too crowded and cluttered.\n[IMPORTANT] You can stop proposing new floor or wall objects if the room is already crowded. sufficient space for walking is critical.\n\nIf you think we need to add more objects to the current room, then you need to propose the objects to add.\nConsider the addition in the following aspects:\n\nFirst aspect:\n- other background objects (including {background_objects_floor_and_wall_description} objects on top of the furniture surfaces placed at the same time)\nwhich are necessary for the completeness in the current room types.\n\nSecond aspect:\n- object combos with the current objects in the room \n(1. chair beside (or small decorated objects on top of) the table/desk, sofa in front of table, etc.)\n(2. Unless the user explicitly restricts unrelated or decorative objects, consider a moderate selection of decorative objects or small everyday items on appropriate supports.\nDo not require every shelf or surface to contain objects. Preserve explicitly empty surfaces and exact counts.\nUse the structured room information for object presence and IDs; an occluded object is not necessarily missing.\nNever request new views or regeneration merely because an object is not visually clear.\n\nNotes:\n- Recommend additions only when they improve the requested room and fit the user's limits.\n- A deliberately minimal room is not incomplete merely because it has empty space.\n- Avoid repetitive clutter, unsupported accessories and blocked circulation.\nOBJECT RESTRICTIONS:\nDO NOT recommend: rugs, mats, carpets, windows, doors, curtains, ceiling-hanging objects, ceiling objects (already installed)\nPlacement locations allowed: ONLY "floor", "wall", or exact object id in the list (e.g. table_001, sofa_002, etc.) (**on top of** the existing object ID in the room, use exact object id in the room, e.g. table_001, sofa_002, etc.) (no "ceiling" or other invalid locations allowed)\n\nStep 3:\nObject Adjustment Analysis (At most 1-2 object adjustment analysis recommendations; Propose the most important;):\nFourthly, you need to think about whether we need to adjust the existing objects in the current room based on the analysis.\nAdjust the room to ensure that:\n- objects are neatly placed, objects of the same category are well ailgned, relationships are reasonable (e.g., chairs face desks), sufficient space exists for walking, and **orientations must** be correct.\n- avoid the situation that: crowded floor, **abnormal size**, objects with collision, incorrect **orientation**, or large items placed oddly (e.g., sofa not against the wall). Large empty space. Blocker in front of furniture.\nYou are allowed to adjust the existing objects in the current room by the following operations:\nOPERATIONS AVAILABLE:\n- MOVE: Relocate and rotate existing object (specify object ID)\n- REMOVE: Remove problematic object (specify object ID)  \n- REPLACE: Replace object with new one (specify object ID)\nPriority GUIDE (0-10):\n- 9-10: Critical issues affecting safety, functionality, or severe spatial problems (e.g., missing essential furniture)\n- 7-8: Major issues affecting usability or comfort (e.g., missing extra objects which can greatly add diversity to the room (including objects on top of some furniture surfaces), missing important functional items)\n- 5-6: Moderate issues affecting aesthetics or minor functionality (e.g., missing some objects which can minimally add diversity to the room, suboptimal orientation and object placement, missing comfort items)\n- 3-4: Minor improvements for better layout (e.g., aesthetic enhancements, additional decorative items)\n- 1-2: Optional nice-to-have suggestions (e.g., minor decorative additions)\n- 0: No issue or lowest priority suggestion\nTOLORENCE:\n- Objects with reasonable sizes are not considered as issues, even they might be a little bit inconsistent.\n\nNow here is the format for json response return.\n\nREQUIRED JSON RESPONSE FORMAT:\n```json\n{{\n    "analysis_summary": {{\n        "detailed_reasoning": "Based on the scene current state, provide detailed reasoning on how to improve the room quality.",\n        "overall_room_rating": "good"\n    }},\n    "object_addition_analysis": {{\n        "detailed_reasoning": "Based on the room analysis summary, provide detailed reasoning on whether we need to add more objects to the current room.",\n        "object_combos_analysis": [\n            {{\n                "object_id": "exact_object_id_in_the_list",\n                "possible_object_combos": [\n                    {{\n                        "new_object_type": "new_object_type_1",\n                        "new_object_quantity": 1,\n                        "new_object_placement_location": "only one of the following: floor | wall | exact object id in the list (e.g. table_001, sofa_002, etc.)",\n                        "new_object_placement_guidance": "placement_guidance",\n                        "new_object_priority": 3\n                    }},\n                    {{\n                        "new_object_type": "new_object_type_2",\n                        "new_object_quantity": 1,\n                        "new_object_placement_location": "only one of the following: floor | wall | exact object id in the list (e.g. table_001, sofa_002, etc.)",\n                        "new_object_placement_guidance": "placement_guidance",\n                        "new_object_priority": 3\n                    }}\n                ],\n                "priority": 3\n            }}\n        ],\n        "background_objects_analysis": [\n            {{\n                "new_background_object_type": "new_background_object_type",\n                "new_background_object_quantity": 1,\n                "new_background_object_placement_location": "only one of the following: floor | wall | exact object id in the list (e.g. table_001, sofa_002, etc.)",\n                "new_background_object_placement_guidance": "placement_guidance",\n                "priority": 3\n            }}\n        ]\n    }},\n    "object_existing_analysis": [\n        {{\n            "object_id": "exact object id in the list (e.g. table_001, sofa_002, etc.)",\n            "object_type": "object_type",\n            "issues_found": [\n                {{\n                    "criteria": "placement|orientation",\n                    "score": 3,\n                    "issue_description": "Problem description",\n                    "suggested_operation": {{\n                        "type": "MOVE|REMOVE|REPLACE",\n                        "target_object_id": "object_id_if_applicable",\n                        "condition": "Positioning description",\n                        "reasoning": "Why this fixes the issue"\n                    }}\n                }}\n            ]\n        }}\n    ]\n}}\n```\n\nReturn valid JSON only, without comments or trailing commas. Quantities and priority scores must be integers. Empty lists are valid when there is no appropriate improvement; do not invent issues to fill the example.\n\nResponse Length Constraints:\nAt most 8-10 object addition analysis recommendations.\nAt most 1-2 object adjustment analysis recommendations.\n\n"""
        content = [{'type': 'text', 'text': prompt}]
        content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': top_down_image_base64}})
        for (i, image_base64) in enumerate(image_data_list):
            content.append({'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image_base64}})
        total_images = 1 + len(image_data_list)
        with timed_stage('critic.semantic.vlm', 'llm'):
            response = call_vlm(vlm_type='qwen', max_tokens=12000, temperature=0.4, messages=[{'role': 'user', 'content': content}])
        response_text = response.content[0].text.strip()
        try:
            response_text = extract_json_from_response(response_text)
            if not response_text:
                raise ValueError('Could not extract JSON content from Claude response')
            claude_analysis = json.loads(response_text)
        except json.JSONDecodeError as e:
            return json.dumps({'success': False, 'error': f"Failed to parse Claude's analysis response as JSON: {str(e)}", 'raw_response': response_text[:500] + '...' if len(response_text) > 500 else response_text, 'room_id': room_id})
        result = {'success': True, 'room_id': room_id, 'room_type': room.room_type, 'notice': 'These are suggestions, not mandatory actions. Follow the explicit user constraints and task feedback; discard conflicting suggestions. Stop when requirements are satisfied or the tool budget is reached. For additions use place_objects_in_room with room_id and a placement_conditions string; for moves use move_one_object_with_condition_in_room with room_id and a condition string.', 'next_step': {'actions': []}}
        modification_actions = []
        highest_priority_adjust = -1
        if 'object_existing_analysis' in claude_analysis:
            for obj_analysis in claude_analysis['object_existing_analysis']:
                obj_id = obj_analysis.get('object_id', 'unknown')
                obj_type = obj_analysis.get('object_type', 'unknown')
                for issue in obj_analysis.get('issues_found', []):
                    score = int(issue.get('score', 5))
                    issue_description = issue.get('issue_description', '')
                    suggested_operation = issue.get('suggested_operation', {})
                    op_type = suggested_operation.get('type', '')
                    target_id = suggested_operation.get('target_object_id', '')
                    condition = suggested_operation.get('condition', '')
                    suggestion_parts = []
                    if condition:
                        suggestion_parts.append(f'({condition})')
                    suggestion = ' '.join(suggestion_parts) if suggestion_parts else 'consider addressing this issue'
                    action_sentence = f'Priority: {score}; {issue_description} for object {obj_id} ({obj_type}). Suggestion: {suggestion}.'
                    recommendation = {'action': action_sentence}
                    if int(score) > int(highest_priority_adjust):
                        highest_priority_adjust = int(score)
                        modification_actions = [recommendation]
        combo_actions = []
        bg_actions = []
        highest_priority_add = -1
        if 'object_addition_analysis' in claude_analysis:
            addition_analysis = claude_analysis['object_addition_analysis']
            if 'object_combos_analysis' in addition_analysis:
                for combo_item in addition_analysis['object_combos_analysis']:
                    priority = int(combo_item.get('priority', 5))
                    highest_priority_add = max(highest_priority_add, priority)
                    combo_action = {}
                    combo_action_str = ''
                    if len(combo_item.get('possible_object_combos', [])) > 0:
                        for object_combo_dict in combo_item.get('possible_object_combos', []):
                            new_object_type = object_combo_dict.get('new_object_type', '')
                            new_object_quantity = object_combo_dict.get('new_object_quantity', '')
                            new_object_placement_location = object_combo_dict.get('new_object_placement_location', '')
                            new_object_placement_guidance = object_combo_dict.get('new_object_placement_guidance', '')
                            new_object_priority = object_combo_dict.get('new_object_priority', '')
                            combo_action_str += f'ADD {new_object_quantity} {new_object_type} on {new_object_placement_location} ({new_object_placement_guidance}); \n'
                    combo_action = {'action': combo_action_str}
                    combo_actions.append(combo_action)
            if 'background_objects_analysis' in addition_analysis:
                for bg_item in addition_analysis['background_objects_analysis']:
                    priority = int(bg_item.get('priority', 5))
                    highest_priority_add = max(highest_priority_add, priority)
                    new_background_object_type = bg_item.get('new_background_object_type', '')
                    new_background_object_quantity = bg_item.get('new_background_object_quantity', '')
                    new_background_object_placement_location = bg_item.get('new_background_object_placement_location', '')
                    new_background_object_placement_guidance = bg_item.get('new_background_object_placement_guidance', '')
                    priority = int(bg_item.get('priority', 5))
                    bg_action_str = f'ADD {new_background_object_quantity} {new_background_object_type} on {new_background_object_placement_location} ({new_background_object_placement_guidance}); '
                    bg_action = {'action': bg_action_str}
                    bg_actions.append(bg_action)
        combo_actions.sort(key=lambda x: x.get('priority', 0), reverse=True)
        bg_actions.sort(key=lambda x: x.get('priority', 0), reverse=True)
        combo_actions = combo_actions
        bg_actions = bg_actions
        object_addition_actions = combo_actions + bg_actions
        object_addition_actions = {'actions': 'Priority: ' + str(highest_priority_add) + '; ' + ' '.join([action['action'] for action in object_addition_actions])}
        top_modifications = modification_actions[:max_recommendations] if max_recommendations > 0 else modification_actions
        if propose_modifications:
            if int(highest_priority_adjust) > int(highest_priority_add):
                result['next_step']['actions'] = top_modifications
            else:
                result['next_step']['actions'] = [object_addition_actions]
        else:
            result['next_step']['actions'] = object_addition_actions
        return json.dumps(result, indent=2)
    except ImportError as e:
        return json.dumps({'success': False, 'error': f'Missing required modules for room rendering: {str(e)}. Make sure nvdiffrast_rendering modules are available.', 'room_id': room_id})
    except anthropic.APIError as e:
        return json.dumps({'success': False, 'error': f'Anthropic API error during analysis: {str(e)}', 'room_id': room_id})
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Unexpected error in room semantic analysis: {str(e)}', 'room_id': room_id})

async def room_physics_critic(room_id: str):
    """
    Criticize the physics of the room.
    """
    global current_layout
    output_path = str(Path(RESULTS_DIR) / f'{current_layout.id}')
    target_room = next((r for r in current_layout.rooms if r.id == room_id), None)
    if target_room is None:
        return {'success': False, 'error': f"Room with ID '{room_id}' not found"}
    room_objects = target_room.objects
    num_objects = len(room_objects)
    ceiling_height = target_room.ceiling_height
    room_objects = [obj for obj in room_objects if obj.position.z + obj.dimensions.height <= ceiling_height]
    print(f'removing over-ceiling objects: {num_objects} -> {len(room_objects)}', file=sys.stderr)
    for room in current_layout.rooms:
        if room.id == room_id:
            room.objects = room_objects
    with timed_stage('critic.physics.export', 'io'):
        export_layout_to_json(current_layout, os.path.join(output_path, f'{current_layout.id}.json'))
    with timed_stage('critic.physics.simulation', 'physics'):
        result = validate_room_stability(output_path, room_id)
    if not isinstance(result, dict):
        return json.dumps(result)
    unstable_ids = [str(object_id) for object_id in result.get('unstable_objects') or []]
    (removed_objects, remaining_objects) = _remove_objects_and_descendants(target_room.objects, unstable_ids)
    result['detected_unstable_object_ids'] = unstable_ids
    if removed_objects:
        target_room.objects = remaining_objects
        removed_ids = [obj.id for obj in removed_objects]
        result['removed_unstable_object_ids'] = removed_ids
        removed_id_set = set(removed_ids)
        unresolved_ids = [object_id for object_id in unstable_ids if object_id not in removed_id_set]
        result['unstable_objects'] = unresolved_ids
        if not unresolved_ids:
            result['next_step'] = 'The detected unstable objects and their descendants were removed from the persisted room layout.'
        print(f'Physics critic removed {len(removed_ids)} unstable objects/descendants: {removed_ids}', file=sys.stderr)
        with timed_stage('critic.physics.persist_removals', 'io'):
            export_layout_to_json(current_layout, os.path.join(output_path, f'{current_layout.id}.json'))
    else:
        result['removed_unstable_object_ids'] = []
    return json.dumps(result)

if __name__ == '__main__':

    mcp.run(transport='stdio')
