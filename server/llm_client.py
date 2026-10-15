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
import anthropic
from typing import Dict, Any, List
from key import ANTHROPIC_API_KEY
from vlm import call_vlm
from validation import validate_room_layout, validate_llm_response_structure, validate_room_only_layout
import base64
import os
import time
import sys
from constants import SERVER_ROOT_DIR
from utils import extract_json_from_response
from datetime import datetime
import asyncio
import networkx as nx
import random
import numpy as np
from room_solver import RectangleContactSolver, RectangleContactRelaxationSolver, RectangleSpec, RectangleLayout

def convert_vertices_to_position_dimensions(layout_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert vertices format back to position + dimensions format.
    
    Input format:
    {
        "rooms": [
            {
                "room_type": "living room",
                "vertices": [{"x": 0, "y": 0}, {"x": 0, "y": 6}, {"x": 4.5, "y": 6}, {"x": 4.5, "y": 0}],
                "height": 2.7
            }
        ]
    }
    
    Output format:
    {
        "rooms": [
            {
                "room_type": "living room", 
                "dimensions": {"width": 4.5, "length": 6, "height": 2.7},
                "position": {"x": 0, "y": 0, "z": 0}
            }
        ]
    }
    """
    if 'rooms' not in layout_data:
        return layout_data
    converted_rooms = []
    for room in layout_data['rooms']:
        if 'vertices' in room and 'height' in room:
            vertices = room['vertices']
            if len(vertices) != 4:
                raise ValueError(f"Room {room.get('room_type', 'unknown')} must have exactly 4 vertices")
            x_coords = [v['x'] for v in vertices]
            y_coords = [v['y'] for v in vertices]
            (min_x, max_x) = (min(x_coords), max(x_coords))
            (min_y, max_y) = (min(y_coords), max(y_coords))
            width = max_x - min_x
            length = max_y - min_y
            height = room['height']
            converted_room = {'room_type': room['room_type'], 'dimensions': {'width': width, 'length': length, 'height': height}, 'position': {'x': min_x, 'y': min_y, 'z': 0}}
            converted_rooms.append(converted_room)
        else:
            converted_rooms.append(room)
    result = layout_data.copy()
    result['rooms'] = converted_rooms
    return result

async def call_llm_for_layout(input_text: str) -> Dict[str, Any]:
    """Call Claude Sonnet 4 to generate room layout from text description."""
    prompt = f'''You are an expert architect and interior designer. Analyze this description and generate an appropriate layout: "{input_text}"\n\nSTEP 1: ANALYZE REQUEST TYPE\nFirst determine if the user is asking for:\n- SINGLE ROOM: Only one specific room (e.g., "design a living room", "create a bedroom", "generate a kitchen")\n- MULTIPLE ROOMS: A complete layout, house, apartment, or multiple connected rooms\n\nSTEP 2: GENERATE APPROPRIATE LAYOUT\nFor SINGLE ROOM requests:\n- Generate just that one room with appropriate dimensions\n- Include doors and windows suitable for that room type\n- Position the room at coordinates (0, 0, 0)\n- Use realistic proportions for the specific room type\n- Consider the room's intended function and furniture layout\n- Example single room dimensions: living room (4.5×6m), bedroom (3.5×4m), kitchen (3×4m)\n\nFor MULTIPLE ROOMS requests:\n- Create a functional, well-designed layout that considers:\n  - Architectural best practices and building codes\n  - Natural light and ventilation\n  - Traffic flow and accessibility\n  - Room adjacencies and relationships\n  - Realistic proportions and dimensions\n\nEXAMPLES OF REQUEST TYPES:\nSINGLE ROOM: "design a master bedroom", "create a modern kitchen", "living room with fireplace"\nMULTIPLE ROOMS: "2 bedroom apartment", "small house", "office layout", "3 bedroom home"\n\nRespond with ONLY a valid JSON object using this exact structure:\n\n```json\n{{\n    "building_style": "description of architectural style (e.g., Modern residential, Traditional colonial, etc.)",\n    "rooms": [\n        {{\n            "room_type": "specific room type (e.g., living room, master bedroom, kitchen, etc.)",\n            "dimensions": {{"width": float, "length": float, "height": float}},\n            "position": {{"x": float, "y": float, "z": 0}},\n            "doors": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "door_type": "standard/sliding/french/pocket/etc."\n                }}\n            ],\n            "windows": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "sill_height": float,\n                    "window_type": "standard/bay/casement/picture/sliding/etc."\n                }}\n            ]\n        }}\n    ]\n}}\n```\n\nCRITICAL REQUIREMENTS:\n- For SINGLE ROOM: Include exactly 1 room in the "rooms" array\n- For MULTIPLE ROOMS: Include all rooms in the "rooms" array with proper layout\n- Use metric dimensions (meters)\n- Standard ceiling height: 2.7m (can vary for special rooms)\n- Door dimensions: height 2.0-2.1m, width 0.7-1.0m\n- Window sill heights: 0.9-1.1m typically\n- position_on_wall: 0.0-1.0 (fraction along wall length)\n- For single rooms: position at (0,0,0) and focus on room functionality\n- For multiple rooms: arrange logically without overlaps and ensure connectivity\n- Include appropriate doors for room access and flow\n- Place windows for natural light and views\n- Consider the specific requirements mentioned in the description\n\nGenerate a layout that truly reflects the input description with appropriate room types, sizes, and relationships.'''
    try:
        response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.3, messages=[{'role': 'user', 'content': prompt}])
        if not response:
            raise ValueError('No response received from Claude API')
        if not hasattr(response, 'content') or not response.content:
            raise ValueError('Claude API response has no content')
        if len(response.content) == 0:
            raise ValueError('Claude API response content is empty')
        if not hasattr(response.content[0], 'text'):
            raise ValueError('Claude API response content has no text attribute')
        response_text = response.content[0].text
        if not response_text or not response_text.strip():
            raise ValueError('Claude API returned empty response text')
        response_text = response_text.strip()
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        layout_data = json.loads(response_text)
        return layout_data
    except anthropic.APIError as e:
        raise ValueError(f'Anthropic API error: {e}')
    except Exception as e:
        raise ValueError(f'Error calling Claude API: {e}')

async def call_llm_for_layout_with_validation(input_text: str, max_attempts: int=1) -> Dict[str, Any]:
    """
    Call Claude to generate room layout with validation and iterative correction.
    
    Args:
        input_text: Original layout description
        max_attempts: Maximum number of correction attempts
        
    Returns:
        Validated layout data
    """
    attempt = 0
    last_response = None
    while attempt < max_attempts:
        try:
            if attempt == 0:
                response = await call_llm_for_layout(input_text)
            elif last_response is None:
                response = await call_llm_for_layout(input_text)
            else:
                validation_result = validate_room_layout(last_response['rooms'])
                correction_prompt = generate_correction_prompt(validation_result, input_text, last_response)
                claude_response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.2, messages=[{'role': 'user', 'content': correction_prompt}])
                if not claude_response:
                    raise ValueError('No response received from Claude API during correction')
                if not hasattr(claude_response, 'content') or not claude_response.content:
                    raise ValueError('Claude API correction response has no content')
                if len(claude_response.content) == 0:
                    raise ValueError('Claude API correction response content is empty')
                if not hasattr(claude_response.content[0], 'text'):
                    raise ValueError('Claude API correction response content has no text attribute')
                response_text = claude_response.content[0].text
                if not response_text or not response_text.strip():
                    raise ValueError('Claude API returned empty correction response text')
                response_text = response_text.strip()
                response_text = extract_json_from_response(response_text)
                if not response_text:
                    raise ValueError('Could not extract JSON content from Claude response')
                response = json.loads(response_text)
            if 'rooms' not in response or not response['rooms']:
                raise ValueError('Invalid response structure')
            validation_result = validate_room_layout(response['rooms'])
            if validation_result['valid']:
                return response
            else:
                last_response = response
                attempt += 1
                if attempt >= max_attempts:
                    response['validation_warning'] = {'message': f'Layout still has issues after {max_attempts} attempts', 'remaining_issues': validation_result['issues']}
                    return response
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise e
            last_response = None
    raise ValueError(f'Failed to generate valid layout after {max_attempts} attempts')

def generate_correction_prompt(validation_result: Dict[str, Any], original_input: str, previous_attempt: Dict[str, Any]) -> str:
    """Generate a correction prompt for Claude based on validation issues."""
    issues_text = ''
    if validation_result['overlaps']:
        overlaps_desc = []
        for overlap in validation_result['overlaps']:
            overlaps_desc.append(f"'{overlap['room1']}' and '{overlap['room2']}' (overlap area: {overlap['overlap_area']:.2f} m²)")
        issues_text += f"\n- OVERLAPPING ROOMS: {', '.join(overlaps_desc)}"
    if validation_result['detached_rooms']:
        issues_text += f"\n- DETACHED ROOMS: {', '.join(validation_result['detached_rooms'])} (these rooms have no shared walls with other rooms)"
    prompt = f'The previous floor plan layout had CRITICAL ARCHITECTURAL ERRORS that must be fixed:\n{issues_text}\n\nREQUIREMENTS FOR CORRECTION:\n1. NO ROOM OVERLAPS: Rooms must not overlap in space. Each room should have distinct boundaries.\n2. CONNECTED LAYOUT: All rooms must be connected through shared walls. No isolated/detached rooms.\n3. REALISTIC ADJACENCIES: Rooms should be logically connected (e.g., kitchen near dining, bedrooms in private areas).\n4. PROPER SPACING: Leave small gaps (0.1m) between non-adjacent rooms if needed.\n\nOriginal description: "{original_input}"\n\nPlease generate a CORRECTED floor plan that fixes these issues. Ensure:\n- Rooms are arranged in a logical, connected layout\n- No overlapping areas\n- All rooms share walls with at least one other room\n- Maintain the same room types and approximate sizes from the original request\n\nRespond with ONLY the JSON layout data in the same format as before.'
    return prompt

async def request_missing_information(original_response: Dict[str, Any], validation_result: Dict[str, Any], input_text: str) -> Dict[str, Any]:
    """
    Ask Claude to provide missing information in the response.
    """
    missing_info = ''
    if validation_result['missing_fields']:
        missing_info += f"\nMISSING FIELDS: {', '.join(validation_result['missing_fields'])}"
    if validation_result['invalid_fields']:
        missing_info += f"\nINVALID FIELDS: {', '.join(validation_result['invalid_fields'])}"
    correction_prompt = f'The previous JSON response was incomplete or had invalid fields:{missing_info}\n\nPlease provide a COMPLETE and CORRECTED JSON response for the layout: "{input_text}"\n\nThe response must include ALL required fields:\nOUTPUT FORMAT:\n```json\n{{\n    "building_style": "architectural style description",\n    "rooms": [\n        {{\n            "room_type": "room name",\n            "dimensions": {{"width": float, "length": float, "height": float}},\n            "position": {{"x": float, "y": float, "z": float}},\n            "doors": [\n                {{\n                    "width": float,\n                    "height": float, \n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "door_type": "standard"\n                }}\n            ],\n            "windows": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float, \n                    "wall_side": "north/south/east/west",\n                    "sill_height": float,\n                    "window_type": "standard"\n                }}\n            ]\n        }}\n    ]\n}}\n```\n\nCRITICAL: Ensure ALL fields are present and properly formatted. Include at least one door per room and appropriate windows.'
    try:
        response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.1, messages=[{'role': 'user', 'content': correction_prompt}])
        if not response or not hasattr(response, 'content') or (not response.content):
            raise ValueError('No valid response from Claude for missing information request')
        response_text = response.content[0].text.strip()
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        corrected_response = json.loads(response_text)
        return corrected_response
    except Exception as e:
        raise ValueError(f'Failed to get missing information from Claude: {str(e)}')

async def call_llm_for_rooms_only(input_text: str) -> Dict[str, Any]:
    """Call Claude Sonnet 4 to generate room layout WITHOUT doors and windows."""
    prompt = f'''You are an expert architect and interior designer. Analyze this description and generate an appropriate ROOM LAYOUT (rooms and walls only, NO doors or windows): "{input_text}"\n\n🚨 CRITICAL LAYOUT CONSTRAINTS 🚨\n- **ABSOLUTELY NO OVERLAPPING ROOMS**: Room boundaries must NEVER intersect or overlap\n- **NO ISOLATED ROOMS**: Every room must be adjacent to at least one other room for connectivity\n\nSTEP 1: ANALYZE REQUEST TYPE\nFirst determine if the user is asking for:\n- SINGLE ROOM: Only one specific room (e.g., "design a living room", "create a bedroom", "generate a kitchen")\n- MULTIPLE ROOMS: A complete layout, house, apartment, or multiple connected rooms\n\nSTEP 2: GENERATE ROOM STRUCTURE ONLY\nFor SINGLE ROOM requests:\n- Generate just that one room with appropriate rectangular vertices\n- Position the room starting at coordinates (0, 0) as the first vertex\n- Use realistic proportions for the specific room type\n- Consider the room's intended function and furniture layout\n- Example single room vertices: living room [(0,0), (0,6), (4.5,6), (4.5,0)], bedroom [(0,0), (0,4), (3.5,4), (3.5,0)]\n- Vertices must form a perfect rectangle with 90-degree corners\n\nFor MULTIPLE ROOMS requests:\n- **MANDATORY**: Ensure NO ROOM OVERLAPS and NO ISOLATED ROOMS using vertex-based checking\n- **VERTEX CONSTRAINT**: When placing each new room, ensure NO vertex of the new room is inside any previously generated room's boundary\n- Create a functional, well-designed layout that considers:\n  - Architectural best practices and building codes\n  - Natural light and ventilation\n  - Traffic flow and accessibility\n  - Room adjacencies and relationships (every room must touch at least one other room)\n  - Realistic proportions using rectangular vertices\n  - **VERIFY**: No room vertices are inside other rooms' boundaries\n  - **VERIFY**: Every room is positioned adjacent to at least one other room\n  - **VERIFY**: All rooms use perfect rectangular vertices with 90-degree corners\n\nCRITICAL: Generate ONLY the room structure (rectangular vertices, height) - NO doors or windows at this stage.\n\nRespond with ONLY a valid JSON object using this exact structure:\n\nOUTPUT FORMAT:\n```json\n{{\n    "building_style": "description of architectural style (e.g., Modern residential, Traditional colonial, etc.)",\n    "rooms": [\n        {{\n            "room_type": "specific room type (e.g., living room, master bedroom, kitchen, etc.)",\n            "vertices": [\n                {{"x": float, "y": float}},\n                {{"x": float, "y": float}},\n                {{"x": float, "y": float}},\n                {{"x": float, "y": float}}\n            ],\n            "height": float\n        }}\n    ]\n}}\n```\n\nCRITICAL REQUIREMENTS:\n- For SINGLE ROOM: Include exactly 1 room in the "rooms" array\n- For MULTIPLE ROOMS: Include all rooms in the "rooms" array with proper layout\n- Use metric coordinates (meters) for all vertices\n- Standard ceiling height: 2.7m (can vary for special rooms)  \n- For single rooms: start vertices at (0,0) and focus on room functionality\n- For multiple rooms: arrange rectangular vertices logically without overlaps and ensure rooms can be connected later\n- **VERTICES FORMAT**: Each room must have exactly 4 vertices forming a perfect rectangle\n- **VERTEX ORDERING**: List vertices in clockwise or counter-clockwise order around the rectangle\n- **ABSOLUTELY NO OVERLAPPING ROOMS**: Ensure no room vertices are inside other rooms and no boundaries intersect\n- **NO ISOLATED ROOMS**: Every room must be positioned so it can be connected to at least one other room via doors (adjacency is essential)\n- **VERTEX-BASED CHECKING**: Verify each room's 4 vertices do not fall inside any other room's boundary\n- 🚨 DOUBLE-CHECK: Before finalizing, verify NO vertex overlaps and NO isolated rooms 🚨\n- DO NOT include doors or windows arrays - these will be added in a separate step\n- Focus on creating a solid architectural foundation with proper rectangular room vertices and relationships\n\nGenerate a room-only layout that truly reflects the input description with appropriate room types, rectangular vertices, and spatial relationships.\n\n🚨 FINAL REMINDER - THESE ARE MANDATORY 🚨\n- **NO OVERLAPPING ROOMS**: Check that no room vertices are inside other rooms and no boundaries intersect\n- **NO ISOLATED ROOMS**: Verify every room can connect to at least one other room\n- **PERFECT RECTANGLES**: Every room must have exactly 4 vertices forming a rectangle with 90-degree corners'''
    try:
        response = call_vlm(vlm_type='openai', max_tokens=16000, temperature=1.0, thinking=True, messages=[{'role': 'user', 'content': prompt}])
        if not response:
            raise ValueError('No response received from Claude API')
        if not hasattr(response, 'content') or not response.content:
            raise ValueError('Claude API response has no content')
        if len(response.content) == 0:
            raise ValueError('Claude API response content is empty')
        response_text = None
        for content in response.content:
            if hasattr(content, 'text'):
                response_text = content.text
                break
        if response_text is None:
            raise ValueError('Claude API response content has no text attribute')
        if not response_text or not response_text.strip():
            raise ValueError('Claude API returned empty response text')
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        layout_data = json.loads(response_text)
        debug_vis_dir = f'{SERVER_ROOT_DIR}/vis'
        os.makedirs(debug_vis_dir, exist_ok=True)
        debug_vis_path = os.path.join(debug_vis_dir, f"room_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        await get_vlm_room_layout_2d_visualization(layout_data, debug_vis_path)
        print(f'🔍 Debug: Room layout visualization saved to {debug_vis_path}', file=sys.stderr)
        layout_data = convert_vertices_to_position_dimensions(layout_data)
        return layout_data
    except anthropic.APIError as e:
        raise ValueError(f'Anthropic API error: {e}')
    except Exception as e:
        raise ValueError(f'Error calling Claude API: {e}')

async def get_vlm_room_layout_2d_visualization(layout_data: Dict[str, Any], debug_vis_path: str) -> None:
    """
    Get a 2D visualization of the room layout using matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Polygon
        import numpy as np
    except ImportError:
        print('Warning: matplotlib not available, skipping visualization', file=sys.stderr)
        return
    (fig, ax) = plt.subplots(1, 1, figsize=(12, 10))
    if 'rooms' not in layout_data or not layout_data['rooms']:
        ax.text(0.5, 0.5, 'No rooms found', ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig(debug_vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray', 'lightcyan', 'wheat', 'lavender', 'mistyrose', 'peachpuff', 'lightsteelblue', 'thistle', 'honeydew', 'azure']
    all_x = []
    all_y = []
    legend_elements = []
    room_vertices = set()
    for (i, room) in enumerate(layout_data['rooms']):
        room_type = room.get('room_type', f'Room {i + 1}')
        if 'vertices' in room:
            vertices = room['vertices']
            if len(vertices) != 4:
                print(f'Warning: Room {room_type} has {len(vertices)} vertices, expected 4', file=sys.stderr)
                continue
            x_coords = [v['x'] for v in vertices]
            y_coords = [v['y'] for v in vertices]
            for v in vertices:
                room_vertices.add((v['x'], v['y']))
        elif 'position' in room and 'dimensions' in room:
            pos = room['position']
            dim = room['dimensions']
            x = pos['x']
            y = pos['y']
            width = dim['width']
            length = dim['length']
            x_coords = [x, x, x + width, x + width]
            y_coords = [y, y + length, y + length, y]
            room_vertices.add((x, y))
            room_vertices.add((x, y + length))
            room_vertices.add((x + width, y + length))
            room_vertices.add((x + width, y))
        else:
            print(f'Warning: Room {room_type} has neither vertices nor position+dimensions', file=sys.stderr)
            continue
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        polygon_coords = list(zip(x_coords, y_coords))
        color = colors[i % len(colors)]
        polygon = Polygon(polygon_coords, closed=True, facecolor=color, edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(polygon)
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        display_name = room_type.replace('_', ' ').title()
        ax.text(center_x, center_y, display_name, ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        height = room.get('height', 'N/A')
        ax.text(center_x, center_y - 0.3, f'H: {height}m', ha='center', va='center', fontsize=7, style='italic', bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.6))
        legend_elements.append(patches.Patch(color=color, label=display_name))
    if room_vertices:
        vertex_x = [v[0] for v in room_vertices]
        vertex_y = [v[1] for v in room_vertices]
        ax.scatter(vertex_x, vertex_y, color='red', s=50, zorder=10, alpha=0.8, edgecolors='darkred', linewidth=1, label='Vertices')
        for (x, y) in room_vertices:
            ax.annotate(f'({x:.1f},{y:.1f})', xy=(x, y), xytext=(3, 3), textcoords='offset points', fontsize=7, color='darkred', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='red'))
        legend_elements.append(patches.Patch(color='red', label=f'Vertices ({len(room_vertices)})'))
    if all_x and all_y:
        margin = max(2.0, (max(all_x) - min(all_x)) * 0.1)
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax.set_xlim(-1, 10)
        ax.set_ylim(-1, 10)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlabel('X Coordinate (meters)', fontsize=12)
    ax.set_ylabel('Y Coordinate (meters)', fontsize=12)
    building_style = layout_data.get('building_style', 'Unknown Style')
    ax.set_title(f"Room Layout Visualization\nBuilding Style: {building_style}\nTotal Rooms: {len(layout_data['rooms'])}", fontsize=14, fontweight='bold', pad=20)
    if legend_elements:
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5), title='Room Types', title_fontsize=12, fontsize=10)
    if all_x and all_y:
        ax.annotate(f'({min(all_x):.1f}, {min(all_y):.1f})', xy=(min(all_x), min(all_y)), xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)
        ax.annotate(f'({max(all_x):.1f}, {max(all_y):.1f})', xy=(max(all_x), max(all_y)), xytext=(-5, -5), textcoords='offset points', fontsize=8, alpha=0.7, ha='right', va='top')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(debug_vis_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'✅ Room layout visualization saved successfully to {debug_vis_path}', file=sys.stderr)
    except Exception as e:
        print(f'❌ Error saving visualization: {e}', file=sys.stderr)
    finally:
        plt.close()

async def call_llm_for_doors_windows(rooms_data: List[Dict[str, Any]], input_text: str) -> Dict[str, Any]:
    """
    Call Claude Sonnet 4 to add doors and windows to an existing room layout using shared walls analysis.
    
    This function:
    1. Analyzes shared walls between rooms and with exterior
    2. Generates a floor plan visualization
    3. Sends both image and data to Claude
    4. Asks Claude for specific door/window placement
    5. Transforms response back to original format
    """
    debug_info = {'prompts': [], 'responses': [], 'steps': []}
    from utils import find_all_shared_walls, generate_floor_plan_visualization, calculate_room_wall_position_from_shared_wall
    shared_walls_info = find_all_shared_walls(rooms_data)
    try:
        visualization_path = generate_floor_plan_visualization(rooms_data)
        with open(visualization_path, 'rb') as image_file:
            image_data = image_file.read()
        debug_vis_dir = f'{SERVER_ROOT_DIR}/vis'
        os.makedirs(debug_vis_dir, exist_ok=True)
        timestamp = int(time.time())
        debug_vis_path = os.path.join(debug_vis_dir, f'floor_plan_{timestamp}.png')
        with open(debug_vis_path, 'wb') as debug_file:
            debug_file.write(image_data)
        print(f'🔍 Debug: Floor plan visualization saved to {debug_vis_path}', file=sys.stderr)
        os.unlink(visualization_path)
    except Exception as e:
        raise ValueError(f'Failed to generate floor plan visualization: {str(e)}')
    rooms_summary = []
    for (i, room) in enumerate(rooms_data):
        room_summary = {'room_index': i, 'room_type': room['room_type'], 'dimensions': f"{room['dimensions']['width']}m × {room['dimensions']['length']}m × {room['dimensions']['height']}m", 'position': f"({room['position']['x']}, {room['position']['y']}, {room['position']['z']})", 'area': f"{room['dimensions']['width'] * room['dimensions']['length']:.1f} sq m"}
        rooms_summary.append(room_summary)
    room_room_walls_summary = []
    for (i, wall) in enumerate(shared_walls_info['room_room_walls']):
        wall_summary = {'shared_wall_index': i, 'room1': f"Room {wall['room1']['index']} ({wall['room1']['room_type']})", 'room2': f"Room {wall['room2']['index']} ({wall['room2']['room_type']})", 'room1_wall_side': wall['room1_wall'], 'room2_wall_side': wall['room2_wall'], 'shared_length': f"{wall['overlap_length']:.2f}m", 'coordinates': f"({wall['x_start']:.1f}, {wall['y_start']:.1f}) to ({wall['x_end']:.1f}, {wall['y_end']:.1f})"}
        room_room_walls_summary.append(wall_summary)
    room_exterior_walls_summary = []
    for (i, wall) in enumerate(shared_walls_info['room_exterior_walls']):
        wall_summary = {'exterior_wall_index': i, 'room': f"Room {wall['room']['index']} ({wall['room']['room_type']})", 'wall_side': wall['wall_side'], 'wall_length': f"{wall['overlap_length']:.2f}m", 'coordinates': f"({wall['x_start']:.1f}, {wall['y_start']:.1f}) to ({wall['x_end']:.1f}, {wall['y_end']:.1f})"}
        room_exterior_walls_summary.append(wall_summary)
    debug_info['steps'].append('Step 4.5: Generating traffic flow analysis')
    traffic_flow_prompt = f'''You are an expert architect analyzing a floor plan to determine optimal traffic flow patterns for door placement.\n\nI'm providing you with:\n1. A floor plan visualization showing the room layout\n2. Detailed room information\n3. Shared wall analysis between rooms\n4. Exterior wall segments\n\nORIGINAL REQUEST: "{input_text}"\n\nROOM LAYOUT:\n{json.dumps(rooms_summary, indent=2)}\n\nSHARED WALLS BETWEEN ROOMS ({len(room_room_walls_summary)} total):\n{json.dumps(room_room_walls_summary, indent=2)}\n\nEXTERIOR WALLS ({len(room_exterior_walls_summary)} total):\n{json.dumps(room_exterior_walls_summary, indent=2)}\n\nTASK: Analyze the floor plan and determine the optimal traffic flow strategy for this layout.\n\nPlease provide:\n\n1. ENTRY STRATEGY:\n   - Which room should have the main entrance door?\n   - Which exterior wall segment is best for the entry door?\n   - Reasoning for entry door placement\n\n2. TRAFFIC FLOW GRAPH:\n   - How should people move through the space from the entry?\n   - Which rooms need direct connections to which other rooms?\n   - What is the circulation hierarchy (main paths vs secondary paths)?\n   - Which shared walls should have connecting doors?\n\n3. CONNECTIVITY REQUIREMENTS:\n   - List all room-to-room connections needed for good traffic flow\n   - Prioritize connections (essential vs optional)\n   - Consider both functional needs and user convenience\n\nFORMAT your response as JSON:\n\n```json\n{{\n    "entry_strategy": {{\n        "entry_room_index": int,\n        "preferred_exterior_wall_index": int,\n        "reasoning": "Why this entry location is optimal"\n    }},\n    "traffic_flow": {{\n        "main_circulation_path": ["room_index_sequence"],\n        "secondary_paths": [["room_index_sequence"]],\n        "circulation_reasoning": "Explanation of flow strategy"\n    }},\n    "required_connections": [\n        {{\n            "shared_wall_index": int,\n            "priority": "essential/important/optional",\n            "reasoning": "Why this connection is needed"\n        }}\n    ]\n}}\n```\n\nFocus on creating an efficient, logical flow that serves the original request: "{input_text}"\n'''
    try:
        debug_info['prompts'].append({'step': '4.5_traffic_flow', 'prompt': traffic_flow_prompt})
        traffic_response = call_vlm(vlm_type='qwen', max_tokens=8000, temperature=0.2, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': traffic_flow_prompt}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64.b64encode(image_data).decode('utf-8')}}]}])
        traffic_response_text = traffic_response.content[0].text.strip()
        debug_info['responses'].append({'step': '4.5_traffic_flow', 'response': traffic_response_text})
        try:
            traffic_flow_data = extract_json_from_response(traffic_response_text)
            if not traffic_flow_data:
                raise ValueError('Could not extract JSON content from Claude response')
            traffic_flow_data = json.loads(traffic_flow_data)
        except json.JSONDecodeError:
            traffic_flow_data = {}
        debug_info['steps'].append('Step 4.5 completed: Traffic flow analysis generated')
        print(f'🔍 Debug: Traffic flow analysis completed', file=sys.stderr)
    except Exception as e:
        traffic_flow_data = {}
        debug_info['steps'].append(f'Step 4.5 failed: {str(e)}')
        print(f'⚠️ Debug: Traffic flow analysis failed: {e}')
    traffic_flow_section = ''
    if traffic_flow_data:
        traffic_flow_section = f'\n\nTRAFFIC FLOW ANALYSIS:\n{json.dumps(traffic_flow_data, indent=2)}\n\nIMPORTANT: Use the traffic flow analysis above to guide your door placement decisions. Pay special attention to:\n- The recommended entry strategy\n- Required connections for good circulation\n- Priority levels of different connections\n'
    prompt = f'''You are an expert architect analyzing a floor plan to add doors and windows. I'm providing you with:\n1. A floor plan visualization image\n2. Room layout data\n3. Shared wall analysis\n4. Traffic flow analysis for optimal circulation\n\nORIGINAL REQUEST: "{input_text}"\n\nROOM LAYOUT:\n{json.dumps(rooms_summary, indent=2)}\n\nSHARED WALLS BETWEEN ROOMS ({len(room_room_walls_summary)} total):\n{json.dumps(room_room_walls_summary, indent=2)}\n\nEXTERIOR WALLS ({len(room_exterior_walls_summary)} total):\n{json.dumps(room_exterior_walls_summary, indent=2)}{traffic_flow_section}\n\nCOORDINATE SYSTEM:\n- X-axis: horizontal (left-right)\n- Y-axis: vertical (bottom-top)\n- Wall sides: north (top), south (bottom), east (right), west (left)\n\nYOUR TASK:\nBased on the floor plan image, data, and traffic flow analysis, provide door and window placement recommendations following these rules:\n\n1. ENTRY DOOR (exactly one for the entire floor plan):\n   - Must be placed on an exterior wall (room-exterior shared wall)\n   - Follow the traffic flow analysis recommendations if available\n   - Predict appropriate entry door material/style that matches the building aesthetic\n   - Specify: room_index, wall_side, center_position (0.0-1.0), width, height, door_material (brief description)\n\n2. CONNECTING DOORS (for room-room shared walls):\n   - Place doors to connect rooms according to traffic flow requirements\n   - Prioritize essential connections over optional ones\n   - Only place on room-room shared walls (between two rooms)\n   - Choose between physical doors or wide openings based on room function:\n     * Use physical doors (opening=false) for: bedrooms, bathrooms, private offices, storage rooms\n     * Use wide openings (opening=true) for: living room-kitchen, living room-dining, open-plan concepts\n   - Predict appropriate door material/style based on room types and design aesthetics\n   - Specify: shared_wall_index, center_position_on_shared_wall (0.0-1.0), width, height, opening (true/false), door_material (brief description)\n\n3. WINDOWS (only on exterior walls):\n   - Place on room-exterior shared walls only\n   - Consider natural light, ventilation, and room function\n   - Specify: exterior_wall_index, center_position (0.0-1.0), width, height, sill_height\n\nFORMATTING:\nReturn your response as a JSON object with this exact structure:\n\n```json\n{{\n    "entry_door": {{\n        "room_index": int,\n        "wall_side": "north/south/east/west",\n        "center_position": float_0_to_1,\n        "width": float_meters,\n        "height": float_meters,\n        "door_type": "entry/main/front",\n        "door_material": "Brief description of door style/material (e.g., solid wood entry door, glass panel door, traditional front door)",\n        "reasoning": "Why this location was chosen"\n    }},\n    "connecting_doors": [\n        {{\n            "shared_wall_index": int,\n            "center_position_on_shared_wall": float_0_to_1,\n            "width": float_meters,\n            "height": float_meters,\n            "door_type": "interior/standard",\n            "opening": bool,\n            "door_material": "Brief description of door style/material (e.g., wooden panel door, glass door, modern steel door)",\n            "reasoning": "Why this connection is needed and why door/opening was chosen"\n        }}\n    ],\n    "windows": [\n        {{\n            "exterior_wall_index": int,\n            "center_position": float_0_to_1,\n            "width": float_meters,\n            "height": float_meters,\n            "sill_height": float_meters,\n            "window_type": "standard",\n            "window_grid": [nx, ny],\n            "window_appearance_description": "string",\n            "glass_color": [r, g, b],\n            "frame_color": [r, g, b],\n            "reasoning": "Why this window placement was chosen"\n        }}\n    ]\n}}\n```\n\nIMPORTANT COLOR SELECTION GUIDANCE:\n- Analyze the building style and room function before choosing colors\n- For TRADITIONAL/CLASSIC buildings: Use warmer, earth-tone colors with natural wood frames\n- For MODERN/CONTEMPORARY buildings: Use cooler, neutral colors with metal or painted frames\n- For INDUSTRIAL buildings: Use darker, muted colors with metal frames\n- Match glass tint to room privacy needs: bedrooms (subtle tints), bathrooms (privacy glass), living areas (clear)\n- Ensure frame colors complement the overall architectural aesthetic\n- Avoid repetitive color choices - vary colors appropriately across different rooms and window types\n\nCONSTRAINTS:\n- Entry door: width 0.8-1.2m, height 2.0-2.1m\n- Interior doors: width 0.7-0.9m, height 2.0-2.1m  \n- Wide openings: width 1.2-3.0m, height 2.0-2.4m (for open-plan connections)\n- Windows: width 0.8-2.0m, height 1.0-1.8m, sill height 0.8-1.2m\n- Window grid [nx, ny]: Define window pane layout (nx=horizontal panes, ny=vertical panes), examples:\n  * Fixed/Picture: [1, 1] - single large pane\n  * Slider: [2, 1] - two horizontal sliding panes\n  * Hung: [1, 2] - two vertical stacked panes\n  * Bay: [2, 2] to [3, 3] - multiple panes in grid\n  * Casement: [1, 1] - single hinged pane\n- Window appearance description: Brief description combining frame style and glass type, examples:\n  * "classic white painted wooden window frame with frosted glass"\n  * "modern black aluminum window frame with translucent glass panels"\n  * "classic white window frame with frosted glass panels"\n  * "classic natural wood grain window frame with frosted glass panes"\n- Window colors: Generate contextually appropriate glass_color and frame_color as RGB values [0-255] based on:\n  * BUILDING STYLE: Traditional/classic → warmer tones, Modern → cooler/neutral tones, Industrial → darker frames\n  * ROOM FUNCTION: Bedrooms → softer tints, Kitchens → clear/bright, Bathrooms → privacy (frosted/tinted)\n  * AESTHETIC HARMONY: Match building style and room character with meaningful color choices\n  \n  GLASS COLOR EXAMPLES (choose based on context):\n  * Clear/Neutral: [245, 247, 250] (modern clear), [252, 251, 244] (warm clear)\n  * Subtle Tints: [240, 248, 255] (alice blue), [176, 224, 230] (powder blue), [173, 216, 230] (light blue)\n  * Privacy Glass: [199, 227, 225] (frosted blue glass), [184, 227, 233] (frosted powder blue), [192, 230, 235] (frosted light blue)\n  * Colored Glass: [135, 206, 250] (sky blue), [166, 227, 224] (ice crystal blue), [200, 233, 238] (pale blue)\n  \n  FRAME COLOR EXAMPLES (choose based on style):\n  * Natural Wood: [49, 36, 26] (dark walnut), [139, 97, 69] (dark oak), [60, 34, 24] (espresso)\n  * Painted Wood: [252, 251, 244] (cream white), [64, 89, 71] (forest green), [71, 82, 107] (navy blue)\n  * Metal Frames: [54, 69, 79] (charcoal), [191, 191, 191] (brushed aluminum), [115, 107, 97] (bronze)\n  * Modern Colors: [89, 97, 107] (graphite), [255, 253, 228] (creamy white), [92, 62, 41] (antique oak)\n\n- Keep doors/windows away from corners (position 0.1-0.9 range)\n- Ensure no overlapping doors/windows on the same wall segment\n- Consider furniture placement and room circulation\n\nFocus on creating a functional and well-lit layout that serves the original request: "{input_text}"\n'''
    debug_info['prompts'].append({'step': '5_main_door_window_placement', 'prompt': prompt})
    try:
        response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.2, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64.b64encode(image_data).decode('utf-8')}}]}])
        if not response or not hasattr(response, 'content') or (not response.content):
            raise ValueError('No response received from Claude API')
        if len(response.content) == 0 or not hasattr(response.content[0], 'text'):
            raise ValueError('Claude API response content is empty or invalid')
        response_text = response.content[0].text
        debug_info['responses'].append({'step': '5_main_door_window_placement', 'response': response_text})
        if not response_text or not response_text.strip():
            raise ValueError('Claude API returned empty response text')
        try:
            claude_response = extract_json_from_response(response_text)
            if not claude_response:
                raise ValueError('Could not extract JSON content from Claude response')
            claude_response = json.loads(claude_response)
        except json.JSONDecodeError as e:
            raise ValueError(f'Could not parse JSON from Claude response: {e}')
        result = {'building_style': 'same as before', 'rooms': []}
        for (i, room_data) in enumerate(rooms_data):
            room_result = {'room_type': room_data['room_type'], 'dimensions': room_data['dimensions'], 'position': room_data['position'], 'doors': [], 'windows': []}
            result['rooms'].append(room_result)
        if 'entry_door' in claude_response and claude_response['entry_door']:
            entry_door = claude_response['entry_door']
            room_index = entry_door['room_index']
            if 0 <= room_index < len(result['rooms']):
                door_info = {'width': entry_door['width'], 'height': entry_door['height'], 'position_on_wall': entry_door['center_position'], 'wall_side': entry_door['wall_side'], 'door_type': entry_door.get('door_type', 'entry'), 'door_material': entry_door.get('door_material', 'standard wooden door')}
                original_width = door_info['width']
                original_position = door_info['position_on_wall']
                if entry_door['wall_side'] in ['north', 'south']:
                    wall_length = rooms_data[room_index]['dimensions']['width']
                else:
                    wall_length = rooms_data[room_index]['dimensions']['length']
                max_width_constraint = wall_length * 0.8
                safe_start = 0.1
                safe_end = 0.9
                max_half_width_from_start = original_position - safe_start
                max_half_width_from_end = safe_end - original_position
                max_half_width = min(max_half_width_from_start, max_half_width_from_end)
                position_based_max_width = max_half_width * 2 * wall_length
                final_width = min(original_width, max_width_constraint, max(0.7, position_based_max_width))
                if final_width < original_width * 0.8:
                    desired_width = min(original_width, max_width_constraint)
                    required_half_width_ratio = desired_width / 2 / wall_length
                    min_center = safe_start + required_half_width_ratio
                    max_center = safe_end - required_half_width_ratio
                    if min_center <= max_center:
                        if original_position < min_center:
                            final_position = min_center
                        elif original_position > max_center:
                            final_position = max_center
                        else:
                            final_position = original_position
                        final_width = desired_width
                    else:
                        final_position = max(safe_start, min(safe_end, original_position))
                else:
                    final_position = max(safe_start, min(safe_end, original_position))
                door_info['width'] = final_width
                door_info['position_on_wall'] = final_position
                result['rooms'][room_index]['doors'].append(door_info)
        if 'connecting_doors' in claude_response and claude_response['connecting_doors']:
            for connecting_door in claude_response['connecting_doors']:
                shared_wall_index = connecting_door['shared_wall_index']
                if 0 <= shared_wall_index < len(shared_walls_info['room_room_walls']):
                    shared_wall = shared_walls_info['room_room_walls'][shared_wall_index]
                    room1_index = shared_wall['room1']['index']
                    room2_index = shared_wall['room2']['index']
                    original_door_width = connecting_door['width']
                    room1_max_width = rooms_data[room1_index]['dimensions']['width'] * 0.8 if shared_wall['room1_wall'] in ['north', 'south'] else rooms_data[room1_index]['dimensions']['length'] * 0.8
                    room2_max_width = rooms_data[room2_index]['dimensions']['width'] * 0.8 if shared_wall['room2_wall'] in ['north', 'south'] else rooms_data[room2_index]['dimensions']['length'] * 0.8
                    shared_wall_max_width = shared_wall['overlap_length'] * 0.8
                    final_door_width = min(original_door_width, room1_max_width, room2_max_width, shared_wall_max_width)
                    original_shared_position = connecting_door['center_position_on_shared_wall']
                    door_half_width_ratio = final_door_width / 2 / shared_wall['overlap_length']
                    min_safe_position = max(0.1, door_half_width_ratio + 0.05)
                    max_safe_position = min(0.9, 1.0 - door_half_width_ratio - 0.05)
                    final_shared_position = max(min_safe_position, min(max_safe_position, original_shared_position))
                    room1_wall_position = calculate_room_wall_position_from_shared_wall(shared_wall, final_shared_position, rooms_data[room1_index], True)
                    room2_wall_position = calculate_room_wall_position_from_shared_wall(shared_wall, final_shared_position, rooms_data[room2_index], False)
                    door1_info = {'width': final_door_width, 'height': connecting_door['height'], 'position_on_wall': room1_wall_position, 'wall_side': shared_wall['room1_wall'], 'door_type': connecting_door.get('door_type', 'interior'), 'opening': connecting_door.get('opening', False), 'door_material': connecting_door.get('door_material', 'standard wooden door')}
                    door2_info = {'width': final_door_width, 'height': connecting_door['height'], 'position_on_wall': room2_wall_position, 'wall_side': shared_wall['room2_wall'], 'door_type': connecting_door.get('door_type', 'interior'), 'opening': connecting_door.get('opening', False), 'door_material': connecting_door.get('door_material', 'standard wooden door')}
                    result['rooms'][room1_index]['doors'].append(door1_info)
                    result['rooms'][room2_index]['doors'].append(door2_info)
        if 'windows' in claude_response and claude_response['windows']:
            for window in claude_response['windows']:
                exterior_wall_index = window['exterior_wall_index']
                if 0 <= exterior_wall_index < len(shared_walls_info['room_exterior_walls']):
                    exterior_wall = shared_walls_info['room_exterior_walls'][exterior_wall_index]
                    room_index = exterior_wall['room']['index']
                    window_info = {'width': window['width'], 'height': window['height'], 'position_on_wall': window['center_position'], 'wall_side': exterior_wall['wall_side'], 'sill_height': window['sill_height'], 'window_type': window.get('window_type', 'standard'), 'window_grid': window.get('window_grid', [1, 1]), 'glass_color': window.get('glass_color', [204, 230, 255]), 'frame_color': window.get('frame_color', [77, 77, 77])}
                    original_window_width = window_info['width']
                    original_window_position = window_info['position_on_wall']
                    if window_info['wall_side'] in ['north', 'south']:
                        room_max_width = rooms_data[room_index]['dimensions']['width'] * 0.8
                    else:
                        room_max_width = rooms_data[room_index]['dimensions']['length'] * 0.8
                    exterior_wall_max_width = exterior_wall['overlap_length'] * 0.8
                    safe_start = 0.1
                    safe_end = 0.9
                    max_half_width_from_start = original_window_position - safe_start
                    max_half_width_from_end = safe_end - original_window_position
                    max_half_width = min(max_half_width_from_start, max_half_width_from_end)
                    position_based_max_width = max_half_width * 2 * exterior_wall['overlap_length']
                    max_width_constraint = min(room_max_width, exterior_wall_max_width)
                    final_window_width = min(original_window_width, max_width_constraint, max(0.8, position_based_max_width))
                    if final_window_width < original_window_width * 0.8:
                        desired_width = min(original_window_width, max_width_constraint)
                        required_half_width_ratio = desired_width / 2 / exterior_wall['overlap_length']
                        min_center = safe_start + required_half_width_ratio
                        max_center = safe_end - required_half_width_ratio
                        if min_center <= max_center:
                            if original_window_position < min_center:
                                final_window_position = min_center
                            elif original_window_position > max_center:
                                final_window_position = max_center
                            else:
                                final_window_position = original_window_position
                            final_window_width = desired_width
                        else:
                            final_window_position = max(safe_start, min(safe_end, original_window_position))
                    else:
                        final_window_position = max(safe_start, min(safe_end, original_window_position))
                    window_info['width'] = final_window_width
                    window_info['position_on_wall'] = final_window_position
                    existing_items = []
                    for door in result['rooms'][room_index]['doors']:
                        if door['wall_side'] == window_info['wall_side']:
                            existing_items.append({'position': door['position_on_wall'], 'width': door['width'], 'type': 'door'})
                    for existing_window in result['rooms'][room_index]['windows']:
                        if existing_window['wall_side'] == window_info['wall_side']:
                            existing_items.append({'position': existing_window['position_on_wall'], 'width': existing_window['width'], 'type': 'window'})
                    if existing_items:
                        wall_length = exterior_wall['overlap_length']
                        window_width_ratio = window_info['width'] / wall_length
                        window_half_width_ratio = window_width_ratio / 2
                        existing_items.sort(key=lambda x: x['position'])
                        safe_position = window_info['position_on_wall']
                        position_found = False
                        occupied_segments = []
                        for item in existing_items:
                            item_width_ratio = item['width'] / wall_length
                            item_half_width_ratio = item_width_ratio / 2
                            margin = 0.05
                            segment_start = max(0.0, item['position'] - item_half_width_ratio - margin)
                            segment_end = min(1.0, item['position'] + item_half_width_ratio + margin)
                            occupied_segments.append((segment_start, segment_end))
                        if occupied_segments:
                            occupied_segments.sort()
                            merged_segments = [occupied_segments[0]]
                            for (current_start, current_end) in occupied_segments[1:]:
                                (last_start, last_end) = merged_segments[-1]
                                if current_start <= last_end:
                                    merged_segments[-1] = (last_start, max(last_end, current_end))
                                else:
                                    merged_segments.append((current_start, current_end))
                        min_required_space = window_width_ratio + 0.1
                        current_start = safe_position - window_half_width_ratio
                        current_end = safe_position + window_half_width_ratio
                        conflicts_with_current = any((not (current_end <= seg_start or current_start >= seg_end) for (seg_start, seg_end) in merged_segments))
                        if not conflicts_with_current and 0.1 <= current_start and (current_end <= 0.9):
                            position_found = True
                        else:
                            for i in range(len(merged_segments) + 1):
                                if i == 0:
                                    gap_start = 0.1
                                    gap_end = merged_segments[0][0] if merged_segments else 0.9
                                elif i == len(merged_segments):
                                    gap_start = merged_segments[-1][1]
                                    gap_end = 0.9
                                else:
                                    gap_start = merged_segments[i - 1][1]
                                    gap_end = merged_segments[i][0]
                                gap_size = gap_end - gap_start
                                if gap_size >= min_required_space:
                                    safe_position = gap_start + gap_size / 2
                                    safe_position = max(0.1, min(0.9, safe_position))
                                    position_found = True
                                    break
                        if not position_found:
                            continue
                        window_info['position_on_wall'] = safe_position
                    room_height = rooms_data[room_index]['dimensions']['height']
                    original_sill_height = window_info['sill_height']
                    original_window_height = window_info['height']
                    total_window_height = original_sill_height + original_window_height
                    if total_window_height > room_height:
                        max_total_height = room_height - 0.1
                        height_ratio = max_total_height / total_window_height
                        window_info['sill_height'] = original_sill_height * height_ratio
                        window_info['height'] = original_window_height * height_ratio
                        min_sill_height = 0.5
                        if window_info['sill_height'] < min_sill_height:
                            window_info['sill_height'] = min_sill_height
                            remaining_height = max_total_height - min_sill_height
                            if remaining_height > 0.3:
                                window_info['height'] = remaining_height
                            else:
                                continue
                    result['rooms'][room_index]['windows'].append(window_info)
        result['debug_info'] = debug_info
        result['debug_info']['steps'].append('Step 6 completed: All doors and windows processed')
        return result
    except anthropic.APIError as e:
        raise ValueError(f'Anthropic API error: {e}')
    except Exception as e:
        raise ValueError(f'Error calling Claude API: {e}')

async def call_llm_for_rooms_with_validation(input_text: str, max_attempts: int=3) -> Dict[str, Any]:
    """
    Call Claude to generate room layout (rooms only) with validation and iterative correction.
    
    Args:
        input_text: Original layout description
        max_attempts: Maximum number of correction attempts
        
    Returns:
        Validated room-only layout data
    """
    attempt = 0
    last_response = None
    while attempt < max_attempts:
        try:
            if attempt == 0:
                response = await call_llm_for_rooms_only_solver(input_text)
            elif last_response is None:
                response = await call_llm_for_rooms_only_solver(input_text)
            else:
                validation_result = validate_room_only_layout(last_response['rooms'])
                correction_prompt = generate_room_correction_prompt(validation_result, input_text, last_response)
                claude_response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.2, messages=[{'role': 'user', 'content': correction_prompt}])
                if not claude_response or not hasattr(claude_response, 'content') or (not claude_response.content):
                    raise ValueError('No response received from Claude API during correction')
                response_text = claude_response.content[0].text
                response_text = extract_json_from_response(response_text)
                if not response_text:
                    raise ValueError('Could not extract JSON content from Claude response')
                response = json.loads(response_text)
            if 'rooms' not in response or not response['rooms']:
                raise ValueError('Invalid response structure')
            validation_result = validate_room_only_layout(response['rooms'])
            if validation_result['valid']:
                return response
            else:
                last_response = response
                attempt += 1
                if attempt >= max_attempts:
                    response['validation_warning'] = {'message': f'Room layout still has issues after {max_attempts} attempts', 'remaining_issues': validation_result['issues']}
                    return response
        except Exception as e:
            attempt += 1
            if attempt >= max_attempts:
                raise e
            last_response = None
    raise ValueError(f'Failed to generate valid room layout after {max_attempts} attempts')

def generate_room_correction_prompt(validation_result: Dict[str, Any], original_input: str, previous_attempt: Dict[str, Any]) -> str:
    """Generate a correction prompt for room-only layout issues."""
    issues = validation_result.get('issues', [])
    overlaps = validation_result.get('overlaps', [])
    detached_rooms = validation_result.get('detached_rooms', [])
    prompt = f'The previous room layout attempt has architectural issues that need correction.\n\nORIGINAL REQUEST: "{original_input}"\n\nPREVIOUS ATTEMPT:\n{json.dumps(previous_attempt, indent=2)}\n\nISSUES FOUND:\n{json.dumps(issues, indent=2)}\n\nSPECIFIC PROBLEMS:\n'
    if overlaps:
        prompt += f'\nROOM OVERLAPS: {overlaps}'
        prompt += "\n- Move rooms so they don't occupy the same space"
        prompt += '\n- Ensure clear separation between room boundaries'
    if detached_rooms:
        prompt += f'\nDETACHED ROOMS: {detached_rooms}'
        prompt += '\n- Arrange rooms so they share walls (adjacent positioning)'
        prompt += '\n- Ensure rooms can be connected with doors later'
    prompt += '\n\nPlease generate a CORRECTED room-only layout that:\n1. Fixes all the issues listed above\n2. Maintains the original intent from the user request\n3. Uses appropriate room sizes and relationships\n4. Ensures no room overlaps\n5. Arranges rooms to share walls where appropriate\n\nReturn the corrected layout using the SAME JSON structure (rooms only, no doors/windows):\n\n{\n    "building_style": "...",\n    "rooms": [\n        {\n            "room_type": "...",\n            "dimensions": {"width": float, "length": float, "height": float},\n            "position": {"x": float, "y": float, "z": 0}\n        }\n    ]\n}'
    return prompt

async def incremental_floor_plan_additions(existing_layout_data: Dict[str, Any], rejected_rooms: List[str], input_text: str) -> Dict[str, Any]:
    """
    Use Claude API to add previously rejected rooms to an existing layout.
    
    Args:
        existing_layout_data: Dict containing successfully added rooms with their vertices
        rejected_rooms: List of room names that were previously rejected
        
    Returns:
        Dict containing the complete layout with existing + newly added rooms
    """
    debug_vis_dir = f'{SERVER_ROOT_DIR}/vis'
    os.makedirs(debug_vis_dir, exist_ok=True)
    existing_vis_path = os.path.join(debug_vis_dir, f"existing_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    await get_vlm_room_layout_2d_visualization(existing_layout_data, existing_vis_path)
    existing_rooms_text = ''
    if 'rooms' in existing_layout_data and existing_layout_data['rooms']:
        existing_rooms_text = 'EXISTING ROOMS IN LAYOUT:\n'
        for (i, room) in enumerate(existing_layout_data['rooms']):
            room_type = room.get('room_type', f'Room_{i + 1}')
            vertices = room['vertices']
            vertices_str = ', '.join([f"({v['x']:.1f},{v['y']:.1f})" for v in vertices])
            existing_rooms_text += f"- {room_type}: vertices [{vertices_str}], height {room.get('height', 2.7)}m\n"
    else:
        existing_rooms_text = 'NO EXISTING ROOMS (starting fresh layout)\n'
    rejected_rooms_text = 'ROOMS TO ADD:\n'
    if rejected_rooms:
        for room_name in rejected_rooms:
            rejected_rooms_text += f'- {room_name}\n'
    else:
        return existing_layout_data
    prompt = f'''You are an expert architect and interior designer. I have an EXISTING ROOM LAYOUT and need you to add SPECIFIC ADDITIONAL ROOMS without overlapping or creating isolated rooms.\n\nORIGINAL USER REQUEST:\n"{input_text.strip()}"\n\nCONTEXT:\nThis existing layout was generated as part of fulfilling the above user request. However, some rooms from the original request could not be added due to collision or isolation issues. Your task is to add these remaining rooms while respecting the existing layout and maintaining the original design intent.\n\nSTEP 1: IDENTIFY ROOMS TO ADD\n{rejected_rooms_text}\n\nSTEP 2: ANALYZE ROOM REQUIREMENTS\nBefore placing these rooms, carefully analyze the ORIGINAL USER REQUEST above to understand:\n- What are the specific functional requirements for each room you need to add?\n- What are the size expectations (compact, medium, large, spacious) for each room type?\n- What are the connectivity and adjacency requirements mentioned in the original request?\n- How should these rooms relate to the existing rooms in terms of circulation and privacy?\n- What are the architectural style and design principles that should guide the placement?\n\nSTEP 3: PLACEMENT CONSTRAINTS\n🚨 CRITICAL LAYOUT CONSTRAINTS 🚨\n- **ABSOLUTELY NO OVERLAPPING ROOMS**: New room vertices must NEVER intersect or overlap with ALL existing rooms\n- **NO ISOLATED ROOMS**: Every new room must be adjacent to at least one existing room OR another new room for connectivity\n- **PERFECT RECTANGLES**: Every new room must have exactly 4 vertices forming a rectangle with 90-degree corners\n- **FLEXIBLE COORDINATE SPACE**: Rooms can be placed at ANY coordinate location including NEGATIVE coordinates (x < 0, y < 0) if needed for optimal layout design\n\n{existing_rooms_text}\n\nLAYOUT REQUIREMENTS:\n- **FULFILL ORIGINAL REQUEST**: Ensure the added rooms serve the functions and relationships described in the original user request\n- **VERTEX-BASED PLACEMENT**: Position new rooms so NO vertex falls inside ALL existing room boundaries\n- **ADJACENCY REQUIRED**: Each new room must touch at least one existing room or another new room (sharing edges/corners)\n- **RECTANGULAR ROOMS ONLY**: All rooms must be perfect rectangles with 90-degree corners\n- **AVOID OVERLAPS**: Carefully check that new room boundaries don't intersect ALL existing rooms\n- **MAINTAIN CONNECTIVITY**: Ensure all new rooms can be connected via doors to the existing layout\n- **FUNCTIONAL RELATIONSHIPS**: Consider the room relationships and circulation patterns mentioned in the original request\n- **COORDINATE FREEDOM**: Use ANY coordinate values including negative numbers (x < 0, y < 0) as needed - rooms are NOT required to be above (0,0)\n\nBUILDING STYLE: {existing_layout_data.get('building_style', 'Modern functional architecture')}\n\nSTEP 4: ROOM PLACEMENT TASK\nFollowing the analysis from Steps 1-3 above, add the specified rooms to the existing layout using rectangular vertices. Your placement must:\n\n**FULFILL ORIGINAL REQUIREMENTS**:\n- Implement the specific functional requirements you identified for each room type\n- Use the appropriate room sizes (compact/medium/large/spacious) as indicated in the original request\n- Respect the connectivity and adjacency relationships described in the original user request\n- Support the circulation patterns and privacy levels mentioned in the original design\n\n**MAINTAIN DESIGN COHERENCE**:\n- Follow the architectural style and design principles from the original request\n- Ensure proper spatial relationships and functional adjacencies\n- Create logical circulation flow that integrates with existing rooms\n- Position rooms to support the overall design concept and usage patterns\n\n**TECHNICAL CONSTRAINTS**:\n- No overlaps with existing rooms and ensure proper connectivity to avoid isolation\n- Use perfect rectangular vertices for all new rooms\n- Feel free to use negative coordinates (x < 0, y < 0) if they create better spatial arrangements\n\nRespond with your thinking process and then a valid JSON object for the NEW ROOMS using this exact structure:\n\n```json\n{{\n    "building_style": "description of architectural style",\n    "rooms": [\n        {{\n            "room_type": "specific room type from the rooms to add list",\n            "vertices": [\n                {{"x": float, "y": float}},\n                {{"x": float, "y": float}},\n                {{"x": float, "y": float}},\n                {{"x": float, "y": float}}\n            ],\n            "height": float\n        }}\n    ]\n}}\n```\n\nCRITICAL VERIFICATION STEPS:\n1. **VERIFY ANALYSIS**: Confirm you have identified the specific requirements for each room from the original user request\n2. **VERIFY REQUIREMENTS FULFILLMENT**: Ensure each new room meets the functional, size, and adjacency requirements from the original request\n3. **CHECK OVERLAPS**: Verify no new room vertex is inside any existing room boundary\n4. **CHECK ADJACENCY**: Verify each new room touches at least one existing room or another new room  \n5. **CHECK RECTANGLES**: Verify all rooms have exactly 4 vertices forming perfect rectangles\n6. **CHECK CONNECTIVITY**: Ensure new rooms support the circulation patterns described in the original request\n\n🚨 FINAL REMINDER - THESE ARE MANDATORY 🚨\n- **NO OVERLAPPING ROOMS**: Check that no new room vertices are inside existing rooms and no boundaries intersect\n- **NO ISOLATED ROOMS**: Verify every new room can connect to existing layout\n- **PERFECT RECTANGLES**: Every room must have exactly 4 vertices forming a rectangle with 90-degree corners\n- **COORDINATE FLEXIBILITY**: Remember you can use ANY coordinates including negative values - optimal placement is more important than staying above (0,0)\n\nGenerate ONLY the new rooms that need to be added - do not repeat existing rooms.'''
    try:
        with open(existing_vis_path, 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        response = call_vlm(vlm_type='qwen', max_tokens=4000, temperature=0.05, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': image_data}}]}])
        if not response or not hasattr(response, 'content') or (not response.content):
            raise ValueError('No response received from Claude API')
        response_text = None
        for content in response.content:
            if hasattr(content, 'text'):
                response_text = content.text
                break
        if not response_text:
            raise ValueError('Claude API response content has no text')
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        new_rooms_data = json.loads(response_text)
        combined_layout = {'building_style': existing_layout_data.get('building_style', new_rooms_data.get('building_style', 'Combined Layout')), 'rooms': []}
        if 'rooms' in existing_layout_data:
            combined_layout['rooms'].extend(existing_layout_data['rooms'])
        if 'rooms' in new_rooms_data:
            combined_layout['rooms'].extend(new_rooms_data['rooms'])
        print(f"✅ Successfully added {len(new_rooms_data.get('rooms', []))} new rooms to existing layout", file=sys.stderr)
        print(f"📊 Total rooms in combined layout: {len(combined_layout['rooms'])}", file=sys.stderr)
        return combined_layout
    except Exception as e:
        print(f'❌ Error in incremental floor plan additions: {e}', file=sys.stderr)
        return existing_layout_data
    finally:
        try:
            if os.path.exists(existing_vis_path):
                os.remove(existing_vis_path)
        except:
            pass

async def call_llm_for_rooms_only_solver(input_text: str) -> Dict[str, Any]:
    """
    Call Claude Sonnet 4 to generate room shapes and adjacency graph, then use RectangleContactSolver 
    to generate the final layout. Same input/output format as call_llm_for_rooms_only.
    """
    prompt = f'You are an expert architect and interior designer. Analyze this description and generate room specifications and adjacency relationships: "{input_text}"\n\n🚨 CRITICAL REQUIREMENTS 🚨\n- Generate ONLY room shapes (rectangles) and adjacency connections\n- NO overlapping rooms, NO isolated rooms\n- Every room must connect to at least one other room\n\nSTEP 1: ANALYZE REQUEST TYPE\nFirst determine if the user is asking for:\n- SINGLE ROOM: Only one specific room (e.g., "design a living room", "create a bedroom", "generate a kitchen")\n- MULTIPLE ROOMS: A complete layout, house, apartment, or multiple connected rooms\n\nSTEP 2: GENERATE ROOM SPECIFICATIONS AND ADJACENCY GRAPH\nFor SINGLE ROOM requests:\n- Generate just that one room with appropriate dimensions\n- Use realistic proportions for the specific room type\n- No adjacency graph needed (single room)\n\nFor MULTIPLE ROOMS requests:\n- Create functional room specifications with realistic dimensions\n- Generate adjacency relationships ensuring connectivity\n- Consider architectural best practices and room relationships\n- Ensure every room is connected to at least one other room\n\nRespond with ONLY a valid JSON object using this exact structure:\n\n```json\n{{\n    "building_style": "description of architectural style (e.g., Modern residential, Traditional colonial, etc.)",\n    "shapes": [\n        {{\n            "room_type": "specific room type (e.g., living room, master bedroom, kitchen, etc.)",\n            "width": float,\n            "height": float\n        }}\n    ],\n    "graph": {{\n        "room_type_1": ["room_type_2", "room_type_3"],\n        "room_type_2": ["room_type_1"],\n        "room_type_3": ["room_type_1"]\n    }}\n}}\n```\n\nCRITICAL REQUIREMENTS:\n- For SINGLE ROOM: Include exactly 1 room in "shapes" array, empty graph object {{}}\n- For MULTIPLE ROOMS: Include all rooms in "shapes" array with proper adjacency in "graph"\n- Use metric units (meters) for all dimensions\n- Standard room dimensions: living room (4-6m x 5-7m), bedroom (3-4m x 4-5m), kitchen (3-4m x 3-5m), etc.\n- Graph keys must exactly match room_type values in shapes array\n- Ensure graph represents a connected structure (no isolated rooms)\n- Room dimensions should be realistic for their intended function\n\nGenerate room specifications and adjacency graph that truly reflects the input description with appropriate room types, dimensions, and spatial relationships.\n\n🚨 FINAL REMINDER 🚨\n- **CONNECTED GRAPH**: Every room must be reachable from every other room\n- **MATCHING NAMES**: Graph keys must exactly match room_type values in shapes\n- **REALISTIC DIMENSIONS**: Use appropriate sizes for each room type'
    try:
        response = call_vlm(vlm_type='openai', max_tokens=16000, temperature=1.0, thinking=True, messages=[{'role': 'user', 'content': prompt}])
        if not response:
            raise ValueError('No response received from Claude API')
        if not hasattr(response, 'content') or not response.content:
            raise ValueError('Claude API response has no content')
        if len(response.content) == 0:
            raise ValueError('Claude API response content is empty')
        response_text = None
        for content in response.content:
            if hasattr(content, 'text'):
                response_text = content.text
                break
        if response_text is None:
            raise ValueError('Claude API response content has no text attribute')
        if not response_text or not response_text.strip():
            raise ValueError('Claude API returned empty response text')
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        solver_data = json.loads(response_text)
        if 'shapes' not in solver_data or 'graph' not in solver_data:
            raise ValueError("Invalid solver data: missing 'shapes' or 'graph' fields")
        shapes = solver_data['shapes']
        graph_dict = solver_data['graph']
        if not shapes:
            raise ValueError('No room shapes provided')
        if len(shapes) == 1:
            room = shapes[0]
            layout_data = {'building_style': solver_data.get('building_style', 'Modern residential'), 'rooms': [{'room_type': room['room_type'], 'dimensions': {'width': room['width'], 'length': room['height'], 'height': 2.7}, 'position': {'x': 0.0, 'y': 0.0, 'z': 0.0}}]}
            return layout_data
        G = nx.Graph()
        room_type_to_index = {room['room_type']: i for (i, room) in enumerate(shapes)}
        index_to_room_type = {i: room['room_type'] for (i, room) in enumerate(shapes)}
        for i in range(len(shapes)):
            G.add_node(i)
        for (room_type, adjacent_types) in graph_dict.items():
            if room_type not in room_type_to_index:
                continue
            room_idx = room_type_to_index[room_type]
            for adj_type in adjacent_types:
                if adj_type in room_type_to_index:
                    adj_idx = room_type_to_index[adj_type]
                    G.add_edge(room_idx, adj_idx)
        if not nx.is_connected(G):
            raise ValueError('Generated graph is not connected - all rooms must be reachable')
        rectangles = {}
        for (i, room) in enumerate(shapes):
            rectangles[i] = RectangleSpec(w=room['width'], h=room['height'])
        solver = RectangleContactRelaxationSolver(G, rectangles)
        (solved_graph, layout) = solver.solve(max_attempts=50)
        converted_rooms = []
        for (i, rect_layout) in layout.items():
            room_type = index_to_room_type[i]
            converted_room = {'room_type': room_type, 'dimensions': {'width': rect_layout.w, 'length': rect_layout.h, 'height': 2.7}, 'position': {'x': rect_layout.x - rect_layout.w / 2, 'y': rect_layout.y - rect_layout.h / 2, 'z': 0.0}}
            converted_rooms.append(converted_room)
        layout_data = {'building_style': solver_data.get('building_style', 'Modern residential'), 'rooms': converted_rooms}
        debug_vis_dir = f'{SERVER_ROOT_DIR}/vis'
        os.makedirs(debug_vis_dir, exist_ok=True)
        debug_vis_path = os.path.join(debug_vis_dir, f"solver_layout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        await get_vlm_room_layout_2d_visualization(layout_data, debug_vis_path)
        print(f'🔍 Debug: Solver layout visualization saved to {debug_vis_path}', file=sys.stderr)
        return layout_data
    except anthropic.APIError as e:
        raise ValueError(f'Anthropic API error: {e}')
    except Exception as e:
        raise ValueError(f'Error in solver-based room generation: {e}')

async def get_doors_windows_layout_visualization(layout_result: Dict[str, Any], save_path: str, mst, entry_room_index: int) -> None:
    """
    Create a comprehensive visualization of the room layout with doors, windows, and MST connectivity.
    Similar style to get_vlm_room_layout_2d_visualization but with door/window details.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch
        import numpy as np
    except ImportError:
        print('Warning: matplotlib not available, skipping doors/windows visualization', file=sys.stderr)
        return
    fig = plt.figure(figsize=(16, 12))
    ax_main = plt.subplot(1, 2, (1, 1))
    ax_legend = plt.subplot(1, 2, 2)
    if 'rooms' not in layout_result or not layout_result['rooms']:
        ax_main.text(0.5, 0.5, 'No rooms found', ha='center', va='center', transform=ax_main.transAxes, fontsize=16)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(0, 1)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    room_colors = ['#FFB6C1', '#98FB98', '#FFB347', '#87CEEB', '#DDA0DD', '#F0E68C', '#FF6347', '#40E0D0', '#FFE4B5', '#98FB98', '#FFA07A', '#20B2AA', '#F5DEB3', '#FFB6C1', '#87CEFA', '#DDA0DD', '#F0E68C', '#FF7F50', '#40E0D0', '#FFFFE0']
    door_color = '#8B4513'
    window_color = '#87CEEB'
    entry_door_color = '#FF4500'
    mst_edge_color = '#FF6347'
    all_x = []
    all_y = []
    room_centers = {}
    legend_elements = []
    rooms_data = layout_result['rooms']
    for (i, room) in enumerate(rooms_data):
        room_type = room.get('room_type', f'Room {i + 1}')
        pos = room['position']
        dim = room['dimensions']
        x = pos['x']
        y = pos['y']
        width = dim['width']
        length = dim['length']
        x_coords = [x, x, x + width, x + width]
        y_coords = [y, y + length, y + length, y]
        center_x = x + width / 2
        center_y = y + length / 2
        room_centers[i] = (center_x, center_y)
        all_x.extend(x_coords)
        all_y.extend(y_coords)
        polygon_coords = list(zip(x_coords, y_coords))
        color = room_colors[i % len(room_colors)]
        edge_color = 'red' if i == entry_room_index else 'black'
        edge_width = 3 if i == entry_room_index else 2
        polygon = Polygon(polygon_coords, closed=True, facecolor=color, edgecolor=edge_color, linewidth=edge_width, alpha=0.8)
        ax_main.add_patch(polygon)
        display_name = room_type.replace('_', ' ').title()
        entry_marker = ' ⭐ [ENTRY]' if i == entry_room_index else ''
        ax_main.text(center_x, center_y, f'{display_name}{entry_marker}', ha='center', va='center', fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=edge_color, linewidth=1))
        ax_main.text(x + 0.1, y + length - 0.1, f'#{i}', ha='left', va='top', fontsize=8, fontweight='bold', color='darkblue', bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        legend_label = f'{display_name} ({i})'
        if i == entry_room_index:
            legend_label += ' ⭐'
        legend_elements.append(patches.Patch(color=color, label=legend_label))
    if mst and mst.edges():
        for edge in mst.edges():
            (room1_idx, room2_idx) = edge
            if room1_idx in room_centers and room2_idx in room_centers:
                (x1, y1) = room_centers[room1_idx]
                (x2, y2) = room_centers[room2_idx]
                ax_main.plot([x1, x2], [y1, y2], color=mst_edge_color, linewidth=3, linestyle='--', alpha=0.8, zorder=5)
                (mid_x, mid_y) = ((x1 + x2) / 2, (y1 + y2) / 2)
    door_count = 0
    window_count = 0
    for (i, room) in enumerate(rooms_data):
        pos = room['position']
        dim = room['dimensions']
        x = pos['x']
        y = pos['y']
        width = dim['width']
        length = dim['length']
        if 'doors' in room:
            room_color = room_colors[i % len(room_colors)]
            for door in room['doors']:
                door_count += 1
                wall_side = door['wall_side']
                position = door['position_on_wall']
                door_width = door['width']
                door_height = door.get('height', 2.0)
                door_type = door.get('door_type', 'interior')
                door_thickness = 0.15
                if wall_side == 'north':
                    door_x = x + position * width - door_width / 2
                    door_y = y + length - door_thickness
                    door_rect_width = door_width
                    door_rect_height = door_thickness
                    door_center_x = door_x + door_width / 2
                    door_center_y = door_y + door_thickness / 2
                    arrow_dx = 0
                    arrow_dy = -0.3
                elif wall_side == 'south':
                    door_x = x + position * width - door_width / 2
                    door_y = y
                    door_rect_width = door_width
                    door_rect_height = door_thickness
                    door_center_x = door_x + door_width / 2
                    door_center_y = door_y + door_thickness / 2
                    arrow_dx = 0
                    arrow_dy = 0.3
                elif wall_side == 'east':
                    door_x = x + width - door_thickness
                    door_y = y + position * length - door_width / 2
                    door_rect_width = door_thickness
                    door_rect_height = door_width
                    door_center_x = door_x + door_thickness / 2
                    door_center_y = door_y + door_width / 2
                    arrow_dx = -0.3
                    arrow_dy = 0
                else:
                    door_x = x
                    door_y = y + position * length - door_width / 2
                    door_rect_width = door_thickness
                    door_rect_height = door_width
                    door_center_x = door_x + door_thickness / 2
                    door_center_y = door_y + door_width / 2
                    arrow_dx = 0.3
                    arrow_dy = 0
                door_color_final = room_color if door_type not in ['entry', 'main', 'front'] else entry_door_color
                door_rect = Rectangle((door_x, door_y), door_rect_width, door_rect_height, facecolor=door_color_final, edgecolor='black', linewidth=2, alpha=0.9)
                ax_main.add_patch(door_rect)
                ax_main.arrow(door_center_x, door_center_y, arrow_dx, arrow_dy, head_width=0.08, head_length=0.08, fc='black', ec='black', linewidth=2)
                label_x = door_center_x
                label_y = door_center_y
                door_symbol = '🚪E' if door_type in ['entry', 'main', 'front'] else '🚪'
                ax_main.text(label_x, label_y, door_symbol, ha='center', va='center', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))
        if 'windows' in room:
            for window in room['windows']:
                window_count += 1
                wall_side = window['wall_side']
                position = window['position_on_wall']
                window_width = window['width']
                window_height = window.get('height', 1.5)
                if wall_side == 'north':
                    win_x = x + position * width - window_width / 2
                    win_y = y + length - 0.15
                    win_rect_width = window_width
                    win_rect_height = 0.3
                elif wall_side == 'south':
                    win_x = x + position * width - window_width / 2
                    win_y = y - 0.15
                    win_rect_width = window_width
                    win_rect_height = 0.3
                elif wall_side == 'east':
                    win_x = x + width - 0.15
                    win_y = y + position * length - window_width / 2
                    win_rect_width = 0.3
                    win_rect_height = window_width
                else:
                    win_x = x - 0.15
                    win_y = y + position * length - window_width / 2
                    win_rect_width = 0.3
                    win_rect_height = window_width
                window_rect = Rectangle((win_x, win_y), win_rect_width, win_rect_height, facecolor=window_color, edgecolor='navy', linewidth=1, alpha=0.9)
                ax_main.add_patch(window_rect)
                label_x = win_x + win_rect_width / 2
                label_y = win_y + win_rect_height / 2
                ax_main.text(label_x, label_y, '🪟', ha='center', va='center', fontsize=6, fontweight='bold')
    if all_x and all_y:
        margin = max(1.0, (max(all_x) - min(all_x)) * 0.15)
        ax_main.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax_main.set_ylim(min(all_y) - margin, max(all_y) + margin)
    else:
        ax_main.set_xlim(-1, 10)
        ax_main.set_ylim(-1, 10)
    ax_main.set_aspect('equal')
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.set_xlabel('X Coordinate (meters)', fontsize=12)
    ax_main.set_ylabel('Y Coordinate (meters)', fontsize=12)
    ax_main.set_title(f'Graph-Based Layout: Doors & Windows\nEntry Room: #{entry_room_index} | Doors: {door_count} | Windows: {window_count}', fontsize=14, fontweight='bold', pad=20)
    legend_elements.append(patches.Patch(color=entry_door_color, label='🚪 Entry Door'))
    legend_elements.append(patches.Patch(color='gray', label='🚪 Interior Door (room color)'))
    legend_elements.append(patches.Patch(color=window_color, label='🪟 Window'))
    legend_elements.append(patches.Patch(color=mst_edge_color, label='MST Connection'))
    legend_elements.append(patches.Patch(color='black', label='➡️ Door Direction'))
    ax_legend.axis('off')
    room_legend_elements = [elem for elem in legend_elements if '🚪' not in elem.get_label() and '🪟' not in elem.get_label() and ('MST' not in elem.get_label())]
    feature_legend_elements = [elem for elem in legend_elements if '🚪' in elem.get_label() or '🪟' in elem.get_label() or 'MST' in elem.get_label()]
    if room_legend_elements:
        legend1 = ax_legend.legend(handles=room_legend_elements, loc='upper left', title='🏠 Rooms', title_fontsize=12, fontsize=10)
        legend1.get_frame().set_facecolor('lightblue')
        legend1.get_frame().set_alpha(0.8)
    if feature_legend_elements:
        legend2 = ax_legend.legend(handles=feature_legend_elements, loc='center left', title='🔧 Features', title_fontsize=12, fontsize=10)
        legend2.get_frame().set_facecolor('lightgreen')
        legend2.get_frame().set_alpha(0.8)
    stats_text = f'📊 Layout Statistics:\n• Total Rooms: {len(rooms_data)}\n• Entry Room: #{entry_room_index}\n• Total Doors: {door_count}\n• Total Windows: {window_count}\n• MST Edges: {(len(mst.edges()) if mst else 0)}\n• Graph-based Algorithm: ✅\n\n🎨 Visualization Features:\n• Doors shown within rooms\n• Door directions with arrows\n• Distinct room colors\n• Entry room highlighted'
    ax_legend.text(0.1, 0.3, stats_text, fontsize=10, bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9), verticalalignment='top')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.02, 0.02, f'Generated: {timestamp}', fontsize=8, alpha=0.6)
    plt.tight_layout()
    try:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f'✅ Doors & windows layout visualization saved successfully to {save_path}', file=sys.stderr)
    except Exception as e:
        print(f'❌ Error saving doors/windows visualization: {e}', file=sys.stderr)
    finally:
        plt.close()

async def call_llm_for_doors_windows_graph_based(rooms_data: List[Dict[str, Any]], input_text: str) -> Dict[str, Any]:
    """
    Graph-based door and window placement using minimum spanning tree approach.
    
    This function:
    1. Finds all shared walls (for connected room doors) and exterior walls (for windows and entry door)
    2. Builds a graph where each vertex is a room, edges exist for rooms with shared walls
    3. Uses LLM to identify the entry room from input text and room types
    4. Calculates MST with entry room as root node
    5. Selects door/window placements based on MST and rules
    6. Uses LLM to generate detailed specifications
    7. Returns in same format as original function
    
    Args:
        rooms_data: List of room data dictionaries
        input_text: Original user request text
        
    Returns:
        Dictionary with same format as call_llm_for_doors_windows
    """
    debug_info = {'prompts': [], 'responses': [], 'steps': []}
    from utils import find_all_shared_walls, generate_floor_plan_visualization, calculate_room_wall_position_from_shared_wall
    debug_info['steps'].append('Step 1: Finding shared walls and exterior walls')
    shared_walls_info = find_all_shared_walls(rooms_data)
    debug_info['steps'].append('Step 2: Building room connectivity graph')
    room_graph = nx.Graph()
    for (i, room) in enumerate(rooms_data):
        room_graph.add_node(i, room_type=room['room_type'])
    for shared_wall in shared_walls_info['room_room_walls']:
        room1_idx = shared_wall['room1']['index']
        room2_idx = shared_wall['room2']['index']
        weight = 1.0 / max(shared_wall['overlap_length'], 0.1)
        room_graph.add_edge(room1_idx, room2_idx, weight=weight, shared_wall_info=shared_wall)
    debug_info['steps'].append('Step 3: Identifying entry room')
    if len(rooms_data) == 1:
        entry_room_index = 0
        debug_info['steps'].append('Step 3: Single room - entry room is room 0')
        print(f'🏠 Single room layout: entry room automatically set to room 0', file=sys.stderr)
    else:
        debug_info['steps'].append('Step 3: Multiple rooms - using LLM to identify entry room')
        room_types_summary = []
        for (i, room) in enumerate(rooms_data):
            room_types_summary.append({'room_index': i, 'room_type': room['room_type'], 'area': f"{room['dimensions']['width'] * room['dimensions']['length']:.1f} sq m"})
        entry_room_prompt = f'You are an expert architect analyzing a floor plan to determine the entry room.\n\nORIGINAL REQUEST: "{input_text}"\n\nAVAILABLE ROOMS:\n{json.dumps(room_types_summary, indent=2)}\n\nTASK: Based on the original request and the room types, determine which room should have the main entrance door.\n\nConsider:\n- Typical entry patterns (living room, foyer, hallway for residential; lobby, reception for commercial)\n- The original request context and building type\n- Logical flow for visitors entering the space\n\nReturn your response as JSON:\n```json\n{{\n    "entry_room_index": int,\n    "reasoning": "Brief explanation of why this room should be the entry point"\n}}\n```\n'
        try:
            debug_info['prompts'].append({'step': '3_entry_room_identification', 'prompt': entry_room_prompt})
            entry_response = call_vlm(vlm_type='qwen', max_tokens=1000, temperature=0.2, messages=[{'role': 'user', 'content': entry_room_prompt}])
            entry_response_text = entry_response.content[0].text.strip()
            debug_info['responses'].append({'step': '3_entry_room_identification', 'response': entry_response_text})
            entry_data = extract_json_from_response(entry_response_text)
            if not entry_data:
                raise ValueError('Could not extract JSON content from Claude response')
            entry_data = json.loads(entry_data)
            entry_room_index = entry_data['entry_room_index']
            if not 0 <= entry_room_index < len(rooms_data):
                entry_room_index = 0
        except Exception as e:
            entry_room_index = 0
            if len(rooms_data) > 1:
                largest_area = 0
                for (i, room) in enumerate(rooms_data):
                    area = room['dimensions']['width'] * room['dimensions']['length']
                    if area > largest_area:
                        largest_area = area
                        entry_room_index = i
            debug_info['steps'].append(f'Step 3 failed: {str(e)}, using fallback entry room {entry_room_index}')
    debug_info['steps'].append('Step 4: Calculating minimum spanning tree')
    if nx.is_connected(room_graph):
        mst = nx.minimum_spanning_tree(room_graph, weight='weight')
    else:
        components = list(nx.connected_components(room_graph))
        mst = nx.Graph()
        for component in components:
            subgraph = room_graph.subgraph(component)
            if len(component) > 1:
                component_mst = nx.minimum_spanning_tree(subgraph, weight='weight')
                mst = nx.union(mst, component_mst)
            else:
                mst.add_node(list(component)[0])
        component_list = [list(comp) for comp in components]
        for i in range(len(component_list) - 1):
            min_distance = float('inf')
            best_edge = None
            for room1 in component_list[i]:
                for room2 in component_list[i + 1]:
                    pos1 = rooms_data[room1]['position']
                    pos2 = rooms_data[room2]['position']
                    distance = ((pos1['x'] - pos2['x']) ** 2 + (pos1['y'] - pos2['y']) ** 2) ** 0.5
                    if distance < min_distance:
                        min_distance = distance
                        best_edge = (room1, room2)
            if best_edge:
                mst.add_edge(best_edge[0], best_edge[1], weight=min_distance, artificial=True)
    debug_info['steps'].append('Step 5: Selecting door and window placements')
    if len(rooms_data) == 1:
        debug_info['steps'].append('Step 5: Single room detected - no connecting doors needed')
        print(f'🏠 Single room layout detected - skipping connecting door placement', file=sys.stderr)
    if len(rooms_data) == 1:
        leaf_nodes = [0]
    else:
        leaf_nodes = [node for node in mst.nodes() if mst.degree(node) == 1]
    if entry_room_index not in leaf_nodes and mst.degree(entry_room_index) > 1:
        leaf_nodes.append(entry_room_index)
    required_doors = []
    for edge in mst.edges():
        (room1, room2) = edge
        for (i, shared_wall) in enumerate(shared_walls_info['room_room_walls']):
            if shared_wall['room1']['index'] == room1 and shared_wall['room2']['index'] == room2 or (shared_wall['room1']['index'] == room2 and shared_wall['room2']['index'] == room1):
                required_doors.append({'shared_wall_index': i, 'wall_size': shared_wall['overlap_length'], 'adjacent_room_types': [shared_wall['room1']['room_type'], shared_wall['room2']['room_type']]})
                break
    remaining_shared_walls = []
    for (i, shared_wall) in enumerate(shared_walls_info['room_room_walls']):
        room1 = shared_wall['room1']['index']
        room2 = shared_wall['room2']['index']
        is_mst_edge = any((door['shared_wall_index'] == i for door in required_doors))
        if not is_mst_edge:
            remaining_shared_walls.append({'shared_wall_index': i, 'wall_size': shared_wall['overlap_length'], 'adjacent_room_types': [shared_wall['room1']['room_type'], shared_wall['room2']['room_type']]})
    additional_doors = []
    if remaining_shared_walls:
        max_additional_doors = max(1, int(len(remaining_shared_walls) * 0.3))
        additional_doors = random.sample(remaining_shared_walls, min(max_additional_doors, len(remaining_shared_walls)))
    all_selected_doors = required_doors + additional_doors
    entry_room_exterior_walls = [i for (i, wall) in enumerate(shared_walls_info['room_exterior_walls']) if wall['room']['index'] == entry_room_index]
    if entry_room_exterior_walls:
        entry_door_wall_idx = min(entry_room_exterior_walls, key=lambda i: shared_walls_info['room_exterior_walls'][i]['overlap_length'])
    elif shared_walls_info['room_exterior_walls']:
        entry_door_wall_idx = 0
        entry_room_index = shared_walls_info['room_exterior_walls'][entry_door_wall_idx]['room']['index']
    else:
        raise ValueError('No exterior walls found in the layout')
    entry_door_info = {'room_index': entry_room_index, 'wall_index': entry_door_wall_idx}
    required_windows = []
    for room_idx in leaf_nodes:
        room_exterior_walls = [i for (i, wall) in enumerate(shared_walls_info['room_exterior_walls']) if wall['room']['index'] == room_idx and i != entry_door_wall_idx]
        if room_exterior_walls:
            best_wall_idx = max(room_exterior_walls, key=lambda i: shared_walls_info['room_exterior_walls'][i]['overlap_length'])
            required_windows.append({'exterior_wall_index': best_wall_idx, 'wall_size': shared_walls_info['room_exterior_walls'][best_wall_idx]['overlap_length'], 'wall_room_type': shared_walls_info['room_exterior_walls'][best_wall_idx]['room']['room_type']})
    remaining_exterior_walls = []
    for (i, wall) in enumerate(shared_walls_info['room_exterior_walls']):
        is_already_selected = any((w['exterior_wall_index'] == i for w in required_windows))
        is_entry_door_wall = i == entry_door_wall_idx
        if not is_already_selected and (not is_entry_door_wall):
            remaining_exterior_walls.append({'exterior_wall_index': i, 'wall_size': wall['overlap_length'], 'wall_room_type': wall['room']['room_type']})
    additional_windows = []
    if remaining_exterior_walls:
        max_additional_windows = max(1, int(len(remaining_exterior_walls) * 0.4))
        additional_windows = random.sample(remaining_exterior_walls, min(max_additional_windows, len(remaining_exterior_walls)))
    all_selected_windows = required_windows + additional_windows
    print(f'🚪 Selected connecting doors: {len(all_selected_doors)}')
    print(f'🪟 Selected windows: {len(all_selected_windows)}')
    print(f'🚪 Entry door wall index: {entry_door_wall_idx} (excluded from window placement)')
    if len(rooms_data) == 1:
        print(f'🏠 Single room layout: only entry door and windows will be placed')
    debug_info['steps'].append('Step 6: Getting detailed door and window specifications from LLM')
    try:
        visualization_path = generate_floor_plan_visualization(rooms_data)
        with open(visualization_path, 'rb') as image_file:
            image_data = image_file.read()
        debug_vis_dir = f'{SERVER_ROOT_DIR}/vis'
        os.makedirs(debug_vis_dir, exist_ok=True)
        timestamp = int(time.time())
        debug_vis_path = os.path.join(debug_vis_dir, f'floor_plan_graph_{timestamp}.png')
        with open(debug_vis_path, 'wb') as debug_file:
            debug_file.write(image_data)
        print(f'🔍 Debug: Graph-based floor plan visualization saved to {debug_vis_path}', file=sys.stderr)
        os.unlink(visualization_path)
    except Exception as e:
        raise ValueError(f'Failed to generate floor plan visualization: {str(e)}')
    connecting_doors_section = ''
    if len(all_selected_doors) > 0:
        connecting_doors_section = f'\nCONNECTING DOORS ({len(all_selected_doors)} total):\n{json.dumps(all_selected_doors, indent=2)}'
    else:
        connecting_doors_section = '\nCONNECTING DOORS: None (single room layout - no connecting doors needed)'
    detailed_prompt = f'''You are an expert architect providing detailed door and window specifications for a floor plan.\n\nORIGINAL REQUEST: "{input_text}"\n\nSELECTED DOOR AND WINDOW PLACEMENTS:\n\nENTRY DOOR:\n- Room Index: {entry_door_info['room_index']} ({rooms_data[entry_door_info['room_index']]['room_type']})\n- Exterior Wall Index: {entry_door_info['wall_index']}\n{connecting_doors_section}\n\nWINDOWS ({len(all_selected_windows)} total):\n{json.dumps(all_selected_windows, indent=2)}\n\nROOM LAYOUT:\n{json.dumps([{'room_index': i, 'room_type': room['room_type'], 'dimensions': f"{room['dimensions']['width']}m × {room['dimensions']['length']}m × {room['dimensions']['height']}m"} for (i, room) in enumerate(rooms_data)], indent=2)}\n\nTASK: Provide detailed specifications for entry door{(', each selected connecting door' if len(all_selected_doors) > 0 else '')} and window. Return EXACTLY this JSON format:\n\n```json\n{{\n    "entry_door": {{\n        "exterior_wall_index": {entry_door_wall_idx},\n        "center_position": float_0_to_1 (Position along the exterior wall segment (0.0 = start, 1.0 = end), choose float number between 0.20 to 0.80 randomly),\n        "width": float_meters,\n        "height": float_meters,\n        "door_type": "entry/main/front",\n        "door_material": "Brief description of door style/material",\n        "reasoning": "Why this location was chosen"\n    }},{(f"""
    "connecting_doors": [
        {{
            "shared_wall_index": int,
            "center_position_on_shared_wall": float_0_to_1 (Position along the shared wall segment (0.0 = start, 1.0 = end), choose float number between 0.20 to 0.80 randomly),
            "width": float_meters,
            "height": float_meters,
            "door_type": "interior/standard",
            "opening": bool,
            "door_material": "Brief description of door style/material",
            "reasoning": "Why this connection is needed"
        }}
    ],""" if len(all_selected_doors) > 0 else """
    "connecting_doors": [],""")}\n    "windows": [\n        {{\n            "exterior_wall_index": int,\n            "center_position": float_0_to_1 (Position along the exterior wall segment (0.0 = start, 1.0 = end), choose float number between 0.20 to 0.80 randomly),\n            "width": float_meters,\n            "height": float_meters,\n            "sill_height": float_meters,\n            "window_type": "standard",\n            "window_grid": [nx, ny],\n            "window_appearance_description": "Detailed description combining frame style and glass type, including the color style of the frame and glass.",\n            "glass_color": [r, g, b],\n            "frame_color": [r, g, b],\n            "reasoning": "Why this window placement was chosen"\n        }}\n    ]\n}}\n```\n\nCONSTRAINTS:\n- Entry door: width 0.8-1.2m, height 2.0-2.1m\n{(f'- Interior doors: width 0.7-0.9m, height 2.0-2.1m' if len(all_selected_doors) > 0 else '')}\n{(f'- Wide openings: width 1.2-3.0m, height 2.0-2.4m' if len(all_selected_doors) > 0 else '')}\n- Windows: width 0.8-3.0m, height 0.8-2.0m, sill_height 0.3-1.2m\n- Window grid: [nx, ny]  where nx=horizontal grid, ny=vertical grid\nExamples:\n\n"fixed": (1, 1),\n"slider": (2, 1),  # 2 horizontal panes, 1 vertical\n"hung": (1, 2),    # 1 horizontal pane, 2 vertical\n"bay_small": (2, 2),\n"bay_medium": (3, 2),\n"bay_large": (3, 3),\n\n- Window colors: Generate DISTINCT and CONTRASTING glass_color and frame_color as RGB values [0-255].\nCRITICAL: Ensure high visual contrast between glass and frame colors - they must be easily distinguishable.\nColors should be consistent with the window appearance description.\n\n  GLASS COLOR EXAMPLES (choose ONE type, ensure contrast with frame):\n  * Clear Glass: [245, 250, 255] (bright clear), [250, 248, 240] (warm clear), [248, 252, 255] (cool clear)\n  * Tinted Glass: [200, 230, 255] (light blue tint), [230, 245, 220] (light green tint), [255, 245, 220] (warm amber)\n  * Privacy Glass: [220, 220, 220] (frosted white), [200, 210, 225] (frosted blue), [210, 200, 190] (frosted bronze)\n  * Colored Glass: [150, 200, 255] (sky blue), [200, 255, 200] (mint green), [255, 220, 180] (warm amber)\n  \n  FRAME COLOR EXAMPLES (choose to CONTRAST with glass color):\n  * Dark Frames: [40, 30, 20] (dark brown), [25, 25, 25] (charcoal black), [60, 40, 30] (dark walnut)\n  * Light Frames: [250, 245, 235] (cream white), [240, 240, 240] (light gray), [245, 240, 225] (off-white)\n  * Colored Frames: [120, 80, 60] (medium brown), [80, 100, 120] (blue-gray), [100, 120, 80] (sage green)\n  * Metal Frames: [180, 180, 180] (brushed aluminum), [160, 140, 120] (bronze), [100, 100, 100] (gunmetal)\n\n  CONTRAST GUIDELINES:\n  * Light glass (245+) → Use dark frame (60 or below)\n  * Dark/colored glass (200 or below) → Use light frame (200+) or contrasting color\n  * Ensure at least 100+ difference in brightness between glass and frame RGB averages\n- Position values are 0.0-1.0 (center of wall)\n{(f'- Use opening=true for wide openings, opening=false for doors with frames' if len(all_selected_doors) > 0 else '')}\n'''
    try:
        debug_info['prompts'].append({'step': '6_detailed_specifications', 'prompt': detailed_prompt})
        response = call_vlm(vlm_type='openai', max_tokens=8000, temperature=0.2, messages=[{'role': 'user', 'content': [{'type': 'text', 'text': detailed_prompt}]}])
        if not response or not hasattr(response, 'content') or (not response.content):
            raise ValueError('No response received from Claude API')
        if len(response.content) == 0 or not hasattr(response.content[0], 'text'):
            raise ValueError('Claude API response content is empty or invalid')
        response_text = response.content[0].text
        debug_info['responses'].append({'step': '6_detailed_specifications', 'response': response_text})
        if not response_text or not response_text.strip():
            raise ValueError('Claude API returned empty response text')
        try:
            claude_response_json = extract_json_from_response(response_text)
            if not claude_response_json:
                raise ValueError('Could not extract JSON content from Claude response')
            claude_response = json.loads(claude_response_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse Claude's JSON response: {e}")
        debug_info['steps'].append('Step 6 completed: Detailed specifications received')
    except Exception as e:
        raise ValueError(f'Error getting detailed specifications: {e}')
    debug_info['steps'].append('Step 7: Formatting output to match original function interface')
    result = {'rooms': []}
    for (i, room_data) in enumerate(rooms_data):
        room_result = {'room_type': room_data['room_type'], 'dimensions': room_data['dimensions'], 'position': room_data['position'], 'doors': [], 'windows': []}
        result['rooms'].append(room_result)
    debug_info['steps'].append('Step 7.1: Filtering and correcting LLM outputs')
    entry_door_corrected = False
    if 'entry_door' not in claude_response or not claude_response['entry_door']:
        entry_door_corrected = True
        debug_info['steps'].append('Step 7.1a: Entry door missing from LLM output, creating one')
        print(f'Step 7.1a: Entry door missing from LLM output, creating one', file=sys.stderr)
    else:
        entry_door = claude_response['entry_door']
        entry_wall_idx = entry_door.get('exterior_wall_index', -1)
        if entry_wall_idx != entry_door_wall_idx:
            entry_door_corrected = True
            debug_info['steps'].append(f'Step 7.1a: Entry door misplaced (wall {entry_wall_idx}), correcting to wall {entry_door_wall_idx}')
            print(f'Step 7.1a: Entry door misplaced (wall {entry_wall_idx}), correcting to wall {entry_door_wall_idx}', file=sys.stderr)
    if entry_door_corrected:
        claude_response['entry_door'] = {'exterior_wall_index': entry_door_wall_idx, 'center_position': float(np.random.uniform(0.2, 0.8)), 'width': 1.0, 'height': 2.1, 'door_type': 'entry', 'door_material': 'solid wood entry door', 'reasoning': 'Auto-generated to match selected entry door location'}
        debug_info['steps'].append(f'Step 7.1a: Created entry door on wall {entry_door_wall_idx}')
        print(f'Step 7.1a: Created entry door on wall {entry_door_wall_idx}', file=sys.stderr)
    connecting_doors_corrected = 0
    if 'connecting_doors' not in claude_response:
        claude_response['connecting_doors'] = []
    llm_door_walls = set()
    if claude_response['connecting_doors']:
        for door in claude_response['connecting_doors']:
            wall_idx = door.get('shared_wall_index', -1)
            if 0 <= wall_idx < len(shared_walls_info['room_room_walls']):
                llm_door_walls.add(wall_idx)
    selected_door_walls = set((door['shared_wall_index'] for door in all_selected_doors))
    missing_doors = selected_door_walls - llm_door_walls
    for wall_idx in missing_doors:
        if wall_idx < len(shared_walls_info['room_room_walls']):
            shared_wall = shared_walls_info['room_room_walls'][wall_idx]
            claude_response['connecting_doors'].append({'shared_wall_index': wall_idx, 'center_position_on_shared_wall': 0.5, 'width': 0.9, 'height': 2.1, 'door_type': 'interior', 'opening': False, 'door_material': 'wooden panel door', 'reasoning': 'Auto-generated to match MST connectivity requirements'})
            connecting_doors_corrected += 1
    if connecting_doors_corrected > 0:
        debug_info['steps'].append(f'Step 7.1b: Added {connecting_doors_corrected} missing connecting doors')
        print(f'Step 7.1b: Added {connecting_doors_corrected} missing connecting doors', file=sys.stderr)
    original_doors = claude_response['connecting_doors'][:]
    claude_response['connecting_doors'] = [door for door in original_doors if door.get('shared_wall_index', -1) in selected_door_walls]
    removed_doors = len(original_doors) - len(claude_response['connecting_doors'])
    if removed_doors > 0:
        debug_info['steps'].append(f'Step 7.1b: Removed {removed_doors} doors not in selected walls')
        print(f'Step 7.1b: Removed {removed_doors} doors not in selected walls', file=sys.stderr)
    windows_corrected = 0
    if 'windows' not in claude_response:
        claude_response['windows'] = []
    original_windows = claude_response['windows'][:]
    claude_response['windows'] = []
    for window in original_windows:
        wall_idx = window.get('exterior_wall_index', -1)
        if not 0 <= wall_idx < len(shared_walls_info['room_exterior_walls']):
            debug_info['steps'].append(f'Step 7.1c: Removed window on invalid wall index {wall_idx}')
            print(f'Step 7.1c: Removed window on invalid wall index {wall_idx}', file=sys.stderr)
            continue
        if wall_idx == entry_door_wall_idx:
            debug_info['steps'].append(f'Step 7.1c: Removed window on entry door wall {wall_idx}')
            print(f'Step 7.1c: Removed window on entry door wall {wall_idx}', file=sys.stderr)
            continue
        claude_response['windows'].append(window)
    removed_windows = len(original_windows) - len(claude_response['windows'])
    if removed_windows > 0:
        debug_info['steps'].append(f'Step 7.1c: Removed {removed_windows} invalid windows')
        print(f'Step 7.1c: Removed {removed_windows} invalid windows', file=sys.stderr)
    if claude_response['connecting_doors']:
        seen_door_walls = set()
        unique_doors = []
        duplicates_removed = 0
        for door in claude_response['connecting_doors']:
            wall_idx = door.get('shared_wall_index', -1)
            if wall_idx not in seen_door_walls:
                seen_door_walls.add(wall_idx)
                unique_doors.append(door)
            else:
                duplicates_removed += 1
        claude_response['connecting_doors'] = unique_doors
        if duplicates_removed > 0:
            debug_info['steps'].append(f'Step 7.1d: Removed {duplicates_removed} duplicate doors on same shared walls')
            print(f'Step 7.1d: Removed {duplicates_removed} duplicate doors on same shared walls', file=sys.stderr)
    if claude_response['windows']:
        seen_window_walls = set()
        unique_windows = []
        duplicates_removed = 0
        for window in claude_response['windows']:
            wall_idx = window.get('exterior_wall_index', -1)
            if wall_idx not in seen_window_walls:
                seen_window_walls.add(wall_idx)
                unique_windows.append(window)
            else:
                duplicates_removed += 1
        claude_response['windows'] = unique_windows
        if duplicates_removed > 0:
            debug_info['steps'].append(f'Step 7.1e: Removed {duplicates_removed} duplicate windows on same exterior walls')
            print(f'Step 7.1e: Removed {duplicates_removed} duplicate windows on same exterior walls', file=sys.stderr)
    debug_info['steps'].append(f"Step 7.1: Filtering completed - Entry door: {('corrected' if entry_door_corrected else 'valid')}, Doors: {len(claude_response['connecting_doors'])}, Windows: {len(claude_response['windows'])}")
    print(f"Step 7.1: Filtering completed - Entry door: {('corrected' if entry_door_corrected else 'valid')}, Doors: {len(claude_response['connecting_doors'])}, Windows: {len(claude_response['windows'])}", file=sys.stderr)
    if 'entry_door' in claude_response and claude_response['entry_door']:
        entry_door = claude_response['entry_door']
        exterior_wall_index = entry_door['exterior_wall_index']
        if 0 <= exterior_wall_index < len(shared_walls_info['room_exterior_walls']):
            exterior_wall = shared_walls_info['room_exterior_walls'][exterior_wall_index]
            room_index = exterior_wall['room']['index']
            door_info = {'width': entry_door['width'], 'height': entry_door['height'], 'position_on_wall': float(np.random.uniform(0.2, 0.8)), 'wall_side': exterior_wall['wall_side'], 'door_type': entry_door.get('door_type', 'entry'), 'opening': False, 'door_material': entry_door.get('door_material', 'solid wood entry door')}
            original_width = door_info['width']
            original_position = door_info['position_on_wall']
            if exterior_wall['direction'] == 'x':
                wall_segment_length = exterior_wall['x_end'] - exterior_wall['x_start']
            else:
                wall_segment_length = exterior_wall['y_end'] - exterior_wall['y_start']
            max_width_constraint = wall_segment_length * 0.8
            safe_start = 0.1
            safe_end = 0.9
            max_half_width_from_start = original_position - safe_start
            max_half_width_from_end = safe_end - original_position
            max_half_width = min(max_half_width_from_start, max_half_width_from_end)
            position_based_max_width = max_half_width * 2 * wall_segment_length
            final_width = min(original_width, max_width_constraint, max(0.7, position_based_max_width))
            if final_width < original_width * 0.8:
                desired_width = min(original_width, max_width_constraint)
                required_half_width_ratio = desired_width / 2 / wall_segment_length
                min_center = safe_start + required_half_width_ratio
                max_center = safe_end - required_half_width_ratio
                if min_center <= max_center:
                    if original_position < min_center:
                        final_position = min_center
                    elif original_position > max_center:
                        final_position = max_center
                    else:
                        final_position = original_position
                    final_width = desired_width
                else:
                    final_position = max(safe_start, min(safe_end, original_position))
            else:
                final_position = max(safe_start, min(safe_end, original_position))
            if exterior_wall['direction'] == 'x':
                segment_start = exterior_wall['x_start']
                segment_length = exterior_wall['x_end'] - exterior_wall['x_start']
                door_absolute_position = segment_start + final_position * segment_length
                room_wall_start = rooms_data[room_index]['position']['x'] if exterior_wall['wall_side'] == 'west' else rooms_data[room_index]['position']['x']
                if exterior_wall['wall_side'] == 'north' or exterior_wall['wall_side'] == 'south':
                    room_wall_start = rooms_data[room_index]['position']['x']
                    room_wall_length = rooms_data[room_index]['dimensions']['width']
                else:
                    room_wall_start = rooms_data[room_index]['position']['y']
                    room_wall_length = rooms_data[room_index]['dimensions']['length']
                wall_relative_position = (door_absolute_position - room_wall_start) / room_wall_length
            else:
                segment_start = exterior_wall['y_start']
                segment_length = exterior_wall['y_end'] - exterior_wall['y_start']
                door_absolute_position = segment_start + final_position * segment_length
                if exterior_wall['wall_side'] == 'east' or exterior_wall['wall_side'] == 'west':
                    room_wall_start = rooms_data[room_index]['position']['y']
                    room_wall_length = rooms_data[room_index]['dimensions']['length']
                else:
                    room_wall_start = rooms_data[room_index]['position']['x']
                    room_wall_length = rooms_data[room_index]['dimensions']['width']
                wall_relative_position = (door_absolute_position - room_wall_start) / room_wall_length
            door_info['width'] = final_width
            door_info['position_on_wall'] = wall_relative_position
            print(f"🚪 Adding ENTRY door to Room {room_index} ({rooms_data[room_index]['room_type']}) {door_info['wall_side']} wall:", file=sys.stderr)
            print(f"   Exterior segment: Y[{exterior_wall['y_start']}, {exterior_wall['y_end']}] (length: {wall_segment_length})", file=sys.stderr)
            print(f'   Segment-relative position: {final_position:.3f} → Absolute position: {door_absolute_position:.3f}', file=sys.stderr)
            print(f'   Wall-relative position: {wall_relative_position:.3f}', file=sys.stderr)
            result['rooms'][room_index]['doors'].append(door_info)
    if 'connecting_doors' in claude_response and claude_response['connecting_doors']:
        for connecting_door in claude_response['connecting_doors']:
            shared_wall_index = connecting_door['shared_wall_index']
            if 0 <= shared_wall_index < len(shared_walls_info['room_room_walls']):
                shared_wall = shared_walls_info['room_room_walls'][shared_wall_index]
                room1_index = shared_wall['room1']['index']
                room2_index = shared_wall['room2']['index']
                original_door_width = connecting_door['width']
                room1_max_width = rooms_data[room1_index]['dimensions']['width'] * 0.8 if shared_wall['room1_wall'] in ['north', 'south'] else rooms_data[room1_index]['dimensions']['length'] * 0.8
                room2_max_width = rooms_data[room2_index]['dimensions']['width'] * 0.8 if shared_wall['room2_wall'] in ['north', 'south'] else rooms_data[room2_index]['dimensions']['length'] * 0.8
                shared_wall_max_width = shared_wall['overlap_length'] * 0.8
                final_door_width = min(original_door_width, room1_max_width, room2_max_width, shared_wall_max_width)
                original_shared_position = connecting_door['center_position_on_shared_wall']
                door_half_width_ratio = final_door_width / 2 / shared_wall['overlap_length']
                min_safe_position = max(0.1, door_half_width_ratio + 0.05)
                max_safe_position = min(0.9, 1.0 - door_half_width_ratio - 0.05)
                final_shared_position = max(min_safe_position, min(max_safe_position, original_shared_position))
                room1_wall_position = calculate_room_wall_position_from_shared_wall(shared_wall, final_shared_position, rooms_data[room1_index], True)
                room2_wall_position = calculate_room_wall_position_from_shared_wall(shared_wall, final_shared_position, rooms_data[room2_index], False)
                if final_door_width < 0.4:
                    continue
                door1_info = {'width': final_door_width, 'height': connecting_door['height'], 'position_on_wall': room1_wall_position, 'wall_side': shared_wall['room1_wall'], 'door_type': connecting_door.get('door_type', 'interior'), 'opening': connecting_door.get('opening', False), 'door_material': connecting_door.get('door_material', 'standard wooden door')}
                door2_info = {'width': final_door_width, 'height': connecting_door['height'], 'position_on_wall': room2_wall_position, 'wall_side': shared_wall['room2_wall'], 'door_type': connecting_door.get('door_type', 'interior'), 'opening': connecting_door.get('opening', False), 'door_material': connecting_door.get('door_material', 'standard wooden door')}
                print(f"🔗 Adding CONNECTING door between Room {room1_index} ({rooms_data[room1_index]['room_type']}) and Room {room2_index} ({rooms_data[room2_index]['room_type']}):", file=sys.stderr)
                print(f"   Shared segment: Y[{shared_wall['y_start']}, {shared_wall['y_end']}] (length: {shared_wall['overlap_length']})", file=sys.stderr)
                print(f'   Segment-relative position: {final_shared_position:.3f}', file=sys.stderr)
                print(f"   Room1 {shared_wall['room1_wall']} wall position: {room1_wall_position:.3f}", file=sys.stderr)
                print(f"   Room2 {shared_wall['room2_wall']} wall position: {room2_wall_position:.3f}", file=sys.stderr)
                result['rooms'][room1_index]['doors'].append(door1_info)
                result['rooms'][room2_index]['doors'].append(door2_info)
    if 'windows' in claude_response and claude_response['windows']:
        for window in claude_response['windows']:
            exterior_wall_index = window['exterior_wall_index']
            if 0 <= exterior_wall_index < len(shared_walls_info['room_exterior_walls']):
                exterior_wall = shared_walls_info['room_exterior_walls'][exterior_wall_index]
                room_index = exterior_wall['room']['index']
                window_info = {'width': window['width'], 'height': window['height'], 'position_on_wall': window['center_position'], 'wall_side': exterior_wall['wall_side'], 'sill_height': window['sill_height'], 'window_type': window.get('window_type', 'standard'), 'window_grid': window.get('window_grid', [1, 1]), 'glass_color': window.get('glass_color', [204, 230, 255]), 'frame_color': window.get('frame_color', [77, 77, 77]), 'window_appearance_description': window.get('window_appearance_description', 'standard window')}
                original_window_width = window_info['width']
                original_window_position = window_info['position_on_wall']
                if window_info['wall_side'] in ['north', 'south']:
                    room_max_width = rooms_data[room_index]['dimensions']['width'] * 0.8
                else:
                    room_max_width = rooms_data[room_index]['dimensions']['length'] * 0.8
                exterior_wall_max_width = exterior_wall['overlap_length'] * 0.8
                safe_start = 0.1
                safe_end = 0.9
                max_half_width_from_start = original_window_position - safe_start
                max_half_width_from_end = safe_end - original_window_position
                max_half_width = min(max_half_width_from_start, max_half_width_from_end)
                position_based_max_width = max_half_width * 2 * exterior_wall['overlap_length']
                max_width_constraint = min(room_max_width, exterior_wall_max_width)
                final_window_width = min(original_window_width, max_width_constraint, max(0.8, position_based_max_width))
                if final_window_width < original_window_width * 0.8:
                    desired_width = min(original_window_width, max_width_constraint)
                    required_half_width_ratio = desired_width / 2 / exterior_wall['overlap_length']
                    min_center = safe_start + required_half_width_ratio
                    max_center = safe_end - required_half_width_ratio
                    if min_center <= max_center:
                        if original_window_position < min_center:
                            final_window_position = min_center
                        elif original_window_position > max_center:
                            final_window_position = max_center
                        else:
                            final_window_position = original_window_position
                        final_window_width = desired_width
                    else:
                        final_window_position = max(safe_start, min(safe_end, original_window_position))
                else:
                    final_window_position = max(safe_start, min(safe_end, original_window_position))
                if exterior_wall['direction'] == 'x':
                    segment_start = exterior_wall['x_start']
                    segment_length = exterior_wall['x_end'] - exterior_wall['x_start']
                    window_absolute_position = segment_start + final_window_position * segment_length
                    if exterior_wall['wall_side'] == 'north' or exterior_wall['wall_side'] == 'south':
                        room_wall_start = rooms_data[room_index]['position']['x']
                        room_wall_length = rooms_data[room_index]['dimensions']['width']
                    else:
                        room_wall_start = rooms_data[room_index]['position']['y']
                        room_wall_length = rooms_data[room_index]['dimensions']['length']
                    wall_relative_position = (window_absolute_position - room_wall_start) / room_wall_length
                else:
                    segment_start = exterior_wall['y_start']
                    segment_length = exterior_wall['y_end'] - exterior_wall['y_start']
                    window_absolute_position = segment_start + final_window_position * segment_length
                    if exterior_wall['wall_side'] == 'east' or exterior_wall['wall_side'] == 'west':
                        room_wall_start = rooms_data[room_index]['position']['y']
                        room_wall_length = rooms_data[room_index]['dimensions']['length']
                    else:
                        room_wall_start = rooms_data[room_index]['position']['x']
                        room_wall_length = rooms_data[room_index]['dimensions']['width']
                    wall_relative_position = (window_absolute_position - room_wall_start) / room_wall_length
                window_info['width'] = final_window_width
                window_info['position_on_wall'] = wall_relative_position
                print(f"🪟 Adding WINDOW to Room {room_index} ({rooms_data[room_index]['room_type']}) {window_info['wall_side']} wall:", file=sys.stderr)
                print(f"   Exterior segment: Y[{exterior_wall['y_start']}, {exterior_wall['y_end']}] (length: {segment_length})", file=sys.stderr)
                print(f'   Segment-relative position: {final_window_position:.3f} → Absolute position: {window_absolute_position:.3f}', file=sys.stderr)
                print(f'   Wall-relative position: {wall_relative_position:.3f}', file=sys.stderr)
                existing_items = []
                for door in result['rooms'][room_index]['doors']:
                    if door['wall_side'] == window_info['wall_side']:
                        existing_items.append({'position': door['position_on_wall'], 'width': door['width'], 'type': 'door'})
                for existing_window in result['rooms'][room_index]['windows']:
                    if existing_window['wall_side'] == window_info['wall_side']:
                        existing_items.append({'position': existing_window['position_on_wall'], 'width': existing_window['width'], 'type': 'window'})
                if existing_items:
                    wall_length = exterior_wall['overlap_length']
                    window_width_ratio = window_info['width'] / wall_length
                    window_half_width_ratio = window_width_ratio / 2
                    existing_items.sort(key=lambda x: x['position'])
                    safe_position = window_info['position_on_wall']
                    position_found = False
                    occupied_segments = []
                    for item in existing_items:
                        item_width_ratio = item['width'] / wall_length
                        item_half_width_ratio = item_width_ratio / 2
                        margin = 0.05
                        segment_start = max(0.0, item['position'] - item_half_width_ratio - margin)
                        segment_end = min(1.0, item['position'] + item_half_width_ratio + margin)
                        occupied_segments.append((segment_start, segment_end))
                    if occupied_segments:
                        occupied_segments.sort()
                        merged_segments = [occupied_segments[0]]
                        for (current_start, current_end) in occupied_segments[1:]:
                            (last_start, last_end) = merged_segments[-1]
                            if current_start <= last_end:
                                merged_segments[-1] = (last_start, max(last_end, current_end))
                            else:
                                merged_segments.append((current_start, current_end))
                    min_required_space = window_width_ratio + 0.1
                    current_start = safe_position - window_half_width_ratio
                    current_end = safe_position + window_half_width_ratio
                    conflicts_with_current = any((not (current_end <= seg_start or current_start >= seg_end) for (seg_start, seg_end) in merged_segments))
                    if not conflicts_with_current and 0.1 <= current_start and (current_end <= 0.9):
                        position_found = True
                    else:
                        for i in range(len(merged_segments) + 1):
                            if i == 0:
                                gap_start = 0.1
                                gap_end = merged_segments[0][0] if merged_segments else 0.9
                            elif i == len(merged_segments):
                                gap_start = merged_segments[-1][1]
                                gap_end = 0.9
                            else:
                                gap_start = merged_segments[i - 1][1]
                                gap_end = merged_segments[i][0]
                            gap_size = gap_end - gap_start
                            if gap_size >= min_required_space:
                                safe_position = gap_start + gap_size / 2
                                safe_position = max(0.1, min(0.9, safe_position))
                                position_found = True
                                break
                    if not position_found:
                        continue
                    window_info['position_on_wall'] = safe_position
                room_height = rooms_data[room_index]['dimensions']['height']
                original_sill_height = window_info['sill_height']
                original_window_height = window_info['height']
                total_window_height = original_sill_height + original_window_height
                if total_window_height > room_height:
                    max_total_height = room_height - 0.1
                    height_ratio = max_total_height / total_window_height
                    window_info['sill_height'] = original_sill_height * height_ratio
                    window_info['height'] = original_window_height * height_ratio
                    min_sill_height = 0.5
                    if window_info['sill_height'] < min_sill_height:
                        window_info['sill_height'] = min_sill_height
                        remaining_height = max_total_height - min_sill_height
                        if remaining_height > 0.3:
                            window_info['height'] = remaining_height
                        else:
                            continue
                result['rooms'][room_index]['windows'].append(window_info)
    result['debug_info'] = debug_info
    result['debug_info']['steps'].append('Step 7 completed: Output formatted successfully')
    debug_info['steps'].append('Graph-based door and window placement completed successfully')
    debug_vis_dir = f'{SERVER_ROOT_DIR}/vis'
    os.makedirs(debug_vis_dir, exist_ok=True)
    graph_vis_path = os.path.join(debug_vis_dir, f"graph_doors_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    await get_doors_windows_layout_visualization(result, graph_vis_path, mst, entry_room_index)
    print(f'🔍 Debug: Graph-based doors & windows visualization saved to {graph_vis_path}', file=sys.stderr)
    return result
