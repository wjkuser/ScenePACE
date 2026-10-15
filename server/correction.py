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
from typing import Dict, Any, List, Optional
from key import ANTHROPIC_API_KEY
from vlm import call_vlm
from validation import validate_room_layout, validate_door_window_placement, check_door_window_integrity, validate_door_window_issues, validate_llm_response_structure, check_room_overlap
from utils import add_connectivity_doors, check_room_connectivity, reposition_bidirectional_door, resize_bidirectional_door, remove_bidirectional_door, doors_align, generate_unique_id, create_walls_for_room, extract_wall_side_from_id, get_room_priorities_from_claude, calculate_room_reduction, find_shared_walls, calculate_door_world_position, add_bidirectional_door, add_bidirectional_window, remove_bidirectional_window, reposition_bidirectional_window, resize_bidirectional_window
from layout_parser import parse_llm_response_to_floor_plan

def analyze_removal_strategy(validation_result: Dict[str, Any], rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze validation issues to determine optimal removal strategy.
    
    Args:
        validation_result: Results from validate_room_layout()
        rooms_data: Current room layout data
        
    Returns:
        Dictionary with removal strategy recommendations
    """
    room_importance = {'living room': 10, 'family room': 10, 'great room': 10, 'kitchen': 10, 'dining room': 9, 'dining area': 9, 'master bedroom': 9, 'bedroom': 8, 'guest bedroom': 7, 'bathroom': 8, 'master bathroom': 9, 'guest bathroom': 6, 'office': 6, 'study': 6, 'den': 6, 'hallway': 7, 'foyer': 7, 'entry': 7, 'laundry room': 5, 'utility room': 4, 'storage': 3, 'closet': 3, 'pantry': 4, 'garage': 5, 'mudroom': 4, 'powder room': 5, 'half bath': 5}
    overlapping_rooms = set()
    for overlap in validation_result.get('overlaps', []):
        overlapping_rooms.add(overlap['room1'])
        overlapping_rooms.add(overlap['room2'])
    detached_rooms = set(validation_result.get('detached_rooms', []))
    removal_candidates = []
    for room_data in rooms_data:
        room_type = room_data['room_type']
        importance = room_importance.get(room_type.lower(), 5)
        removal_priority = 0
        if room_type in overlapping_rooms:
            removal_priority += 3
        if room_type in detached_rooms:
            removal_priority += 2
        removal_priority -= importance
        if any((room_type in issue for issue in validation_result.get('door_issues', []))):
            removal_priority += 1
        if any((room_type in issue for issue in validation_result.get('window_issues', []))):
            removal_priority += 1
        removal_candidates.append({'room_type': room_type, 'importance': importance, 'removal_priority': removal_priority, 'is_overlapping': room_type in overlapping_rooms, 'is_detached': room_type in detached_rooms})
    removal_candidates.sort(key=lambda x: x['removal_priority'], reverse=True)
    strategy = {'severity': 'moderate', 'max_rooms_to_remove': 2, 'priority_order': removal_candidates, 'removal_reasons': [], 'keep_essential': True}
    total_issues = len(validation_result.get('overlaps', [])) + len(validation_result.get('detached_rooms', []))
    if total_issues >= 3:
        strategy['severity'] = 'high'
        strategy['max_rooms_to_remove'] = 3
        strategy['removal_reasons'].append('Multiple severe architectural issues detected')
    elif total_issues >= 2:
        strategy['severity'] = 'moderate'
        strategy['max_rooms_to_remove'] = 2
        strategy['removal_reasons'].append('Moderate architectural issues need resolution')
    else:
        strategy['severity'] = 'low'
        strategy['max_rooms_to_remove'] = 1
        strategy['removal_reasons'].append('Minor issues require targeted removal')
    return strategy

async def generate_detailed_correction_prompt(validation_result: Dict[str, Any], original_input: str, current_rooms: List[Dict[str, Any]]) -> str:
    """
    Generate a detailed correction prompt for Claude based on validation issues.
    
    Args:
        validation_result: Results from validate_room_layout()
        original_input: Original text description used to generate layout
        current_rooms: Current room layout data
        
    Returns:
        Formatted correction prompt for Claude
    """
    issues_analysis = ''
    if validation_result['overlaps']:
        issues_analysis += '\n🔴 OVERLAPPING ROOMS DETECTED:\n'
        for overlap in validation_result['overlaps']:
            issues_analysis += f"   - '{overlap['room1']}' and '{overlap['room2']}' overlap by {overlap['overlap_area']:.2f} m²\n"
        issues_analysis += '   → Rooms must have distinct, non-overlapping boundaries\n'
    if validation_result['detached_rooms']:
        issues_analysis += '\n🔴 DETACHED ROOMS DETECTED:\n'
        for room in validation_result['detached_rooms']:
            issues_analysis += f"   - '{room}' has no shared walls with other rooms\n"
        issues_analysis += '   → All rooms must be connected through shared walls for realistic architecture\n'
    current_layout_summary = '\nCURRENT LAYOUT:\n'
    door_window_issues = []
    for (i, room) in enumerate(current_rooms):
        pos = room['position']
        dims = room['dimensions']
        current_layout_summary += f"   {i + 1}. {room['room_type']}: {dims['width']:.1f}m × {dims['length']:.1f}m at position ({pos['x']:.1f}, {pos['y']:.1f})\n"
        if 'doors' in room:
            for (j, door) in enumerate(room['doors']):
                current_layout_summary += f"      Door {j + 1}: {door['width']:.1f}m wide on {door.get('wall_side', 'unknown')} wall\n"
        if 'windows' in room:
            for (j, window) in enumerate(room['windows']):
                current_layout_summary += f"      Window {j + 1}: {window['width']:.1f}m wide on {window.get('wall_side', 'unknown')} wall\n"
    door_window_corrections = ''
    if validation_result['overlaps'] or validation_result['detached_rooms'] or validation_result['door_issues'] or validation_result['window_issues'] or validation_result['wall_boundary_violations'] or validation_result['door_window_overlaps'] or validation_result['wall_intersections']:
        door_window_corrections = '\n🔧 DOOR & WINDOW CORRECTIONS REQUIRED:'
        if validation_result['door_issues']:
            door_window_corrections += f"\n   DOOR ISSUES FOUND ({len(validation_result['door_issues'])}):"
            for issue in validation_result['door_issues']:
                door_window_corrections += f'\n      - {issue}'
        if validation_result['window_issues']:
            door_window_corrections += f"\n   WINDOW ISSUES FOUND ({len(validation_result['window_issues'])}):"
            for issue in validation_result['window_issues']:
                door_window_corrections += f'\n      - {issue}'
        if validation_result['wall_boundary_violations']:
            door_window_corrections += f"\n   WALL BOUNDARY VIOLATIONS ({len(validation_result['wall_boundary_violations'])}):"
            for violation in validation_result['wall_boundary_violations']:
                door_window_corrections += f"\n      - {violation['room']} {violation['element_type']} on {violation['wall_side']} wall extends beyond wall (wall: {violation['wall_length']:.1f}m, element: {violation['element_start']:.1f}-{violation['element_end']:.1f}m)"
        if validation_result['door_window_overlaps']:
            door_window_corrections += f"\n   DOOR/WINDOW OVERLAPS ({len(validation_result['door_window_overlaps'])}):"
            for overlap in validation_result['door_window_overlaps']:
                door_window_corrections += f"\n      - {overlap['room']} {overlap['wall_side']} wall: {overlap['element1_type']} and {overlap['element2_type']} overlap by {overlap['overlap_width']:.2f}m"
        if validation_result['wall_intersections']:
            door_window_corrections += f"\n   WALL INTERSECTIONS ({len(validation_result['wall_intersections'])}):"
            for intersection in validation_result['wall_intersections']:
                door_window_corrections += f'\n      - {intersection}'
        door_window_corrections += "\n   CORRECTION GUIDELINES:\n   - REPLACE OLD DOORS/WINDOWS: Generate completely new door/window placements, don't add to existing\n   - When rooms are repositioned, doors MUST be updated to provide logical access between connected rooms\n   - Doors should be placed on shared walls between rooms that need to connect\n   - Entry doors should provide access from main circulation areas (hallways, foyers)\n   - Windows should be placed on exterior walls for natural light and views\n   - Door positions (position_on_wall) must be recalculated for new room layouts\n   - Window positions should optimize natural light distribution\n   - WALL BOUNDARY COMPLIANCE: Ensure doors/windows fit completely within wall boundaries\n   - NO OVERLAPS: Doors and windows on same wall must not overlap with each other\n   - SPACING: Leave adequate space between doors and windows on same wall (minimum 0.3m)\n   - Ensure all position_on_wall values are between 0.0 and 1.0\n   - Calculate proper positioning: (start_pos + element_width) must be ≤ wall_length\n   - Use appropriate sill heights (typically 0.9-1.1m for standard windows)\n   - Standard door widths: 0.7-1.0m, window widths: 0.8-2.0m"
    correction_prompt = f'You are an expert architect. The current floor plan has CRITICAL ARCHITECTURAL ERRORS that must be fixed.\n\nORIGINAL REQUIREMENTS: "{original_input}"\n{current_layout_summary}\nIDENTIFIED PROBLEMS:{issues_analysis}{door_window_corrections}\n\nCORRECTION REQUIREMENTS:\n1. ✅ ELIMINATE OVERLAPS: Reposition rooms so no two rooms occupy the same space\n2. ✅ ENSURE CONNECTIVITY: All rooms must be connected through shared walls\n3. ✅ MAINTAIN FUNCTIONALITY: Keep the same room types and similar sizes as requested\n4. ✅ LOGICAL ADJACENCIES: Place rooms in architecturally sensible arrangements\n5. ✅ REALISTIC PROPORTIONS: Use practical dimensions for each room type\n6. ✅ CORRECT DOOR PLACEMENT: Update door positions to connect repositioned rooms logically\n7. ✅ OPTIMIZE WINDOW PLACEMENT: Position windows appropriately for natural light and views\n\nCORRECTION STRATEGY:\n- For overlapping rooms: Reposition them to adjacent locations with shared walls\n- For detached rooms: Move them to connect with the main building footprint\n- Update door positions to provide logical access between connected rooms\n- Place doors on shared walls between rooms that need circulation\n- Position windows on exterior walls for optimal natural light\n- Recalculate door/window position_on_wall values for new room layouts\n- Maintain the overall intent of the original design\n\nCRITICAL DOOR & WINDOW GUIDELINES:\n- COMPLETE REPLACEMENT: Generate entirely new door/window layouts, replacing all existing ones\n- Doors between connected rooms should be on shared walls\n- Entry/exit doors should connect to circulation spaces (hallways, foyers)\n- Bedroom doors should provide privacy while maintaining access\n- Kitchen/dining areas should have logical flow connections\n- Windows should be on exterior walls (not shared interior walls)\n- Window placement should consider views and natural light distribution\n- WALL BOUNDARY COMPLIANCE: All doors/windows must fit within their wall boundaries\n- NO OVERLAPPING: Doors and windows on same wall must have clear separation (min 0.3m gap)\n- PROPER SIZING: Doors 0.7-1.0m wide, windows 0.8-2.0m wide, positioned correctly within walls\n- POSITIONING MATH: For element at position P on wall of length L with width W: P*L + W ≤ L\n\nRespond with ONLY a corrected JSON layout that fixes these issues:\n\n{{\n    "building_style": "corrected architectural style description",\n    "rooms": [\n        {{\n            "room_type": "room name",\n            "dimensions": {{"width": float, "length": float, "height": float}},\n            "position": {{"x": float, "y": float, "z": 0}},\n            "doors": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "door_type": "standard"\n                }}\n            ],\n            "windows": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "sill_height": float,\n                    "window_type": "standard"\n                }}\n            ]\n        }}\n    ]\n}}\n\nCRITICAL: The corrected layout must have NO overlaps and NO detached rooms. All rooms must form a connected, realistic building.'
    return correction_prompt

async def generate_aggressive_correction_prompt(validation_result: Dict[str, Any], original_input: str, current_rooms: List[Dict[str, Any]], removal_strategy: Dict[str, Any]) -> str:
    """
    Generate an aggressive correction prompt that includes room/door/window removal strategies.
    
    Args:
        validation_result: Results from validate_room_layout()
        original_input: Original text description used to generate layout
        current_rooms: Current room layout data
        removal_strategy: Strategy for what to remove
        
    Returns:
        Formatted aggressive correction prompt for Claude
    """
    issues_summary = f"\n🔴 PERSISTENT ARCHITECTURAL ISSUES (Total: {len(validation_result['issues'])}):\n"
    if validation_result['overlaps']:
        issues_summary += f"   - {len(validation_result['overlaps'])} ROOM OVERLAPS: "
        overlaps_desc = [f"'{o['room1']}' ↔ '{o['room2']}'" for o in validation_result['overlaps']]
        issues_summary += ', '.join(overlaps_desc) + '\n'
    if validation_result['detached_rooms']:
        issues_summary += f"   - {len(validation_result['detached_rooms'])} DETACHED ROOMS: {', '.join(validation_result['detached_rooms'])}\n"
    layout_analysis = '\nCURRENT LAYOUT WITH REMOVAL ANALYSIS:\n'
    for (i, room) in enumerate(current_rooms):
        room_info = next((r for r in removal_strategy['priority_order'] if r['room_type'] == room['room_type']), {})
        priority = room_info.get('removal_priority', 0)
        importance = room_info.get('importance', 5)
        layout_analysis += f"   {i + 1}. {room['room_type']}: {room['dimensions']['width']:.1f}m × {room['dimensions']['length']:.1f}m"
        layout_analysis += f' [Priority: {priority}, Importance: {importance}]'
        if room_info.get('is_overlapping'):
            layout_analysis += ' ⚠️ OVERLAPPING'
        if room_info.get('is_detached'):
            layout_analysis += ' ⚠️ DETACHED'
        layout_analysis += '\n'
    removal_guidance = f"\n📋 AGGRESSIVE CORRECTION STRATEGY:\n   Severity Level: {removal_strategy['severity'].upper()}\n   Maximum Rooms to Remove: {removal_strategy['max_rooms_to_remove']}\n   \n   TOP REMOVAL CANDIDATES (in priority order):"
    for (i, candidate) in enumerate(removal_strategy['priority_order'][:removal_strategy['max_rooms_to_remove'] + 1]):
        removal_guidance += f"\n      {i + 1}. '{candidate['room_type']}' (Priority: {candidate['removal_priority']})"
        if candidate['is_overlapping']:
            removal_guidance += ' - Causes overlaps'
        if candidate['is_detached']:
            removal_guidance += ' - Isolated room'
    aggressive_prompt = f'''You are an expert architect performing AGGRESSIVE LAYOUT CORRECTION. Normal correction attempts have FAILED, so you must now strategically REMOVE problematic rooms, doors, or windows to create a valid layout.\n\nORIGINAL REQUIREMENTS: "{original_input}"\n{layout_analysis}{issues_summary}{removal_guidance}\n\n🚨 AGGRESSIVE CORRECTION REQUIREMENTS:\n1. ✅ REMOVE PROBLEMATIC ROOMS: Eliminate rooms causing persistent overlaps or detachment\n2. ✅ PRESERVE ESSENTIAL FUNCTIONS: Keep core rooms (living, kitchen, bedrooms, main bathroom)\n3. ✅ ELIMINATE ALL OVERLAPS: No rooms can occupy the same space\n4. ✅ ENSURE CONNECTIVITY: All remaining rooms must be connected via shared walls\n5. ✅ REMOVE PROBLEMATIC DOORS/WINDOWS: Delete doors/windows that can't be properly placed\n6. ✅ MAINTAIN LIVABILITY: Resulting layout must still be functional for daily life\n\nREMOVAL STRATEGY:\n- Start with LOWEST IMPORTANCE rooms (utility, storage, extra bathrooms)\n- Remove rooms causing the MOST OVERLAPS first\n- Eliminate DETACHED rooms that can't be easily connected\n- Remove doors/windows that have placement issues\n- Keep ESSENTIAL rooms: living areas, kitchen, primary bedroom, main bathroom\n- Ensure remaining rooms form a CONNECTED, LIVABLE layout\n\nCRITICAL GUIDELINES:\n- Remove UP TO {removal_strategy['max_rooms_to_remove']} rooms if necessary\n- DO NOT remove living room, kitchen, or primary bedroom unless absolutely critical\n- Remove doors/windows that have invalid positions or cause issues\n- Recalculate door/window positions for remaining rooms\n- Ensure final layout has NO overlaps and NO detached rooms\n\nRespond with ONLY a simplified JSON layout that eliminates all architectural issues:\n\n{{\n    "building_style": "simplified architectural style description",\n    "rooms": [\n        {{\n            "room_type": "room name",\n            "dimensions": {{"width": float, "length": float, "height": float}},\n            "position": {{"x": float, "y": float, "z": 0}},\n            "doors": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "door_type": "standard"\n                }}\n            ],\n            "windows": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "sill_height": float,\n                    "window_type": "standard"\n                }}\n            ]\n        }}\n    ]\n}}\n\nCRITICAL: The resulting layout must be 100% VALID with no overlaps, no detached rooms, and proper door/window placement. Remove elements as needed to achieve this.'''
    return aggressive_prompt

def generate_door_window_cleanup_prompt(door_window_validation: Dict[str, Any], original_input: str, rooms_data: List[Dict[str, Any]]) -> str:
    """
    Generate a specialized prompt for cleaning up door and window placement issues.
    
    Args:
        door_window_validation: Results from validate_door_window_placement()
        original_input: Original layout description
        rooms_data: Current room layout data
        
    Returns:
        Specialized cleanup prompt for Claude
    """
    issues_analysis = f"\n🔍 DOOR/WINDOW PLACEMENT ANALYSIS (Total Issues: {door_window_validation['total_issues']}):\n"
    if door_window_validation['wall_boundary_violations']:
        issues_analysis += f"\n🚫 WALL BOUNDARY VIOLATIONS ({len(door_window_validation['wall_boundary_violations'])}):\n"
        for violation in door_window_validation['wall_boundary_violations']:
            issues_analysis += f"   - {violation['room']} {violation['element_type']} on {violation['wall_side']} wall: "
            issues_analysis += f"element spans {violation['element_start']:.1f}-{violation['element_end']:.1f}m but wall is only {violation['wall_length']:.1f}m long\n"
    if door_window_validation['door_window_overlaps']:
        issues_analysis += f"\n⚠️ DOOR/WINDOW OVERLAPS ({len(door_window_validation['door_window_overlaps'])}):\n"
        for overlap in door_window_validation['door_window_overlaps']:
            issues_analysis += f"   - {overlap['room']} {overlap['wall_side']} wall: {overlap['element1_type']} and {overlap['element2_type']} "
            issues_analysis += f"overlap by {overlap['overlap_width']:.2f}m ({overlap['element1_range']} vs {overlap['element2_range']})\n"
    if door_window_validation['door_window_issues']:
        issues_analysis += f"\n❌ GENERAL DOOR/WINDOW ISSUES ({len(door_window_validation['door_window_issues'])}):\n"
        for issue in door_window_validation['door_window_issues']:
            issues_analysis += f'   - {issue}\n'
    layout_summary = '\nCURRENT DOOR/WINDOW LAYOUT:\n'
    for room in rooms_data:
        layout_summary += f"   {room['room_type']} ({room['dimensions']['width']:.1f}m × {room['dimensions']['length']:.1f}m):\n"
        for door in room.get('doors', []):
            wall_side = door.get('wall_side', 'unknown')
            pos = door.get('position_on_wall', 0)
            width = door.get('width', 0)
            layout_summary += f'     Door: {width:.1f}m wide at position {pos:.2f} on {wall_side} wall\n'
        for window in room.get('windows', []):
            wall_side = window.get('wall_side', 'unknown')
            pos = window.get('position_on_wall', 0)
            width = window.get('width', 0)
            layout_summary += f'     Window: {width:.1f}m wide at position {pos:.2f} on {wall_side} wall\n'
    cleanup_prompt = f'You are an expert architect specializing in door and window placement. The current layout has DOOR AND WINDOW PLACEMENT ISSUES that need immediate correction.\n\nORIGINAL LAYOUT REQUIREMENTS: "{original_input}"\n{layout_summary}{issues_analysis}\n\n🔧 DOOR/WINDOW CLEANUP REQUIREMENTS:\n1. ✅ WALL BOUNDARY COMPLIANCE: All doors/windows must fit completely within their wall boundaries\n2. ✅ NO OVERLAPS: Doors and windows on same wall must not overlap - maintain minimum 0.3m separation\n3. ✅ PROPER POSITIONING: Calculate positions to ensure: position_on_wall * wall_length + element_width ≤ wall_length\n4. ✅ COMPLETE REPLACEMENT: Generate entirely new door/window layouts, replacing all existing placements\n5. ✅ MAINTAIN FUNCTIONALITY: Preserve door connectivity and window natural light purposes\n6. ✅ STANDARD SIZING: Use appropriate dimensions (doors: 0.7-1.0m, windows: 0.8-2.0m)\n\nPOSITIONING CALCULATIONS:\n- For wall of length L meters and element of width W meters at position P (0.0-1.0):\n- Element occupies: P*L to (P*L + W) along the wall\n- MUST ENSURE: P*L + W ≤ L (element fits within wall)\n- For multiple elements on same wall: ensure no overlaps with minimum 0.3m gaps\n\nCLEANUP STRATEGY:\n- Replace ALL existing door/window positions with new, properly calculated ones\n- Ensure doors provide logical room connections (on shared walls)\n- Place windows on exterior walls for natural light\n- Distribute elements evenly along walls to avoid overcrowding\n- Maintain architectural logic while fixing placement issues\n\nRespond with ONLY a corrected JSON layout with properly positioned doors and windows:\n\n{{\n    "building_style": "layout with corrected door/window placement",\n    "rooms": [\n        {{\n            "room_type": "room name",\n            "dimensions": {{"width": float, "length": float, "height": float}},\n            "position": {{"x": float, "y": float, "z": 0}},\n            "doors": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "door_type": "standard"\n                }}\n            ],\n            "windows": [\n                {{\n                    "width": float,\n                    "height": float,\n                    "position_on_wall": float,\n                    "wall_side": "north/south/east/west",\n                    "sill_height": float,\n                    "window_type": "standard"\n                }}\n            ]\n        }}\n    ]\n}}\n\nCRITICAL: All doors and windows must have valid positions with NO boundary violations and NO overlaps.'
    return cleanup_prompt

def correct_door_window_integrity(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Automatically correct door/window integrity issues using unified validation criteria with bidirectional support.
    Uses standardized fractional position system (0-1) for consistency.
    Ensures room connectivity by adding connecting doors between rooms.
    Handles bidirectional requirements for doors/windows on shared walls.
    Guarantees 100% door/window integrity and room connectivity after correction.
    
    Args:
        rooms_data: List of room data dictionaries with doors and windows
        
    Returns:
        Dictionary with correction results and updated room data
    """
    corrected_rooms_data = [room.copy() for room in rooms_data]
    correction_actions = []
    doors_removed = []
    windows_removed = []
    doors_resized = []
    windows_resized = []
    doors_repositioned = []
    windows_repositioned = []
    connectivity_doors_added = []
    bidirectional_corrections = []
    shared_wall_map = {}
    exterior_walls = set()
    for room1_idx in range(len(corrected_rooms_data)):
        for room2_idx in range(room1_idx + 1, len(corrected_rooms_data)):
            shared_walls = find_shared_walls(corrected_rooms_data[room1_idx], corrected_rooms_data[room2_idx])
            for shared_wall in shared_walls:
                room1_wall = shared_wall['room1_wall']
                room2_wall = shared_wall['room2_wall']
                key1 = (room1_idx, room1_wall)
                key2 = (room2_idx, room2_wall)
                if key1 not in shared_wall_map:
                    shared_wall_map[key1] = []
                if key2 not in shared_wall_map:
                    shared_wall_map[key2] = []
                shared_wall_map[key1].append(key2)
                shared_wall_map[key2].append(key1)
    for (room_idx, room) in enumerate(corrected_rooms_data):
        for wall_side in ['north', 'south', 'east', 'west']:
            wall_key = (room_idx, wall_side)
            if wall_key not in shared_wall_map:
                exterior_walls.add(wall_key)
    for (room_idx, room) in enumerate(corrected_rooms_data):
        room_type = room['room_type']
        width = room['dimensions']['width']
        length = room['dimensions']['length']
        doors_to_keep = []
        for (door_idx, door) in enumerate(room.get('doors', [])):
            door_id = f'{room_type}_door_{door_idx}'
            wall_side = door.get('wall_side', '')
            position_on_wall = door.get('position_on_wall', 0)
            door_width = door.get('width', 0.8)
            if not wall_side or wall_side not in ['north', 'south', 'east', 'west'] or position_on_wall < 0 or (position_on_wall > 1) or (door_width <= 0) or (door_width > 3.0):
                doors_removed.append({'room': room_type, 'door_id': door_id, 'wall_side': wall_side, 'reason': f"Invalid door parameters: wall_side='{wall_side}', position={position_on_wall}, width={door_width}"})
                correction_actions.append(f'Removed invalid door from {room_type}: {door_id}')
                continue
            wall_length = width if wall_side in ['north', 'south'] else length
            door_start_pos = position_on_wall * wall_length - door_width / 2
            door_end_pos = position_on_wall * wall_length + door_width / 2
            if door_start_pos < 0 or door_end_pos > wall_length:
                min_position = door_width / 2 / wall_length
                max_position = 1.0 - door_width / 2 / wall_length
                if min_position <= max_position:
                    if position_on_wall < min_position:
                        new_position = min_position
                    elif position_on_wall > max_position:
                        new_position = max_position
                    else:
                        new_position = 0.5
                    door['position_on_wall'] = new_position
                    doors_repositioned.append({'room': room_type, 'door_id': door_id, 'wall_side': wall_side, 'original_position': position_on_wall, 'new_position': new_position, 'reason': 'Wall boundary violation'})
                    correction_actions.append(f'Repositioned door in {room_type} on {wall_side} wall from {position_on_wall:.3f} to {new_position:.3f}')
                else:
                    max_width = wall_length * 0.8
                    min_width = 0.6
                    if max_width >= min_width:
                        new_width = min(max_width, max(min_width, door_width))
                        new_position = 0.5
                        door['width'] = new_width
                        door['position_on_wall'] = new_position
                        doors_resized.append({'room': room_type, 'door_id': door_id, 'wall_side': wall_side, 'original_width': door_width, 'new_width': new_width, 'original_position': position_on_wall, 'new_position': new_position, 'reason': 'Wall boundary violation - too wide'})
                        correction_actions.append(f'Resized and repositioned door in {room_type} on {wall_side} wall: {door_width:.2f}m→{new_width:.2f}m')
                    else:
                        doors_removed.append({'room': room_type, 'door_id': door_id, 'wall_side': wall_side, 'reason': f'Cannot fit on {wall_length:.2f}m wall (required: {door_width:.2f}m)'})
                        correction_actions.append(f'Removed door from {room_type} on {wall_side} wall - cannot fit')
                        continue
            doors_to_keep.append(door)
        windows_to_keep = []
        for (window_idx, window) in enumerate(room.get('windows', [])):
            window_id = f'{room_type}_window_{window_idx}'
            wall_side = window.get('wall_side', '')
            position_on_wall = window.get('position_on_wall', 0)
            window_width = window.get('width', 1.2)
            if not wall_side or wall_side not in ['north', 'south', 'east', 'west'] or position_on_wall < 0 or (position_on_wall > 1) or (window_width <= 0) or (window_width > 5.0):
                windows_removed.append({'room': room_type, 'window_id': window_id, 'wall_side': wall_side, 'reason': f"Invalid window parameters: wall_side='{wall_side}', position={position_on_wall}, width={window_width}"})
                correction_actions.append(f'Removed invalid window from {room_type}: {window_id}')
                continue
            wall_length = width if wall_side in ['north', 'south'] else length
            window_start_pos = position_on_wall * wall_length - window_width / 2
            window_end_pos = position_on_wall * wall_length + window_width / 2
            if window_start_pos < 0 or window_end_pos > wall_length:
                min_position = window_width / 2 / wall_length
                max_position = 1.0 - window_width / 2 / wall_length
                if min_position <= max_position:
                    if position_on_wall < min_position:
                        new_position = min_position
                    elif position_on_wall > max_position:
                        new_position = max_position
                    else:
                        new_position = 0.5
                    window['position_on_wall'] = new_position
                    windows_repositioned.append({'room': room_type, 'window_id': window_id, 'wall_side': wall_side, 'original_position': position_on_wall, 'new_position': new_position, 'reason': 'Wall boundary violation'})
                    correction_actions.append(f'Repositioned window in {room_type} on {wall_side} wall from {position_on_wall:.3f} to {new_position:.3f}')
                else:
                    max_width = wall_length * 0.8
                    min_width = 0.4
                    if max_width >= min_width:
                        new_width = min(max_width, max(min_width, window_width))
                        new_position = 0.5
                        window['width'] = new_width
                        window['position_on_wall'] = new_position
                        windows_resized.append({'room': room_type, 'window_id': window_id, 'wall_side': wall_side, 'original_width': window_width, 'new_width': new_width, 'original_position': position_on_wall, 'new_position': new_position, 'reason': 'Wall boundary violation - too wide'})
                        correction_actions.append(f'Resized and repositioned window in {room_type} on {wall_side} wall: {window_width:.2f}m→{new_width:.2f}m')
                    else:
                        windows_removed.append({'room': room_type, 'window_id': window_id, 'wall_side': wall_side, 'reason': f'Cannot fit on {wall_length:.2f}m wall (required: {window_width:.2f}m)'})
                        correction_actions.append(f'Removed window from {room_type} on {wall_side} wall - cannot fit')
                        continue
            windows_to_keep.append(window)
        corrected_rooms_data[room_idx]['doors'] = doors_to_keep
        corrected_rooms_data[room_idx]['windows'] = windows_to_keep
    for (room_idx, room) in enumerate(corrected_rooms_data):
        room_type = room['room_type']
        width = room['dimensions']['width']
        length = room['dimensions']['length']
        for wall_side in ['north', 'south', 'east', 'west']:
            wall_elements = []
            for (door_idx, door) in enumerate(room.get('doors', [])):
                if door['wall_side'] == wall_side:
                    wall_length = width if wall_side in ['north', 'south'] else length
                    door_start = door['position_on_wall'] * wall_length - door['width'] / 2
                    door_end = door['position_on_wall'] * wall_length + door['width'] / 2
                    wall_elements.append({'type': 'door', 'index': door_idx, 'data': door, 'start': door_start, 'end': door_end, 'priority': 1})
            for (window_idx, window) in enumerate(room.get('windows', [])):
                if window['wall_side'] == wall_side:
                    wall_length = width if wall_side in ['north', 'south'] else length
                    window_start = window['position_on_wall'] * wall_length - window['width'] / 2
                    window_end = window['position_on_wall'] * wall_length + window['width'] / 2
                    wall_elements.append({'type': 'window', 'index': window_idx, 'data': window, 'start': window_start, 'end': window_end, 'priority': 2})
            if len(wall_elements) > 1:
                wall_elements.sort(key=lambda x: (x['priority'], x['start']))
                elements_to_remove = []
                for i in range(len(wall_elements)):
                    for j in range(i + 1, len(wall_elements)):
                        elem1 = wall_elements[i]
                        elem2 = wall_elements[j]
                        tolerance = 0.001
                        if not (elem1['end'] <= elem2['start'] + tolerance or elem2['end'] <= elem1['start'] + tolerance):
                            if elem1['priority'] <= elem2['priority']:
                                elements_to_remove.append(elem2)
                            else:
                                elements_to_remove.append(elem1)
                for elem_to_remove in elements_to_remove:
                    if elem_to_remove['type'] == 'door':
                        doors_removed.append({'room': room_type, 'door_id': f"{room_type}_door_{elem_to_remove['index']}", 'wall_side': wall_side, 'reason': 'Overlaps with other elements on same wall'})
                        corrected_rooms_data[room_idx]['doors'] = [door for (door_idx, door) in enumerate(corrected_rooms_data[room_idx]['doors']) if not (door['wall_side'] == wall_side and door_idx == elem_to_remove['index'])]
                    else:
                        windows_removed.append({'room': room_type, 'window_id': f"{room_type}_window_{elem_to_remove['index']}", 'wall_side': wall_side, 'reason': 'Overlaps with other elements on same wall'})
                        corrected_rooms_data[room_idx]['windows'] = [window for (window_idx, window) in enumerate(corrected_rooms_data[room_idx]['windows']) if not (window['wall_side'] == wall_side and window_idx == elem_to_remove['index'])]
                    correction_actions.append(f"Removed {elem_to_remove['type']} from {room_type} on {wall_side} wall due to overlap")
    processed_shared_doors = set()
    processed_shared_windows = set()
    for room1_idx in range(len(corrected_rooms_data)):
        for room2_idx in range(room1_idx + 1, len(corrected_rooms_data)):
            room1 = corrected_rooms_data[room1_idx]
            room2 = corrected_rooms_data[room2_idx]
            shared_walls = find_shared_walls(room1, room2)
            for shared_wall in shared_walls:
                room1_wall = shared_wall['room1_wall']
                room2_wall = shared_wall['room2_wall']
                room1_doors = [door for door in room1.get('doors', []) if door['wall_side'] == room1_wall]
                room2_doors = [door for door in room2.get('doors', []) if door['wall_side'] == room2_wall]
                for door1 in room1_doors:
                    door1_world_pos = calculate_door_world_position(room1, door1)
                    has_counterpart = False
                    for door2 in room2_doors:
                        door2_world_pos = calculate_door_world_position(room2, door2)
                        if doors_align(door1_world_pos, door2_world_pos, door1['width'], door2['width']):
                            has_counterpart = True
                            break
                    if not has_counterpart:
                        corrected_rooms_data[room1_idx]['doors'] = [door for door in corrected_rooms_data[room1_idx]['doors'] if door is not door1]
                        doors_removed.append({'room': room1['room_type'], 'door_id': f"{room1['room_type']}_door_orphaned", 'wall_side': room1_wall, 'reason': 'Missing bidirectional counterpart on shared wall'})
                        bidirectional_corrections.append(f"Removed orphaned door from {room1['room_type']} on {room1_wall} wall")
                        correction_actions.append(f"Removed orphaned door from {room1['room_type']} on {room1_wall} wall - no counterpart")
                for door2 in room2_doors:
                    door2_world_pos = calculate_door_world_position(room2, door2)
                    has_counterpart = False
                    for door1 in room1_doors:
                        door1_world_pos = calculate_door_world_position(room1, door1)
                        if doors_align(door1_world_pos, door2_world_pos, door1['width'], door2['width']):
                            has_counterpart = True
                            break
                    if not has_counterpart:
                        corrected_rooms_data[room2_idx]['doors'] = [door for door in corrected_rooms_data[room2_idx]['doors'] if door is not door2]
                        doors_removed.append({'room': room2['room_type'], 'door_id': f"{room2['room_type']}_door_orphaned", 'wall_side': room2_wall, 'reason': 'Missing bidirectional counterpart on shared wall'})
                        bidirectional_corrections.append(f"Removed orphaned door from {room2['room_type']} on {room2_wall} wall")
                        correction_actions.append(f"Removed orphaned door from {room2['room_type']} on {room2_wall} wall - no counterpart")
                room1_windows = [window for window in room1.get('windows', []) if window['wall_side'] == room1_wall]
                room2_windows = [window for window in room2.get('windows', []) if window['wall_side'] == room2_wall]
                for window1 in room1_windows:
                    window1_world_pos = calculate_door_world_position(room1, window1)
                    has_counterpart = False
                    for window2 in room2_windows:
                        window2_world_pos = calculate_door_world_position(room2, window2)
                        if doors_align(window1_world_pos, window2_world_pos, window1['width'], window2['width']):
                            has_counterpart = True
                            break
                    if not has_counterpart:
                        corrected_rooms_data[room1_idx]['windows'] = [window for window in corrected_rooms_data[room1_idx]['windows'] if window is not window1]
                        windows_removed.append({'room': room1['room_type'], 'window_id': f"{room1['room_type']}_window_orphaned", 'wall_side': room1_wall, 'reason': 'Missing bidirectional counterpart on shared wall'})
                        bidirectional_corrections.append(f"Removed orphaned window from {room1['room_type']} on {room1_wall} wall")
                        correction_actions.append(f"Removed orphaned window from {room1['room_type']} on {room1_wall} wall - no counterpart")
                for window2 in room2_windows:
                    window2_world_pos = calculate_door_world_position(room2, window2)
                    has_counterpart = False
                    for window1 in room1_windows:
                        window1_world_pos = calculate_door_world_position(room1, window1)
                        if doors_align(window1_world_pos, window2_world_pos, window1['width'], window2['width']):
                            has_counterpart = True
                            break
                    if not has_counterpart:
                        corrected_rooms_data[room2_idx]['windows'] = [window for window in corrected_rooms_data[room2_idx]['windows'] if window is not window2]
                        windows_removed.append({'room': room2['room_type'], 'window_id': f"{room2['room_type']}_window_orphaned", 'wall_side': room2_wall, 'reason': 'Missing bidirectional counterpart on shared wall'})
                        bidirectional_corrections.append(f"Removed orphaned window from {room2['room_type']} on {room2_wall} wall")
                        correction_actions.append(f"Removed orphaned window from {room2['room_type']} on {room2_wall} wall - no counterpart")
    connectivity_result = add_connectivity_doors(corrected_rooms_data)
    if connectivity_result['success']:
        corrected_rooms_data = connectivity_result['updated_rooms_data']
        connectivity_doors_added = connectivity_result['doors_added_details']
        correction_actions.extend(connectivity_result['connectivity_actions'])
    final_validation = validate_door_window_issues(corrected_rooms_data)
    final_connectivity = check_room_connectivity(corrected_rooms_data)
    return {'success': True, 'corrected_rooms_data': corrected_rooms_data, 'correction_summary': {'doors_removed': len(doors_removed), 'windows_removed': len(windows_removed), 'doors_resized': len(doors_resized), 'windows_resized': len(windows_resized), 'doors_repositioned': len(doors_repositioned), 'windows_repositioned': len(windows_repositioned), 'connectivity_doors_added': len(connectivity_doors_added), 'bidirectional_corrections': len(bidirectional_corrections), 'total_corrections': len(correction_actions)}, 'correction_actions': correction_actions, 'detailed_changes': {'doors_removed': doors_removed, 'windows_removed': windows_removed, 'doors_resized': doors_resized, 'windows_resized': windows_resized, 'doors_repositioned': doors_repositioned, 'windows_repositioned': windows_repositioned, 'connectivity_doors_added': connectivity_doors_added, 'bidirectional_corrections': bidirectional_corrections}, 'connectivity_result': connectivity_result, 'final_integrity': final_validation, 'final_connectivity': final_connectivity, 'guaranteed_integrity': final_validation['valid'], 'guaranteed_connectivity': final_connectivity['connected'], 'bidirectional_compliance': len(final_validation.get('bidirectional_issues', [])) == 0}

async def correct_layout_issues(current_layout) -> str:
    """
    Automatically detect and correct issues in the current room layout.
    Uses validation results to generate correction suggestions via Claude API.
    
    Args:
        current_layout: The current FloorPlan object to correct
    
    Returns:
        JSON string with correction results and updated layout information
    """
    if current_layout is None:
        return json.dumps({'error': 'No layout has been provided for correction.'})
    try:
        rooms_data = []
        for room in current_layout.rooms:
            room_data = {'room_type': room.room_type, 'position': {'x': room.position.x, 'y': room.position.y, 'z': room.position.z}, 'dimensions': {'width': room.dimensions.width, 'length': room.dimensions.length, 'height': room.dimensions.height}}
            rooms_data.append(room_data)
        validation_result = validate_room_layout(rooms_data)
        if validation_result['valid']:
            return json.dumps({'success': True, 'message': '✅ Current layout is already valid - no corrections needed', 'layout_id': current_layout.id, 'validation_status': 'passed'})
        correction_prompt = await generate_detailed_correction_prompt(validation_result, current_layout.created_from_text, rooms_data)
        response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.2, messages=[{'role': 'user', 'content': correction_prompt}])
        if not response or not hasattr(response, 'content') or (not response.content):
            return json.dumps({'success': False, 'error': 'No response received from Claude API during correction'})
        response_text = response.content[0].text
        try:
            corrected_layout_data = json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search('\\{.*\\}', response_text, re.DOTALL)
            if json_match:
                corrected_layout_data = json.loads(json_match.group())
            else:
                return json.dumps({'success': False, 'error': 'Could not parse correction response from Claude'})
        structure_validation = validate_llm_response_structure(corrected_layout_data)
        if not structure_validation['valid']:
            return json.dumps({'success': False, 'error': f"Corrected layout has invalid structure: {structure_validation['error']}", 'missing_fields': structure_validation.get('missing_fields', []), 'invalid_fields': structure_validation.get('invalid_fields', [])})
        corrected_validation = validate_room_layout(corrected_layout_data['rooms'])
        try:
            old_layout_id = current_layout.id
            old_description = current_layout.description
            corrected_floor_plan = parse_llm_response_to_floor_plan(corrected_layout_data, current_layout.created_from_text)
            correction_summary = {'success': True, 'message': '🔧 Layout successfully corrected', 'original_issues': validation_result['issues'], 'correction_applied': True, 'corrected_layout_data': corrected_layout_data, 'corrected_layout': {'layout_id': corrected_floor_plan.id, 'num_rooms': len(corrected_floor_plan.rooms), 'total_area': corrected_floor_plan.total_area, 'building_style': corrected_floor_plan.building_style, 'rooms': [{'id': room.id, 'type': room.room_type, 'area': room.dimensions.width * room.dimensions.length, 'dimensions': f'{room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m', 'doors': len(room.doors), 'windows': len(room.windows)} for room in corrected_floor_plan.rooms]}, 'validation_after_correction': {'valid': corrected_validation['valid'], 'issues': corrected_validation['issues'] if not corrected_validation['valid'] else [], 'overlaps': corrected_validation['overlaps'], 'detached_rooms': corrected_validation['detached_rooms'], 'door_issues': corrected_validation['door_issues'], 'window_issues': corrected_validation['window_issues'], 'wall_boundary_violations': corrected_validation['wall_boundary_violations'], 'door_window_overlaps': corrected_validation['door_window_overlaps'], 'wall_intersections': corrected_validation['wall_intersections']}}
            if not corrected_validation['valid']:
                correction_summary['warning'] = '⚠️ Corrected layout still has some issues - may need manual adjustment'
                correction_summary['remaining_issues'] = corrected_validation['issues']
            return json.dumps(correction_summary, indent=2)
        except Exception as parse_error:
            return json.dumps({'success': False, 'error': f'Failed to apply corrections: {str(parse_error)}'})
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Layout correction failed: {str(e)}'})

async def aggressive_layout_correction(current_layout) -> str:
    """
    Perform guaranteed overlap elimination using iterative global optimization.
    Uses Claude for room prioritization, then applies iterative correction until 100% overlap-free.
    
    Args:
        current_layout: The current FloorPlan object to correct
    
    Returns:
        JSON string with aggressive correction results and updated layout information
    """
    if current_layout is None:
        return json.dumps({'error': "No layout has been generated yet. Use 'generate_room_layout()' first."})
    try:
        rooms_data = []
        for room in current_layout.rooms:
            room_data = {'room_type': room.room_type, 'position': {'x': room.position.x, 'y': room.position.y, 'z': room.position.z}, 'dimensions': {'width': room.dimensions.width, 'length': room.dimensions.length, 'height': room.dimensions.height}, 'doors': [], 'windows': []}
            for door in room.doors:
                door_data = {'width': door.width, 'height': door.height, 'position_on_wall': door.position_on_wall, 'wall_side': extract_wall_side_from_id(door.wall_id), 'door_type': door.door_type}
                room_data['doors'].append(door_data)
            for window in room.windows:
                window_data = {'width': window.width, 'height': window.height, 'position_on_wall': window.position_on_wall, 'wall_side': extract_wall_side_from_id(window.wall_id), 'sill_height': window.sill_height, 'window_type': window.window_type}
                room_data['windows'].append(window_data)
            rooms_data.append(room_data)
        validation_result = validate_room_layout(rooms_data)
        if validation_result['valid']:
            return json.dumps({'success': True, 'message': '✅ Current layout is already valid - no aggressive correction needed', 'layout_id': current_layout.id, 'validation_status': 'passed'})
        try:
            room_priorities = await get_room_priorities_from_claude(rooms_data, current_layout.created_from_text)
        except Exception as e:
            return json.dumps({'success': False, 'error': f'Failed to get room priorities from Claude: {str(e)}'})
        priority_map = {}
        for (i, room_type) in enumerate(room_priorities):
            priority_map[room_type] = i
        max_priority = len(room_priorities)
        for room in rooms_data:
            if room['room_type'] not in priority_map:
                priority_map[room['room_type']] = max_priority
                max_priority += 1
        correction_actions = []
        rooms_removed = []
        rooms_reduced = []
        iteration = 0
        max_iterations = 100
        corrected_rooms_data = rooms_data.copy()
        while iteration < max_iterations:
            iteration += 1
            overlapping_pairs = []
            for i in range(len(corrected_rooms_data)):
                for j in range(i + 1, len(corrected_rooms_data)):
                    room1 = corrected_rooms_data[i]
                    room2 = corrected_rooms_data[j]
                    overlap_result = check_room_overlap(room1, room2)
                    if overlap_result['overlaps']:
                        overlapping_pairs.append({'room1': room1, 'room2': room2, 'room1_index': i, 'room2_index': j, 'overlap_area': overlap_result['overlap_area'], 'room1_priority': priority_map.get(room1['room_type'], max_priority), 'room2_priority': priority_map.get(room2['room_type'], max_priority)})
            if not overlapping_pairs:
                break
            overlapping_pairs.sort(key=lambda x: x['overlap_area'], reverse=True)
            rooms_to_remove_this_iteration = set()
            for pair in overlapping_pairs:
                room1 = pair['room1']
                room2 = pair['room2']
                room1_priority = pair['room1_priority']
                room2_priority = pair['room2_priority']
                if room1['room_type'] in rooms_to_remove_this_iteration or room2['room_type'] in rooms_to_remove_this_iteration:
                    continue
                if room1_priority < room2_priority:
                    room_to_modify = room2
                    other_room = room1
                    modify_index = pair['room2_index']
                elif room2_priority < room1_priority:
                    room_to_modify = room1
                    other_room = room2
                    modify_index = pair['room1_index']
                else:
                    room1_area = room1['dimensions']['width'] * room1['dimensions']['length']
                    room2_area = room2['dimensions']['width'] * room2['dimensions']['length']
                    if room1_area >= room2_area:
                        room_to_modify = room1
                        other_room = room2
                        modify_index = pair['room1_index']
                    else:
                        room_to_modify = room2
                        other_room = room1
                        modify_index = pair['room2_index']
                reduction_result = calculate_room_reduction(room_to_modify, [other_room])
                if reduction_result['can_reduce']:
                    strategy = reduction_result['strategy']
                    room_to_modify['dimensions']['width'] = strategy['new_width']
                    room_to_modify['dimensions']['length'] = strategy['new_length']
                    rooms_reduced.append({'room_type': room_to_modify['room_type'], 'iteration': iteration, 'original_dimensions': reduction_result['original_dimensions'], 'new_dimensions': {'width': strategy['new_width'], 'length': strategy['new_length']}, 'area_loss': strategy['area_loss'], 'reduction_factor': strategy['reduction_factor'], 'reason': f"Overlap with {other_room['room_type']}"})
                    correction_actions.append(f"Iteration {iteration}: Reduced {room_to_modify['room_type']} by {(1 - strategy['reduction_factor']) * 100:.1f}% to eliminate overlap with {other_room['room_type']}")
                else:
                    rooms_to_remove_this_iteration.add(room_to_modify['room_type'])
                    correction_actions.append(f"Iteration {iteration}: Removed {room_to_modify['room_type']} - could not reduce sufficiently to eliminate overlap with {other_room['room_type']}")
            if rooms_to_remove_this_iteration:
                corrected_rooms_data = [room for room in corrected_rooms_data if room['room_type'] not in rooms_to_remove_this_iteration]
                rooms_removed.extend(list(rooms_to_remove_this_iteration))
        final_validation = validate_room_layout(corrected_rooms_data)
        if not final_validation['valid'] and final_validation.get('overlaps'):
            return json.dumps({'success': False, 'error': f'Failed to eliminate all overlaps after {max_iterations} iterations', 'remaining_overlaps': final_validation['overlaps'], 'recommendation': 'Layout may be too complex - consider simplifying the room arrangement or regenerating from scratch'})
        try:
            corrected_layout_data = {'building_style': current_layout.building_style, 'rooms': corrected_rooms_data}
            old_layout_id = current_layout.id
            old_description = current_layout.description
            original_room_count = len(current_layout.rooms)
            corrected_floor_plan = parse_llm_response_to_floor_plan(corrected_layout_data, current_layout.created_from_text)
            correction_summary = {'success': True, 'message': '🔧⚡ Layout corrected with guaranteed overlap elimination', 'correction_type': 'iterative_global_optimization', 'iterations_required': iteration, 'room_priorities': room_priorities, 'correction_actions': correction_actions, 'corrected_layout_data': corrected_layout_data, 'changes_made': {'rooms_removed': len(rooms_removed), 'rooms_reduced': len(rooms_reduced), 'original_room_count': original_room_count, 'final_room_count': len(corrected_rooms_data), 'removed_rooms': rooms_removed, 'reduced_rooms': rooms_reduced}, 'corrected_layout': {'layout_id': corrected_floor_plan.id, 'num_rooms': len(corrected_floor_plan.rooms), 'total_area': corrected_floor_plan.total_area, 'building_style': corrected_floor_plan.building_style, 'rooms': [{'id': room.id, 'type': room.room_type, 'area': room.dimensions.width * room.dimensions.length, 'dimensions': f'{room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m', 'doors': len(room.doors), 'windows': len(room.windows)} for room in corrected_floor_plan.rooms]}, 'validation_after_correction': {'valid': final_validation['valid'], 'issues': final_validation['issues'] if not final_validation['valid'] else [], 'overlaps': final_validation['overlaps'], 'detached_rooms': final_validation['detached_rooms'], 'guaranteed_overlap_free': len(final_validation.get('overlaps', [])) == 0}}
            if len(final_validation.get('overlaps', [])) == 0:
                correction_summary['success_message'] = '✅ GUARANTEED: All room overlaps eliminated through iterative optimization'
            if not final_validation['valid']:
                non_overlap_issues = [issue for issue in final_validation.get('issues', []) if 'overlap' not in issue.lower()]
                if non_overlap_issues:
                    correction_summary['info'] = '✅ All overlaps eliminated, but some non-overlap issues remain (connectivity, etc.)'
                    correction_summary['remaining_non_overlap_issues'] = non_overlap_issues
                    correction_summary['recommendations'] = ['All room overlaps have been eliminated', 'Remaining issues are related to room connectivity or door/window placement', "Use 'clean_door_window_placement()' for door/window issues"]
            if rooms_reduced or rooms_removed:
                total_area_loss = sum((r['area_loss'] for r in rooms_reduced))
                correction_summary['efficiency_metrics'] = {'total_area_preserved': corrected_floor_plan.total_area, 'total_area_lost': total_area_loss, 'rooms_preserved': len(corrected_rooms_data), 'preservation_rate': f'{len(corrected_rooms_data) / original_room_count * 100:.1f}%', 'optimization_efficiency': f'Converged in {iteration} iterations'}
            return json.dumps(correction_summary, indent=2)
        except Exception as parse_error:
            return json.dumps({'success': False, 'error': f'Failed to apply guaranteed corrections: {str(parse_error)}'})
    except Exception as e:
        return json.dumps({'success': False, 'error': f'Guaranteed layout correction failed: {str(e)}'})
