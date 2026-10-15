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
from typing import List
from models import Point3D, Dimensions, Wall
from typing import Dict, Any, Optional
from key import ANTHROPIC_API_KEY
from vlm import call_vlm
import anthropic
import json
from validation import check_room_overlap
from models import FloorPlan, Room, Door, Window, Wall, Object, Euler
from dataclasses import asdict
import trimesh
import numpy as np
import os
from constants import RESULTS_DIR

def calculate_door_world_position(room: Dict[str, Any], door: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the world coordinates of a door using unified fractional position system (0-1).
    
    Args:
        room: Room data containing position and dimensions
        door: Door data containing wall_side and position_on_wall (0-1 fractional position)
        
    Returns:
        Dictionary with world coordinates and door bounds
    """
    room_pos = room['position']
    room_dims = room['dimensions']
    wall_side = door['wall_side']
    position_on_wall = door['position_on_wall']
    door_width = door['width']
    if wall_side == 'north':
        wall_length = room_dims['width']
        door_center_x = room_pos['x'] + position_on_wall * wall_length
        door_center_y = room_pos['y'] + room_dims['length']
        door_start_x = door_center_x - door_width / 2
        door_end_x = door_center_x + door_width / 2
        return {'center_x': door_center_x, 'center_y': door_center_y, 'start_x': door_start_x, 'end_x': door_end_x, 'wall_side': wall_side}
    elif wall_side == 'south':
        wall_length = room_dims['width']
        door_center_x = room_pos['x'] + position_on_wall * wall_length
        door_center_y = room_pos['y']
        door_start_x = door_center_x - door_width / 2
        door_end_x = door_center_x + door_width / 2
        return {'center_x': door_center_x, 'center_y': door_center_y, 'start_x': door_start_x, 'end_x': door_end_x, 'wall_side': wall_side}
    elif wall_side == 'east':
        wall_length = room_dims['length']
        door_center_x = room_pos['x'] + room_dims['width']
        door_center_y = room_pos['y'] + position_on_wall * wall_length
        door_start_y = door_center_y - door_width / 2
        door_end_y = door_center_y + door_width / 2
        return {'center_x': door_center_x, 'center_y': door_center_y, 'start_y': door_start_y, 'end_y': door_end_y, 'wall_side': wall_side}
    else:
        wall_length = room_dims['length']
        door_center_x = room_pos['x']
        door_center_y = room_pos['y'] + position_on_wall * wall_length
        door_start_y = door_center_y - door_width / 2
        door_end_y = door_center_y + door_width / 2
        return {'center_x': door_center_x, 'center_y': door_center_y, 'start_y': door_start_y, 'end_y': door_end_y, 'wall_side': wall_side}

def add_bidirectional_door(room1_idx: int, room2_idx: int, connection_info: Dict[str, Any], rooms_data: List[Dict[str, Any]], door_width: float=0.9) -> Dict[str, Any]:
    """
    Add a connecting door between two rooms on both sides.
    
    Args:
        room1_idx: Index of first room
        room2_idx: Index of second room
        connection_info: Connection information from find_best_connection_point
        rooms_data: List of room data
        door_width: Width of the door to add
        
    Returns:
        Dictionary with operation result and door information
    """
    if not connection_info['possible']:
        return {'success': False, 'error': 'No valid connection point found between rooms'}
    conn = connection_info['connection']
    room1 = rooms_data[room1_idx]
    room2 = rooms_data[room2_idx]
    door1 = {'width': door_width, 'height': 2.1, 'position_on_wall': conn['room1_local_pos'], 'wall_side': conn['room1_wall'], 'door_type': 'connecting'}
    door2 = {'width': door_width, 'height': 2.1, 'position_on_wall': conn['room2_local_pos'], 'wall_side': conn['room2_wall'], 'door_type': 'connecting'}
    if 'doors' not in rooms_data[room1_idx]:
        rooms_data[room1_idx]['doors'] = []
    if 'doors' not in rooms_data[room2_idx]:
        rooms_data[room2_idx]['doors'] = []
    rooms_data[room1_idx]['doors'].append(door1)
    rooms_data[room2_idx]['doors'].append(door2)
    return {'success': True, 'room1_idx': room1_idx, 'room2_idx': room2_idx, 'door1': door1, 'door2': door2, 'door1_idx': len(rooms_data[room1_idx]['doors']) - 1, 'door2_idx': len(rooms_data[room2_idx]['doors']) - 1}

def remove_bidirectional_door(room_idx: int, door_idx: int, rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Remove a connecting door from both sides.
    
    Args:
        room_idx: Index of the room containing the door
        door_idx: Index of the door to remove
        rooms_data: List of room data
        
    Returns:
        Dictionary with operation result
    """
    if door_idx >= len(rooms_data[room_idx]['doors']):
        return {'success': False, 'error': 'Door index out of range'}
    door = rooms_data[room_idx]['doors'][door_idx]
    connecting_door_info = find_connecting_door(room_idx, door, rooms_data)
    removed_door = rooms_data[room_idx]['doors'].pop(door_idx)
    result = {'success': True, 'removed_doors': [{'room_idx': room_idx, 'door_idx': door_idx, 'door': removed_door}]}
    if connecting_door_info:
        other_room_idx = connecting_door_info['room_idx']
        other_door_idx = connecting_door_info['door_idx']
        if other_door_idx < len(rooms_data[other_room_idx]['doors']):
            other_removed_door = rooms_data[other_room_idx]['doors'].pop(other_door_idx)
            result['removed_doors'].append({'room_idx': other_room_idx, 'door_idx': other_door_idx, 'door': other_removed_door})
    return result

def reposition_bidirectional_door(room_idx: int, door_idx: int, new_position: float, rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reposition a connecting door on both sides.
    
    Args:
        room_idx: Index of the room containing the door
        door_idx: Index of the door to reposition
        new_position: New position on wall
        rooms_data: List of room data
        
    Returns:
        Dictionary with operation result
    """
    if door_idx >= len(rooms_data[room_idx]['doors']):
        return {'success': False, 'error': 'Door index out of range'}
    door = rooms_data[room_idx]['doors'][door_idx]
    old_position = door['position_on_wall']
    connecting_door_info = find_connecting_door(room_idx, door, rooms_data)
    rooms_data[room_idx]['doors'][door_idx]['position_on_wall'] = new_position
    result = {'success': True, 'repositioned_doors': [{'room_idx': room_idx, 'door_idx': door_idx, 'old_position': old_position, 'new_position': new_position}]}
    if connecting_door_info:
        other_room_idx = connecting_door_info['room_idx']
        other_door_idx = connecting_door_info['door_idx']
        shared_wall = connecting_door_info['shared_wall']
        if door['wall_side'] == shared_wall['room1_wall']:
            room1 = rooms_data[room_idx]
            room2 = rooms_data[other_room_idx]
            if door['wall_side'] in ['north', 'south']:
                world_x = room1['position']['x'] + new_position
                other_new_position = world_x - room2['position']['x']
            else:
                world_y = room1['position']['y'] + new_position
                other_new_position = world_y - room2['position']['y']
        else:
            room1 = rooms_data[other_room_idx]
            room2 = rooms_data[room_idx]
            if door['wall_side'] in ['north', 'south']:
                world_x = room2['position']['x'] + new_position
                other_new_position = world_x - room1['position']['x']
            else:
                world_y = room2['position']['y'] + new_position
                other_new_position = world_y - room1['position']['y']
        if other_door_idx < len(rooms_data[other_room_idx]['doors']):
            old_other_position = rooms_data[other_room_idx]['doors'][other_door_idx]['position_on_wall']
            rooms_data[other_room_idx]['doors'][other_door_idx]['position_on_wall'] = other_new_position
            result['repositioned_doors'].append({'room_idx': other_room_idx, 'door_idx': other_door_idx, 'old_position': old_other_position, 'new_position': other_new_position})
    return result

def resize_bidirectional_door(room_idx: int, door_idx: int, new_width: float, rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resize a connecting door on both sides.
    
    Args:
        room_idx: Index of the room containing the door
        door_idx: Index of the door to resize
        new_width: New width for the door
        rooms_data: List of room data
        
    Returns:
        Dictionary with operation result
    """
    if door_idx >= len(rooms_data[room_idx]['doors']):
        return {'success': False, 'error': 'Door index out of range'}
    door = rooms_data[room_idx]['doors'][door_idx]
    old_width = door['width']
    connecting_door_info = find_connecting_door(room_idx, door, rooms_data)
    rooms_data[room_idx]['doors'][door_idx]['width'] = new_width
    result = {'success': True, 'resized_doors': [{'room_idx': room_idx, 'door_idx': door_idx, 'old_width': old_width, 'new_width': new_width}]}
    if connecting_door_info:
        other_room_idx = connecting_door_info['room_idx']
        other_door_idx = connecting_door_info['door_idx']
        if other_door_idx < len(rooms_data[other_room_idx]['doors']):
            old_other_width = rooms_data[other_room_idx]['doors'][other_door_idx]['width']
            rooms_data[other_room_idx]['doors'][other_door_idx]['width'] = new_width
            result['resized_doors'].append({'room_idx': other_room_idx, 'door_idx': other_door_idx, 'old_width': old_other_width, 'new_width': new_width})
    return result

def find_connecting_door(room_idx: int, door: Dict[str, Any], rooms_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the corresponding connecting door on the other side of a shared wall.
    
    Args:
        room_idx: Index of the room containing the door
        door: Door data dictionary
        rooms_data: List of all room data
        
    Returns:
        Dictionary with connected room info and door, or None if not a connecting door
    """
    if door.get('door_type') != 'connecting':
        return None
    current_room = rooms_data[room_idx]
    for (other_room_idx, other_room) in enumerate(rooms_data):
        if other_room_idx == room_idx:
            continue
        shared_walls = find_shared_walls(current_room, other_room)
        if not shared_walls:
            continue
        for shared_wall in shared_walls:
            if door['wall_side'] == shared_wall['room1_wall']:
                corresponding_wall = shared_wall['room2_wall']
            elif door['wall_side'] == shared_wall['room2_wall']:
                corresponding_wall = shared_wall['room1_wall']
            else:
                continue
            for (other_door_idx, other_door) in enumerate(other_room.get('doors', [])):
                if other_door['wall_side'] == corresponding_wall and other_door.get('door_type') == 'connecting':
                    door1_world_pos = calculate_door_world_position(current_room, door)
                    door2_world_pos = calculate_door_world_position(other_room, other_door)
                    if doors_align(door1_world_pos, door2_world_pos, door['width'], other_door['width']):
                        return {'room_idx': other_room_idx, 'door_idx': other_door_idx, 'door': other_door, 'room': other_room, 'shared_wall': shared_wall}
    return None

def doors_align(door1_pos: Dict[str, Any], door2_pos: Dict[str, Any], door1_width: float, door2_width: float) -> bool:
    """
    Check if two doors on opposite sides of a shared wall align properly.
    
    Args:
        door1_pos: World position of first door
        door2_pos: World position of second door
        door1_width: Width of first door
        door2_width: Width of second door
        
    Returns:
        True if doors align and create a valid connection
    """
    tolerance = 0.1
    if door1_pos['wall_side'] in ['north', 'south'] and door2_pos['wall_side'] in ['north', 'south']:
        if abs(door1_pos['center_y'] - door2_pos['center_y']) > tolerance:
            return False
        door1_start = door1_pos['start_x']
        door1_end = door1_pos['end_x']
        door2_start = door2_pos['start_x']
        door2_end = door2_pos['end_x']
        overlap_start = max(door1_start, door2_start)
        overlap_end = min(door1_end, door2_end)
        overlap_width = overlap_end - overlap_start
        return overlap_width >= 0.6
    elif door1_pos['wall_side'] in ['east', 'west'] and door2_pos['wall_side'] in ['east', 'west']:
        if abs(door1_pos['center_x'] - door2_pos['center_x']) > tolerance:
            return False
        door1_start = door1_pos['start_y']
        door1_end = door1_pos['end_y']
        door2_start = door2_pos['start_y']
        door2_end = door2_pos['end_y']
        overlap_start = max(door1_start, door2_start)
        overlap_end = min(door1_end, door2_end)
        overlap_width = overlap_end - overlap_start
        return overlap_width >= 0.6
    return False

def generate_unique_id(prefix: str) -> str:
    """Generate a unique ID with the given prefix."""
    import uuid
    return f'{prefix}_{str(uuid.uuid4())[:8]}'

def create_walls_for_room(room_data: dict, room_id: str) -> List[Wall]:
    """Generate walls for a room based on its dimensions and position."""
    pos = room_data['position']
    dims = room_data['dimensions']
    walls = []
    wall_configs = [('north', Point3D(pos['x'], pos['y'] + dims['length'], pos['z']), Point3D(pos['x'] + dims['width'], pos['y'] + dims['length'], pos['z'])), ('south', Point3D(pos['x'], pos['y'], pos['z']), Point3D(pos['x'] + dims['width'], pos['y'], pos['z'])), ('east', Point3D(pos['x'] + dims['width'], pos['y'], pos['z']), Point3D(pos['x'] + dims['width'], pos['y'] + dims['length'], pos['z'])), ('west', Point3D(pos['x'], pos['y'], pos['z']), Point3D(pos['x'], pos['y'] + dims['length'], pos['z']))]
    for (direction, start, end) in wall_configs:
        wall = Wall(id=generate_unique_id(f'wall_{room_id}_{direction}'), start_point=start, end_point=end, height=dims['height'])
        walls.append(wall)
    return walls

def extract_wall_side_from_id(wall_id: str) -> str:
    """
    Extract wall direction from wall ID.
    
    Args:
        wall_id: Wall ID string (e.g., "wall_room_abc123_north")
        
    Returns:
        Wall direction (north/south/east/west) or "unknown"
    """
    if '_north' in wall_id:
        return 'north'
    elif '_south' in wall_id:
        return 'south'
    elif '_east' in wall_id:
        return 'east'
    elif '_west' in wall_id:
        return 'west'
    else:
        return 'unknown'

def find_shared_walls(room1: Dict[str, Any], room2: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Find shared wall segments between two rooms.
    
    Args:
        room1: First room data
        room2: Second room data
        
    Returns:
        List of shared wall information
    """
    pos1 = room1['position']
    dims1 = room1['dimensions']
    pos2 = room2['position']
    dims2 = room2['dimensions']
    room1_bounds = {'x_min': pos1['x'], 'x_max': pos1['x'] + dims1['width'], 'y_min': pos1['y'], 'y_max': pos1['y'] + dims1['length']}
    room2_bounds = {'x_min': pos2['x'], 'x_max': pos2['x'] + dims2['width'], 'y_min': pos2['y'], 'y_max': pos2['y'] + dims2['length']}
    shared_walls = []
    tolerance = 0.01
    if abs(room1_bounds['y_max'] - room2_bounds['y_min']) < tolerance:
        overlap_x_min = max(room1_bounds['x_min'], room2_bounds['x_min'])
        overlap_x_max = min(room1_bounds['x_max'], room2_bounds['x_max'])
        if overlap_x_max > overlap_x_min + tolerance:
            shared_walls.append({'room1_wall': 'north', 'room2_wall': 'south', 'overlap_start': overlap_x_min, 'overlap_end': overlap_x_max, 'overlap_length': overlap_x_max - overlap_x_min})
    if abs(room1_bounds['y_min'] - room2_bounds['y_max']) < tolerance:
        overlap_x_min = max(room1_bounds['x_min'], room2_bounds['x_min'])
        overlap_x_max = min(room1_bounds['x_max'], room2_bounds['x_max'])
        if overlap_x_max > overlap_x_min + tolerance:
            shared_walls.append({'room1_wall': 'south', 'room2_wall': 'north', 'overlap_start': overlap_x_min, 'overlap_end': overlap_x_max, 'overlap_length': overlap_x_max - overlap_x_min})
    if abs(room1_bounds['x_max'] - room2_bounds['x_min']) < tolerance:
        overlap_y_min = max(room1_bounds['y_min'], room2_bounds['y_min'])
        overlap_y_max = min(room1_bounds['y_max'], room2_bounds['y_max'])
        if overlap_y_max > overlap_y_min + tolerance:
            shared_walls.append({'room1_wall': 'east', 'room2_wall': 'west', 'overlap_start': overlap_y_min, 'overlap_end': overlap_y_max, 'overlap_length': overlap_y_max - overlap_y_min})
    if abs(room1_bounds['x_min'] - room2_bounds['x_max']) < tolerance:
        overlap_y_min = max(room1_bounds['y_min'], room2_bounds['y_min'])
        overlap_y_max = min(room1_bounds['y_max'], room2_bounds['y_max'])
        if overlap_y_max > overlap_y_min + tolerance:
            shared_walls.append({'room1_wall': 'west', 'room2_wall': 'east', 'overlap_start': overlap_y_min, 'overlap_end': overlap_y_max, 'overlap_length': overlap_y_max - overlap_y_min})
    return shared_walls

def check_room_connectivity(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Check if all rooms are connected through doors using bidirectional door matching.
    Two rooms are connected only if both have doors on their shared wall that align.
    
    Args:
        rooms_data: List of room data dictionaries with doors and windows
        
    Returns:
        Dictionary with connectivity analysis
    """
    if len(rooms_data) <= 1:
        return {'connected': True, 'components': [list(range(len(rooms_data)))], 'isolated_rooms': [], 'missing_connections': [], 'door_connections': []}
    adjacency = {i: set() for i in range(len(rooms_data))}
    door_connections = []
    for room1_idx in range(len(rooms_data)):
        for room2_idx in range(room1_idx + 1, len(rooms_data)):
            room1 = rooms_data[room1_idx]
            room2 = rooms_data[room2_idx]
            shared_walls = find_shared_walls(room1, room2)
            if not shared_walls:
                continue
            for shared_wall in shared_walls:
                room1_wall = shared_wall['room1_wall']
                room2_wall = shared_wall['room2_wall']
                overlap_start = shared_wall['overlap_start']
                overlap_end = shared_wall['overlap_end']
                room1_doors = [door for door in room1.get('doors', []) if door['wall_side'] == room1_wall]
                room2_doors = [door for door in room2.get('doors', []) if door['wall_side'] == room2_wall]
                for door1 in room1_doors:
                    door1_world_pos = calculate_door_world_position(room1, door1)
                    for door2 in room2_doors:
                        door2_world_pos = calculate_door_world_position(room2, door2)
                        if doors_align(door1_world_pos, door2_world_pos, door1['width'], door2['width']):
                            adjacency[room1_idx].add(room2_idx)
                            adjacency[room2_idx].add(room1_idx)
                            door_connections.append({'room1': room1['room_type'], 'room2': room2['room_type'], 'room1_wall': room1_wall, 'room2_wall': room2_wall, 'room1_door_pos': door1['position_on_wall'], 'room2_door_pos': door2['position_on_wall'], 'connection_type': 'bidirectional_doors'})
                            break
                    else:
                        continue
                    break
    visited = set()
    components = []

    def dfs(room_idx, component):
        if room_idx in visited:
            return
        visited.add(room_idx)
        component.append(room_idx)
        for neighbor in adjacency[room_idx]:
            dfs(neighbor, component)
    for room_idx in range(len(rooms_data)):
        if room_idx not in visited:
            component = []
            dfs(room_idx, component)
            components.append(component)
    connected = len(components) == 1
    isolated_rooms = []
    missing_connections = []
    if not connected:
        for component in components:
            if len(component) == 1:
                isolated_rooms.append(component[0])
        for i in range(len(components)):
            for j in range(i + 1, len(components)):
                missing_connections.append({'component_1': components[i], 'component_2': components[j], 'rooms_1': [rooms_data[idx]['room_type'] for idx in components[i]], 'rooms_2': [rooms_data[idx]['room_type'] for idx in components[j]]})
    return {'connected': connected, 'components': components, 'isolated_rooms': isolated_rooms, 'missing_connections': missing_connections, 'door_connections': door_connections, 'adjacency': {str(k): list(v) for (k, v) in adjacency.items()}}

def find_best_connection_point(room1: Dict[str, Any], room2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Find the best point to add a door connecting two rooms using unified fractional position system (0-1).
    
    Args:
        room1: First room data
        room2: Second room data
        
    Returns:
        Dictionary with connection details using fractional positions
    """
    pos1 = room1['position']
    dims1 = room1['dimensions']
    pos2 = room2['position']
    dims2 = room2['dimensions']
    room1_bounds = {'x_min': pos1['x'], 'x_max': pos1['x'] + dims1['width'], 'y_min': pos1['y'], 'y_max': pos1['y'] + dims1['length']}
    room2_bounds = {'x_min': pos2['x'], 'x_max': pos2['x'] + dims2['width'], 'y_min': pos2['y'], 'y_max': pos2['y'] + dims2['length']}
    connections = []
    if abs(room1_bounds['y_max'] - room2_bounds['y_min']) < 0.1:
        overlap_x_min = max(room1_bounds['x_min'], room2_bounds['x_min'])
        overlap_x_max = min(room1_bounds['x_max'], room2_bounds['x_max'])
        if overlap_x_max > overlap_x_min:
            overlap_center_x = overlap_x_min + (overlap_x_max - overlap_x_min) / 2
            room1_fractional_pos = (overlap_center_x - pos1['x']) / dims1['width']
            room2_fractional_pos = (overlap_center_x - pos2['x']) / dims2['width']
            connections.append({'room1_wall': 'north', 'room2_wall': 'south', 'overlap_length': overlap_x_max - overlap_x_min, 'position_range': (overlap_x_min, overlap_x_max), 'room1_local_pos': room1_fractional_pos, 'room2_local_pos': room2_fractional_pos})
    if abs(room1_bounds['y_min'] - room2_bounds['y_max']) < 0.1:
        overlap_x_min = max(room1_bounds['x_min'], room2_bounds['x_min'])
        overlap_x_max = min(room1_bounds['x_max'], room2_bounds['x_max'])
        if overlap_x_max > overlap_x_min:
            overlap_center_x = overlap_x_min + (overlap_x_max - overlap_x_min) / 2
            room1_fractional_pos = (overlap_center_x - pos1['x']) / dims1['width']
            room2_fractional_pos = (overlap_center_x - pos2['x']) / dims2['width']
            connections.append({'room1_wall': 'south', 'room2_wall': 'north', 'overlap_length': overlap_x_max - overlap_x_min, 'position_range': (overlap_x_min, overlap_x_max), 'room1_local_pos': room1_fractional_pos, 'room2_local_pos': room2_fractional_pos})
    if abs(room1_bounds['x_max'] - room2_bounds['x_min']) < 0.1:
        overlap_y_min = max(room1_bounds['y_min'], room2_bounds['y_min'])
        overlap_y_max = min(room1_bounds['y_max'], room2_bounds['y_max'])
        if overlap_y_max > overlap_y_min:
            overlap_center_y = overlap_y_min + (overlap_y_max - overlap_y_min) / 2
            room1_fractional_pos = (overlap_center_y - pos1['y']) / dims1['length']
            room2_fractional_pos = (overlap_center_y - pos2['y']) / dims2['length']
            connections.append({'room1_wall': 'east', 'room2_wall': 'west', 'overlap_length': overlap_y_max - overlap_y_min, 'position_range': (overlap_y_min, overlap_y_max), 'room1_local_pos': room1_fractional_pos, 'room2_local_pos': room2_fractional_pos})
    if abs(room1_bounds['x_min'] - room2_bounds['x_max']) < 0.1:
        overlap_y_min = max(room1_bounds['y_min'], room2_bounds['y_min'])
        overlap_y_max = min(room1_bounds['y_max'], room2_bounds['y_max'])
        if overlap_y_max > overlap_y_min:
            overlap_center_y = overlap_y_min + (overlap_y_max - overlap_y_min) / 2
            room1_fractional_pos = (overlap_center_y - pos1['y']) / dims1['length']
            room2_fractional_pos = (overlap_center_y - pos2['y']) / dims2['length']
            connections.append({'room1_wall': 'west', 'room2_wall': 'east', 'overlap_length': overlap_y_max - overlap_y_min, 'position_range': (overlap_y_min, overlap_y_max), 'room1_local_pos': room1_fractional_pos, 'room2_local_pos': room2_fractional_pos})
    if connections:
        best_connection = max(connections, key=lambda x: x['overlap_length'])
        return {'possible': True, 'connection': best_connection}
    else:
        return {'possible': False, 'reason': 'Rooms do not share adjacent walls'}

def add_connectivity_doors(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Add doors to ensure all rooms are connected.
    
    Args:
        rooms_data: List of room data dictionaries
        
    Returns:
        Dictionary with updated room data and connectivity actions
    """
    connectivity_actions = []
    doors_added = []
    max_iterations = 10
    iteration = 0
    updated_rooms_data = [room.copy() for room in rooms_data]
    while iteration < max_iterations:
        iteration += 1
        connectivity = check_room_connectivity(updated_rooms_data)
        if connectivity['connected']:
            break
        if connectivity['missing_connections']:
            missing_conn = connectivity['missing_connections'][0]
            comp1_rooms = missing_conn['component_1']
            comp2_rooms = missing_conn['component_2']
            best_connection = None
            best_room_pair = None
            for room1_idx in comp1_rooms:
                for room2_idx in comp2_rooms:
                    room1 = updated_rooms_data[room1_idx]
                    room2 = updated_rooms_data[room2_idx]
                    connection = find_best_connection_point(room1, room2)
                    if connection['possible']:
                        if best_connection is None or connection['connection']['overlap_length'] > best_connection['connection']['overlap_length']:
                            best_connection = connection
                            best_room_pair = (room1_idx, room2_idx)
            if best_connection and best_room_pair:
                (room1_idx, room2_idx) = best_room_pair
                room1 = updated_rooms_data[room1_idx]
                room2 = updated_rooms_data[room2_idx]
                door_width = min(0.9, best_connection['connection']['overlap_length'] * 0.8)
                door_result = add_bidirectional_door(room1_idx, room2_idx, best_connection, updated_rooms_data, door_width)
                if not door_result['success']:
                    connectivity_actions.append(f"Iteration {iteration}: Failed to add connecting door - {door_result['error']}")
                    break
                doors_added.append({'room1': room1['room_type'], 'room2': room2['room_type'], 'room1_wall': best_connection['connection']['room1_wall'], 'room2_wall': best_connection['connection']['room2_wall'], 'door_width': door_width, 'overlap_length': best_connection['connection']['overlap_length']})
                connectivity_actions.append(f"Iteration {iteration}: Added connecting door between {room1['room_type']} and {room2['room_type']} (overlap: {best_connection['connection']['overlap_length']:.2f}m)")
            else:
                connectivity_actions.append(f'Iteration {iteration}: Could not find valid connection between components')
                break
        else:
            connectivity_actions.append(f'Iteration {iteration}: Connectivity analysis error')
            break
    final_connectivity = check_room_connectivity(updated_rooms_data)
    return {'success': True, 'connected': final_connectivity['connected'], 'iterations_required': iteration, 'doors_added': len(doors_added), 'connectivity_actions': connectivity_actions, 'doors_added_details': doors_added, 'updated_rooms_data': updated_rooms_data, 'final_connectivity': final_connectivity}

def calculate_room_reduction(room: Dict[str, Any], overlapping_rooms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate how to reduce a room's dimensions to eliminate overlaps with other rooms.
    Also tries repositioning as an alternative to reduction.
    
    Args:
        room: Room to be reduced
        overlapping_rooms: List of rooms that overlap with this room
        
    Returns:
        Dictionary with reduction strategy
    """
    original_width = room['dimensions']['width']
    original_length = room['dimensions']['length']
    original_area = original_width * original_length
    min_width = 1.5
    min_length = 1.5
    min_area = min_width * min_length
    strategies = []
    for reduction_factor in [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3]:
        new_width = original_width * reduction_factor
        new_length = original_length * reduction_factor
        if new_width < min_width or new_length < min_length:
            continue
        test_room = {'room_type': room['room_type'], 'position': room['position'].copy(), 'dimensions': {'width': new_width, 'length': new_length, 'height': room['dimensions']['height']}}
        has_overlaps = False
        for overlapping_room in overlapping_rooms:
            overlap_result = check_room_overlap(test_room, overlapping_room)
            if overlap_result['overlaps']:
                has_overlaps = True
                break
        if not has_overlaps:
            strategies.append({'reduction_factor': reduction_factor, 'new_width': new_width, 'new_length': new_length, 'new_area': new_width * new_length, 'area_loss': original_area - new_width * new_length, 'viable': True, 'method': 'uniform_reduction'})
    for width_factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
        new_width = original_width * width_factor
        if new_width < min_width:
            continue
        test_room = {'room_type': room['room_type'], 'position': room['position'].copy(), 'dimensions': {'width': new_width, 'length': original_length, 'height': room['dimensions']['height']}}
        has_overlaps = False
        for overlapping_room in overlapping_rooms:
            overlap_result = check_room_overlap(test_room, overlapping_room)
            if overlap_result['overlaps']:
                has_overlaps = True
                break
        if not has_overlaps:
            strategies.append({'reduction_factor': width_factor, 'new_width': new_width, 'new_length': original_length, 'new_area': new_width * original_length, 'area_loss': original_area - new_width * original_length, 'viable': True, 'method': 'width_reduction'})
    for length_factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
        new_length = original_length * length_factor
        if new_length < min_length:
            continue
        test_room = {'room_type': room['room_type'], 'position': room['position'].copy(), 'dimensions': {'width': original_width, 'length': new_length, 'height': room['dimensions']['height']}}
        has_overlaps = False
        for overlapping_room in overlapping_rooms:
            overlap_result = check_room_overlap(test_room, overlapping_room)
            if overlap_result['overlaps']:
                has_overlaps = True
                break
        if not has_overlaps:
            strategies.append({'reduction_factor': length_factor, 'new_width': original_width, 'new_length': new_length, 'new_area': original_width * new_length, 'area_loss': original_area - original_width * new_length, 'viable': True, 'method': 'length_reduction'})
    if strategies:
        best_strategy = min(strategies, key=lambda x: x['area_loss'])
        return {'can_reduce': True, 'strategy': best_strategy, 'original_dimensions': {'width': original_width, 'length': original_length}, 'all_strategies': strategies}
    else:
        return {'can_reduce': False, 'reason': 'No viable reduction found that eliminates overlaps while maintaining minimum room size', 'original_dimensions': {'width': original_width, 'length': original_length}, 'min_required': {'width': min_width, 'length': min_length}}

async def get_room_priorities_from_claude(rooms_data: List[Dict[str, Any]], original_description: str) -> List[str]:
    """
    Query Claude to get a strict priority ranking of rooms from most important to least important.
    
    Args:
        rooms_data: List of room data dictionaries
        original_description: Original layout description for context
        
    Returns:
        List of room types in order of decreasing priority (most important first)
    """
    room_list = []
    for (i, room) in enumerate(rooms_data):
        room_info = f"Room {i + 1}: {room['room_type']} ({room['dimensions']['width']}m × {room['dimensions']['length']}m)"
        room_list.append(room_info)
    prompt = f'You are an architectural expert. Given this layout description and list of rooms, provide a strict priority ranking of rooms from MOST IMPORTANT to LEAST IMPORTANT.\n\nORIGINAL LAYOUT DESCRIPTION: "{original_description}"\n\nROOMS TO PRIORITIZE:\n{chr(10).join(room_list)}\n\nConsider these factors for prioritization:\n1. ESSENTIAL LIVING FUNCTIONS: Bedrooms, kitchens, living rooms, bathrooms are typically high priority\n2. ORIGINAL INTENT: Rooms specifically mentioned in the description should be higher priority\n3. ARCHITECTURAL NECESSITY: Load-bearing or structurally important spaces\n4. FUNCTIONAL HIERARCHY: Primary > Secondary > Utility spaces\n5. SIZE EFFICIENCY: Larger rooms that serve core functions vs smaller utility spaces\n\nProvide ONLY a JSON array with room types in strict decreasing priority order:\n["room_type_1", "room_type_2", "room_type_3", ...]\n\nExample: ["master bedroom", "kitchen", "living room", "bathroom", "closet", "storage"]\n\nRespond with ONLY the JSON array, no other text.'
    try:
        response = call_vlm(vlm_type='qwen', max_tokens=3000, temperature=0.1, messages=[{'role': 'user', 'content': prompt}])
        response_text = response.content[0].text.strip()
        response_text = extract_json_from_response(response_text)
        if not response_text:
            raise ValueError('Could not extract JSON content from Claude response')
        priorities = json.loads(response_text)
        return priorities
    except Exception as e:
        raise ValueError(f'Failed to get room priorities from Claude: {str(e)}')

def add_bidirectional_window(room1_idx: int, room2_idx: int, connection_info: Dict[str, Any], rooms_data: List[Dict[str, Any]], window_width: float=1.2) -> Dict[str, Any]:
    """
    Add a connecting window between two rooms on both sides.
    
    Args:
        room1_idx: Index of first room
        room2_idx: Index of second room
        connection_info: Connection information from find_best_connection_point
        rooms_data: List of room data
        window_width: Width of the window to add
        
    Returns:
        Dictionary with operation result and window information
    """
    if not connection_info['possible']:
        return {'success': False, 'error': 'No valid connection point found between rooms'}
    conn = connection_info['connection']
    room1 = rooms_data[room1_idx]
    room2 = rooms_data[room2_idx]
    window1 = {'width': window_width, 'height': 1.2, 'position_on_wall': conn['room1_local_pos'], 'wall_side': conn['room1_wall'], 'sill_height': 0.9, 'window_type': 'connecting'}
    window2 = {'width': window_width, 'height': 1.2, 'position_on_wall': conn['room2_local_pos'], 'wall_side': conn['room2_wall'], 'sill_height': 0.9, 'window_type': 'connecting'}
    if 'windows' not in rooms_data[room1_idx]:
        rooms_data[room1_idx]['windows'] = []
    if 'windows' not in rooms_data[room2_idx]:
        rooms_data[room2_idx]['windows'] = []
    rooms_data[room1_idx]['windows'].append(window1)
    rooms_data[room2_idx]['windows'].append(window2)
    return {'success': True, 'room1_idx': room1_idx, 'room2_idx': room2_idx, 'window1': window1, 'window2': window2, 'window1_idx': len(rooms_data[room1_idx]['windows']) - 1, 'window2_idx': len(rooms_data[room2_idx]['windows']) - 1}

def remove_bidirectional_window(room_idx: int, window_idx: int, rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Remove a connecting window from both sides.
    
    Args:
        room_idx: Index of the room containing the window
        window_idx: Index of the window to remove
        rooms_data: List of room data
        
    Returns:
        Dictionary with operation result
    """
    if window_idx >= len(rooms_data[room_idx]['windows']):
        return {'success': False, 'error': 'Window index out of range'}
    window = rooms_data[room_idx]['windows'][window_idx]
    connecting_window_info = find_connecting_window(room_idx, window, rooms_data)
    removed_window = rooms_data[room_idx]['windows'].pop(window_idx)
    result = {'success': True, 'removed_windows': [{'room_idx': room_idx, 'window_idx': window_idx, 'window': removed_window}]}
    if connecting_window_info:
        other_room_idx = connecting_window_info['room_idx']
        other_window_idx = connecting_window_info['window_idx']
        if other_window_idx < len(rooms_data[other_room_idx]['windows']):
            other_removed_window = rooms_data[other_room_idx]['windows'].pop(other_window_idx)
            result['removed_windows'].append({'room_idx': other_room_idx, 'window_idx': other_window_idx, 'window': other_removed_window})
    return result

def reposition_bidirectional_window(room_idx: int, window_idx: int, new_position: float, rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Reposition a connecting window on both sides.
    
    Args:
        room_idx: Index of the room containing the window
        window_idx: Index of the window to reposition
        new_position: New fractional position on wall (0-1)
        rooms_data: List of room data
        
    Returns:
        Dictionary with operation result
    """
    if window_idx >= len(rooms_data[room_idx]['windows']):
        return {'success': False, 'error': 'Window index out of range'}
    window = rooms_data[room_idx]['windows'][window_idx]
    old_position = window['position_on_wall']
    connecting_window_info = find_connecting_window(room_idx, window, rooms_data)
    rooms_data[room_idx]['windows'][window_idx]['position_on_wall'] = new_position
    result = {'success': True, 'repositioned_windows': [{'room_idx': room_idx, 'window_idx': window_idx, 'old_position': old_position, 'new_position': new_position}]}
    if connecting_window_info:
        other_room_idx = connecting_window_info['room_idx']
        other_window_idx = connecting_window_info['window_idx']
        shared_wall = connecting_window_info['shared_wall']
        if window['wall_side'] == shared_wall['room1_wall']:
            room1 = rooms_data[room_idx]
            room2 = rooms_data[other_room_idx]
            if window['wall_side'] in ['north', 'south']:
                wall_length = room1['dimensions']['width']
                world_x = room1['position']['x'] + new_position * wall_length
                other_wall_length = room2['dimensions']['width']
                other_new_position = (world_x - room2['position']['x']) / other_wall_length
            else:
                wall_length = room1['dimensions']['length']
                world_y = room1['position']['y'] + new_position * wall_length
                other_wall_length = room2['dimensions']['length']
                other_new_position = (world_y - room2['position']['y']) / other_wall_length
        else:
            room1 = rooms_data[other_room_idx]
            room2 = rooms_data[room_idx]
            if window['wall_side'] in ['north', 'south']:
                wall_length = room2['dimensions']['width']
                world_x = room2['position']['x'] + new_position * wall_length
                other_wall_length = room1['dimensions']['width']
                other_new_position = (world_x - room1['position']['x']) / other_wall_length
            else:
                wall_length = room2['dimensions']['length']
                world_y = room2['position']['y'] + new_position * wall_length
                other_wall_length = room1['dimensions']['length']
                other_new_position = (world_y - room1['position']['y']) / other_wall_length
        if other_window_idx < len(rooms_data[other_room_idx]['windows']):
            old_other_position = rooms_data[other_room_idx]['windows'][other_window_idx]['position_on_wall']
            rooms_data[other_room_idx]['windows'][other_window_idx]['position_on_wall'] = other_new_position
            result['repositioned_windows'].append({'room_idx': other_room_idx, 'window_idx': other_window_idx, 'old_position': old_other_position, 'new_position': other_new_position})
    return result

def resize_bidirectional_window(room_idx: int, window_idx: int, new_width: float, rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Resize a connecting window on both sides.
    
    Args:
        room_idx: Index of the room containing the window
        window_idx: Index of the window to resize
        new_width: New width for the window
        rooms_data: List of room data
        
    Returns:
        Dictionary with operation result
    """
    if window_idx >= len(rooms_data[room_idx]['windows']):
        return {'success': False, 'error': 'Window index out of range'}
    window = rooms_data[room_idx]['windows'][window_idx]
    old_width = window['width']
    connecting_window_info = find_connecting_window(room_idx, window, rooms_data)
    rooms_data[room_idx]['windows'][window_idx]['width'] = new_width
    result = {'success': True, 'resized_windows': [{'room_idx': room_idx, 'window_idx': window_idx, 'old_width': old_width, 'new_width': new_width}]}
    if connecting_window_info:
        other_room_idx = connecting_window_info['room_idx']
        other_window_idx = connecting_window_info['window_idx']
        if other_window_idx < len(rooms_data[other_room_idx]['windows']):
            old_other_width = rooms_data[other_room_idx]['windows'][other_window_idx]['width']
            rooms_data[other_room_idx]['windows'][other_window_idx]['width'] = new_width
            result['resized_windows'].append({'room_idx': other_room_idx, 'window_idx': other_window_idx, 'old_width': old_other_width, 'new_width': new_width})
    return result

def find_connecting_window(room_idx: int, window: Dict[str, Any], rooms_data: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Find the corresponding connecting window on the other side of a shared wall.
    
    Args:
        room_idx: Index of the room containing the window
        window: Window data dictionary
        rooms_data: List of all room data
        
    Returns:
        Dictionary with connected room info and window, or None if not a connecting window
    """
    if window.get('window_type') != 'connecting':
        return None
    current_room = rooms_data[room_idx]
    for (other_room_idx, other_room) in enumerate(rooms_data):
        if other_room_idx == room_idx:
            continue
        shared_walls = find_shared_walls(current_room, other_room)
        if not shared_walls:
            continue
        for shared_wall in shared_walls:
            if window['wall_side'] == shared_wall['room1_wall']:
                corresponding_wall = shared_wall['room2_wall']
            elif window['wall_side'] == shared_wall['room2_wall']:
                corresponding_wall = shared_wall['room1_wall']
            else:
                continue
            for (other_window_idx, other_window) in enumerate(other_room.get('windows', [])):
                if other_window['wall_side'] == corresponding_wall and other_window.get('window_type') == 'connecting':
                    window1_world_pos = calculate_door_world_position(current_room, window)
                    window2_world_pos = calculate_door_world_position(other_room, other_window)
                    if doors_align(window1_world_pos, window2_world_pos, window['width'], other_window['width']):
                        return {'room_idx': other_room_idx, 'window_idx': other_window_idx, 'window': other_window, 'room': other_room, 'shared_wall': shared_wall}
    return None

def export_layout_to_json(layout: FloorPlan, export_path: str):
    """
    Export a FloorPlan object to JSON file.
    
    Args:
        layout: FloorPlan object to export
        export_path: Path where the JSON file will be saved
    """
    layout_dict = asdict(layout)
    with open(export_path, 'w') as f:
        json.dump(layout_dict, f, indent=4)

def export_layout_to_mesh(layout: FloorPlan, export_path: str):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
        export_path: Path where the mesh file will be saved (supports .obj, .ply, .stl, etc.)
    """
    from objects.get_objects import get_object_mesh
    floor_meshes = []
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    object_meshes = []
    processed_doors = set()
    processed_windows = set()
    for room in layout.rooms:
        floor_mesh = create_floor_mesh(room)
        floor_meshes.append(floor_mesh)
        (room_wall_meshes, room_door_meshes, room_window_meshes) = create_room_meshes_with_openings(room, processed_doors, processed_windows)
        wall_meshes.extend(room_wall_meshes)
        door_meshes.extend(room_door_meshes)
        window_meshes.extend(room_window_meshes)
        for obj in room.objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id, layout.id)
            if obj_mesh is not None:
                transformed_mesh = apply_object_transform(obj_mesh, obj)
                object_meshes.append(transformed_mesh)
    all_meshes = []
    gray_meshes = floor_meshes + wall_meshes
    if gray_meshes:
        combined_gray = trimesh.util.concatenate(gray_meshes)
        combined_gray.visual.face_colors = [128, 128, 128, 255]
        all_meshes.append(combined_gray)
    if door_meshes:
        combined_doors = trimesh.util.concatenate(door_meshes)
        combined_doors.visual.face_colors = [255, 0, 0, 255]
        all_meshes.append(combined_doors)
    if window_meshes:
        combined_windows = trimesh.util.concatenate(window_meshes)
        combined_windows.visual.face_colors = [0, 0, 255, 255]
        all_meshes.append(combined_windows)
    if object_meshes:
        combined_objects = trimesh.util.concatenate(object_meshes)
        combined_objects.visual.face_colors = [0, 255, 0, 255]
        all_meshes.append(combined_objects)
    if all_meshes:
        final_mesh = trimesh.util.concatenate(all_meshes)
        final_mesh.export(export_path)
        print(f'Floor plan with {len(object_meshes)} objects exported to {export_path}')
    else:
        print('No meshes to export')

def create_floor_mesh(room: Room) -> trimesh.Trimesh:
    """Create a floor mesh for a room."""
    pos = room.position
    dims = room.dimensions
    floor_thickness = 0.1
    floor_box = trimesh.creation.box(extents=[dims.width, dims.length, floor_thickness], transform=trimesh.transformations.translation_matrix([pos.x + dims.width / 2, pos.y + dims.length / 2, pos.z - floor_thickness / 2]))
    return floor_box

def create_room_meshes_with_openings(room: Room, processed_doors: set, processed_windows: set):
    """
    Create wall meshes with door and window openings cut out using boolean operations.
    
    Returns:
        Tuple of (wall_meshes, door_meshes, window_meshes)
    """
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    for wall in room.walls:
        wall_mesh = create_wall_mesh(wall)
        wall_doors = [door for door in room.doors if door.wall_id == wall.id]
        wall_windows = [window for window in room.windows if window.wall_id == wall.id]
        for door in wall_doors:
            door_id = get_door_unique_id(room, door)
            if door_id not in processed_doors:
                door_mesh = create_door_mesh(wall, door)
                if not door.opening:
                    door_meshes.append(door_mesh)
                processed_doors.add(door_id)
                try:
                    wall_mesh = wall_mesh.difference(door_mesh, engine='manifold')
                except:
                    opening_mesh = create_door_opening_mesh(wall, door)
                    try:
                        wall_mesh = wall_mesh.difference(opening_mesh, engine='manifold')
                    except:
                        pass
        for window in wall_windows:
            window_id = get_window_unique_id(room, window)
            if window_id not in processed_windows:
                window_mesh = create_window_mesh(wall, window)
                window_meshes.append(window_mesh)
                processed_windows.add(window_id)
                try:
                    wall_mesh = wall_mesh.difference(window_mesh, engine='manifold')
                except:
                    opening_mesh = create_window_opening_mesh(wall, window)
                    try:
                        wall_mesh = wall_mesh.difference(opening_mesh, engine='manifold')
                    except:
                        pass
        wall_meshes.append(wall_mesh)
    return (wall_meshes, door_meshes, window_meshes)

def create_wall_mesh(wall: Wall) -> trimesh.Trimesh:
    """Create a wall mesh from wall definition."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    wall_length = np.linalg.norm(wall_vector)
    wall_direction = wall_vector / wall_length
    wall_center = (start + end) / 2
    wall_center[2] = wall.start_point.z + wall.height / 2
    wall_box = trimesh.creation.box(extents=[wall_length, wall.thickness, wall.height])
    if abs(wall_direction[0]) < 0.001:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    else:
        rotation_matrix = np.eye(4)
    transform = trimesh.transformations.translation_matrix(wall_center) @ rotation_matrix
    wall_box.apply_transform(transform)
    return wall_box

def create_door_mesh(wall: Wall, door: Door) -> trimesh.Trimesh:
    """Create a door mesh positioned on the wall."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    door_position_3d = start + wall_vector * door.position_on_wall
    door_position_3d[2] = wall.start_point.z + door.height / 2
    door_box = trimesh.creation.box(extents=[door.width, wall.thickness * 1.1, door.height])
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        door_box.apply_transform(rotation_matrix)
    door_box.apply_translation(door_position_3d)
    return door_box

def create_window_mesh(wall: Wall, window: Window) -> trimesh.Trimesh:
    """Create a window mesh positioned on the wall."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    window_position_3d = start + wall_vector * window.position_on_wall
    window_position_3d[2] = wall.start_point.z + window.sill_height + window.height / 2
    window_box = trimesh.creation.box(extents=[window.width, wall.thickness * 1.1, window.height])
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        window_box.apply_transform(rotation_matrix)
    window_box.apply_translation(window_position_3d)
    return window_box

def create_door_opening_mesh(wall: Wall, door: Door) -> trimesh.Trimesh:
    """Create a door opening mesh for boolean subtraction."""
    return create_door_mesh(wall, door)

def create_window_opening_mesh(wall: Wall, window: Window) -> trimesh.Trimesh:
    """Create a window opening mesh for boolean subtraction."""
    return create_window_mesh(wall, window)

def get_door_unique_id(room: Room, door: Door) -> str:
    """Generate unique ID for a door to avoid processing bidirectional doors twice."""
    if door.door_type == 'connecting':
        wall_id = door.wall_id
        position = door.position_on_wall
        return f'connecting_door_{wall_id}_{position:.3f}'
    else:
        return f'door_{room.id}_{door.id}'

def get_window_unique_id(room: Room, window: Window) -> str:
    """Generate unique ID for a window to avoid processing bidirectional windows twice."""
    if window.window_type == 'connecting':
        wall_id = window.wall_id
        position = window.position_on_wall
        return f'connecting_window_{wall_id}_{position:.3f}'
    else:
        return f'window_{room.id}_{window.id}'

def dict_to_floor_plan(layout_data: dict) -> FloorPlan:
    """
    Convert a dictionary (from JSON) back to a FloorPlan object.
    
    Args:
        layout_data: Dictionary containing the floor plan data
        
    Returns:
        FloorPlan object reconstructed from the dictionary
        
    Raises:
        ValueError: If the data structure is invalid or incomplete
    """
    try:
        rooms = []
        for room_data in layout_data['rooms']:
            room = dict_to_room(room_data)
            rooms.append(room)
        floor_plan = FloorPlan(id=layout_data['id'], rooms=rooms, total_area=layout_data['total_area'], building_style=layout_data['building_style'], description=layout_data['description'], created_from_text=layout_data['created_from_text'], policy_analysis=layout_data.get('policy_analysis', None))
        return floor_plan
    except KeyError as e:
        raise ValueError(f'Missing required field in layout data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting layout data: {e}')

def dict_to_room(room_data: dict) -> Room:
    """
    Convert a dictionary to a Room object.
    
    Args:
        room_data: Dictionary containing room data
        
    Returns:
        Room object reconstructed from the dictionary
    """
    try:
        position = Point3D(x=room_data['position']['x'], y=room_data['position']['y'], z=room_data['position']['z'])
        dimensions = Dimensions(width=room_data['dimensions']['width'], length=room_data['dimensions']['length'], height=room_data['dimensions']['height'])
        walls = []
        for wall_data in room_data['walls']:
            wall = dict_to_wall(wall_data)
            walls.append(wall)
        doors = []
        for door_data in room_data['doors']:
            door = dict_to_door(door_data)
            doors.append(door)
        windows = []
        for window_data in room_data['windows']:
            window = dict_to_window(window_data)
            windows.append(window)
        objects = []
        for object_data in room_data.get('objects', []):
            obj = dict_to_object(object_data)
            objects.append(obj)
        room = Room(id=room_data['id'], room_type=room_data['room_type'], position=position, dimensions=dimensions, walls=walls, doors=doors, objects=objects, windows=windows, floor_material=room_data.get('floor_material', 'hardwood'), ceiling_height=room_data.get('ceiling_height', 2.7))
        return room
    except KeyError as e:
        raise ValueError(f'Missing required field in room data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting room data: {e}')

def dict_to_wall(wall_data: dict) -> Wall:
    """
    Convert a dictionary to a Wall object.
    
    Args:
        wall_data: Dictionary containing wall data
        
    Returns:
        Wall object reconstructed from the dictionary
    """
    try:
        start_point = Point3D(x=wall_data['start_point']['x'], y=wall_data['start_point']['y'], z=wall_data['start_point']['z'])
        end_point = Point3D(x=wall_data['end_point']['x'], y=wall_data['end_point']['y'], z=wall_data['end_point']['z'])
        wall = Wall(id=wall_data['id'], start_point=start_point, end_point=end_point, height=wall_data['height'], thickness=wall_data.get('thickness', 0.1), material=wall_data.get('material', 'drywall'))
        return wall
    except KeyError as e:
        raise ValueError(f'Missing required field in wall data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting wall data: {e}')

def dict_to_door(door_data: dict) -> Door:
    """
    Convert a dictionary to a Door object.
    
    Args:
        door_data: Dictionary containing door data
        
    Returns:
        Door object reconstructed from the dictionary
    """
    try:
        door = Door(id=door_data['id'], wall_id=door_data['wall_id'], position_on_wall=door_data['position_on_wall'], width=door_data['width'], height=door_data['height'], door_type=door_data.get('door_type', 'standard'), opens_inward=door_data.get('opens_inward', True), opening=door_data.get('opening', False), door_material=door_data.get('door_material', 'wood'))
        return door
    except KeyError as e:
        raise ValueError(f'Missing required field in door data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting door data: {e}')

def dict_to_window(window_data: dict) -> Window:
    """
    Convert a dictionary to a Window object.
    
    Args:
        window_data: Dictionary containing window data
        
    Returns:
        Window object reconstructed from the dictionary
    """
    try:
        window = Window(id=window_data['id'], wall_id=window_data['wall_id'], position_on_wall=window_data['position_on_wall'], width=window_data['width'], height=window_data['height'], sill_height=window_data['sill_height'], window_type=window_data.get('window_type', 'standard'), window_material=window_data.get('window_material', 'standard'))
        return window
    except KeyError as e:
        raise ValueError(f'Missing required field in window data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting window data: {e}')

def dict_to_object(object_data: dict) -> Object:
    """
    Convert a dictionary to an Object object.
    
    Args:
        object_data: Dictionary containing object data
        
    Returns:
        Object object reconstructed from the dictionary
    """
    try:
        position = Point3D(x=object_data['position']['x'], y=object_data['position']['y'], z=object_data['position']['z'])
        rotation = Euler(x=object_data['rotation']['x'], y=object_data['rotation']['y'], z=object_data['rotation']['z'])
        dimensions = Dimensions(width=object_data['dimensions']['width'], length=object_data['dimensions']['length'], height=object_data['dimensions']['height'])
        obj = Object(id=object_data['id'], room_id=object_data['room_id'], type=object_data['type'], description=object_data['description'], position=position, rotation=rotation, dimensions=dimensions, source=object_data['source'], source_id=object_data['source_id'], place_id=object_data['place_id'], mass=object_data.get('mass', 1.0), placement_constraints=object_data.get('placement_constraints', None))
        return obj
    except KeyError as e:
        raise ValueError(f'Missing required field in object data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting object data: {e}')

def apply_object_transform(mesh: trimesh.Trimesh, obj: Object) -> trimesh.Trimesh:
    """
    Apply position and rotation transforms to an object mesh.
    
    Args:
        mesh: The original mesh (untransformed)
        obj: Object containing position and rotation information
        
    Returns:
        Transformed mesh positioned and rotated according to object properties
    """
    transformed_mesh = mesh.copy()
    rx_rad = np.radians(obj.rotation.x)
    ry_rad = np.radians(obj.rotation.y)
    rz_rad = np.radians(obj.rotation.z)
    rotation_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
    rotation_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
    rotation_z = trimesh.transformations.rotation_matrix(rz_rad, [0, 0, 1])
    combined_rotation = rotation_z @ rotation_y @ rotation_x
    translation = trimesh.transformations.translation_matrix([obj.position.x, obj.position.y, obj.position.z])
    final_transform = translation @ combined_rotation
    transformed_mesh.apply_transform(final_transform)
    return transformed_mesh

def export_layout_to_mesh_dict_list(layout: FloorPlan):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
        export_path: Path where the mesh file will be saved (supports .obj, .ply, .stl, etc.)
    """

    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f'{RESULTS_DIR}/{layout_id}/{source}/{source_id}.ply'
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None
    mesh_dict_list = {}
    mesh_idx_to_object_id = {}
    mesh_idx = 0
    floor_meshes = []
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    object_meshes = []
    processed_doors = set()
    processed_windows = set()
    for room in layout.rooms:
        floor_mesh = create_floor_mesh(room)
        floor_meshes.append(floor_mesh)
        (room_wall_meshes, room_door_meshes, room_window_meshes) = create_room_meshes_with_openings(room, processed_doors, processed_windows)
        wall_meshes.extend(room_wall_meshes)
        door_meshes.extend(room_door_meshes)
        window_meshes.extend(room_window_meshes)
    background_meshes = []
    gray_meshes = floor_meshes + wall_meshes
    if gray_meshes:
        combined_gray = trimesh.util.concatenate(gray_meshes)
        combined_gray.visual.face_colors = [128, 128, 128, 255]
        background_meshes.append(combined_gray)
    if door_meshes:
        combined_doors = trimesh.util.concatenate(door_meshes)
        combined_doors.visual.face_colors = [255, 0, 0, 255]
        background_meshes.append(combined_doors)
    if window_meshes:
        combined_windows = trimesh.util.concatenate(window_meshes)
        combined_windows.visual.face_colors = [0, 0, 255, 255]
        background_meshes.append(combined_windows)
    background_mesh = trimesh.util.concatenate(background_meshes)
    mesh_dict_list[mesh_idx] = {'mesh': background_mesh, 'static': True}
    mesh_idx_to_object_id[mesh_idx] = 'background'
    mesh_idx += 1
    for room in layout.rooms:
        for obj in room.objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id, layout.id)
            if obj_mesh is not None:
                transformed_mesh = apply_object_transform(obj_mesh, obj)
                mesh_dict_list[mesh_idx] = {'mesh': transformed_mesh, 'static': False}
                mesh_idx_to_object_id[mesh_idx] = obj.id
                mesh_idx += 1
    return (mesh_dict_list, mesh_idx_to_object_id)

def find_all_shared_walls(rooms_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Find all shared walls in the floor plan, including both room-room and room-exterior walls.
    
    Args:
        rooms_data: List of room data dictionaries
        
    Returns:
        Dictionary containing:
        - room_room_walls: List of walls shared between two rooms
        - room_exterior_walls: List of walls shared between room and exterior
    """
    room_room_walls = []
    room_exterior_walls = []
    for i in range(len(rooms_data)):
        for j in range(i + 1, len(rooms_data)):
            room1 = rooms_data[i]
            room2 = rooms_data[j]
            shared_walls = find_shared_walls(room1, room2)
            for shared_wall in shared_walls:
                wall_info = {'type': 'room_room', 'room1': {'index': i, 'id': room1.get('id', f'room_{i}'), 'name': room1.get('name', room1['room_type']), 'room_type': room1['room_type']}, 'room2': {'index': j, 'id': room2.get('id', f'room_{j}'), 'name': room2.get('name', room2['room_type']), 'room_type': room2['room_type']}, 'room1_wall': shared_wall['room1_wall'], 'room2_wall': shared_wall['room2_wall'], 'direction': 'x' if shared_wall['room1_wall'] in ['north', 'south'] else 'y', 'overlap_length': shared_wall['overlap_length']}
                if shared_wall['room1_wall'] in ['north', 'south']:
                    wall_info['x_start'] = shared_wall['overlap_start']
                    wall_info['x_end'] = shared_wall['overlap_end']
                    if shared_wall['room1_wall'] == 'north':
                        wall_info['y_start'] = room1['position']['y'] + room1['dimensions']['length']
                        wall_info['y_end'] = room1['position']['y'] + room1['dimensions']['length']
                    else:
                        wall_info['y_start'] = room1['position']['y']
                        wall_info['y_end'] = room1['position']['y']
                else:
                    wall_info['y_start'] = shared_wall['overlap_start']
                    wall_info['y_end'] = shared_wall['overlap_end']
                    if shared_wall['room1_wall'] == 'east':
                        wall_info['x_start'] = room1['position']['x'] + room1['dimensions']['width']
                        wall_info['x_end'] = room1['position']['x'] + room1['dimensions']['width']
                    else:
                        wall_info['x_start'] = room1['position']['x']
                        wall_info['x_end'] = room1['position']['x']
                room_room_walls.append(wall_info)
    for (i, room) in enumerate(rooms_data):
        room_pos = room['position']
        room_dims = room['dimensions']
        room_walls = {'north': {'x_start': room_pos['x'], 'x_end': room_pos['x'] + room_dims['width'], 'y_start': room_pos['y'] + room_dims['length'], 'y_end': room_pos['y'] + room_dims['length'], 'direction': 'x'}, 'south': {'x_start': room_pos['x'], 'x_end': room_pos['x'] + room_dims['width'], 'y_start': room_pos['y'], 'y_end': room_pos['y'], 'direction': 'x'}, 'east': {'x_start': room_pos['x'] + room_dims['width'], 'x_end': room_pos['x'] + room_dims['width'], 'y_start': room_pos['y'], 'y_end': room_pos['y'] + room_dims['length'], 'direction': 'y'}, 'west': {'x_start': room_pos['x'], 'x_end': room_pos['x'], 'y_start': room_pos['y'], 'y_end': room_pos['y'] + room_dims['length'], 'direction': 'y'}}
        for (wall_side, wall_coords) in room_walls.items():
            shared_segments = []
            for shared_wall in room_room_walls:
                if shared_wall['room1']['index'] == i and shared_wall['room1_wall'] == wall_side or (shared_wall['room2']['index'] == i and shared_wall['room2_wall'] == wall_side):
                    if wall_coords['direction'] == 'x':
                        shared_segments.append({'start': shared_wall['x_start'], 'end': shared_wall['x_end']})
                    else:
                        shared_segments.append({'start': shared_wall['y_start'], 'end': shared_wall['y_end']})
            shared_segments.sort(key=lambda x: x['start'])
            if wall_coords['direction'] == 'x':
                wall_start = wall_coords['x_start']
                wall_end = wall_coords['x_end']
            else:
                wall_start = wall_coords['y_start']
                wall_end = wall_coords['y_end']
            exterior_segments = []
            current_pos = wall_start
            for segment in shared_segments:
                if current_pos < segment['start']:
                    exterior_segments.append({'start': current_pos, 'end': segment['start']})
                current_pos = max(current_pos, segment['end'])
            if not shared_segments:
                exterior_segments.append({'start': wall_start, 'end': wall_end})
            elif current_pos < wall_end:
                exterior_segments.append({'start': current_pos, 'end': wall_end})
            for segment in exterior_segments:
                if segment['end'] - segment['start'] > 0.1:
                    wall_info = {'type': 'room_exterior', 'room': {'index': i, 'id': room.get('id', f'room_{i}'), 'name': room.get('name', room['room_type']), 'room_type': room['room_type']}, 'wall_side': wall_side, 'direction': wall_coords['direction'], 'overlap_length': segment['end'] - segment['start']}
                    if wall_coords['direction'] == 'x':
                        wall_info['x_start'] = segment['start']
                        wall_info['x_end'] = segment['end']
                        wall_info['y_start'] = wall_coords['y_start']
                        wall_info['y_end'] = wall_coords['y_end']
                    else:
                        wall_info['x_start'] = wall_coords['x_start']
                        wall_info['x_end'] = wall_coords['x_end']
                        wall_info['y_start'] = segment['start']
                        wall_info['y_end'] = segment['end']
                    room_exterior_walls.append(wall_info)
    return {'room_room_walls': room_room_walls, 'room_exterior_walls': room_exterior_walls, 'total_room_room_walls': len(room_room_walls), 'total_room_exterior_walls': len(room_exterior_walls)}

def generate_floor_plan_visualization(rooms_data: List[Dict[str, Any]], save_path: str=None) -> str:
    """
    Generate a floor plan visualization for Claude API input.
    
    Args:
        rooms_data: List of room data dictionaries
        save_path: Optional path to save the visualization
        
    Returns:
        Path to the saved visualization image
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.patches import Rectangle
        import tempfile
        import os
        (fig, ax) = plt.subplots(1, 1, figsize=(12, 10))
        colors = {'living room': '#FFB6C1', 'bedroom': '#ADD8E6', 'kitchen': '#98FB98', 'bathroom': '#F0E68C', 'dining room': '#DDA0DD', 'office': '#F4A460', 'closet': '#D3D3D3', 'hallway': '#FFEFD5', 'garage': '#C0C0C0', 'default': '#E6E6FA'}

        def get_room_color(room_type: str) -> str:
            return colors.get(room_type.lower(), colors['default'])
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for room in rooms_data:
            room_min_x = room['position']['x']
            room_min_y = room['position']['y']
            room_max_x = room['position']['x'] + room['dimensions']['width']
            room_max_y = room['position']['y'] + room['dimensions']['length']
            min_x = min(min_x, room_min_x)
            min_y = min(min_y, room_min_y)
            max_x = max(max_x, room_max_x)
            max_y = max(max_y, room_max_y)
        padding = 1.0
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        for (i, room) in enumerate(rooms_data):
            color = get_room_color(room['room_type'])
            room_rect = Rectangle((room['position']['x'], room['position']['y']), room['dimensions']['width'], room['dimensions']['length'], facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
            ax.add_patch(room_rect)
            center_x = room['position']['x'] + room['dimensions']['width'] / 2
            center_y = room['position']['y'] + room['dimensions']['length'] / 2
            room_label = f"{room['room_type'].title()}\n(Room {i})"
            ax.text(center_x, center_y, room_label, ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)', fontsize=12)
        ax.set_ylabel('Y (meters)', fontsize=12)
        ax.set_title('Floor Plan Layout - Room Structure Only', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        legend_elements = []
        used_types = set()
        for room in rooms_data:
            if room['room_type'] not in used_types:
                color = get_room_color(room['room_type'])
                legend_elements.append(patches.Patch(color=color, label=room['room_type'].title()))
                used_types.add(room['room_type'])
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        if save_path is None:
            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
            save_path = temp_file.name
            temp_file.close()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    except ImportError:
        raise ValueError('matplotlib is required for floor plan visualization')
    except Exception as e:
        raise ValueError(f'Error generating floor plan visualization: {str(e)}')

def transform_shared_wall_position_to_room_wall(shared_wall_info: Dict[str, Any], shared_wall_position: float, room_index: int) -> float:
    """
    Transform a position on a shared wall (0-1 range) to a position on an individual room's wall (0-1 range).
    
    Args:
        shared_wall_info: Information about the shared wall
        shared_wall_position: Position on the shared wall (0-1 range)
        room_index: Index of the room (0 for room1, 1 for room2)
        
    Returns:
        Position on the individual room's wall (0-1 range)
    """
    if shared_wall_info['type'] != 'room_room':
        raise ValueError('This function is only for room-room shared walls')
    if room_index not in [0, 1]:
        raise ValueError('room_index must be 0 (room1) or 1 (room2)')
    if shared_wall_info['direction'] == 'x':
        shared_start = shared_wall_info['x_start']
        shared_end = shared_wall_info['x_end']
    else:
        shared_start = shared_wall_info['y_start']
        shared_end = shared_wall_info['y_end']
    world_position = shared_start + (shared_end - shared_start) * shared_wall_position
    if room_index == 0:
        room_info = shared_wall_info['room1']
        wall_side = shared_wall_info['room1_wall']
    else:
        room_info = shared_wall_info['room2']
        wall_side = shared_wall_info['room2_wall']
    if shared_wall_info['direction'] == 'x':
        if room_index == 0:
            room_data = {'position': {'x': shared_wall_info['x_start'] - shared_wall_info['room1']['dimensions']['width'] if wall_side == 'east' else shared_wall_info['x_start']}}
        else:
            room_data = {'position': {'x': shared_wall_info['x_start'] - shared_wall_info['room2']['dimensions']['width'] if wall_side == 'east' else shared_wall_info['x_start']}}
        room_wall_position = shared_wall_position
    else:
        room_wall_position = shared_wall_position
    return max(0.0, min(1.0, room_wall_position))

def calculate_room_wall_position_from_shared_wall(shared_wall_info: Dict[str, Any], shared_wall_position: float, room_data: Dict[str, Any], room_is_room1: bool) -> float:
    """
    Calculate the position on a room's wall based on a position on a shared wall.
    
    Args:
        shared_wall_info: Information about the shared wall
        shared_wall_position: Position on the shared wall (0-1 range)
        room_data: The room data dictionary
        room_is_room1: True if this is room1, False if room2
        
    Returns:
        Position on the room's wall (0-1 range)
    """
    if shared_wall_info['type'] != 'room_room':
        raise ValueError('This function is only for room-room shared walls')
    if shared_wall_info['direction'] == 'x':
        shared_start = shared_wall_info['x_start']
        shared_end = shared_wall_info['x_end']
    else:
        shared_start = shared_wall_info['y_start']
        shared_end = shared_wall_info['y_end']
    world_position = shared_start + (shared_end - shared_start) * shared_wall_position
    if room_is_room1:
        wall_side = shared_wall_info['room1_wall']
    else:
        wall_side = shared_wall_info['room2_wall']
    room_pos = room_data['position']
    room_dims = room_data['dimensions']
    if wall_side in ['north', 'south']:
        wall_start = room_pos['x']
        wall_length = room_dims['width']
        room_wall_position = (world_position - wall_start) / wall_length
    else:
        wall_start = room_pos['y']
        wall_length = room_dims['length']
        room_wall_position = (world_position - wall_start) / wall_length
    return max(0.0, min(1.0, room_wall_position))

def clean_json_comments(json_str: str) -> str:
    """
    Remove C-style comments (// and /* */ comment text) from a JSON string while preserving 
    any comment markers that appear inside string values.
    
    Args:
        json_str: JSON string that may contain C-style comments
        
    Returns:
        Cleaned JSON string without comments
    """
    in_string = False
    in_block_comment = False
    escape_next = False
    result = []
    i = 0
    while i < len(json_str):
        char = json_str[i]
        if escape_next:
            result.append(char)
            escape_next = False
            i += 1
        elif char == '\\' and in_string:
            result.append(char)
            escape_next = True
            i += 1
        elif char == '"' and (not in_block_comment):
            in_string = not in_string
            result.append(char)
            i += 1
        elif not in_string and (not in_block_comment) and (char == '/') and (i + 1 < len(json_str)):
            next_char = json_str[i + 1]
            if next_char == '/':
                i += 2
                while i < len(json_str) and json_str[i] != '\n':
                    i += 1
            elif next_char == '*':
                in_block_comment = True
                i += 2
            else:
                result.append(char)
                i += 1
        elif in_block_comment and char == '*' and (i + 1 < len(json_str)) and (json_str[i + 1] == '/'):
            in_block_comment = False
            i += 2
        elif in_block_comment:
            i += 1
        else:
            result.append(char)
            i += 1
    result_str = ''.join(result)
    lines = result_str.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.rstrip()
        if line:
            cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def extract_json_from_response(response_text: str) -> Optional[str]:
    """
    Robustly extract JSON content from Claude's response, handling various formats.
    
    Args:
        response_text: Raw response text from Claude
        
    Returns:
        Extracted JSON string or None if not found
    """
    import re
    json_pattern = '```json\\s*\\n(.*?)\\n```'
    matches = re.findall(json_pattern, response_text, re.DOTALL | re.IGNORECASE)
    if matches:
        content = matches[-1].strip()
        return clean_json_comments(content)
    code_block_pattern = '```(?:json)?\\s*\\n?(.*?)\\n?```'
    matches = re.findall(code_block_pattern, response_text, re.DOTALL | re.IGNORECASE)
    for match in matches:
        content = match.strip()
        if content.startswith('json\n'):
            content = content[5:].strip()
        elif content.startswith('json '):
            content = content[5:].strip()
        if content.startswith('{') and content.endswith('}'):
            cleaned_content = clean_json_comments(content)
            try:
                json.loads(cleaned_content)
                return cleaned_content
            except json.JSONDecodeError:
                continue
    json_object_pattern = '\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}'
    matches = re.findall(json_object_pattern, response_text, re.DOTALL)
    if matches:
        matches = sorted(matches, key=len, reverse=True)
        for match in matches:
            match = match.strip()
            cleaned_content = clean_json_comments(match)
            try:
                parsed = json.loads(cleaned_content)
                return cleaned_content
            except json.JSONDecodeError:
                pass
    lines = response_text.split('\n')
    json_lines = []
    brace_count = 0
    capturing = False
    for line in lines:
        line = line.strip()
        if not capturing and line.startswith('{'):
            capturing = True
            json_lines = [line]
            brace_count = line.count('{') - line.count('}')
        elif capturing:
            json_lines.append(line)
            brace_count += line.count('{') - line.count('}')
            if brace_count <= 0:
                json_content = '\n'.join(json_lines)
                cleaned_content = clean_json_comments(json_content)
                try:
                    json.loads(cleaned_content)
                    return cleaned_content
                except json.JSONDecodeError:
                    capturing = False
                    json_lines = []
                    brace_count = 0
    for i in range(len(response_text)):
        if response_text[i] == '{':
            for j in range(i + 1, len(response_text)):
                if response_text[j] == '}':
                    candidate = response_text[i:j + 1]
                    cleaned_content = clean_json_comments(candidate)
                    try:
                        parsed = json.loads(cleaned_content)
                        if isinstance(parsed, dict) and len(parsed) > 1:
                            return cleaned_content
                    except json.JSONDecodeError:
                        continue
    return None

def get_layout_from_scene_save_dir(scene_save_dir: str) -> FloorPlan:
    """
    Load a room layout from JSON data and set it as the current layout.
    """
    global current_layout
    layout_id = os.path.basename(scene_save_dir)
    json_file_path = os.path.join(scene_save_dir, f'{layout_id}.json')
    with open(json_file_path, 'r') as f:
        layout_data = json.load(f)
    floor_plan = dict_to_floor_plan(layout_data)
    return floor_plan

def get_layout_from_scene_json_path(json_file_path: str) -> FloorPlan:
    """
    Load a room layout from JSON data and set it as the current layout.
    """
    global current_layout
    with open(json_file_path, 'r') as f:
        layout_data = json.load(f)
    floor_plan = dict_to_floor_plan(layout_data)
    return floor_plan
