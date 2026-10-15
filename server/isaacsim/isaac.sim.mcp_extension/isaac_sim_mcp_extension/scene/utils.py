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
from sympy import im
from typing import Dict
from isaac_sim_mcp_extension.scene.models import FloorPlan, Room, Wall, Door, Object, Window, Point3D, Dimensions, Euler
import trimesh
import numpy as np
import os
import xatlas
import sys
import json
print('os.path.abspath(__file__):', os.path.abspath(__file__))
isaac_ext_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print('isaac_ext_dir:', isaac_ext_dir)
if os.path.islink(isaac_ext_dir):
    isaac_ext_dir = os.readlink(isaac_ext_dir)
print('isaac_ext_dir:', isaac_ext_dir)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(isaac_ext_dir))))
SERVER_ROOT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(isaac_ext_dir))), 'server')
print('SERVER_ROOT_DIR:', SERVER_ROOT_DIR)

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
        window = Window(id=window_data['id'], wall_id=window_data['wall_id'], position_on_wall=window_data['position_on_wall'], width=window_data['width'], height=window_data['height'], sill_height=window_data['sill_height'], window_type=window_data.get('window_type', 'standard'))
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
        obj = Object(id=object_data['id'], room_id=object_data['room_id'], type=object_data['type'], description=object_data['description'], position=position, rotation=rotation, dimensions=dimensions, source=object_data['source'], source_id=object_data['source_id'], place_id=object_data['place_id'], mass=object_data.get('mass', 1.0))
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
        floor_plan = FloorPlan(id=layout_data['id'], rooms=rooms, total_area=layout_data['total_area'], building_style=layout_data['building_style'], description=layout_data['description'], created_from_text=layout_data['created_from_text'])
        return floor_plan
    except KeyError as e:
        raise ValueError(f'Missing required field in layout data: {e}')
    except Exception as e:
        raise ValueError(f'Error converting layout data: {e}')

def create_floor_mesh(room: Room) -> trimesh.Trimesh:
    """Create a floor mesh for a room."""
    pos = room.position
    dims = room.dimensions
    floor_thickness = 0.1
    floor_box = trimesh.creation.box(extents=[dims.width, dims.length, floor_thickness], transform=trimesh.transformations.translation_matrix([pos.x + dims.width / 2, pos.y + dims.length / 2, pos.z - floor_thickness / 2]))
    return floor_box

def create_ceiling_mesh(room: Room) -> trimesh.Trimesh:
    """Create a ceiling mesh for a room."""
    pos = room.position
    dims = room.dimensions
    ceiling_thickness = 0.1
    ceiling_box = trimesh.creation.box(extents=[dims.width, dims.length, ceiling_thickness], transform=trimesh.transformations.translation_matrix([pos.x + dims.width / 2, pos.y + dims.length / 2, pos.z + dims.height + ceiling_thickness / 2]))
    return ceiling_box

def create_room_meshes_with_openings(room: Room, processed_doors: set, processed_windows: set):
    """
    Create wall meshes with door and window openings cut out using boolean operations.
    
    Returns:
        Tuple of (wall_meshes, door_meshes, window_meshes)
    """
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    wall_ids = []
    door_ids = []
    window_ids = []
    for wall in room.walls:
        wall_mesh = create_wall_mesh(wall, room)
        wall_ids.append(wall.id)
        wall_doors = [door for door in room.doors if door.wall_id == wall.id]
        wall_windows = [window for window in room.windows if window.wall_id == wall.id]
        for door in wall_doors:
            door_id = get_door_unique_id(room, door)
            if door_id not in processed_doors:
                door_mesh = create_door_mesh(wall, door, door_size_offset=0.11)
                if not door.opening:
                    door_meshes.append(door_mesh)
                processed_doors.add(door_id)
                door_ids.append(door_id)
                try:
                    wall_mesh = wall_mesh.difference(door_mesh, engine='manifold')
                except:
                    opening_mesh = create_door_opening_mesh(wall, door)
                    try:
                        wall_mesh = wall_mesh.difference(opening_mesh, engine='manifold')
                    except:
                        print(f'Boolean operation failed for door {door.id} on wall {wall.id}')
                        print(f'Boolean operation failed for door {door.id} on wall {wall.id}', file=sys.stderr)
                        pass
        for window in wall_windows:
            window_id = get_window_unique_id(room, window)
            if window_id not in processed_windows:
                window_mesh = create_window_mesh(wall, window)
                window_meshes.append(window_mesh)
                processed_windows.add(window_id)
                window_ids.append(window.id)
                try:
                    wall_mesh = wall_mesh.difference(window_mesh, engine='manifold')
                except:
                    opening_mesh = create_window_opening_mesh(wall, window)
                    try:
                        wall_mesh = wall_mesh.difference(opening_mesh, engine='manifold')
                    except:
                        print(f'Boolean operation failed for window {window.id} on wall {wall.id}')
                        print(f'Boolean operation failed for window {window.id} on wall {wall.id}', file=sys.stderr)
                        pass
        wall_meshes.append(wall_mesh)
    return (wall_meshes, door_meshes, window_meshes, wall_ids, door_ids, window_ids)

def create_wall_mesh(wall: Wall, room: Room) -> trimesh.Trimesh:
    """Create a wall mesh from wall definition."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    wall_length = np.linalg.norm(wall_vector)
    wall_direction = wall_vector / wall_length
    room_center = np.array([room.position.x + room.dimensions.width / 2, room.position.y + room.dimensions.length / 2, room.position.z])
    wall_center = (start + end) / 2
    normal1 = np.array([wall_direction[1], -wall_direction[0], 0])
    normal2 = np.array([-wall_direction[1], wall_direction[0], 0])
    wall_to_room = room_center - wall_center
    if np.dot(normal1, wall_to_room) > 0:
        inward_normal = normal1
    else:
        inward_normal = normal2
    half_thickness = wall.thickness / 2
    wall_center[2] = wall.start_point.z + wall.height / 2
    wall_center_offset = wall_center + inward_normal * (half_thickness / 2)
    wall_box = trimesh.creation.box(extents=[wall_length, half_thickness, wall.height])
    if abs(wall_direction[0]) < 0.001:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
    else:
        rotation_matrix = np.eye(4)
    transform = trimesh.transformations.translation_matrix(wall_center_offset) @ rotation_matrix
    wall_box.apply_transform(transform)
    return wall_box

def create_door_mesh(wall: Wall, door: Door, size_scale: float=1.0, thickness_scale: float=1.0, door_size_offset: float=0.0) -> trimesh.Trimesh:
    """Create a door mesh positioned on the wall."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    door_position_3d = start + wall_vector * door.position_on_wall
    door_position_3d[2] = wall.start_point.z + door.height / 2
    door_box = trimesh.creation.box(extents=[door.width * size_scale + door_size_offset, wall.thickness * thickness_scale, door.height * size_scale + door_size_offset])
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    if abs(wall_direction[0]) < 0.001:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        door_box.apply_transform(rotation_matrix)
    door_box.apply_translation(door_position_3d)
    return door_box

def create_door_frame_mesh(wall: Wall, door: Door, size_scale: float=1.0, thickness_scale: float=1.0, frame_width: float=0.05) -> trimesh.Trimesh:
    """Create a door frame mesh with left, right, and top edges surrounding the door."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    wall_direction = wall_vector / np.linalg.norm(wall_vector)
    door_position_3d = start + wall_vector * door.position_on_wall
    door_position_3d[2] = wall.start_point.z + door.height / 2
    door_width = door.width * size_scale
    door_height = door.height * size_scale
    door_thickness = wall.thickness * thickness_scale
    frame_thickness = wall.thickness * thickness_scale
    gap_width = (door.width - door_width) / 2
    gap_height = door.height - door_height
    frame_meshes = []
    left_frame = trimesh.creation.box(extents=[frame_width, frame_thickness, door_height + gap_height + frame_width])
    left_offset = np.array([-(door_width / 2 + frame_width / 2), 0, (gap_height + frame_width) / 2])
    right_frame = trimesh.creation.box(extents=[frame_width, frame_thickness, door_height + gap_height + frame_width])
    right_offset = np.array([door_width / 2 + frame_width / 2, 0, (gap_height + frame_width) / 2])
    top_frame = trimesh.creation.box(extents=[door_width + 2 * frame_width, frame_thickness, frame_width])
    top_offset = np.array([0, 0, door_height / 2 + frame_width / 2])
    left_frame.apply_translation(left_offset)
    right_frame.apply_translation(right_offset)
    top_frame.apply_translation(top_offset)
    frame_meshes = [left_frame, right_frame, top_frame]
    combined_frame = trimesh.util.concatenate(frame_meshes)
    if abs(wall_direction[0]) < 0.001:
        rotation_matrix = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 0, 1])
        combined_frame.apply_transform(rotation_matrix)
    combined_frame.apply_translation(door_position_3d)
    return combined_frame

def create_window_mesh(wall: Wall, window: Window) -> trimesh.Trimesh:
    """Create a window mesh positioned on the wall."""
    import numpy as np
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_vector = end - start
    window_position_3d = start + wall_vector * window.position_on_wall
    window_position_3d[2] = wall.start_point.z + window.sill_height + window.height / 2
    window_box = trimesh.creation.box(extents=[window.width, wall.thickness * 1.0, window.height])
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
    import os
    import pickle
    layout_save_dir = os.environ.get('SAGE_RESULTS_DIR', os.path.join(SERVER_ROOT_DIR, 'results/'))

    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}.ply'
        watertight_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_wt.ply'
        if source == 'asset_library' and os.path.exists(watertight_path):
            return trimesh.load(watertight_path)
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None

    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl'
        texture_map_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png'
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, 'rb') as f:
                tex_coords = pickle.load(f)
            return {'vts': tex_coords['vts'], 'fts': tex_coords['fts'], 'texture_map_path': texture_map_path}
        else:
            return None
    mesh_info_dict = {}
    floor_meshes = []
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    object_meshes = []
    processed_doors = set()
    processed_windows = set()
    for room in layout.rooms:
        floor_mesh = create_floor_mesh(room)
        floor_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.floor_material}.png'
        floor_mesh_tex_coords = create_floor_mesh_tex_coords(floor_mesh)
        mesh_info_dict[f'floor_{room.id}'] = {'mesh': floor_mesh, 'static': True, 'texture': {'vts': floor_mesh_tex_coords['vts'], 'fts': floor_mesh_tex_coords['fts'], 'texture_map_path': floor_mesh_texture_map_path}}
        (room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids) = create_room_meshes_with_openings(room, processed_doors, processed_windows)
        wall_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png'
        for (wall_id, wall_mesh) in zip(room_wall_ids, room_wall_meshes):
            wall_mesh_tex_coords = create_wall_mesh_tex_coords(wall_mesh)
            mesh_info_dict[f'{wall_id}'] = {'mesh': wall_mesh, 'static': True, 'texture': {'vts': wall_mesh_tex_coords['vts'], 'fts': wall_mesh_tex_coords['fts'], 'texture_map_path': wall_mesh_texture_map_path}}
        ceiling_mesh = create_ceiling_mesh(room)
        ceiling_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png'
        ceiling_mesh_tex_coords = create_wall_mesh_tex_coords(ceiling_mesh)
        mesh_info_dict[f'floor_{room.id}_ceiling'] = {'mesh': ceiling_mesh, 'static': True, 'texture': {'vts': ceiling_mesh_tex_coords['vts'], 'fts': ceiling_mesh_tex_coords['fts'], 'texture_map_path': ceiling_mesh_texture_map_path}}
        for (window_id, window_mesh) in zip(room_window_ids, room_window_meshes):
            window_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{window_id}_texture.png'
            window_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{window_id}_tex_coords.pkl'
            with open(window_mesh_tex_coords_save_path, 'rb') as f:
                window_mesh_tex_coords = pickle.load(f)
            mesh_info_dict[f'{window_id}'] = {'mesh': window_mesh, 'static': True, 'texture': {'vts': window_mesh_tex_coords['vts'], 'fts': window_mesh_tex_coords['fts'], 'texture_map_path': window_mesh_texture_map_path}}
    for room in layout.rooms:
        for obj in room.objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id, layout.id)
            if obj_mesh is not None:
                transformed_mesh = apply_object_transform(obj_mesh, obj)
                texture_info = get_object_mesh_texture(obj.source, obj.source_id, layout.id)
                print(f'obj.place_id: {obj.place_id}, obj.id: {obj.id}, obj.type: {obj.type}, obj.description: {obj.description}')
                mesh_info_dict[obj.id] = {'mesh': transformed_mesh, 'static': False if obj.place_id != 'wall' else True, 'texture': texture_info, 'mass': getattr(obj, 'mass', 1.0), 'transform': {'position': {'x': obj.position.x, 'y': obj.position.y, 'z': obj.position.z}, 'rotation': {'x': obj.rotation.x, 'y': obj.rotation.y, 'z': obj.rotation.z}}}
    door_center_list = []
    for room in layout.rooms:
        wall_map = {wall.id: wall for wall in room.walls}
        for door in room.doors:
            if door.opening:
                continue
            wall = wall_map.get(door.wall_id)
            assert wall is not None, f'Wall {door.wall_id} not found'
            start_point = wall.start_point
            end_point = wall.end_point
            position_on_wall = door.position_on_wall
            door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall
            door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall
            duplicate_door = False
            for (door_center_prev_x, door_center_prev_y) in door_center_list:
                if abs(door_center_x - door_center_prev_x) < 0.01 and abs(door_center_y - door_center_prev_y) < 0.01:
                    duplicate_door = True
                    break
            if duplicate_door:
                continue
            door_center_list.append((door_center_x, door_center_y))
            thickness_scale = 0.95
            size_scale = 0.95
            door_width_original = door.width
            door_thickness = wall.thickness * thickness_scale
            delta_s = 0.5 * (1 - size_scale) * door_width_original
            delta_r_min = max(0, ((0.5 * door_thickness) ** 2 - delta_s ** 2) / (2 * delta_s))
            delta_r = delta_r_min * 1.1
            door_size_offset_calculated = (size_scale - 1) * door_width_original
            door_mesh = create_door_mesh(wall, door, size_scale=1.0, thickness_scale=thickness_scale, door_size_offset=door_size_offset_calculated)
            door_frame_mesh = create_door_frame_mesh(wall, door, size_scale=1.0, thickness_scale=1.05, frame_width=0.05)
            start = np.array([wall.start_point.x, wall.start_point.y, 0])
            end = np.array([wall.end_point.x, wall.end_point.y, 0])
            wall_vector = end - start
            wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
            door_center_point = start + wall_vector * door.position_on_wall
            door_start_point = door_center_point - wall_vector_norm * door.width / 2
            door_rotate_axis_point_lower = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_lower[2] = 0.0
            door_rotate_axis_point_upper = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_upper[2] = door.height
            door_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_texture.png'
            door_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_tex_coords.pkl'
            with open(door_mesh_tex_coords_save_path, 'rb') as f:
                door_mesh_tex_coords = pickle.load(f)
            texture_info = {'vts': door_mesh_tex_coords['vts'], 'fts': door_mesh_tex_coords['fts'], 'texture_map_path': door_mesh_texture_map_path}
            mesh_info_dict[f'{door.id}'] = {'mesh': door_mesh, 'static': False, 'articulation': (door_rotate_axis_point_lower, door_rotate_axis_point_upper), 'texture': texture_info}
            door_frame_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_texture.png'
            door_frame_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_tex_coords.pkl'
            if not os.path.exists(door_frame_tex_coords_save_path):
                door_frame_texture_map_path = door_mesh_texture_map_path
                door_frame_tex_coords_save_path = door_mesh_tex_coords_save_path
            with open(door_frame_tex_coords_save_path, 'rb') as f:
                door_frame_tex_coords = pickle.load(f)
            door_frame_texture_info = {'vts': door_frame_tex_coords['vts'], 'fts': door_frame_tex_coords['fts'], 'texture_map_path': door_frame_texture_map_path}
            mesh_info_dict[f'{door.id}_frame'] = {'mesh': door_frame_mesh, 'static': True, 'texture': door_frame_texture_info}
    return mesh_info_dict

def export_layout_to_mesh_dict_list_no_object_transform(layout: FloorPlan):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
        export_path: Path where the mesh file will be saved (supports .obj, .ply, .stl, etc.)
    """
    import os
    import pickle
    layout_save_dir = os.environ.get('SAGE_RESULTS_DIR', os.path.join(SERVER_ROOT_DIR, 'results/'))

    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}.ply'
        watertight_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_wt.ply'
        if source == 'asset_library' and os.path.exists(watertight_path):
            return trimesh.load(watertight_path)
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None

    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl'
        texture_map_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png'
        texture_pbr_params_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_pbr_parameters.json'
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, 'rb') as f:
                tex_coords = pickle.load(f)
            tex_dict = {'vts': tex_coords['vts'], 'fts': tex_coords['fts'], 'texture_map_path': texture_map_path}
            if os.path.exists(texture_pbr_params_path):
                with open(texture_pbr_params_path, 'r') as f:
                    pbr_parameters = json.load(f)
                tex_dict['pbr_parameters'] = pbr_parameters
            return tex_dict
        else:
            return None
    mesh_info_dict = {}
    floor_meshes = []
    wall_meshes = []
    door_meshes = []
    window_meshes = []
    object_meshes = []
    processed_doors = set()
    processed_windows = set()
    for room in layout.rooms:
        floor_mesh = create_floor_mesh(room)
        floor_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.floor_material}.png'
        floor_mesh_tex_coords = create_floor_mesh_tex_coords(floor_mesh)
        mesh_info_dict[f'floor_{room.id}'] = {'mesh': floor_mesh, 'static': True, 'texture': {'vts': floor_mesh_tex_coords['vts'], 'fts': floor_mesh_tex_coords['fts'], 'texture_map_path': floor_mesh_texture_map_path}}
        (room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids) = create_room_meshes_with_openings(room, processed_doors, processed_windows)
        wall_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png'
        for (wall_id, wall_mesh) in zip(room_wall_ids, room_wall_meshes):
            wall_mesh_tex_coords = create_wall_mesh_tex_coords(wall_mesh)
            mesh_info_dict[f'{wall_id}'] = {'mesh': wall_mesh, 'static': True, 'texture': {'vts': wall_mesh_tex_coords['vts'], 'fts': wall_mesh_tex_coords['fts'], 'texture_map_path': wall_mesh_texture_map_path}}
        ceiling_mesh = create_ceiling_mesh(room)
        ceiling_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png'
        ceiling_mesh_tex_coords = create_wall_mesh_tex_coords(ceiling_mesh)
        mesh_info_dict[f'floor_{room.id}_ceiling'] = {'mesh': ceiling_mesh, 'static': True, 'texture': {'vts': ceiling_mesh_tex_coords['vts'], 'fts': ceiling_mesh_tex_coords['fts'], 'texture_map_path': ceiling_mesh_texture_map_path}}
        for (window_id, window_mesh) in zip(room_window_ids, room_window_meshes):
            window_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{window_id}_texture.png'
            window_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{window_id}_tex_coords.pkl'
            with open(window_mesh_tex_coords_save_path, 'rb') as f:
                window_mesh_tex_coords = pickle.load(f)
            mesh_info_dict[f'{window_id}'] = {'mesh': window_mesh, 'static': True, 'texture': {'vts': window_mesh_tex_coords['vts'], 'fts': window_mesh_tex_coords['fts'], 'texture_map_path': window_mesh_texture_map_path, 'pbr_parameters': {'roughness': 0.05, 'metallic': 0.0}}}
    for room in layout.rooms:
        for obj in room.objects:
            obj_mesh = get_object_mesh(obj.source, obj.source_id, layout.id)
            if obj_mesh is not None:
                texture_info = get_object_mesh_texture(obj.source, obj.source_id, layout.id)
                mesh_info_dict[obj.id] = {'mesh': obj_mesh, 'static': False if obj.place_id != 'wall' else True, 'texture': texture_info, 'mass': getattr(obj, 'mass', 1.0), 'transform': {'position': {'x': obj.position.x, 'y': obj.position.y, 'z': obj.position.z}, 'rotation': {'x': obj.rotation.x, 'y': obj.rotation.y, 'z': obj.rotation.z}}}
    door_center_list = []
    for room in layout.rooms:
        wall_map = {wall.id: wall for wall in room.walls}
        for door in room.doors:
            if door.opening:
                continue
            wall = wall_map.get(door.wall_id)
            assert wall is not None, f'Wall {door.wall_id} not found'
            start_point = wall.start_point
            end_point = wall.end_point
            position_on_wall = door.position_on_wall
            door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall
            door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall
            duplicate_door = False
            for (door_center_prev_x, door_center_prev_y) in door_center_list:
                if abs(door_center_x - door_center_prev_x) < 0.01 and abs(door_center_y - door_center_prev_y) < 0.01:
                    duplicate_door = True
                    break
            if duplicate_door:
                continue
            door_center_list.append((door_center_x, door_center_y))
            thickness_scale = 0.95
            size_scale = 0.95
            door_width_original = door.width
            door_thickness = wall.thickness * thickness_scale
            delta_s = 0.5 * (1 - size_scale) * door_width_original
            delta_r_min = max(0, ((0.5 * door_thickness) ** 2 - delta_s ** 2) / (2 * delta_s))
            delta_r = delta_r_min * 1.1
            door_size_offset_calculated = (size_scale - 1) * door_width_original
            door_mesh = create_door_mesh(wall, door, size_scale=1.0, thickness_scale=thickness_scale, door_size_offset=door_size_offset_calculated)
            door_frame_mesh = create_door_frame_mesh(wall, door, size_scale=1.0, thickness_scale=1.05, frame_width=0.05)
            start = np.array([wall.start_point.x, wall.start_point.y, 0])
            end = np.array([wall.end_point.x, wall.end_point.y, 0])
            wall_vector = end - start
            wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
            door_center_point = start + wall_vector * door.position_on_wall
            door_start_point = door_center_point - wall_vector_norm * door.width / 2
            door_rotate_axis_point_lower = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_lower[2] = 0.0
            door_rotate_axis_point_upper = door_start_point + wall_vector_norm * (delta_s + delta_r)
            door_rotate_axis_point_upper[2] = door.height
            door_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_texture.png'
            door_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_tex_coords.pkl'
            with open(door_mesh_tex_coords_save_path, 'rb') as f:
                door_mesh_tex_coords = pickle.load(f)
            texture_info = {'vts': door_mesh_tex_coords['vts'], 'fts': door_mesh_tex_coords['fts'], 'texture_map_path': door_mesh_texture_map_path}
            mesh_info_dict[f'{door.id}'] = {'mesh': door_mesh, 'static': False, 'articulation': (door_rotate_axis_point_lower, door_rotate_axis_point_upper), 'texture': texture_info}
            door_frame_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_texture.png'
            door_frame_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_tex_coords.pkl'
            if not os.path.exists(door_frame_tex_coords_save_path):
                door_frame_texture_map_path = door_mesh_texture_map_path
                door_frame_tex_coords_save_path = door_mesh_tex_coords_save_path
            with open(door_frame_tex_coords_save_path, 'rb') as f:
                door_frame_tex_coords = pickle.load(f)
            door_frame_texture_info = {'vts': door_frame_tex_coords['vts'], 'fts': door_frame_tex_coords['fts'], 'texture_map_path': door_frame_texture_map_path}
            mesh_info_dict[f'{door.id}_frame'] = {'mesh': door_frame_mesh, 'static': True, 'texture': door_frame_texture_info}
    return mesh_info_dict

def create_floor_mesh_tex_coords(floor_mesh: trimesh.Trimesh) -> dict:
    """
    Generate texture coordinates for a floor mesh using xatlas.
    
    Args:
        floor_mesh: Trimesh object representing the floor
        
    Returns:
        Dictionary with 'vts' (texture coordinates) and 'fts' (face texture indices)
    """
    try:
        atlas = xatlas.Atlas()
        vertices = floor_mesh.vertices.astype(np.float32)
        faces = floor_mesh.faces.astype(np.uint32)
        atlas.add_mesh(vertices, faces)
        atlas.generate()
        (vmapping, indices, uvs) = atlas.get_mesh(0)
        vts = uvs
        fts = indices
        return {'vts': vts, 'fts': fts}
    except Exception as e:
        print(f'Error generating texture coordinates for floor mesh: {e}')
        return _simple_planar_uv_mapping(floor_mesh)

def create_wall_mesh_tex_coords(wall_mesh: trimesh.Trimesh) -> dict:
    """
    Generate texture coordinates for a wall mesh using xatlas.
    
    Args:
        wall_mesh: Trimesh object representing the wall
        
    Returns:
        Dictionary with 'vts' (texture coordinates) and 'fts' (face texture indices)
    """
    try:
        atlas = xatlas.Atlas()
        vertices = wall_mesh.vertices.astype(np.float32)
        faces = wall_mesh.faces.astype(np.uint32)
        atlas.add_mesh(vertices, faces)
        atlas.generate()
        (vmapping, indices, uvs) = atlas.get_mesh(0)
        vts = uvs
        fts = indices
        return {'vts': vts, 'fts': fts}
    except Exception as e:
        print(f'Error generating texture coordinates for wall mesh: {e}')
        return _simple_planar_uv_mapping(wall_mesh)

def _simple_planar_uv_mapping(mesh: trimesh.Trimesh) -> dict:
    """
    Fallback function for simple planar UV mapping when xatlas fails.
    
    Args:
        mesh: Trimesh object
        
    Returns:
        Dictionary with 'vts' (texture coordinates) and 'fts' (face texture indices)
    """
    bounds = mesh.bounds
    min_coords = bounds[0]
    max_coords = bounds[1]
    vertices = mesh.vertices
    u = (vertices[:, 0] - min_coords[0]) / (max_coords[0] - min_coords[0])
    v = (vertices[:, 1] - min_coords[1]) / (max_coords[1] - min_coords[1])
    u = np.clip(u, 0, 1)
    v = np.clip(v, 0, 1)
    vts = np.column_stack([u, v])
    fts = mesh.faces
    return {'vts': vts, 'fts': fts}

def export_single_room_layout_to_mesh_dict_list(layout: FloorPlan, room_id: str):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
    """
    import os
    import pickle
    layout_save_dir = os.environ.get('SAGE_RESULTS_DIR', os.path.join(SERVER_ROOT_DIR, 'results/'))

    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}.ply'
        watertight_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_wt.ply'
        if source == 'asset_library' and os.path.exists(watertight_path):
            return trimesh.load(watertight_path)
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None

    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl'
        texture_map_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png'
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, 'rb') as f:
                tex_coords = pickle.load(f)
            return {'vts': tex_coords['vts'], 'fts': tex_coords['fts'], 'texture_map_path': texture_map_path}
        else:
            return None
    mesh_info_dict = {}
    processed_doors = set()
    processed_windows = set()
    room = next((room for room in layout.rooms if room.id == room_id))
    floor_mesh = create_floor_mesh(room)
    floor_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.floor_material}.png'
    floor_mesh_tex_coords = create_floor_mesh_tex_coords(floor_mesh)
    mesh_info_dict[f'floor_{room.id}'] = {'mesh': floor_mesh, 'static': True, 'texture': {'vts': floor_mesh_tex_coords['vts'], 'fts': floor_mesh_tex_coords['fts'], 'texture_map_path': floor_mesh_texture_map_path}}
    (room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids) = create_room_meshes_with_openings(room, processed_doors, processed_windows)
    wall_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png'
    for (wall_id, wall_mesh) in zip(room_wall_ids, room_wall_meshes):
        wall_mesh_tex_coords = create_wall_mesh_tex_coords(wall_mesh)
        mesh_info_dict[f'{wall_id}'] = {'mesh': wall_mesh, 'static': True, 'texture': {'vts': wall_mesh_tex_coords['vts'], 'fts': wall_mesh_tex_coords['fts'], 'texture_map_path': wall_mesh_texture_map_path}}
    ceiling_mesh = create_ceiling_mesh(room)
    ceiling_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{room.walls[0].material}.png'
    ceiling_mesh_tex_coords = create_wall_mesh_tex_coords(ceiling_mesh)
    mesh_info_dict[f'floor_{room.id}_ceiling'] = {'mesh': ceiling_mesh, 'static': True, 'texture': {'vts': ceiling_mesh_tex_coords['vts'], 'fts': ceiling_mesh_tex_coords['fts'], 'texture_map_path': ceiling_mesh_texture_map_path}}
    for (window_id, window_mesh) in zip(room_window_ids, room_window_meshes):
        window_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{window_id}_texture.png'
        window_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{window_id}_tex_coords.pkl'
        with open(window_mesh_tex_coords_save_path, 'rb') as f:
            window_mesh_tex_coords = pickle.load(f)
        mesh_info_dict[f'{window_id}'] = {'mesh': window_mesh, 'static': True, 'texture': {'vts': window_mesh_tex_coords['vts'], 'fts': window_mesh_tex_coords['fts'], 'texture_map_path': window_mesh_texture_map_path}}
    for obj in room.objects:
        obj_mesh = get_object_mesh(obj.source, obj.source_id, layout.id)
        if obj_mesh is not None:
            transformed_mesh = apply_object_transform(obj_mesh, obj)
            texture_info = get_object_mesh_texture(obj.source, obj.source_id, layout.id)
            mesh_info_dict[obj.id] = {'mesh': transformed_mesh, 'static': False if obj.place_id != 'wall' else True, 'texture': texture_info, 'mass': getattr(obj, 'mass', 1.0)}
    door_center_list = []
    wall_map = {wall.id: wall for wall in room.walls}
    for door in room.doors:
        if door.opening:
            continue
        wall = wall_map.get(door.wall_id)
        assert wall is not None, f'Wall {door.wall_id} not found'
        start_point = wall.start_point
        end_point = wall.end_point
        position_on_wall = door.position_on_wall
        door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall
        door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall
        duplicate_door = False
        for (door_center_prev_x, door_center_prev_y) in door_center_list:
            if abs(door_center_x - door_center_prev_x) < 0.01 and abs(door_center_y - door_center_prev_y) < 0.01:
                duplicate_door = True
                break
        if duplicate_door:
            continue
        door_center_list.append((door_center_x, door_center_y))
        thickness_scale = 0.95
        size_scale = 0.95
        door_width_original = door.width
        door_thickness = wall.thickness * thickness_scale
        delta_s = 0.5 * (1 - size_scale) * door_width_original
        delta_r_min = max(0, ((0.5 * door_thickness) ** 2 - delta_s ** 2) / (2 * delta_s))
        delta_r = delta_r_min * 1.1
        door_size_offset_calculated = (size_scale - 1) * door_width_original
        door_mesh = create_door_mesh(wall, door, size_scale=1.0, thickness_scale=thickness_scale, door_size_offset=door_size_offset_calculated)
        door_frame_mesh = create_door_frame_mesh(wall, door, size_scale=1.0, thickness_scale=1.05, frame_width=0.05)
        start = np.array([wall.start_point.x, wall.start_point.y, 0])
        end = np.array([wall.end_point.x, wall.end_point.y, 0])
        wall_vector = end - start
        wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
        door_center_point = start + wall_vector * door.position_on_wall
        door_start_point = door_center_point - wall_vector_norm * door.width / 2
        door_rotate_axis_point_lower = door_start_point + wall_vector_norm * (delta_s + delta_r)
        door_rotate_axis_point_lower[2] = 0.0
        door_rotate_axis_point_upper = door_start_point + wall_vector_norm * (delta_s + delta_r)
        door_rotate_axis_point_upper[2] = door.height
        door_mesh_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_texture.png'
        door_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_tex_coords.pkl'
        with open(door_mesh_tex_coords_save_path, 'rb') as f:
            door_mesh_tex_coords = pickle.load(f)
        texture_info = {'vts': door_mesh_tex_coords['vts'], 'fts': door_mesh_tex_coords['fts'], 'texture_map_path': door_mesh_texture_map_path}
        mesh_info_dict[f'{door.id}'] = {'mesh': door_mesh, 'static': False, 'articulation': (door_rotate_axis_point_lower, door_rotate_axis_point_upper), 'texture': texture_info}
        door_frame_texture_map_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_texture.png'
        door_frame_tex_coords_save_path = f'{layout_save_dir}/{layout.id}/materials/{door.door_material}_frame_tex_coords.pkl'
        if not os.path.exists(door_frame_tex_coords_save_path):
            door_frame_texture_map_path = door_mesh_texture_map_path
            door_frame_tex_coords_save_path = door_mesh_tex_coords_save_path
        with open(door_frame_tex_coords_save_path, 'rb') as f:
            door_frame_tex_coords = pickle.load(f)
        door_frame_texture_info = {'vts': door_frame_tex_coords['vts'], 'fts': door_frame_tex_coords['fts'], 'texture_map_path': door_frame_texture_map_path}
        mesh_info_dict[f'{door.id}_frame'] = {'mesh': door_frame_mesh, 'static': True, 'texture': door_frame_texture_info}
    return mesh_info_dict

def export_single_room_layout_to_mesh_dict_list_from_room(room: Room, layout_id: str):
    """
    Export a FloorPlan object to a mesh file using trimesh.
    Creates gray boxes for walls/floors, red boxes for doors, blue boxes for windows,
    and includes actual object meshes with their transforms.
    Uses boolean operations to cut door/window openings in walls.
    
    Args:
        layout: FloorPlan object to export
    """
    import os
    import pickle
    layout_save_dir = os.environ.get('SAGE_RESULTS_DIR', os.path.join(SERVER_ROOT_DIR, 'results/'))

    def get_object_mesh(source, source_id, layout_id):
        object_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}.ply'
        watertight_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_wt.ply'
        if source == 'asset_library' and os.path.exists(watertight_path):
            return trimesh.load(watertight_path)
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None

    def get_object_mesh_texture(source, source_id, layout_id):
        tex_coords_save_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_tex_coords.pkl'
        texture_map_path = f'{layout_save_dir}/{layout_id}/{source}/{source_id}_texture.png'
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, 'rb') as f:
                tex_coords = pickle.load(f)
            return {'vts': tex_coords['vts'], 'fts': tex_coords['fts'], 'texture_map_path': texture_map_path}
        else:
            return None
    mesh_info_dict = {}
    processed_doors = set()
    processed_windows = set()
    floor_mesh = create_floor_mesh(room)
    floor_mesh_texture_map_path = f'{layout_save_dir}/{layout_id}/materials/{room.floor_material}.png'
    floor_mesh_tex_coords = create_floor_mesh_tex_coords(floor_mesh)
    mesh_info_dict[f'floor_{room.id}'] = {'mesh': floor_mesh, 'static': True, 'texture': {'vts': floor_mesh_tex_coords['vts'], 'fts': floor_mesh_tex_coords['fts'], 'texture_map_path': floor_mesh_texture_map_path}}
    (room_wall_meshes, room_door_meshes, room_window_meshes, room_wall_ids, room_door_ids, room_window_ids) = create_room_meshes_with_openings(room, processed_doors, processed_windows)
    wall_mesh_texture_map_path = f'{layout_save_dir}/{layout_id}/materials/{room.walls[0].material}.png'
    for (wall_id, wall_mesh) in zip(room_wall_ids, room_wall_meshes):
        wall_mesh_tex_coords = create_wall_mesh_tex_coords(wall_mesh)
        mesh_info_dict[f'{wall_id}'] = {'mesh': wall_mesh, 'static': True, 'texture': {'vts': wall_mesh_tex_coords['vts'], 'fts': wall_mesh_tex_coords['fts'], 'texture_map_path': wall_mesh_texture_map_path}}
    ceiling_mesh = create_ceiling_mesh(room)
    ceiling_mesh_texture_map_path = f'{layout_save_dir}/{layout_id}/materials/{room.walls[0].material}.png'
    ceiling_mesh_tex_coords = create_wall_mesh_tex_coords(ceiling_mesh)
    mesh_info_dict[f'floor_{room.id}_ceiling'] = {'mesh': ceiling_mesh, 'static': True, 'texture': {'vts': ceiling_mesh_tex_coords['vts'], 'fts': ceiling_mesh_tex_coords['fts'], 'texture_map_path': ceiling_mesh_texture_map_path}}
    for (window_id, window_mesh) in zip(room_window_ids, room_window_meshes):
        window_mesh_texture_map_path = f'{layout_save_dir}/{layout_id}/materials/{window_id}_texture.png'
        window_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout_id}/materials/{window_id}_tex_coords.pkl'
        with open(window_mesh_tex_coords_save_path, 'rb') as f:
            window_mesh_tex_coords = pickle.load(f)
        mesh_info_dict[f'{window_id}'] = {'mesh': window_mesh, 'static': True, 'texture': {'vts': window_mesh_tex_coords['vts'], 'fts': window_mesh_tex_coords['fts'], 'texture_map_path': window_mesh_texture_map_path}}
    for obj in room.objects:
        obj_mesh = get_object_mesh(obj.source, obj.source_id, layout_id)
        if obj_mesh is not None:
            transformed_mesh = apply_object_transform(obj_mesh, obj)
            texture_info = get_object_mesh_texture(obj.source, obj.source_id, layout_id)
            mesh_info_dict[obj.id] = {'mesh': transformed_mesh, 'static': False if obj.place_id != 'wall' else True, 'texture': texture_info, 'mass': getattr(obj, 'mass', 1.0)}
    door_center_list = []
    wall_map = {wall.id: wall for wall in room.walls}
    for door in room.doors:
        if door.opening:
            continue
        wall = wall_map.get(door.wall_id)
        assert wall is not None, f'Wall {door.wall_id} not found'
        start_point = wall.start_point
        end_point = wall.end_point
        position_on_wall = door.position_on_wall
        door_center_x = start_point.x + (end_point.x - start_point.x) * position_on_wall
        door_center_y = start_point.y + (end_point.y - start_point.y) * position_on_wall
        duplicate_door = False
        for (door_center_prev_x, door_center_prev_y) in door_center_list:
            if abs(door_center_x - door_center_prev_x) < 0.01 and abs(door_center_y - door_center_prev_y) < 0.01:
                duplicate_door = True
                break
        if duplicate_door:
            continue
        door_center_list.append((door_center_x, door_center_y))
        thickness_scale = 0.95
        size_scale = 0.95
        door_width_original = door.width
        door_thickness = wall.thickness * thickness_scale
        delta_s = 0.5 * (1 - size_scale) * door_width_original
        delta_r_min = max(0, ((0.5 * door_thickness) ** 2 - delta_s ** 2) / (2 * delta_s))
        delta_r = delta_r_min * 1.1
        door_size_offset_calculated = (size_scale - 1) * door_width_original
        door_mesh = create_door_mesh(wall, door, size_scale=1.0, thickness_scale=thickness_scale, door_size_offset=door_size_offset_calculated)
        door_frame_mesh = create_door_frame_mesh(wall, door, size_scale=1.0, thickness_scale=1.05, frame_width=0.05)
        start = np.array([wall.start_point.x, wall.start_point.y, 0])
        end = np.array([wall.end_point.x, wall.end_point.y, 0])
        wall_vector = end - start
        wall_vector_norm = wall_vector / np.linalg.norm(wall_vector)
        door_center_point = start + wall_vector * door.position_on_wall
        door_start_point = door_center_point - wall_vector_norm * door.width / 2
        door_rotate_axis_point_lower = door_start_point + wall_vector_norm * (delta_s + delta_r)
        door_rotate_axis_point_lower[2] = 0.0
        door_rotate_axis_point_upper = door_start_point + wall_vector_norm * (delta_s + delta_r)
        door_rotate_axis_point_upper[2] = door.height
        door_mesh_texture_map_path = f'{layout_save_dir}/{layout_id}/materials/{door.door_material}_texture.png'
        door_mesh_tex_coords_save_path = f'{layout_save_dir}/{layout_id}/materials/{door.door_material}_tex_coords.pkl'
        with open(door_mesh_tex_coords_save_path, 'rb') as f:
            door_mesh_tex_coords = pickle.load(f)
        texture_info = {'vts': door_mesh_tex_coords['vts'], 'fts': door_mesh_tex_coords['fts'], 'texture_map_path': door_mesh_texture_map_path}
        mesh_info_dict[f'{door.id}'] = {'mesh': door_mesh, 'static': False, 'articulation': (door_rotate_axis_point_lower, door_rotate_axis_point_upper), 'texture': texture_info}
        door_frame_texture_map_path = f'{layout_save_dir}/{layout_id}/materials/{door.door_material}_frame_texture.png'
        door_frame_tex_coords_save_path = f'{layout_save_dir}/{layout_id}/materials/{door.door_material}_frame_tex_coords.pkl'
        if not os.path.exists(door_frame_tex_coords_save_path):
            door_frame_texture_map_path = door_mesh_texture_map_path
            door_frame_tex_coords_save_path = door_mesh_tex_coords_save_path
        with open(door_frame_tex_coords_save_path, 'rb') as f:
            door_frame_tex_coords = pickle.load(f)
        door_frame_texture_info = {'vts': door_frame_tex_coords['vts'], 'fts': door_frame_tex_coords['fts'], 'texture_map_path': door_frame_texture_map_path}
        mesh_info_dict[f'{door.id}_frame'] = {'mesh': door_frame_mesh, 'static': True, 'texture': door_frame_texture_info}
    return mesh_info_dict

def get_single_object_mesh_info_dict(scene_save_dir, source, source_id):
    import os
    import pickle

    def get_object_mesh(source, source_id):
        object_save_path = f'{scene_save_dir}/{source}/{source_id}.ply'
        watertight_path = f'{scene_save_dir}/{source}/{source_id}_wt.ply'
        if source == 'asset_library' and os.path.exists(watertight_path):
            return trimesh.load(watertight_path)
        if os.path.exists(object_save_path):
            return trimesh.load(object_save_path)
        else:
            return None

    def get_object_mesh_texture(source, source_id):
        tex_coords_save_path = f'{scene_save_dir}/{source}/{source_id}_tex_coords.pkl'
        texture_map_path = f'{scene_save_dir}/{source}/{source_id}_texture.png'
        if os.path.exists(tex_coords_save_path):
            with open(tex_coords_save_path, 'rb') as f:
                tex_coords = pickle.load(f)
            return {'vts': tex_coords['vts'], 'fts': tex_coords['fts'], 'texture_map_path': texture_map_path}
        else:
            return None
    obj_mesh = get_object_mesh(source, source_id)
    if obj_mesh is not None:
        texture_info = get_object_mesh_texture(source, source_id)
        if texture_info is None:
            return None
        mesh_info_dict = {'mesh': obj_mesh, 'static': False, 'texture': texture_info}
        return mesh_info_dict
    else:
        return None

def apply_object_transform_direct(mesh: trimesh.Trimesh, position: Dict[str, float], rotation: Dict[str, float], degrees: bool=True) -> trimesh.Trimesh:
    """
    Apply position and rotation transforms to an object mesh.
    
    Args:
        mesh: The original mesh (untransformed)
        obj: Object containing position and rotation information
        
    Returns:
        Transformed mesh positioned and rotated according to object properties
    """
    transformed_mesh = mesh.copy()
    if degrees:
        rx_rad = np.radians(rotation['x'])
        ry_rad = np.radians(rotation['y'])
        rz_rad = np.radians(rotation['z'])
    else:
        rx_rad = rotation['x']
        ry_rad = rotation['y']
        rz_rad = rotation['z']
    rotation_x = trimesh.transformations.rotation_matrix(rx_rad, [1, 0, 0])
    rotation_y = trimesh.transformations.rotation_matrix(ry_rad, [0, 1, 0])
    rotation_z = trimesh.transformations.rotation_matrix(rz_rad, [0, 0, 1])
    combined_rotation = rotation_z @ rotation_y @ rotation_x
    translation = trimesh.transformations.translation_matrix([position['x'], position['y'], position['z']])
    final_transform = translation @ combined_rotation
    transformed_mesh.apply_transform(final_transform)
    return transformed_mesh
