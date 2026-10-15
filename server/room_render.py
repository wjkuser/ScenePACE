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
from models import FloorPlan, Room, Wall
from nvdiffrast_rendering.mesh import get_mesh_dict_list_from_single_room, build_mesh_dict
from nvdiffrast_rendering.render import rasterize_mesh_dict_list_with_uv_efficient, rasterize_mesh_dict_list_with_uv_efficient_uv_diff
from nvdiffrast_rendering.camera import get_camera_perspective_projection_matrix, get_intrinsic, build_camera_matrix, get_mvp_matrix, get_camera_orthogonal_projection_matrix, get_full_view_camera_sampling
from nvdiffrast_rendering.context import get_glctx
from tex_utils import export_single_room_layout_to_mesh_dict_list
from PIL import Image
import os
import numpy as np
import torch

def get_camera_view_direction(camera_pos, lookat_pos):
    """Calculate normalized view direction from camera to lookat position."""
    direction = lookat_pos - camera_pos
    direction = direction / np.linalg.norm(direction)
    return direction

def get_wall_normal(wall: Wall, room: Room):
    """Calculate the outward normal vector of a wall."""
    start = np.array([wall.start_point.x, wall.start_point.y, wall.start_point.z])
    end = np.array([wall.end_point.x, wall.end_point.y, wall.end_point.z])
    wall_dir = end - start
    wall_dir = wall_dir / np.linalg.norm(wall_dir)
    up = np.array([0, 0, 1])
    normal = np.cross(wall_dir, up)
    room_center = np.array([room.position.x + room.dimensions.width / 2, room.position.y + room.dimensions.length / 2, room.position.z])
    wall_center = (start + end) / 2
    to_center = room_center - wall_center
    if np.dot(normal[:2], to_center[:2]) > 0:
        normal = -normal
    return normal

def should_exclude_wall(wall: Wall, room: Room, camera_pos, lookat_pos):
    """Determine if a wall should be excluded based on camera view direction."""
    wall_normal = get_wall_normal(wall, room)
    view_dir = get_camera_view_direction(camera_pos, lookat_pos)
    dot_product = np.dot(view_dir[:2], wall_normal[:2])
    return dot_product < -0.3

def filter_mesh_info_dict_by_walls(mesh_info_dict, walls_to_exclude):
    """Remove specific walls from the mesh_info_dict."""
    filtered_dict = {}
    excluded_wall_ids = {wall.id for wall in walls_to_exclude}
    for (mesh_id, mesh_info) in mesh_info_dict.items():
        is_excluded_wall = any((wall_id in mesh_id for wall_id in excluded_wall_ids))
        if not is_excluded_wall:
            filtered_dict[mesh_id] = mesh_info
    return filtered_dict

def _build_full_mesh_dicts(layout: FloorPlan, room: Room):
    """Load room meshes and textures once for all camera views."""
    mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, room.id)
    tagged_mesh_dicts = []
    for (mesh_id, mesh_info) in mesh_info_dict.items():
        with Image.open(mesh_info['texture']['texture_map_path']) as texture_image:
            texture_map = np.asarray(texture_image).copy() / 255.0
        tagged_mesh_dicts.append((mesh_id, build_mesh_dict(mesh_info['mesh'].vertices, mesh_info['mesh'].faces, mesh_info['texture']['vts'], mesh_info['texture']['fts'], texture_map)))
    return tagged_mesh_dicts

def _select_excluding_walls(tagged_mesh_dicts, walls_to_exclude):
    excluded_wall_ids = {wall.id for wall in walls_to_exclude}
    return [mesh_dict for (mesh_id, mesh_dict) in tagged_mesh_dicts if not any((wall_id in mesh_id for wall_id in excluded_wall_ids))]

def get_filtered_mesh_dict_list(layout: FloorPlan, room: Room, camera_pos, lookat_pos):
    """Get mesh_dict_list with front walls removed based on camera angle."""
    mesh_info_dict = export_single_room_layout_to_mesh_dict_list(layout, room.id)
    walls_to_exclude = [wall for wall in room.walls if should_exclude_wall(wall, room, camera_pos, lookat_pos)]
    filtered_mesh_info_dict = filter_mesh_info_dict_by_walls(mesh_info_dict, walls_to_exclude)
    mesh_dict_list = []
    for mesh_info in filtered_mesh_info_dict.values():
        vertices = mesh_info['mesh'].vertices
        faces = mesh_info['mesh'].faces
        vts = mesh_info['texture']['vts']
        fts = mesh_info['texture']['fts']
        texture_map_pil = Image.open(mesh_info['texture']['texture_map_path'])
        texture_map = np.array(texture_map_pil) / 255.0
        mesh_dict = build_mesh_dict(vertices, faces, vts, fts, texture_map)
        mesh_dict_list.append(mesh_dict)
    return mesh_dict_list

def render_room_four_top_view(layout: FloorPlan, room_id: str, resolution=768):
    if os.environ.get('SAGE_RENDER_BACKEND') == 'isaac':
        from isaac_room_render import render_isaac_views
        views = render_isaac_views(layout, room_id, resolution, 'edges')
        return views
    mesh_dict_list = get_mesh_dict_list_from_single_room(layout, room_id)
    intrinsic = get_intrinsic(80, resolution, resolution)
    projection_matrix = get_camera_perspective_projection_matrix(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], resolution, resolution, 0.001, 100.0)
    glctx = get_glctx()
    all_rooms = layout.rooms
    room = next((room for room in all_rooms if room.id == room_id))
    room_position = np.array([room.position.x, room.position.y, room.position.z])
    room_height = room.dimensions.height * 1.5
    room_width = room.dimensions.width
    room_length = room.dimensions.length
    room_top_corners = [room_position + np.array([0, 0, room_height]), room_position + np.array([room_width, 0, room_height]), room_position + np.array([0, room_length, room_height]), room_position + np.array([room_width, room_length, room_height])]
    room_lookat_corners = [room_position + np.array([room_width * 0.5, room_length * 0.5, 0]), room_position + np.array([room_width * 0.5, room_length * 0.5, 0]), room_position + np.array([room_width * 0.5, room_length * 0.5, 0]), room_position + np.array([room_width * 0.5, room_length * 0.5, 0])]
    all_rgb = []
    for (top_corner, lookat_corner) in zip(room_top_corners, room_lookat_corners):
        camera_matrix = build_camera_matrix(torch.from_numpy(top_corner).float(), torch.from_numpy(lookat_corner).float(), torch.from_numpy(np.array([0, 0, 1])).float())
        mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)
        (valid, instance_id, rgb) = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, mvp_matrix, glctx, (resolution, resolution))
        rgb = rgb.cpu().numpy().clip(0, 1)
        all_rgb.append(rgb)
    return all_rgb

def render_room_four_edges_view(layout: FloorPlan, room_id: str, resolution=1024):
    if os.environ.get('SAGE_RENDER_BACKEND') == 'isaac':
        from isaac_room_render import render_isaac_views
        views = render_isaac_views(layout, room_id, resolution, 'edges')
        return views
    fov = 35.0
    aspect_ratio = 16 / 9
    res_width = resolution
    res_height = int(res_width / aspect_ratio)
    all_rooms = layout.rooms
    room = next((room for room in all_rooms if room.id == room_id))
    room_position = np.array([room.position.x, room.position.y, room.position.z])
    room_height = room.dimensions.height
    room_width = room.dimensions.width
    room_length = room.dimensions.length
    room_center = np.array([room_position[0] + room_width / 2, room_position[1] + room_length / 2, room_height / 2]).reshape(-1).tolist()
    room_scales = np.array([room_width, room_length, room_height]).reshape(-1).tolist()
    intrinsic = get_intrinsic(fov, res_height, res_width)
    projection_matrix = get_camera_perspective_projection_matrix(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], res_height, res_width, 0.01, np.array(room_scales).max() * 5.0)
    glctx = get_glctx()
    horizontal_angle_list = [0, 90, 180, 270]
    vertical_angle = 45.0
    tagged_mesh_dicts = _build_full_mesh_dicts(layout, room)
    all_rgb = []
    for horizontal_angle in horizontal_angle_list:
        (camera_pos, camera_lookat, _) = get_full_view_camera_sampling(room_center, room_scales, (res_height, res_width), horizontal_angle, vertical_angle, fov, mode='adjustable')
        camera_matrix = build_camera_matrix(torch.from_numpy(camera_pos).float(), torch.from_numpy(camera_lookat).float(), torch.from_numpy(np.array([0, 0, 1])).float())
        mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)
        walls_to_exclude = [wall for wall in room.walls if should_exclude_wall(wall, room, camera_pos, camera_lookat)]
        mesh_dict_list = _select_excluding_walls(tagged_mesh_dicts, walls_to_exclude)
        (valid, instance_id, rgb) = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, mvp_matrix, glctx, (res_height, res_width))
        rgb = rgb.cpu().numpy().clip(0, 1)
        all_rgb.append(rgb)
    return all_rgb

def render_room_top_orthogonal_view(layout: FloorPlan, room_id: str, resolution=1024):
    if os.environ.get('SAGE_RENDER_BACKEND') == 'isaac':
        from isaac_room_render import render_isaac_views
        views = render_isaac_views(layout, room_id, resolution, 'top')
        return views[0]
    mesh_dict_list = get_mesh_dict_list_from_single_room(layout, room_id)
    glctx = get_glctx()
    all_rooms = layout.rooms
    room = next((room for room in all_rooms if room.id == room_id))
    room_position = np.array([room.position.x, room.position.y, room.position.z])
    room_height = room.dimensions.height * 1.5
    room_width = room.dimensions.width
    room_length = room.dimensions.length
    camera_position = np.array([room_position[0] + room_width / 2, room_position[1] + room_length / 2, room_height])
    camera_lookat = np.array([room_position[0] + room_width / 2, room_position[1] + room_length / 2, 0])
    camera_up = np.array([0, 1, 0])
    camera_matrix = build_camera_matrix(torch.from_numpy(camera_position).float(), torch.from_numpy(camera_lookat).float(), torch.from_numpy(camera_up).float())
    projection_matrix = get_camera_orthogonal_projection_matrix(0.001, 100.0, room_width * 0.5, room_length * 0.5)
    mvp_matrix = get_mvp_matrix(camera_matrix, projection_matrix)
    W = int(room_width * 1024)
    H = int(room_length * 1024)
    if max(H, W) > resolution:
        scale = resolution / max(H, W)
        H = int(H * scale)
        W = int(W * scale)
    (valid, instance_id, rgb) = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, mvp_matrix, glctx, (H, W))
    rgb = rgb.cpu().numpy().clip(0, 1)
    return rgb

def render_room_five_views(layout: FloorPlan, room_id: str, edges_resolution: int=1920, top_resolution: int=1024):
    """Render four edge views and one top view from one shared mesh build."""
    if os.environ.get('SAGE_RENDER_BACKEND') == 'isaac':
        from isaac_room_render import render_isaac_views
        views = render_isaac_views(layout, room_id, edges_resolution, 'five')
        return (views[:4], views[4])
    room = next((room for room in layout.rooms if room.id == room_id))
    tagged_mesh_dicts = _build_full_mesh_dicts(layout, room)
    full_mesh_dict_list = [mesh_dict for (_, mesh_dict) in tagged_mesh_dicts]
    glctx = get_glctx()
    room_position = np.array([room.position.x, room.position.y, room.position.z])
    room_scales = np.array([room.dimensions.width, room.dimensions.length, room.dimensions.height])
    room_center = (room_position + room_scales / 2.0).tolist()
    fov = 35.0
    res_width = edges_resolution
    res_height = int(res_width / (16 / 9))
    intrinsic = get_intrinsic(fov, res_height, res_width)
    projection_matrix = get_camera_perspective_projection_matrix(intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3], res_height, res_width, 0.01, room_scales.max() * 5.0)
    edge_views = []
    for horizontal_angle in (0, 90, 180, 270):
        (camera_pos, camera_lookat, _) = get_full_view_camera_sampling(room_center, room_scales.tolist(), (res_height, res_width), horizontal_angle, 45.0, fov, mode='adjustable')
        camera_matrix = build_camera_matrix(torch.from_numpy(camera_pos).float(), torch.from_numpy(camera_lookat).float(), torch.from_numpy(np.array([0, 0, 1])).float())
        walls_to_exclude = [wall for wall in room.walls if should_exclude_wall(wall, room, camera_pos, camera_lookat)]
        mesh_dict_list = _select_excluding_walls(tagged_mesh_dicts, walls_to_exclude)
        (_, _, rgb) = rasterize_mesh_dict_list_with_uv_efficient(mesh_dict_list, get_mvp_matrix(camera_matrix, projection_matrix), glctx, (res_height, res_width))
        edge_views.append(rgb.cpu().numpy().clip(0, 1))
    camera_position = np.array([room_position[0] + room_scales[0] / 2, room_position[1] + room_scales[1] / 2, room_scales[2] * 1.5])
    camera_lookat = np.array([room_position[0] + room_scales[0] / 2, room_position[1] + room_scales[1] / 2, 0])
    camera_matrix = build_camera_matrix(torch.from_numpy(camera_position).float(), torch.from_numpy(camera_lookat).float(), torch.from_numpy(np.array([0, 1, 0])).float())
    projection_matrix = get_camera_orthogonal_projection_matrix(0.001, 100.0, room_scales[0] * 0.5, room_scales[1] * 0.5)
    width = int(room_scales[0] * 1024)
    height = int(room_scales[1] * 1024)
    if max(height, width) > top_resolution:
        scale = top_resolution / max(height, width)
        (height, width) = (int(height * scale), int(width * scale))
    (_, _, top_view) = rasterize_mesh_dict_list_with_uv_efficient(full_mesh_dict_list, get_mvp_matrix(camera_matrix, projection_matrix), glctx, (height, width))
    return (edge_views, top_view.cpu().numpy().clip(0, 1))
