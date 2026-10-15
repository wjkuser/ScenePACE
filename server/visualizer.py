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
"""
Room Layout Visualizer

This script provides visualization capabilities for the FloorPlan data structure.
It can create both 2D floor plan views and 3D room visualizations.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import asdict
import argparse
import sys
from pathlib import Path
from utils import export_layout_to_json
import os
import datetime
from shapely.geometry import Polygon, Point
sys.path.append(str(Path(__file__).parent))
from models import FloorPlan, Room, Wall, Door, Window, Point3D, Dimensions, Object

def dict_to_dataclass(data_dict: Dict[str, Any]) -> FloorPlan:
    """Convert dictionary representation back to dataclass objects."""

    def dict_to_point3d(point_dict):
        return Point3D(**point_dict)

    def dict_to_dimensions(dim_dict):
        return Dimensions(**dim_dict)

    def dict_to_wall(wall_dict):
        wall_dict['start_point'] = dict_to_point3d(wall_dict['start_point'])
        wall_dict['end_point'] = dict_to_point3d(wall_dict['end_point'])
        return Wall(**wall_dict)

    def dict_to_door(door_dict):
        return Door(**door_dict)

    def dict_to_window(window_dict):
        return Window(**window_dict)

    def dict_to_room(room_dict):
        room_dict['position'] = dict_to_point3d(room_dict['position'])
        room_dict['dimensions'] = dict_to_dimensions(room_dict['dimensions'])
        room_dict['walls'] = [dict_to_wall(wall) for wall in room_dict['walls']]
        room_dict['doors'] = [dict_to_door(door) for door in room_dict['doors']]
        room_dict['windows'] = [dict_to_window(window) for window in room_dict['windows']]
        return Room(**room_dict)
    data_dict['rooms'] = [dict_to_room(room) for room in data_dict['rooms']]
    return FloorPlan(**data_dict)

class LayoutVisualizer:
    """Visualizes room layouts in 2D and 3D."""

    def __init__(self, floor_plan: FloorPlan):
        self.floor_plan = floor_plan
        self.colors = {'living room': '#FFB6C1', 'bedroom': '#ADD8E6', 'kitchen': '#98FB98', 'bathroom': '#F0E68C', 'dining room': '#DDA0DD', 'office': '#F4A460', 'closet': '#D3D3D3', 'hallway': '#FFEFD5', 'garage': '#C0C0C0', 'default': '#E6E6FA'}

    def get_room_color(self, room_type: str) -> str:
        """Get color for room type."""
        return self.colors.get(room_type.lower(), self.colors['default'])

    def calculate_layout_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate the overall bounds of the layout."""
        min_x = min_y = float('inf')
        max_x = max_y = float('-inf')
        for room in self.floor_plan.rooms:
            room_min_x = room.position.x
            room_min_y = room.position.y
            room_max_x = room.position.x + room.dimensions.width
            room_max_y = room.position.y + room.dimensions.length
            min_x = min(min_x, room_min_x)
            min_y = min(min_y, room_min_y)
            max_x = max(max_x, room_max_x)
            max_y = max(max_y, room_max_y)
        return (min_x, min_y, max_x, max_y)

    def visualize_2d_floor_plan(self, save_path: str=None, show: bool=True):
        """Create a 2D floor plan visualization."""
        (fig, ax) = plt.subplots(1, 1, figsize=(12, 10))
        (min_x, min_y, max_x, max_y) = self.calculate_layout_bounds()
        padding = 1.0
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding
        for room in self.floor_plan.rooms:
            self._draw_room_2d(ax, room)
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_aspect('equal')
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.grid(True, alpha=0.3)
        legend_elements = []
        used_types = set()
        for room in self.floor_plan.rooms:
            if room.room_type not in used_types:
                color = self.get_room_color(room.room_type)
                legend_elements.append(patches.Patch(color=color, label=room.room_type.title()))
                used_types.add(room.room_type)
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'2D floor plan saved to: {save_path}')
        if show:
            plt.show()
        return (fig, ax)

    def _draw_room_2d(self, ax, room: Room):
        """Draw a single room in 2D."""
        color = self.get_room_color(room.room_type)
        room_rect = Rectangle((room.position.x, room.position.y), room.dimensions.width, room.dimensions.length, facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(room_rect)
        center_x = room.position.x + room.dimensions.width / 2
        center_y = room.position.y + room.dimensions.length / 2
        ax.text(center_x, center_y, room.room_type.title(), ha='center', va='center', fontsize=10, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        for wall in room.walls:
            self._draw_wall_2d(ax, wall)
        for door in room.doors:
            self._draw_door_2d(ax, door, room.walls)
        for window in room.windows:
            self._draw_window_2d(ax, window, room.walls)
        for object in room.objects:
            self._draw_object_2d(ax, object)

    def _draw_wall_2d(self, ax, wall: Wall):
        """Draw a wall in 2D."""
        ax.plot([wall.start_point.x, wall.end_point.x], [wall.start_point.y, wall.end_point.y], 'k-', linewidth=4, alpha=0.8)

    def _draw_door_2d(self, ax, door: Door, walls: List[Wall]):
        """Draw a door in 2D."""
        wall = next((w for w in walls if w.id == door.wall_id), None)
        if not wall:
            return
        wall_vector = np.array([wall.end_point.x - wall.start_point.x, wall.end_point.y - wall.start_point.y])
        wall_length = np.linalg.norm(wall_vector)
        wall_unit = wall_vector / wall_length
        door_center = np.array([wall.start_point.x, wall.start_point.y]) + door.position_on_wall * wall_vector
        door_half_width = door.width / 2 * wall_unit
        door_start = door_center - door_half_width
        door_end = door_center + door_half_width
        ax.plot([door_start[0], door_end[0]], [door_start[1], door_end[1]], 'white', linewidth=6)
        ax.plot([door_start[0], door_end[0]], [door_start[1], door_end[1]], 'brown', linewidth=3)
        door_label = 'O' if door.opening else 'D'
        ax.text(door_center[0], door_center[1], door_label, ha='center', va='center', fontsize=8, color='brown', fontweight='bold')

    def _draw_window_2d(self, ax, window: Window, walls: List[Wall]):
        """Draw a window in 2D."""
        wall = next((w for w in walls if w.id == window.wall_id), None)
        if not wall:
            return
        wall_vector = np.array([wall.end_point.x - wall.start_point.x, wall.end_point.y - wall.start_point.y])
        wall_length = np.linalg.norm(wall_vector)
        wall_unit = wall_vector / wall_length
        window_center = np.array([wall.start_point.x, wall.start_point.y]) + window.position_on_wall * wall_vector
        window_half_width = window.width / 2 * wall_unit
        window_start = window_center - window_half_width
        window_end = window_center + window_half_width
        ax.plot([window_start[0], window_end[0]], [window_start[1], window_end[1]], 'lightblue', linewidth=4)
        ax.text(window_center[0], window_center[1], 'W', ha='center', va='center', fontsize=8, color='blue', fontweight='bold')

    def _draw_object_2d(self, ax, object: Object):
        """Draw an object in 2D."""
        rotation = object.rotation
        position = object.position
        dimensions = object.dimensions
        if rotation.z == 0 or rotation.z == 180:
            object_width = dimensions.width
            object_length = dimensions.length
        elif rotation.z == 90 or rotation.z == 270:
            object_width = dimensions.length
            object_length = dimensions.width
        else:
            raise ValueError(f'Invalid rotation: {rotation.z}')
        object_rect = Rectangle((position.x - object_width / 2, position.y - object_length / 2), object_width, object_length, facecolor='gray', edgecolor='black', linewidth=2, alpha=0.7)
        ax.add_patch(object_rect)
        ax.text(position.x, position.y, object.type, ha='center', va='center', fontsize=8, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    def visualize_3d_wireframe(self, save_path: str=None, show: bool=True):
        """Create a 3D wireframe visualization."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        for room in self.floor_plan.rooms:
            self._draw_room_3d(ax, room)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(f'3D View: {self.floor_plan.building_style}')
        (min_x, min_y, max_x, max_y) = self.calculate_layout_bounds()
        max_z = max((room.position.z + room.dimensions.height for room in self.floor_plan.rooms))
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)
        ax.set_zlim(0, max_z)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'3D wireframe saved to: {save_path}')
        if show:
            plt.show()
        return (fig, ax)

    def _draw_room_3d(self, ax, room: Room):
        """Draw a room in 3D wireframe."""
        (x, y, z) = (room.position.x, room.position.y, room.position.z)
        (w, l, h) = (room.dimensions.width, room.dimensions.length, room.dimensions.height)
        vertices = [[x, y, z], [x + w, y, z], [x + w, y + l, z], [x, y + l, z], [x, y, z + h], [x + w, y, z + h], [x + w, y + l, z + h], [x, y + l, z + h]]
        faces = [[vertices[j] for j in [0, 1, 2, 3]], [vertices[j] for j in [4, 5, 6, 7]], [vertices[j] for j in [0, 1, 5, 4]], [vertices[j] for j in [2, 3, 7, 6]], [vertices[j] for j in [1, 2, 6, 5]], [vertices[j] for j in [0, 3, 7, 4]]]
        room_color = self.get_room_color(room.room_type)
        poly3d = [[vertices[j] for j in [0, 1, 2, 3]]]
        ax.add_collection3d(Poly3DCollection(poly3d, alpha=0.3, facecolor=room_color, edgecolor='black'))
        edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]
        for edge in edges:
            points = [vertices[edge[0]], vertices[edge[1]]]
            ax.plot3D(*zip(*points), 'k-', alpha=0.6)
        center_x = x + w / 2
        center_y = y + l / 2
        ax.text(center_x, center_y, z + h / 2, room.room_type.title(), fontsize=10, ha='center')

    def generate_summary_report(self) -> str:
        """Generate a text summary of the floor plan."""
        report = f'\nFLOOR PLAN SUMMARY\n==================\n\nLayout ID: {self.floor_plan.id}\nBuilding Style: {self.floor_plan.building_style}\nTotal Area: {self.floor_plan.total_area:.2f} m²\nDescription: {self.floor_plan.description}\n\nROOMS ({len(self.floor_plan.rooms)} total):\n'
        for (i, room) in enumerate(self.floor_plan.rooms, 1):
            area = room.dimensions.width * room.dimensions.length
            report += f'\n{i}. {room.room_type.title()} (ID: {room.id})\n   - Dimensions: {room.dimensions.width:.1f}m × {room.dimensions.length:.1f}m × {room.dimensions.height:.1f}m\n   - Area: {area:.2f} m²\n   - Position: ({room.position.x:.1f}, {room.position.y:.1f}, {room.position.z:.1f})\n   - Doors: {len(room.doors)}\n   - Windows: {len(room.windows)}\n   - Floor Material: {room.floor_material}\n'
        return report

class RoomVisualizer:
    """Visualizes a room in 2D."""

    def __init__(self, room: Room, layout: FloorPlan=None):
        self.room = room
        self.layout = layout

    def visualize_2d(self, save_path: str=None, show: bool=True):
        """
        Create a 2D visualization of the room layout with objects, doors, and windows.
        
        Args:
            save_path: Optional path to save the visualization
            show: Whether to display the plot
            
        Returns:
            Path to saved visualization if save_path provided, otherwise None
        """
        max_wall_thickness_cm = 5
        if self.room.walls:
            max_wall_thickness_cm = max((wall.thickness * 0.5 * 100 for wall in self.room.walls))
        inner_width_cm = self.room.dimensions.width * 100 - 2 * max_wall_thickness_cm
        inner_length_cm = self.room.dimensions.length * 100 - 2 * max_wall_thickness_cm
        room_vertices = [(max_wall_thickness_cm, max_wall_thickness_cm), (max_wall_thickness_cm, max_wall_thickness_cm + inner_length_cm), (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm + inner_length_cm), (max_wall_thickness_cm + inner_width_cm, max_wall_thickness_cm)]
        room_poly = Polygon(room_vertices)
        initial_state = self._get_room_elements()
        fontsize = 6
        plt.rcParams['font.size'] = fontsize
        (fig, ax) = plt.subplots(figsize=(12, 10))
        (x, y) = room_poly.exterior.xy
        ax.plot(x, y, '-', label='Room Boundary', color='black', linewidth=3)
        ax.fill(x, y, color='lightgray', alpha=0.2)
        colors = ['brown', 'cyan', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
        color_idx = 0
        arrows_to_draw = []
        for (object_id, (center, rotation, vertices, _)) in initial_state.items():
            (center_x, center_y) = center
            obj_poly = Polygon(vertices)
            (x_coords, y_coords) = obj_poly.exterior.xy
            if object_id.startswith('door-'):
                ax.plot(x_coords, y_coords, '-', linewidth=3, color='brown')
                ax.fill(x_coords, y_coords, color='brown', alpha=0.5)
                ax.text(center_x, center_y, object_id, fontsize=fontsize, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            elif object_id.startswith('window-'):
                ax.plot(x_coords, y_coords, '-', linewidth=2, color='cyan')
                ax.fill(x_coords, y_coords, color='cyan', alpha=0.5)
                ax.text(center_x, center_y, object_id, fontsize=fontsize, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
            else:
                current_color = colors[color_idx % len(colors)]
                ax.plot(x_coords, y_coords, '-', linewidth=2, color=current_color)
                ax.fill(x_coords, y_coords, color=current_color, alpha=0.4)
                label_text = f"{object_id.replace('existing-', '')}"
                label_text = label_text[:13] + '\n' + label_text[13:]
                label_text = label_text[:-8] + '\n' + label_text[-8:]
                ax.text(center_x, center_y, label_text, fontsize=fontsize, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.9))
                arrows_to_draw.append((center_x, center_y, rotation, current_color))
                color_idx += 1
        for (center_x, center_y, rotation, current_color) in arrows_to_draw:
            arrow_length = 20
            head_width = 6
            head_length = 4
            arrow_face_color = 'black'
            arrow_edge_color = 'black'
            shadow_offset = 1
            if rotation == 0:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, 0, arrow_length, head_width=head_width + 2, head_length=head_length + 2, fc=arrow_edge_color, ec=arrow_edge_color, alpha=0.8, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, 0, arrow_length, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=1.0, linewidth=2, zorder=11)
            elif rotation == 90:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, -arrow_length, 0, head_width=head_width + 2, head_length=head_length + 2, fc=arrow_edge_color, ec=arrow_edge_color, alpha=0.8, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, -arrow_length, 0, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=1.0, linewidth=2, zorder=11)
            elif rotation == 180:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, 0, -arrow_length, head_width=head_width + 2, head_length=head_length + 2, fc=arrow_edge_color, ec=arrow_edge_color, alpha=0.8, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, 0, -arrow_length, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=1.0, linewidth=2, zorder=11)
            elif rotation == 270:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, arrow_length, 0, head_width=head_width + 2, head_length=head_length + 2, fc=arrow_edge_color, ec=arrow_edge_color, alpha=0.8, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, arrow_length, 0, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=1.0, linewidth=2, zorder=11)
        title_text = f'Room Layout: {self.room.room_type.title()}\nRoom ID: {self.room.id}'
        if arrows_to_draw:
            title_text += '\n\n→ Black arrows indicate object facing direction'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (cm)', fontsize=12)
        ax.set_ylabel('Y Position (cm)', fontsize=12)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.7, linewidth=1.2, color='gray', linestyle='-')
        ax.legend(fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=12, width=2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        if arrows_to_draw:
            legend_text = '→ Object Direction'
            ax_pos = ax.get_position()
            legend_x = ax_pos.x1 - 0.02
            legend_y = ax_pos.y1 + 0.02
            fig.text(legend_x, legend_y, legend_text, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black'))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
            print(f'Room visualization saved to: {save_path}')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path if save_path else None

    def _get_room_elements(self) -> Dict[str, Tuple]:
        """
        Get room elements (doors, windows, objects) in the format expected by visualization.
        Similar to get_door_window_placements but includes objects.
        
        Returns:
            Dictionary mapping element_id to (center, rotation, vertices, score)
        """
        initial_state = {}
        room = self.room
        wall_map = {wall.id: wall for wall in self.room.walls}
        for (i, door) in enumerate(self.room.doors):
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
            door_identifier = f'opening-{i}' if door.opening else f'door-{i}'
            initial_state[door_identifier] = ((door_open_center_x_cm, door_open_center_y_cm), 0, door_vertices, 1)
        for (i, window) in enumerate(self.room.windows):
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
            initial_state[f'window-{i}'] = ((window_open_center_x_cm, window_open_center_y_cm), 0, window_vertices, 0.3)
        for obj in self.room.objects:
            obj_x_cm = (obj.position.x - self.room.position.x) * 100
            obj_y_cm = (obj.position.y - self.room.position.y) * 100
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
            obj_vertices = [(obj_x_cm - obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2), (obj_x_cm + obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2), (obj_x_cm - obj_length_x_cm / 2, obj_y_cm + obj_length_y_cm / 2), (obj_x_cm - obj_length_x_cm / 2, obj_y_cm - obj_length_y_cm / 2)]
            initial_state[f'{obj.id}'] = ((obj_x_cm, obj_y_cm), obj.rotation.z, obj_vertices, 1)
        return initial_state

    def visualize_2d_render(self, save_path: str=None, show: bool=True):
        """
        Visualize the room in 2D render with overlay elements.
        """
        try:
            from room_render import render_room_top_orthogonal_view
            from models import FloorPlan
        except ImportError as e:
            print(f'Error importing required modules: {e}')
            return None
        try:
            rgb = render_room_top_orthogonal_view(self.layout, self.room.id)
            rgb = rgb[::-1].copy()
        except Exception as e:
            print(f'Error rendering room: {e}')
            return self.visualize_2d(save_path, show)
        initial_state = self._get_room_elements()
        fontsize = 6
        plt.rcParams['font.size'] = fontsize
        (fig, ax) = plt.subplots(figsize=(12, 10))
        room_width_cm = self.room.dimensions.width * 100
        room_length_cm = self.room.dimensions.length * 100
        ax.imshow(rgb, extent=[0, room_width_cm, 0, room_length_cm], origin='lower', alpha=0.95, aspect='equal')
        colors = ['brown', 'cyan', 'red', 'blue', 'green', 'orange', 'purple', 'pink']
        color_idx = 0
        arrows_to_draw = []
        for (object_id, (center, rotation, vertices, _)) in initial_state.items():
            (center_x, center_y) = center
            obj_poly = Polygon(vertices)
            (x_coords, y_coords) = obj_poly.exterior.xy
            if object_id.startswith('door-') or object_id.startswith('opening-'):
                ax.plot(x_coords, y_coords, '-', linewidth=3, color='brown')
                ax.fill(x_coords, y_coords, color='brown', alpha=0.3)
                ax.text(center_x, center_y, object_id, fontsize=fontsize, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
            elif object_id.startswith('window-'):
                ax.plot(x_coords, y_coords, '-', linewidth=2, color='cyan')
                ax.fill(x_coords, y_coords, color='cyan', alpha=0.3)
                ax.text(center_x, center_y, object_id, fontsize=fontsize, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
            else:
                current_color = colors[color_idx % len(colors)]
                ax.plot(x_coords, y_coords, '-', linewidth=2, color=current_color)
                ax.fill(x_coords, y_coords, color=current_color, alpha=0.25)
                label_text = f"{object_id.replace('existing-', '')}"
                if len(label_text) > 13:
                    label_text = label_text[:13] + '\n' + label_text[13:]
                if len(label_text) > 21:
                    label_text = label_text[:-8] + '\n' + label_text[-8:]
                ax.text(center_x, center_y, label_text, fontsize=fontsize, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
                arrows_to_draw.append((center_x, center_y, rotation, current_color))
                color_idx += 1
        for (center_x, center_y, rotation, current_color) in arrows_to_draw:
            arrow_length = 20
            head_width = 6
            head_length = 4
            arrow_face_color = 'white'
            arrow_edge_color = 'black'
            shadow_offset = 1
            if rotation == 0:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, 0, arrow_length, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.4, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, 0, arrow_length, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=0.7, linewidth=2, zorder=11)
            elif rotation == 90:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, -arrow_length, 0, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.4, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, -arrow_length, 0, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=0.7, linewidth=2, zorder=11)
            elif rotation == 180:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, 0, -arrow_length, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.4, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, 0, -arrow_length, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=0.7, linewidth=2, zorder=11)
            elif rotation == 270:
                ax.arrow(center_x + shadow_offset, center_y + shadow_offset, arrow_length, 0, head_width=head_width + 2, head_length=head_length + 2, fc='black', ec='black', alpha=0.4, linewidth=2, zorder=10)
                ax.arrow(center_x, center_y, arrow_length, 0, head_width=head_width, head_length=head_length, fc=arrow_face_color, ec=arrow_edge_color, alpha=0.7, linewidth=2, zorder=11)
        title_text = f'Room Layout with Render: {self.room.room_type.title()}\nRoom ID: {self.room.id}'
        if arrows_to_draw:
            title_text += '\n\n→ White arrows indicate object facing direction'
        ax.set_title(title_text, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X Position (cm)', fontsize=12)
        ax.set_ylabel('Y Position (cm)', fontsize=12)
        ax.set_aspect('equal', 'box')
        ax.grid(True, alpha=0.7, linewidth=1.2, color='gray', linestyle='-')
        ax.set_xlim(0, room_width_cm)
        ax.set_ylim(0, room_length_cm)
        ax.tick_params(axis='both', which='major', labelsize=12, width=2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        if arrows_to_draw:
            legend_text = '→ Object Direction'
            ax_pos = ax.get_position()
            legend_x = ax_pos.x1 - 0.02
            legend_y = ax_pos.y1 + 0.02
            fig.text(legend_x, legend_y, legend_text, fontsize=11, verticalalignment='bottom', horizontalalignment='right', fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.6, edgecolor='black'))
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=150, facecolor='white')
            print(f'Room visualization with render saved to: {save_path}')
        if show:
            plt.show()
        else:
            plt.close(fig)
        return save_path if save_path else None

def visualize_room_2d(room: Room, save_path: str=None):
    """
    Create a 2D visualization of a single room layout.
    
    Args:
        room: Room object to visualize
        save_path: Optional path to save the visualization
        
    Returns:
        Path to saved visualization if save_path provided, otherwise None
    """
    visualizer = RoomVisualizer(room)
    return visualizer.visualize_2d(save_path=save_path, show=False)

def visualize_room_2d_render(room: Room, layout: FloorPlan, save_path: str=None):
    """
    Create a 2D render visualization of a single room layout.
    
    Args:
        room: Room object to visualize
        save_path: Optional path to save the visualization
        
    Returns:
        Path to saved visualization if save_path provided, otherwise None
    """
    visualizer = RoomVisualizer(room, layout)
    return visualizer.visualize_2d_render(save_path=save_path, show=False)
