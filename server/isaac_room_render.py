"""Request RTX views from the local Isaac simulation service."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
from constants import RESULTS_DIR
from isaacsim.isaac_mcp.server import get_isaac_connection

def render_isaac_views(layout, room_id, resolution=1024, views='five'):
    room = next((room for room in layout.rooms if room.id == room_id))
    scene_dir = Path(RESULTS_DIR) / layout.id
    with tempfile.TemporaryDirectory(prefix='sage-render-') as directory:
        path = Path(directory) / 'room.json'
        path.write_text(json.dumps(asdict(room)))
        result = get_isaac_connection().send_command('render_room_views', {'scene_save_dir': str(scene_dir), 'room_dict_save_path': str(path), 'output_dir': directory, 'resolution': resolution, 'views': views})
        return [np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255 for path in result['paths']]
