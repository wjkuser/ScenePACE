"""Explicit category vocabulary shared by matching and diversity accounting."""
import re

ALIASES = {
    'coffee_table': {'coffeetable', 'table'}, 'dining_table': {'diningtable', 'table'},
    'floor_lamp': {'floorlamp', 'lamp'}, 'nightstand': {'nightstand', 'bedsidetable'},
    'pencil_holder': {'pencilholder', 'penholder'}, 'bookshelf': {'bookshelf', 'bookcase', 'bookshelves'},
    'mug': {'mug', 'ceramicmug', 'cup'}, 'sofa': {'sofa', 'couch'},
    'chair': {'chair', 'diningchair', 'officechair'},
    'plant': {'plant', 'pottedplant', 'houseplant', 'decorativeplant'},
    'artwork': {'artwork', 'painting', 'framedart', 'wallart'},
    'desk_lamp': {'desklamp', 'tablelamp'}, 'clock': {'clock', 'wallclock'},
    'wardrobe': {'wardrobe', 'closet'},
    'rug': {'rug', 'carpet', 'arearug'}, 'cushion': {'cushion', 'throwpillow', 'decorativecushion', 'sofacushion'},
}
CATEGORIES = set(ALIASES) | {
    'armchair', 'dresser', 'sideboard', 'office_chair', 'meeting_table', 'easel',
    'work_table', 'stool', 'storage_rack', 'paint_palette', 'notebook', 'student_desk',
    'teacher_desk', 'whiteboard', 'workbench', 'tool_cabinet', 'toolbox', 'piano',
    'music_stand', 'metronome', 'lab_bench', 'storage_cabinet', 'shelving_unit', 'test_tube_rack',
    'reading_table', 'bed', 'desk', 'cabinet', 'apple', 'banana', 'book', 'bowl', 'vase', 'plant',
    'cushion', 'pillow', 'rug', 'artwork', 'clock', 'basket', 'tray', 'bottle',
    'plate', 'teapot', 'tissue_box', 'side_table', 'ottoman', 'bench', 'mirror',
}


def token(value):
    return re.sub(r'[^a-z0-9]', '', re.sub(r'[_\s-]+\d+$', '', value.lower()))


def category_matches(category, actual):
    names = ALIASES.get(category, set()) | {token(category)}
    return token(actual) in names | {name + 's' for name in names}


def canonical_category(actual):
    exact = [c for c in sorted(CATEGORIES) if token(c) == token(actual)]
    matches = exact or [c for c in sorted(CATEGORIES) if category_matches(c, actual)]
    # Ambiguous "table" must not manufacture two distinct diversity categories.
    return matches[0] if matches else None
