"""Convert Labelme annotations to class-index PNG masks and a COCO export."""
import argparse
import json
import math
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('data/labels2/train'))
    parser.add_argument('--output', type=Path, default=Path('data/semantic_labeled'))
    args = parser.parse_args()
    files = sorted(args.input.glob('*.json'))
    if not files:
        raise ValueError('No Labelme JSON files found')
    records = []
    for file in files:
        data = json.loads(file.read_text(encoding='utf-8'))
        image = (file.parent / data['imagePath']).resolve()
        if not image.is_file():
            raise FileNotFoundError(image)
        with Image.open(image) as source:
            if source.size != (data['imageWidth'], data['imageHeight']):
                raise ValueError(f'Image dimensions differ: {file}')
        for shape in data['shapes']:
            if shape['label'] not in NAMES:
                raise ValueError(f'Unknown class in {file}: {shape["label"]}')
            if shape['shape_type'] not in ('polygon', 'rectangle', 'oriented_rectangle', 'circle'):
                raise ValueError(f'Unsupported shape in {file}: {shape["shape_type"]}')
        records.append((file, data, image))
    masks = args.output / 'masks' / 'train'
    images = args.output / 'images' / 'train'
    coco_dir = args.output / 'coco' / 'train'
    for folder in (masks, images, coco_dir):
        folder.mkdir(parents=True, exist_ok=True)
    coco = {'images': [], 'annotations': [], 'categories': [
        {'id': i + 1, 'name': name} for i, name in enumerate(NAMES)]}
    for image_id, (file, data, image) in enumerate(records):
        width, height = data['imageWidth'], data['imageHeight']
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        for shape in data['shapes']:
            label = NAMES.index(shape['label']) + 1
            points = [tuple(p) for p in shape['points']]
            kind = shape['shape_type']
            if kind == 'circle':
                (x, y), (px, py) = points
                radius = math.hypot(px - x, py - y)
                draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=label)
                points = [(x + radius * math.cos(t * math.tau / 72),
                           y + radius * math.sin(t * math.tau / 72)) for t in range(72)]
            else:
                if kind == 'rectangle':
                    (x1, y1), (x2, y2) = points
                    points = [(min(x1,x2),min(y1,y2)), (max(x1,x2),min(y1,y2)),
                              (max(x1,x2),max(y1,y2)), (min(x1,x2),max(y1,y2))]
                draw.polygon(points, fill=label)
            xs, ys = zip(*points)
            area = abs(sum(points[i][0]*points[(i+1)%len(points)][1] -
                           points[(i+1)%len(points)][0]*points[i][1]
                           for i in range(len(points)))) / 2
            coco['annotations'].append({'id': len(coco['annotations']) + 1,
                'image_id': image_id, 'category_id': label, 'iscrowd': 0,
                'segmentation': [[v for point in points for v in point]],
                'bbox': [min(xs), min(ys), max(xs)-min(xs), max(ys)-min(ys)], 'area': area})
        mask.save(masks / (image.stem + '.png'))
        shutil.copy2(image, images / image.name)
        shutil.copy2(image, coco_dir / image.name)
        coco['images'].append({'id': image_id, 'file_name': image.name, 'width': width, 'height': height})
    (coco_dir / '_annotations.coco.json').write_text(json.dumps(coco, ensure_ascii=False), encoding='utf-8')
    mapping = {'0': 'background', **{str(i+1): n for i, n in enumerate(NAMES)}}
    (args.output / 'classes.json').write_text(json.dumps(mapping, indent=2), encoding='utf-8')
    print(f'Converted {len(records)} images, {len(coco["annotations"])} shapes to {args.output.resolve()}')


if __name__ == '__main__':
    main()
