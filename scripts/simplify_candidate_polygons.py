"""Simplify filtered Labelme polygons to at most six vertices."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def simplify(points):
    points = [list(point) for point in points]
    if len(points) > 1 and points[0] == points[-1]:
        points.pop()
    while len(points) > 6:
        vertices = np.asarray(points, dtype=float)
        previous = np.roll(vertices, 1, axis=0)
        following = np.roll(vertices, -1, axis=0)
        left = vertices - previous
        right = following - vertices
        areas = np.abs(left[:, 0] * right[:, 1] - left[:, 1] * right[:, 0])
        points.pop(int(np.argmin(areas)))
    return points


def main():
    source = ROOT / 'data/segmentation_candidates_filtered'
    destination = ROOT / 'data/segmentation_candidates_max6'
    files = sorted(source.glob('*/*.json'))
    changed = total = 0
    for file in files:
        record = json.loads(file.read_text(encoding='utf-8'))
        for shape in record['shapes']:
            if shape['shape_type'] == 'polygon':
                original = shape['points']
                shape['points'] = simplify(original)
                changed += len(original) != len(shape['points'])
                total += 1
        target = destination / file.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, ensure_ascii=False), encoding='utf-8')
    print(f'Files={len(files)}, polygons={total}, simplified={changed}, output={destination}')
    for file in destination.glob('*/*.json'):
        record = json.loads(file.read_text(encoding='utf-8'))
        assert all(3 <= len(shape['points']) <= 6 for shape in record['shapes'])
    print('Verified every polygon has 3 to 6 vertices.')


if __name__ == '__main__':
    main()
