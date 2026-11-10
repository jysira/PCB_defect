"""Export the reviewed Labelme dataset to normalized YOLO instance polygons."""
import argparse
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


def polygon(shape):
    points = shape['points']
    kind = shape['shape_type']
    if kind == 'circle':
        (x,y),(a,b) = points
        radius = math.hypot(a-x,b-y)
        return [[x+radius*math.cos(i*math.tau/32),y+radius*math.sin(i*math.tau/32)] for i in range(32)]
    if kind == 'rectangle':
        (x,y),(a,b) = points
        return [[min(x,a),min(y,b)],[max(x,a),min(y,b)],
                [max(x,a),max(y,b)],[min(x,a),max(y,b)]]
    if kind in {'polygon','oriented_rectangle'}:
        return points
    raise ValueError(f'Unsupported shape type: {kind}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'data/pcb_reviewed_300')
    parser.add_argument('--output',type=Path,default=ROOT/'data/pcb_reviewed_300_yolo_seg')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Choose a new output directory: {args.output}')
    records = []
    counts = {}
    for split in ('train','val','test'):
        files = sorted((args.source/'labels'/split).glob('*.json'))
        if not files:
            raise ValueError(f'No annotations for {split}')
        images = {p.stem:p for p in (args.source/'images'/split).iterdir() if p.is_file()}
        counts[split] = {'images':len(files),'polygons':0}
        for file in files:
            data = json.loads(file.read_text(encoding='utf-8'))
            image = images[file.stem]
            width,height = data['imageWidth'],data['imageHeight']
            if width <= 0 or height <= 0:
                raise ValueError(f'Invalid dimensions: {file}')
            rows = []
            for shape in data['shapes']:
                label = NAMES.index(shape['label'])
                points = polygon(shape)
                points = [[min(1.,max(0.,float(x)/width)),min(1.,max(0.,float(y)/height))] for x,y in points]
                if points and points[0] == points[-1]:
                    points = points[:-1]
                if len(set(map(tuple,points))) < 3:
                    raise ValueError(f'Degenerate polygon: {file}')
                area = abs(sum(points[i][0]*points[(i+1)%len(points)][1]-points[(i+1)%len(points)][0]*points[i][1] for i in range(len(points))))
                if area <= 1e-12:
                    raise ValueError(f'Zero area polygon: {file}')
                rows.append(str(label)+' '+' '.join(f'{v:.8f}' for point in points for v in point))
            counts[split]['polygons'] += len(rows)
            records.append((split,image,file.stem,rows))
    for split,image,stem,rows in records:
        image_dir = args.output/'images'/split
        label_dir = args.output/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True)
        label_dir.mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,image_dir/image.name)
        (label_dir/(stem+'.txt')).write_text('\n'.join(rows)+('\n' if rows else ''),encoding='utf-8')
    config = 'path: '+json.dumps(args.output.resolve().as_posix(),ensure_ascii=False)+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'
    config += ''.join(f'  {i}: {name}\n' for i,name in enumerate(NAMES))
    (args.output/'dataset.yaml').write_text(config,encoding='utf-8')
    (args.output/'conversion_summary.json').write_text(json.dumps(counts,indent=2),encoding='utf-8')
    print(json.dumps(counts,indent=2))
    print(args.output.resolve())


if __name__ == '__main__':
    main()
