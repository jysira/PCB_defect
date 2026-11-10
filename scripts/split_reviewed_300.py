"""Split the first 300 reviewed Labelme files by filename, with crop grouping."""
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


def main():
    source = ROOT / 'data/segmentation_candidates_max6/test'
    output = ROOT / 'data/pcb_reviewed_300'
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    files = sorted(source.glob('*.json'), key=lambda p: p.name)[:300]
    if len(files) != 300:
        raise ValueError('Expected at least 300 JSON files')
    images = {p.stem:p for p in (ROOT/'data/images').rglob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp'}}
    groups = defaultdict(list)
    for file in files:
        data = json.loads(file.read_text(encoding='utf-8'))
        image = images[file.stem]
        match = re.search(r'(\d{2}_(?:missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)_\d+_\d+_600)',file.stem)
        group = match.group(1) if match else file.stem
        for shape in data['shapes']:
            if shape['label'] not in NAMES or shape['shape_type'] not in {'polygon','rectangle','circle','oriented_rectangle'}:
                raise ValueError(f'Unsupported annotation: {file}')
        groups[group].append((file,data,image))
    buckets = defaultdict(list)
    for key, records in groups.items():
        category = next((n for n in NAMES if f'_{n}_' in key),'other')
        buckets[category].append((key,records))
    rng = random.Random(42)
    assigned = []
    for category, entries in sorted(buckets.items()):
        rng.shuffle(entries)
        count = sum(len(records) for _,records in entries)
        used = 0
        for group, records in entries:
            midpoint = used + len(records)/2
            split = 'train' if midpoint <= count*.7 else 'val' if midpoint <= count*.9 else 'test'
            assigned.extend((split,group,*record) for record in records)
            used += len(records)
    manifest = []
    for split,group,file,data,image in assigned:
        for kind in ('images','labels','masks'):
            (output/kind/split).mkdir(parents=True,exist_ok=True)
        shutil.copy2(image,output/'images'/split/image.name)
        data['imagePath'] = str(Path('../../images')/split/image.name)
        data['imageData'] = None
        (output/'labels'/split/file.name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
        mask = Image.new('L',(data['imageWidth'],data['imageHeight']),0)
        draw = ImageDraw.Draw(mask)
        for shape in data['shapes']:
            points = [tuple(p) for p in shape['points']]
            label = NAMES.index(shape['label'])+1
            if shape['shape_type'] == 'circle':
                (x,y),(a,b) = points
                radius = ((x-a)**2+(y-b)**2)**.5
                draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill=label)
            elif shape['shape_type'] == 'rectangle':
                (x,y),(a,b) = points
                draw.rectangle((min(x,a),min(y,b),max(x,a),max(y,b)),fill=label)
            else:
                draw.polygon(points,fill=label)
        mask.save(output/'masks'/split/(image.stem+'.png'))
        manifest.append({'split':split,'group':group,'source_annotation':str(file),'image':image.name})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'classes.json').write_text(json.dumps(['background']+NAMES,indent=2),encoding='utf-8')
    print(output)
    print(dict(Counter(row['split'] for row in manifest)))
    print('First:', files[0].name, 'Last:',files[-1].name)
    assert len(manifest)==300
    assert all(len({r['split'] for r in manifest if r['group']==g})==1 for g in groups)


if __name__ == '__main__':
    main()
