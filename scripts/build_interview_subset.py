"""Create a reproducible, class-balanced 300-image annotation subset."""
import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


def main():
    output = ROOT / 'data/pcb_interview_300'
    image_lookup = {p.stem: p for p in (ROOT / 'data/images').rglob('*')
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}}
    candidates = {name: {} for name in NAMES}
    for file in sorted((ROOT / 'data/segmentation_candidates_max6').glob('*/*.json')):
        record = json.loads(file.read_text(encoding='utf-8'))
        image = image_lookup.get(file.stem)
        if image is None:
            continue
        match = re.search(r'(\d{2}_(?:missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)_\d+_\d+_600)', image.stem)
        if not match or not record['shapes']:
            continue
        group = match.group(1)
        category = next(name for name in NAMES if f'_{name}_' in group)
        # Choose one representative per original crop, across all old splits.
        candidates[category].setdefault(group, (file, record, image))
    rng = random.Random(42)
    manifest = []
    for name in NAMES:
        groups = list(candidates[name])
        rng.shuffle(groups)
        if len(groups) < 50:
            raise ValueError(f'{name}: only {len(groups)} unique crops')
        for index, group in enumerate(groups[:50]):
            split = 'train' if index < 35 else 'val' if index < 45 else 'test'
            file, record, image = candidates[name][group]
            folder = output / split
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, folder / image.name)
            record['imagePath'] = image.name
            record['imageData'] = None
            record.setdefault('flags', {})['needs_human_review'] = True
            (folder / file.name).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
            label = ROOT / 'data/labels' / file.parent.name / (file.stem+'.txt')
            labels = output / 'detection_labels' / split
            labels.mkdir(parents=True, exist_ok=True)
            shutil.copy2(label, labels / label.name)
            manifest.append({'split':split,'class':name,'group':group,'image':image.name,
                             'original_image':str(image),'annotation_status':'candidate_requires_review'})
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'README.txt').write_text(
        '300 images, six classes, 50 images per class.\n'
        'Train: 210 (35/class); val: 60 (10/class); test: 30 (5/class).\n'
        'Each split contains images and matching Labelme JSON. Open the split directory in Labelme.\n'
        'All polygon annotations are model candidates and require manual review.\n'
        'One representative per filename-derived original crop; this is not board-level separation.\n'
        'Original detector/seed-model training used some of these images. This subset must not be used\n'
        'to claim independent generalization of those existing models. Train a new model for evaluation.\n',encoding='utf-8')
    print(output)
    for split in ('train','val','test'):
        print(split, sum(row['split']==split for row in manifest))


if __name__ == '__main__':
    main()
