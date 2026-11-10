"""Merge PCB masks with unified IDs and original-image grouped splits."""
import hashlib
import json
import random
import re
import shutil
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['background','missing_hole','mouse_bite','open_circuit','short','spur','spurious_copper']


def main():
    output = ROOT/'data/pcb_semantic_merged_grouped'
    if output.exists():
        raise FileExistsError(output)
    sources = [('reviewed',ROOT/'data/pcb_reviewed_300',list(range(7))),
               ('six_class',ROOT/'semantic_pcb/dataset',list(range(7))),
               ('five_class',ROOT/'semantic_final/dataset',[0,1,2,3,5,6])]
    records = []
    hashes = {}
    duplicates = []
    groups = defaultdict(list)
    for source,folder,mapping in sources:
        for image in sorted((folder/'images').rglob('*')):
            if image.suffix.lower() not in {'.jpg','.png','.jpeg','.bmp'}:
                continue
            split = image.parent.name
            mask_path = folder/'masks'/split/(image.stem+'.png')
            with Image.open(image) as rgb, Image.open(mask_path) as mask:
                pixels = np.asarray(rgb.convert('RGB'))
                raw = np.asarray(mask)
                if raw.ndim!=2 or raw.shape!=pixels.shape[:2]:
                    raise ValueError(f'Invalid mask dimensions: {mask_path}')
                if not set(np.unique(raw)) <= set(range(len(mapping)))|{255}:
                    raise ValueError(f'Unexpected mask IDs: {mask_path}')
                remapped = np.full(raw.shape,255,dtype=np.uint8)
                for old,new in enumerate(mapping):
                    remapped[raw==old] = new
                digest = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
            # Optical board number + defect class + original sample index groups
            # exports, crops and augmentations conservatively across sources.
            match = re.search(r'(\d{2})_(missing_hole|mouse_bite|open_circuit|short|spur|spurious_copper)_(\d+)',image.stem,re.I)
            if not match:
                raise ValueError(f'Cannot determine original group: {image}')
            group = '_'.join(match.groups()).lower()
            if digest in hashes:
                previous = records[hashes[digest]]
                duplicates.append({'dropped':str(image),'kept':str(previous['image']),
                    'mask_conflict':not np.array_equal(remapped,previous['mask'])})
                # Link matching pixels even when source names differ.
                previous['aliases'].add(group)
                continue
            hashes[digest] = len(records)
            records.append({'image':image,'mask':remapped,'source':source,'group':group,'aliases':{group}})
    # Union groups joined by byte-identical RGB images.
    parent = {}
    def find(key):
        parent.setdefault(key,key)
        if parent[key]!=key:
            parent[key]=find(parent[key])
        return parent[key]
    for record in records:
        for alias in record['aliases']:
            parent[find(alias)] = find(record['group'])
    for record in records:
        groups[find(record['group'])].append(record)
    buckets = defaultdict(list)
    for key,items in groups.items():
        category = next(n for n in NAMES[1:] if f'_{n}_' in key)
        buckets[category].append((key,items))
    rng = random.Random(42)
    manifest = []
    stats = defaultdict(Counter)
    for category,entries in sorted(buckets.items()):
        rng.shuffle(entries)
        total = sum(len(items) for _,items in entries)
        used = 0
        for group,items in entries:
            midpoint = used+len(items)/2
            split = 'train' if midpoint < total*.7 else 'val' if midpoint < total*.9 else 'test'
            used += len(items)
            for record in items:
                image = record['image']
                name = record['source']+'__'+image.name
                for kind in ('images','masks'):
                    (output/kind/split).mkdir(parents=True,exist_ok=True)
                shutil.copy2(image,output/'images'/split/name)
                Image.fromarray(record['mask']).save(output/'masks'/split/(Path(name).stem+'.png'))
                stats[split].update([category])
                manifest.append({'split':split,'group':group,'source':record['source'],'original':str(image),'image':name})
    config = f'path: {json.dumps(output.as_posix(),ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nmasks_dir: masks\nnames:\n'
    config += ''.join(f'  {i}: {name}\n' for i,name in enumerate(NAMES))
    (output/'dataset.yaml').write_text(config,encoding='utf-8')
    report = {'counts':dict(Counter(r['split'] for r in manifest)), 'category_counts':dict(stats),
        'kept_sources':dict(Counter(r['source'] for r in manifest)), 'groups':len(groups),
        'duplicates_removed':len(duplicates),'conflicting_duplicates':sum(d['mask_conflict'] for d in duplicates),
        'duplicate_details':duplicates,'manifest':manifest}
    (output/'merge_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    for group in groups:
        assert len({r['split'] for r in manifest if r['group']==group}) == 1
    print({k:v for k,v in report.items() if k not in {'manifest','duplicate_details'}})


if __name__ == '__main__':
    main()
