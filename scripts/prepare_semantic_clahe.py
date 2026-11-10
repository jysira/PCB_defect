"""Apply LAB-L CLAHE to the merged semantic dataset, preserving masks and splits."""
import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--clip-limit',type=float,default=2.0)
    parser.add_argument('--tile-size',type=int,default=8)
    args = parser.parse_args()
    if args.clip_limit <= 0 or args.tile_size <= 0:
        parser.error('CLAHE parameters must be positive')
    source = ROOT/'data/pcb_semantic_merged_grouped'
    output = ROOT/'data/pcb_semantic_merged_clahe'
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    config = yaml.safe_load((source/'dataset.yaml').read_text(encoding='utf-8'))
    clahe = cv2.createCLAHE(clipLimit=args.clip_limit,tileGridSize=(args.tile_size,args.tile_size))
    counts = {}
    for split in ('train','val','test'):
        files = sorted((source/'images'/split).glob('*'))
        counts[split] = 0
        (output/'images'/split).mkdir(parents=True,exist_ok=True)
        (output/'masks'/split).mkdir(parents=True,exist_ok=True)
        for path in files:
            if path.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:
                continue
            image = cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f'Unreadable image: {path}')
            lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
            lab[:,:,0] = clahe.apply(lab[:,:,0])
            image = cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
            # Lossless PNG avoids additional JPEG artifacts around tiny defects.
            target = output/'images'/split/(path.stem+'.png')
            success,encoded = cv2.imencode('.png',image)
            if not success:
                raise RuntimeError(target)
            encoded.tofile(target)
            mask = source/'masks'/split/(path.stem+'.png')
            shutil.copy2(mask,output/'masks'/split/mask.name)
            counts[split] += 1
        print(split,counts[split],flush=True)
    config['path'] = output.as_posix()
    (output/'dataset.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    report = {'source':str(source),'counts':counts,'color_space':'LAB L channel',
              'clip_limit':args.clip_limit,'tile_grid_size':[args.tile_size,args.tile_size],
              'image_format':'lossless PNG','masks':'copied unchanged'}
    (output/'preprocessing.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(output,flush=True)


if __name__ == '__main__':
    main()
