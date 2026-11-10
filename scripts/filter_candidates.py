"""Constrain bootstrap masks to existing YOLO boxes for manual correction."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


def main():
    source = ROOT / 'data/segmentation_candidates'
    destination = ROOT / 'data/segmentation_candidates_filtered'
    before = after = missing = 0
    files = sorted(source.glob('*/*.json'))
    for file in files:
        record = json.loads(file.read_text(encoding='utf-8'))
        before += len(record['shapes'])
        relative = file.relative_to(source)
        label_file = ROOT / 'data/labels' / relative.with_suffix('.txt')
        if not label_file.is_file():
            missing += 1
            continue
        mask = cv2.imread(str(file.parent / 'masks' / (file.stem+'.png')), 0)
        h,w = mask.shape
        filtered = np.zeros_like(mask)
        shapes = []
        for line in label_file.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            cls,x,y,bw,bh = map(float, line.split())
            cls = int(cls)
            x1,y1 = max(0,int((x-bw/2)*w)), max(0,int((y-bh/2)*h))
            x2,y2 = min(w,int(np.ceil((x+bw/2)*w))), min(h,int(np.ceil((y+bh/2)*h)))
            # Existing detection class determines the semantic class in this ROI.
            region = (mask[y1:y2,x1:x2] > 0).astype('uint8')
            contours,_ = cv2.findContours(region,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = max(contours,key=cv2.contourArea)
            if cv2.contourArea(contour) < 5:
                continue
            contour = contour + np.array([[[x1,y1]]])
            polygon = cv2.approxPolyDP(contour,.75,True).reshape(-1,2)
            if len(polygon) < 3:
                continue
            cv2.fillPoly(filtered,[contour],cls+1)
            shapes.append({'label':NAMES[cls],'points':polygon.tolist(),'group_id':None,
                'shape_type':'polygon','flags':{},'description':'Detection-box constrained candidate; review boundary and missed defects'})
        record['shapes'] = shapes
        target = destination / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        (target.parent/'masks').mkdir(exist_ok=True)
        target.write_text(json.dumps(record),encoding='utf-8')
        cv2.imwrite(str(target.parent/'masks'/(file.stem+'.png')),filtered)
        after += len(shapes)
    print(f'Images={len(files)} missing_detection_labels={missing} polygons_before={before} polygons_after={after}')
    print(destination)


if __name__ == '__main__':
    main()
