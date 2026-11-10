"""Train a small semantic model and generate editable Labelme candidates."""
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['background', 'missing_hole', 'mouse_bite', 'open_circuit', 'short', 'spur', 'spurious_copper']


class Segmenter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=4, dilation=4), nn.ReLU(), nn.Conv2d(32, 7, 1))

    def forward(self, x):
        return self.layers(x)


def main():
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    folder = ROOT / 'data/semantic_labeled'
    pairs = []
    counts = np.zeros(7)
    for path in sorted((folder / 'images/train').glob('*')):
        image = cv2.imread(str(path))
        mask = cv2.imread(str(folder / 'masks/train' / (path.stem + '.png')), 0)
        pairs.append((image, mask))
        counts += np.bincount(mask.ravel(), minlength=7)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1))
    weights = np.minimum(weights / weights[0], 40)
    model = Segmenter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    output = ROOT / 'runs/segmentation/bootstrap'
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(80):
        model.train()
        losses = []
        for _ in range(15):
            images, masks = [], []
            for _ in range(4):
                image, mask = random.choice(pairs)
                ys, xs = np.where(mask > 0)
                if len(xs) and random.random() < 0.8:
                    index = random.randrange(len(xs))
                    x, y = int(xs[index]), int(ys[index])
                else:
                    y, x = random.randrange(image.shape[0]), random.randrange(image.shape[1])
                x = max(0, min(x-64, image.shape[1]-128))
                y = max(0, min(y-64, image.shape[0]-128))
                crop, target = image[y:y+128, x:x+128], mask[y:y+128, x:x+128]
                if random.random() < .5:
                    crop, target = crop[:, ::-1], target[:, ::-1]
                images.append(crop.transpose(2,0,1).copy())
                masks.append(target.copy())
            inputs = torch.tensor(np.stack(images), device=device).float()/255
            targets = torch.tensor(np.stack(masks), device=device).long()
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch+1}/80 loss={np.mean(losses):.4f}', flush=True)
    torch.save({'model': model.state_dict(), 'names': NAMES}, output / 'model.pt')
    model.eval()
    paths = sorted(p for p in (ROOT / 'data/images').rglob('*') if p.suffix.lower() in {'.jpg','.png','.jpeg','.bmp'})
    destination = ROOT / 'data/segmentation_candidates'
    for index, path in enumerate(paths):
        relative = path.relative_to(ROOT / 'data/images')
        target = destination / relative.parent
        (target / 'masks').mkdir(parents=True, exist_ok=True)
        image = cv2.imread(str(path))
        h,w = image.shape[:2]
        with torch.no_grad():
            tensor = torch.from_numpy(image.transpose(2,0,1).copy()).unsqueeze(0).to(device).float()/255
            prediction = model(tensor).argmax(1)[0].cpu().numpy().astype('uint8')
        cv2.imwrite(str(target / 'masks' / (path.stem+'.png')), prediction)
        shapes = []
        for label in range(1,7):
            contours, _ = cv2.findContours((prediction==label).astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 5:
                    continue
                polygon = cv2.approxPolyDP(contour, .75, True).reshape(-1,2)
                if len(polygon) >= 3:
                    shapes.append({'label':NAMES[label], 'points':polygon.tolist(), 'group_id':None,
                        'shape_type':'polygon', 'flags':{}, 'description':'model candidate; requires human review'})
        record = {'version':'5.0.1','flags':{},'shapes':shapes,'imagePath':str(path.resolve()),
                  'imageData':None,'imageHeight':h,'imageWidth':w}
        (target / (path.stem+'.json')).write_text(json.dumps(record), encoding='utf-8')
        if index % 100 == 0:
            print(f'Candidates {index+1}/{len(paths)}', flush=True)
    print(f'DONE {len(paths)} images: {destination}', flush=True)


if __name__ == '__main__':
    main()
