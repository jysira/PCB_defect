"""Train the CLAHE PCB semantic segmentation experiment."""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=150)
    parser.add_argument('--patience',type=int,default=30)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--device',default='0')
    args = parser.parse_args()
    data = ROOT/'data/pcb_semantic_merged_clahe/dataset.yaml'
    if not data.is_file():
        raise FileNotFoundError('Run scripts/prepare_semantic_clahe.py first')
    YOLO('yolo26n-sem.pt',task='semantic').train(data=str(data),epochs=args.epochs,
        patience=args.patience,imgsz=args.imgsz,batch=args.batch,optimizer='AdamW',lr0=args.lr,
        device=args.device,workers=0,seed=42,project=str(ROOT/'runs/semantic_clahe'),name='pcb')


if __name__ == '__main__':
    main()
