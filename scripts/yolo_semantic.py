"""Train, validate or predict PCB pixel classes with YOLO26-sem."""
import argparse

from common import ROOT
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','val','predict'])
    parser.add_argument('--model',default='yolo26n-sem.pt')
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--patience',type=int,default=30)
    parser.add_argument('--batch',type=int,default=-1)
    parser.add_argument('--imgsz',type=int,default=640)
    parser.add_argument('--device',default='0')
    parser.add_argument('--source',default=str(ROOT/'data/pcb_semantic_merged_grouped/images/test'))
    parser.add_argument('--data',default=str(ROOT/'data/pcb_semantic_merged_grouped/dataset.yaml'))
    parser.add_argument('--split',choices=['val','test'],default='test')
    args = parser.parse_args()
    model = YOLO(args.model,task='semantic')
    options = dict(imgsz=args.imgsz,device=args.device,project=str(ROOT/'runs/02_segmentation/merged'))
    data = args.data
    if args.mode == 'train':
        model.train(data=data,epochs=args.epochs,patience=args.patience,batch=args.batch,workers=0,seed=42,
                    name='pcb_yolo26n',**options)
    elif args.mode == 'val':
        model.val(data=data,split=args.split,batch=args.batch,save_json=True,**options)
    else:
        model.predict(source=args.source,save=True,**options)


if __name__ == '__main__':
    main()
