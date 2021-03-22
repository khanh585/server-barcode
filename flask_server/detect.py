import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import load_classifier, time_synchronized
from utils.cut_barcode import getWarp
from flask_server import app
from utils.read_code import read_barcode
from utils.bot_drawer import returnResult
from model_obj.doi_tuong import DoiTuong






def detect(src_path, img_size,save_img=False):

    source, view_img, save_txt, imgsz = src_path, False, True, img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(app.config['SAVE_IMAGES_PATH']) / 'exp', exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = app.config['DEVICE']
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = app.config['MODEL']
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
    if half:
        model.half()  # to FP16


    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 222) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    datas = []
    for path, img, im0s, vid_cap, frame in dataset:

        if not app.config["STREAM"] and app.config["CAP_VIDEO"] != None:
            dataset.stopStream(0)
            break
       
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, 0.3, 0.45, classes=None, agnostic=False)
        t2 = time_synchronized()


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string


                # list code in frame
                # Write results
                img_copy = im0.copy()
                list_dt = []
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if False else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        dt = DoiTuong(label.split(' ')[0], xyxy)
                        dt_image = getWarp(img_copy, dt.toBbox(), dt.width, dt.height)

                        # cut box
                        dt.setImage(dt_image)
                        list_dt.append(dt)
                        plot_one_box(dt.position, im0, label=dt.name, color=colors[int(cls)], line_thickness=3)
                        
            


            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            list_dt = make_order(list_dt)
            datas.append(list_dt)

            # Stream results
            if view_img:
                cv2.imwrite('stream/demo.jpg', im0)
                # cv2.imshow(str(p), im0)
                # cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
                    path = source.replace('.mp4', '.jpg')
                    path = path.replace('\\videos\\', '\\images\\')
                    paths = path.rsplit('.')
                    path = paths[0] + '_%s'%(frame+1) + '.' + paths[1]
                    cv2.imwrite(path, im0)
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


    return save_path, returnResult(datas)


def make_order(list_doituong):
    return  sorted(list_doituong, key=lambda dt: dt.position[0])


# Namespace(
# agnostic_nms=False, 
# augment=False, 
# classes=None, 
# conf_thres=0.25, 
# device='', 
# exist_ok=False, 
# img_size=640, 
# iou_thres=0.45, 
# name='exp', 
# project='runs/detect', 
# save_conf=False, 
# save_txt=False, 
# source='data/images', 
# update=False, 
# view_img=False, 
# weights='yolov5l.pt')

# python detect.py --weights runs/train/exp2/weights/last.pt --source data/test/