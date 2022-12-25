import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolo.models.common import DetectMultiBackend
from yolo.utils.general import check_img_size, non_max_suppression, scale_coords
from yolo.utils.torch_utils import select_device

# weights =  'D:\\cv_samurai_yolo\\yolo\\weights\\best.pt' # model_yolo.pt path(s)
# source = 'data/images/bus.jpg' # file/dir/URL/glob0 for webcam
# device = '' # cuda device i.e. 0 or 0,1,2,3 or cpu
imgsz = [640, 640] # inference size (heightwidth)
conf_thres = 0.1 # confidence threshold
iou_thres = 0.45 # NMS IOU threshold
max_det = 1000 # maximum detections per image
classes = None # filter by class: --class 0, or --class 0 2 3
agnostic_nms = False # class-agnostic NMS
augment = False # augmented inference
half = False # use FP16 half-precision inference
dnn = False # use OpenCV DNN for ONNX inference

# Load model_yolo
device = select_device("")
model_yolo = DetectMultiBackend('G:\\Samurai\\_adaptationToTheSteamVersion\\yolo\\weights\\best.pt', device=device, dnn=dnn)
model_yolo.eval()
stride, names, pt, jit, onnx, engine = model_yolo.stride, model_yolo.names, model_yolo.pt, model_yolo.jit, model_yolo.onnx, model_yolo.engine
imgsz = check_img_size(imgsz, s=stride)  # check image size

# Half
half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt or jit:
    model_yolo.model_yolo.half() if half else model_yolo.model.float()
model_yolo.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def readScreenShot4States(im0s, imgsz=imgsz, stride=stride, pt=pt):
    ## im0s is directly read using cv2.imread in BGR color style

    # # Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    # Run inference
    # for path, im, im0s, _, _ in dataset:
    im = letterbox(im0s, 640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im/= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred_clses, pred_confs, pred_boxes = [], [], []

    # Inference
    pred = model_yolo(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            # print(im0s.shape)
            # print(type(im0s.shape))
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], (im0s.shape[1], im0s.shape[0], 3)).round()

            # Write results
            for *xyxy, conf, cls in det:
                pred_clses.append(int(cls.detach().cpu().numpy()))
                pred_confs.append(conf.detach().cpu().numpy().reshape(-1)[0])
                pred_boxes.append([i.detach().cpu().numpy().reshape(-1)[0] for i in xyxy])

    return pred_clses, pred_confs, pred_boxes
    ## pred_boxes格式：左上纵坐标横坐标，右下纵坐标横坐标

if __name__ == "__main__":
    print( readScreenShot4States(cv2.imread("G:\\Samurai\\yolo\\data\\images_samurai\\000004.jpg")) )