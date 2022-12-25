import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolo.models.common import DetectMultiBackend
from yolo.utils.general import check_img_size, non_max_suppression, scale_coords
from yolo.utils.torch_utils import select_device

# weights =  'G:\\cv_samurai_yolo\\yolo\\weights\\best.pt_detailed' # model_yolo_detailed.pt_detailed path(s)
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
dnn = False # use OpenCV DNN for onnx_detailed inference

# Load model_yolo_detailed
device = select_device("")
model_yolo_detailed = DetectMultiBackend('G:\\Samurai\\_Github\\Train_and_Fight\\yolo\\weights\\best_detailed.pt', device=device, dnn=dnn)
model_yolo_detailed.eval()
stride_detailed, names_detailed, pt_detailed, jit_detailed, onnx_detailed, engine_detailed = model_yolo_detailed.stride, model_yolo_detailed.names, model_yolo_detailed.pt, model_yolo_detailed.jit, model_yolo_detailed.onnx, model_yolo_detailed.engine
imgsz = check_img_size(imgsz, s=stride_detailed)  # check image size

# Half
half &= (pt_detailed or jit_detailed or engine_detailed) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt_detailed or jit_detailed:
    model_yolo_detailed.model_yolo_detailed.half() if half else model_yolo_detailed.model.float()
# model_yolo_detailed.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

def letterbox_detailed(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride_detailed-multiple constraints
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
        dw, dh = np.mod(dw, stride_detailed), np.mod(dh, stride_detailed)  # wh padding
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

def readScreenShot4States_detailed(im0s, imgsz=imgsz, stride=stride_detailed, pt=pt_detailed):
    ## im0s is directly read using cv2.imread in BGR color style

    # # Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride_detailed=stride_detailed, auto=pt_detailed)

    # Run inference
    # for path, im, im0s, _, _ in dataset:
    im = letterbox_detailed(im0s, 640, stride=32, auto=True)[0]
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im/= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred_clses, pred_confs, pred_boxes = [], [], []

    # Inference
    pred = model_yolo_detailed(im, augment=augment)
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    for i, det in enumerate(pred):  # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            # print(im0s.shape)
            # print(type(im0s.shape))
            # print(im.shape)
            # print(det[:, :4])
            # print(im0s.shape)
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], (im0s.shape[0], im0s.shape[1], 3)).round()

            # Write results
            for *xyxy, conf, cls in det:
                pred_clses.append(int(cls.detach().cpu().numpy()))
                pred_confs.append(conf.detach().cpu().numpy().reshape(-1)[0])
                pred_boxes.append([i.detach().cpu().numpy().reshape(-1)[0] for i in xyxy])

    return pred_clses, pred_confs, pred_boxes
    ## pred_boxes格式：左上纵坐标横坐标，右下纵坐标横坐标

if __name__ == "__main__":
    detailed_states = ['1_u', '1_i', '1_o', '1_j', '1_k', '1_upSlash', '1_downSlash', '1_aeroStrike', '1_aeroKick',
                       '1_idle', '1_hit', '1_down', '1_back', '1_parry', '1_jump', '1_dashForward', '1_rollBack',
                       '1_ultimate', '1_crescentMoon', '1_winePot', '1_whirlWind', '1_earthquake',
                       '2_u', '2_i', '2_o', '2_j', '2_k', '2_upSlash', '2_downSlash', '2_aeroStrike', '2_aeroKick',
                       '2_idle', '2_hit', '2_down', '2_back', '2_parry', '2_jump', '2_dashForward', '2_rollBack',
                       '2_ultimate', '2_snowfall', '2_fakeSnowfall', '2_swallowSlash', '2_ghostlyStrike',
                       '1_meleeThrow', '1_rangeThrow', '1_parried',
                       '2_meleeThrow', '2_rangeThrow', '2_parried', '1_faint', '2_faint']

    pred_clses, pred_confs, pred_boxes = readScreenShot4States_detailed(cv2.imread("G:\\Datasets\\labelledSamuraiImagesByHands_20220406\\images\\02338.png"))
    for i in range(len(pred_boxes)):
        print(detailed_states[pred_clses[i]], pred_confs[i])
        print(pred_boxes[i])

    