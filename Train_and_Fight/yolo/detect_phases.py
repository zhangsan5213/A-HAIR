import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolo.models.common import DetectMultiBackend
from yolo.utils.general import check_img_size, non_max_suppression, scale_coords
from yolo.utils.torch_utils import select_device

# weights =  'D:\\cv_samurai_yolo\\yolo\\weights\\best.pt_detailed' # model_yolo_detailed.pt_detailed path(s)
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
model_yolo_detailed = DetectMultiBackend('G:\\Samurai\\_Github\\Train_and_Fight\\yolo\\weights\\best_phases.pt', device=device, dnn=dnn)
model_yolo_detailed.eval()
stride_detailed, names_detailed, pt_detailed, jit_detailed, onnx_detailed, engine_detailed = model_yolo_detailed.stride, model_yolo_detailed.names, model_yolo_detailed.pt, model_yolo_detailed.jit, model_yolo_detailed.onnx, model_yolo_detailed.engine
imgsz = check_img_size(imgsz, s=stride_detailed)  # check image size

# Half
half &= (pt_detailed or jit_detailed or engine_detailed) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
if pt_detailed or jit_detailed:
    model_yolo_detailed.model_yolo_detailed.half() if half else model_yolo_detailed.model.float()
model_yolo_detailed.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup

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
    detailed_states = ['1_u_mid', '1_u_dmg', '1_i_mid', '1_i_dmg', '1_o_mid', '1_o_dmg', '1_k_mid', '1_k_dmg', '1_l_mid', '1_l_dmg',
                       '1_air_sword_light_mid', '1_air_sword_light_dmg', '1_air_sword_heavy_mid', '1_air_sword_heavy_dmg',
                       '1_air_kick_i_mid', '1_air_kick_i_dmg', '1_air_kick_o_mid', '1_air_kick_o_dmg',
                       '1_cres_bgn', '1_cres_dmg', '1_cres_end', '1_earthquake_bgn', '1_earthquake_dmg', '1_earthquake_end',
                       '1_winepot_mid', '1_winepot_dmg', '1_whirlwind_bgn', '1_whirlwind_end',
                       '1_squat_sword_light_mid', '1_squat_sword_light_dmg', '1_squat_sword_heavy_mid', '1_squat_sword_heavy_dmg',
                       '1_squat_kick_light_mid', '1_squat_kick_light_dmg', '1_squat_kick_heavy_mid', '1_squat_kick_heavy_dmg',
                       '1_throw', '1_hit', '1_down', '1_stun', '1_jump', '1_fall', '1_idle', '1_dash', '1_roll',
                       '1_squat', '1_parry', '1_parried', '1_projectile',
                       '2_u_mid', '2_u_dmg', '2_i_mid', '2_i_dmg', '2_o_mid', '2_o_dmg', '2_k_mid', '2_k_dmg', '2_l_mid', '2_l_dmg',
                       '2_air_sword_away_mid', '2_air_sword_away_dmg', '2_air_sword_near_mid', '2_air_sword_near_dmg', '2_air_kick_mid', '2_air_kick_dmg',
                       '2_ghost_bgn', '2_ghost_j_dmg', '2_ghost_k_dmg', '2_ghost_l_dmg', '2_ghost_k_end', '2_ghost_l_end',
                       '2_snow_bgn', '2_snow_dmg', '2_snow_end', '2_swallow_mid', '2_swallow_dmg',
                       '2_squat_away_sword_light_mid', '2_squat_away_sword_light_dmg', '2_squat_away_sword_heavy_mid', '2_squat_away_sword_heavy_dmg',
                       '2_squat_near_sword_light_mid', '2_squat_near_sword_light_dmg', '2_squat_near_sword_heavy_mid', '2_squat_near_sword_heavy_dmg',
                       '2_squat_kick_light_mid', '2_squat_kick_light_dmg', '2_squat_kick_heavy_mid', '2_squat_kick_heavy_dmg',
                       '2_throw', '2_hit', '2_down', '2_stun', '2_jump', '2_fall', '2_idle', '2_dash', '2_roll', 
                       '2_squat', '2_parry', '2_parried', '2_projectile']

    pred_clses, pred_confs, pred_boxes = readScreenShot4States_detailed(cv2.imread("G:\\Datasets\\labelledSamuraiImagesByHands_20220406\\images\\02338.png"))
    for i in range(len(pred_boxes)):
        print(detailed_states[pred_clses[i]], pred_confs[i])
        print(pred_boxes[i])

    