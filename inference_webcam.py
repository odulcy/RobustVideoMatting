"""
Inference on webcams: Use a model on webcam input.

Once launched, the script is in background collection mode.
Press B to toggle between background capture mode and matting mode. The frame shown when B is pressed is used as background for matting.
Press Q to exit.

Example:

    python inference_webcam.py \
        --model-type mattingrefine \
        --model-backbone resnet50 \
        --model-checkpoint "PATH_TO_CHECKPOINT" \
        --resolution 1280 720

"""

import argparse, os, shutil, time
import cv2
import torch

from threading import Thread, Lock
from tqdm import tqdm
from PIL import Image

from model import MattingNetwork
from utils import Displayer

# --------------- Arguments ---------------


parser = argparse.ArgumentParser(description='Inference from web-cam')

parser.add_argument('--model-type', type=str, required=True, choices=['mattingbase', 'mattingrefine'])
parser.add_argument('--model-backbone', type=str, required=True, choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=True)
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.7)

parser.add_argument('--precision', type=str, default='float32', choices=['float32', 'float16'])
parser.add_argument('--backend', type=str, default='pytorch', choices=['pytorch', 'torchscript'])
parser.add_argument('--hide-fps', action='store_true')
parser.add_argument('--resolution', type=int, nargs=2, metavar=('width', 'height'), default=(1280, 720))
parser.add_argument('--camera-device', type=int, default=0)
parser.add_argument('--background-color', type=float, nargs=3, metavar=('red', 'green', 'blue'), default=(0,1,0))
parser.add_argument('--crop-width', type=float, default=1.0)
parser.add_argument('--downsample-ratio', type=float, default=None)

args = parser.parse_args()


# ----------- Utility classes -------------


# A wrapper that reads data from cv2.VideoCapture in its own thread to optimize.
# Use .read() in a tight loop to get the newest frame
class Camera:
    def __init__(self, device_id=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(device_id)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.success_reading, self.frame = self.capture.read()
        self.read_lock = Lock()
        self.thread = Thread(target=self.__update, args=())
        self.thread.daemon = True
        self.thread.start()

    def __update(self):
        while self.success_reading:
            grabbed, frame = self.capture.read()
            with self.read_lock:
                self.success_reading = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return frame

    def __exit__(self, exec_type, exc_value, traceback):
        self.capture.release()


# --------------- Main ---------------


## Load model
#if args.precision == 'float32':
#    precision = torch.float32
#else:
#    precision = torch.float16
#
#if args.backend == 'torchscript':
#    model = torch.jit.load(args.model_checkpoint)
#    model.model_backbone = args.model_backbone
#    model.backbone_scale = args.model_backbone_scale
#    model.refine_mode = args.model_refine_mode
#    model.refine_sample_pixels = args.model_refine_sample_pixels
#    model.refine_threshold = args.model_refine_threshold
#else:
#    if args.model_type == 'mattingbase':
#        model = MattingBase(args.model_backbone)
#    if args.model_type == 'mattingrefine':
#        model = MattingRefine(
#            args.model_backbone,
#            args.model_backbone_scale,
#            args.model_refine_mode,
#            args.model_refine_sample_pixels,
#            args.model_refine_threshold)
#
#    model.load_state_dict(torch.load(args.model_checkpoint), strict=False)

model = MattingNetwork('mobilenetv3').cuda().eval()
model.load_state_dict(torch.load('rvm_mobilenetv3.pth'))

r,g,b = args.background_color
width, height = args.resolution
cam = Camera(device_id=args.camera_device, width=width, height=height)

crop_width = args.crop_width
middle = int(cam.width // 2)
crop_start = middle - int((cam.width * crop_width)//2)
crop_end = middle + int((cam.width * crop_width)//2)

dsp = Displayer('MattingV2', int(cam.width * crop_width), cam.height, show_info=(not args.hide_fps), virtual_webcam=True)

rec = [None] * 4                                      # Initial recurrent states.
downsample_ratio = args.downsample_ratio              # Adjust based on your video.

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that
    the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)


def cv2_frame_to_cuda(frame):
    """Frame as float64"""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(frame)[:,crop_start:crop_end,:]
    return img.cuda().div(255).permute((2, 0, 1)).unsqueeze_(0)


with torch.no_grad():
    while True:
        bgr = None
        colored_bgr = None
        while True: # grab bgr
            frame = cam.read()
            key = dsp.step(frame[:,crop_start:crop_end,:])
            if key == ord('b'):
                bgr = cv2_frame_to_cuda(cam.read())
                colored_bgr = torch.zeros_like(bgr).cuda()
                colored_bgr[:,0,:,:] = r
                colored_bgr[:,1,:,:] = g
                colored_bgr[:,2,:,:] = b
                break
            if key == ord('f'):
                dsp.show_info = not dsp.show_info
            elif key == ord('q'):
                exit()
        while True: # matting
            frame = cam.read()
            src = cv2_frame_to_cuda(frame)
            if downsample_ratio is None:
                downsample_ratio = auto_downsample_ratio(*src.shape[2:])
            fgr, pha, *rec = model(src, *rec, downsample_ratio)
            res = pha * fgr + (1 - pha) * colored_bgr
            res = res.mul(255).byte().permute(0, 2, 3, 1).cpu().numpy()[0]
            res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
            key = dsp.step(res)
            if key == ord('b'):
                break
            if key == ord('f'):
                dsp.show_info = not dsp.show_info
            elif key == ord('q'):
                exit()
