python inference_webcam.py \
    --model-backbone-scale 0.25 \
    --model-type mattingrefine \
    --model-backbone resnet50 \
    --model-checkpoint pytorch_resnet50.pth \
    --camera-device 0 \
    --resolution 1280 720 \
    --crop-width 0.75
