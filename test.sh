# python inference.py \
#     --variant resnet50 \
#     --checkpoint "models/rvm_resnet50.pth" \
#     --device cuda \
#     --input-source "videos/jensen.mp4" \
#     --output-type video \
#     --output-composition "output/composition.mp4" \
#     --output-alpha "output/alpha.mp4" \
#     --output-foreground "output/foreground.mp4" \
#     --output-video-mbps 4 \
#     --seq-chunk 1

python inference.py \
    --variant mobilenetv3 \
    --checkpoint "models/rvm_mobilenetv3.pth" \
    --device cuda \
    --input-source "videos/microsoft.mp4" \
    --output-type video \
    --output-composition "output/composition.mp4" \
    --output-alpha "output/alpha.mp4" \
    --output-foreground "output/foreground.mp4" \
    --output-video-mbps 4 \
    --seq-chunk 1
