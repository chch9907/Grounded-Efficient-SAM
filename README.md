# Grounded-Efficient-SAM

This is the minimal code for Grounded-EfficientSAM forked from [Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything/tree/main/GroundingDINO).

To run the code, first download pre-trained model weights:

Download weights for GroundingDINO
```python
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

```
Download weights for EffcientSAM
```python
wget https://huggingface.co/spaces/yunyangx/EfficientSAM/blob/main/efficientsam_s_gpu.jit

```

Then follow the README.md in GroundingDINO to install groundingdino.