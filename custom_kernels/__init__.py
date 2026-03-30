import torch

# Enable cuDNN benchmark globally for all custom kernels.
# Since our models use fixed input sizes (e.g. 128x128, 512x512),
# this allows cuDNN to heuristically select the fastest convolution
# algorithm on the first run, significantly speeding up subsequent inferences.
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
