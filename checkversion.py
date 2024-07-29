import torch;
print("CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_properties(0))
print("TORCH: ", torch.__version__)
