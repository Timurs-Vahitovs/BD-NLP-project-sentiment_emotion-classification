# check_gpu.py
import torch

print(f"CUDA pieejams : {torch.cuda.is_available()}")
print(f"GPU skaits : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU nosaukums : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"PyTorch versija : {torch.__version__}")