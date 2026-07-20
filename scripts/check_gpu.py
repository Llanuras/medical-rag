import torch


print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("GPU name:", torch.cuda.get_device_name(0))
    print("GPU memory GB:", torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024)
else:
    print("No GPU detected. This is normal in no-GPU mode.")
