# for each layer of the pretrained wavlm weight, we want to write out their name, shape, std, and mean

import torch
import numpy as np

from transformers import WavLMModel

model = WavLMModel.from_pretrained("wavlm-base-plus", local_files_only=True)

with open("huggingface-wavlm-base-plus.weight.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name} {list(param.shape)} {param.std().item():.3f} {param.mean().item():.3f}\n")

