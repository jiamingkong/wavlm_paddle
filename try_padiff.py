from padiff import auto_diff
import torch
import paddle
from wavlm import WavLM, WavLMConfig

config = WavLMConfig()
layer = WavLM(config)
layer.eval()

from unilm.WavLM import WavLM, WavLMConfig

# load the pre-trained checkpoints
checkpoint = torch.load('./wavlm-base-plus/WavLM-Base-Official.pt')
cfg = WavLMConfig(checkpoint['cfg'])
module = WavLM(cfg)
# module.load_state_dict(checkpoint['model'])
module.eval()

inp = paddle.rand((100, 100)).numpy().astype("float32")
inp = ({'x': paddle.to_tensor(inp)},
     {'y': torch.as_tensor(inp) })

auto_diff(layer, module, inp, auto_weights=True, options={'atol': 1e-4, 'rtol':0, 'compare_mode': 'strict', 'single_step':False})