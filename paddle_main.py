"""
启动方法1：使用unilm 实现
"""
from wavlm import WavLM, WavLMConfig
import paddle
config = WavLMConfig()

model = WavLM(config)
model.load_dict(paddle.load("wavlm-base-plus/wavlm-base-paddle.pdparams"))
model.eval()


# randomly shoot 1,16000 samples into the model
import paddle

input_ids = paddle.normal(shape=(1, 16000), mean=0, std=1)

# feed the input into the model
with paddle.no_grad():
    output = model.extract_features(input_ids)[0]
    print(output)
    print(f"The output shape is {output.shape}")






with open("paddle-wavlm-base-plus.weight.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name} {list(param.shape)} {param.std().item():.3f} {param.mean().item():.3f}\n")

