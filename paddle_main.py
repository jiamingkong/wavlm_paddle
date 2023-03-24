"""
启动方法1：使用unilm 实现
"""
from wavlm import WavLM, WavLMConfig

config = WavLMConfig()

model = WavLM(config)

model.eval()

"""
启动方法2：使用transformers实现
"""
# from wavlm_transformers.configuration_wavlm_paddle import WavLMConfig
# from wavlm_transformers.modeling_wavlm_paddle import WavLMModel

# config = WavLMConfig()

# model = WavLMModel(config)


# randomly shoot 1,16000 samples into the model
import paddle

input_ids = paddle.normal(shape=(1, 16000), mean=0, std=1)

# feed the input into the model
with paddle.no_grad():
    output = model.extract_features(input_ids)[0]

    print(f"The output shape is {output.shape}")






with open("paddle-wavlm-base-plus.weight.txt", "w") as f:
    for name, param in model.named_parameters():
        f.write(f"{name} {list(param.shape)} {param.std().item():.3f} {param.mean().item():.3f}\n")

