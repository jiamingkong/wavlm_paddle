from transformers import WavLMModel, Wav2Vec2Processor
import torch
from datasets import load_dataset

# dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
# dataset = dataset.sort("id")
# sampling_rate = dataset.features["audio"].sampling_rate

# processor = Wav2Vec2Processor.from_pretrained("wavlm-base-plus", local_files_only=True)
model = WavLMModel.from_pretrained("wavlm-base-plus", local_files_only=True)

# audio file is decoded on the fly
# inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
# inputs = model._get_feat_extract_input(dataset[0]["audio"]["array"], sampling_rate=sampling_rate)

inputs = {
    # "input_values": torch.tensor(dataset[0]["audio"]["array"]).unsqueeze(0),
    "input_values": torch.rand(1, 16000)
}
with torch.no_grad():
    outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
print(list(last_hidden_states.shape)) # [1,292,768]