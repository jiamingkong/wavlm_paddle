# q.shape:  [1, 12, 49, 64] k.shape:  [1, 12, 49, 64] v.shape:  [1, 12, 49, 64] attn_mask.shape:  [1, 12, 49, 49]

import paddle
import torch
import numpy as np
from wavlm.functional import scaled_dot_product_attention
from torch.nn.functional import scaled_dot_product_attention as torch_scaled_dot_product_attention

# use numpy to initiate the input
q = np.random.rand(1, 12, 49, 64).astype("float32")
k = np.random.rand(1, 12, 49, 64).astype("float32")
v = np.random.rand(1, 12, 49, 64).astype("float32")
attn_mask = np.zeros((1, 12, 49, 49)).astype("float32")

# convert to paddle tensor
paddle_q = paddle.to_tensor(q)
paddle_k = paddle.to_tensor(k)
paddle_v = paddle.to_tensor(v)
paddle_attn_mask = paddle.to_tensor(attn_mask)

# convert to torch tensor
torch_q = torch.as_tensor(q)
torch_k = torch.as_tensor(k)
torch_v = torch.as_tensor(v)
torch_attn_mask = torch.as_tensor(attn_mask)

# use paddle to calculate the output
paddle_out = scaled_dot_product_attention(paddle_q, paddle_k, paddle_v, attn_mask=paddle_attn_mask, dropout_p=0, is_causal=False)
paddle_out = paddle_out.numpy()

# use torch to calculate the output
torch_out = torch_scaled_dot_product_attention(torch_q, torch_k, torch_v, attn_mask=torch_attn_mask, dropout_p=0, is_causal=False)

# compare the output
print("paddle_out.shape: ", paddle_out.shape)
print("torch_out.shape: ", torch_out.shape)
print("paddle_out: ", paddle_out)
print("torch_out: ", torch_out)

print("paddle_out == torch_out: ", np.allclose(paddle_out, torch_out, atol=1e-1, rtol=1e-2))