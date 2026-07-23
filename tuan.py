import torch
import inspect
import torch.nn as nn

"""
proj = nn.Conv2d(3, 48, kernel_size=7, padding=3, stride=4)
unfold = nn.Unfold(kernel_size=4, stride=4)
image = torch.randn(1, 3, 224, 224)

out_proj = proj(image)
# out_proj.shape # torch.Size([1, 48, 56, 56])
print(out_proj.shape)

out_unfold = unfold(out_proj)
# out_unfold.shape # torch.Size([1, 768, 196])
print(out_unfold.shape)
# =============
x = torch.randn(10)
print(x)

proj_drop = nn.Dropout(p=0.1)
proj_drop_inplace = nn.Dropout(p=0.1, inplace=True)

x = proj_drop_inplace(x)
print(x)
"""

# print(inspect.getsource(torch.nn.MultiheadAttention))
print(inspect.getfile(torch.nn.MultiheadAttention))
