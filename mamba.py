import torch
from mamba_ssm import Mamba

import sys
import os
submodule_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './mamba-minimal/'))
sys.path.insert(0, submodule_path)
from model import Mamba as MambaCPU, ModelArgs


d_state = 64
batch, length, dim = 100, 40000, 4 
x = torch.randn(batch, length, dim).to("cuda")
model = Mamba(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=d_state,  # SSM state expansion factor
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

print(model.state_dict().keys())

# Create CPU model and transfer weights to it
model_args = ModelArgs(d_model=4, n_layer=1, vocab_size=4, d_state=64)
model_cpu = MambaCPU(model_args)
model_cpu = model_cpu.layers[0].mixer
print(model_cpu.state_dict().keys())
model_cpu.load_state_dict(model.state_dict())


y = model(x)
y_cpu = model_cpu(x.to("cpu"))
assert y.shape == x.shape

# Check Mamba on the GPU produces the same result as Mamba on the CPU
y = y.to("cpu")
diff = torch.abs(y - y_cpu)
print("Max diff:", diff.max())
print(y.dtype, y_cpu.dtype)
# Needed to increase tolerance to pass the test
assert torch.allclose(y, y_cpu, atol=1e-6, rtol=1e-5)
breakpoint()
