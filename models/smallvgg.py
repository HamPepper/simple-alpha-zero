import sys
import torch
import torch.nn.functional as F
import numpy as np
sys.path.append("..")
from model import Model


class SmallVGG(Model):

  def __init__(self, input_shape, output_shape):
    super(SmallVGG, self).__init__(input_shape, output_shape)
    num_hidden_units = 768
    self.input_shape = input_shape
    self.output_shape = output_shape
    self.conv1 = torch.nn.Conv2d(input_shape[-1], 64, 4)
    self.conv2 = torch.nn.Conv2d(64, 128, 2)
    self.p_head = torch.nn.Linear(num_hidden_units, np.prod(output_shape))
    self.v_head = torch.nn.Linear(num_hidden_units, 1)

  def forward(self, x):
    batch_size = len(x)
    this_output_shape = tuple([batch_size] + list(self.output_shape))
    x = x.permute(0,3,1,2) # NHWC -> NCHW

    # Network
    relu_conv1 = F.relu(self.conv1(x))
    relu_conv2 = F.relu(self.conv2(relu_conv1))
    flat = torch.flatten(relu_conv2, start_dim=1)

    # Outputs
    p_logits = self.p_head(flat).view(this_output_shape)
    v = torch.tanh(self.v_head(flat))

    return p_logits, v
