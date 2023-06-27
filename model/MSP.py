from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
import torch
import torch.nn.functional as F
import torch.nn as nn
LRELU_SLOPE = 0.1

def get_padding(kernel_size, dilation=1):
     return int((kernel_size*dilation - dilation)/2)

class DiscriminatorP(torch.nn.Module):
     def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
          super(DiscriminatorP, self).__init__()
          self.period = period
          norm_f = weight_norm if use_spectral_norm == False else spectral_norm
          self.convs = nn.ModuleList([
               norm_f(Conv2d(1, 16, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
               norm_f(Conv2d(16, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
               norm_f(Conv2d(32, 64, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
               norm_f(Conv2d(64, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0)))
          ])

     def forward(self, x):

          # 1d to 2d
          b, c, t = x.shape
          if t % self.period != 0: # pad first
               n_pad = self.period - (t % self.period)
               x = F.pad(x, (0, n_pad), "reflect")
               t = t + n_pad
          x = x.view(b, c, t // self.period, self.period)

          for l in self.convs:
               x = l(x)
               x = F.leaky_relu(x, LRELU_SLOPE)
          x = torch.flatten(x, 2, -1)
          return x


class MultiPeriodDiscriminator(torch.nn.Module):
     def __init__(self):
          super(MultiPeriodDiscriminator, self).__init__()
          self.discriminators = nn.ModuleList([
               DiscriminatorP(3),
               DiscriminatorP(5),
               DiscriminatorP(7)
          ])

     def forward(self, y):
          y_d_rs = []
          for i, d in enumerate(self.discriminators):
               y_d_r = d(y)
               y_d_rs.append(y_d_r)

          return y_d_rs


class DiscriminatorS(torch.nn.Module):
     def __init__(self, use_spectral_norm=False):
          super(DiscriminatorS, self).__init__()
          norm_f = weight_norm if use_spectral_norm == False else spectral_norm
          self.convs = nn.ModuleList([
               norm_f(Conv1d(1, 8, 17, 1, padding=8)),
               norm_f(Conv1d(8, 16, 11, 2, groups=4, padding=5)),
               norm_f(Conv1d(16, 32, 7, 2, groups=4, padding=3)),
               norm_f(Conv1d(32, 64, 5, 2, groups=16, padding=2)),
               norm_f(Conv1d(64, 128, 3, 4, groups=16, padding=1)),
          ])

     def forward(self, x):
          for l in self.convs:
               x = l(x)
               x = F.leaky_relu(x, LRELU_SLOPE)
          return x


class MultiScaleDiscriminator(torch.nn.Module):
     def __init__(self):
          super(MultiScaleDiscriminator, self).__init__()
          self.discriminators = nn.ModuleList([
               DiscriminatorS(use_spectral_norm=True),
               DiscriminatorS(),
               DiscriminatorS(),
          ])
          self.meanpools = nn.ModuleList([
               AvgPool1d(4, 2, padding=2),
               AvgPool1d(4, 2, padding=2)
          ])

     def forward(self, y):
          y_d_rs = []
          for i, d in enumerate(self.discriminators):
               if i != 0:
                    y = self.meanpools[i-1](y)
               y_d_r = d(y)
               y_d_rs.append(y_d_r)

          return y_d_rs

class MultiScalePeriod(torch.nn.Module):
     def __init__(self):
          super(MultiScalePeriod, self).__init__()
          self.msd = MultiScaleDiscriminator()
          self.mpd = MultiPeriodDiscriminator()
          
          self.pos_feat = nn.Linear(286,80)

     def forward(self, x):
          y = self.msd(x) + self.mpd(x)
          y = torch.cat(y, dim=2)
          y = self.pos_feat(y)
          return y
          
if __name__ == '__main__':

     # Generate random input tensor
     input_tensor = torch.randn(16, 1, 3000)

     msp = MultiScalePeriod()
     
     y = msp(input_tensor)
     
     print(y.shape)