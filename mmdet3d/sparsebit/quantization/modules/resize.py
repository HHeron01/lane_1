import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.sparsebit.quantization.modules import QuantOpr, register_qmodule


@register_qmodule(sources=[nn.Upsample])
class QUpsample(QuantOpr):
    def __init__(self, org_module=None, config=None):
        super(QUpsample, self).__init__()
        self.scale_factor = org_module.scale_factor
        self.size = org_module.size
        self.mode = org_module.mode
        self._repr_info = "QUpsample, mode: {} ".format(self.mode)
        # print("QUpsample:", org_module)

    def build_quantizer(self, config):
        """
        force the bit of resize oprs is 8bit
        """
        QuantOpr.build_quantizer(self, config)
        if self.mode == "nearest":
            self.input_quantizer.set_fake_fused()
            # self.input_quantizer.set_bit(bit=8)
        else:
            self.input_quantizer.set_bit(bit=8)

    def forward(self, x_in, *args):
        x_in = self.input_quantizer(x_in)
        # try:
        out = F.interpolate(x_in, size=self.size, scale_factor=self.scale_factor, mode=self.mode)
        # except:
        #     print("plz fix interp input!!!")
        return out


@register_qmodule(sources=[F.interpolate])
class QInterpolate(QuantOpr):  # hack
    def __init__(self, org_module=None, config=None):
        super(QInterpolate, self).__init__()
        if isinstance(org_module, nn.Module):
            raise NotImplementedError
        else:
            self.mode = org_module.kwargs["mode"]
        self._repr_info = "QInterpolate, mode: {} ".format(self.mode)

    def build_quantizer(self, config):
        """
        force the bit of resize oprs is 8bit
        """
        QuantOpr.build_quantizer(self, config)
        if self.mode == "nearest":
            self.input_quantizer.set_fake_fused()
            # self.input_quantizer.set_bit(bit=8)
        else:
            self.input_quantizer.set_bit(bit=8)

    def forward(self, x_in, *args, **kwargs):
        x_in = self.input_quantizer(x_in)
        out = F.interpolate(x_in, *args, **kwargs)
        return out
