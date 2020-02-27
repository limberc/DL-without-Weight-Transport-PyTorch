import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function


class KPConvFunction(Function):
    def __init__(self, stride, padding, dilation, groups):
        __class__.stride = stride
        __class__.padding = padding
        __class__.dilation = dilation
        __class__.groups = groups

    @staticmethod
    # same as reference linear function, but with additional fa tensor for backward
    def forward(ctx, input, weight, weight_fa, bias=None):
        ctx.save_for_backward(input, weight, weight_fa, bias)
        output = F.conv2d(input, weight=weight, bias=bias, stride=__class__.stride, padding=__class__.padding,
                          dilation=__class__.dilation, groups=__class__.groups)
        return output

    def backward(ctx, grad_output):
        input, weight, weight_fa, bias = ctx.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = F.grad.conv2d_input(input_size=input.shape, weight=weight_fa, grad_output=grad_output,
                                             stride=__class__.stride, padding=__class__.padding,
                                             dilation=__class__.dilation, groups=__class__.groups)
        if ctx.needs_input_grad[1]:
            grad_weight = F.grad.conv2d_weight(input=input, weight_size=weight_fa.shape, grad_output=grad_output,
                                               stride=__class__.stride, padding=__class__.padding,
                                               groups=__class__.groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)
        # Update the backward matrices of the Kolen-Pollack algorithm
        grad_weight_fa = grad_weight

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class KPConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True):
        super(KPConv, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding,
            dilation, groups, bias)
        self.weight_fa = nn.Parameter(torch.rand(self.weight.shape).to(self.weight.device),
                                      requires_grad=False)
        # weight initialization
        torch.nn.init.kaiming_uniform_(self.weight)
        torch.nn.init.kaiming_uniform_(self.weight_fa)
        if bias:
            torch.nn.init.constant_(self.bias, 1)

    def forward(self, input):
        output = KPConvFunction(self.stride, self.padding, self.dilation, self.groups).apply(input, self.weight,
                                                                                             self.weight_fa,
                                                                                             self.bias)
        return output
