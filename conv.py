import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from torch.nn import init
from torch.nn.modules.utils import _pair


class FAConvFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, weight_fa, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, weight_fa, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        output = F.conv2d(input, weight=weight, bias=bias, stride=stride, padding=padding,
                          dilation=dilation, groups=groups)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_weight_fa = grad_bias = None
        if ctx.needs_input_grad[0]:
            # calculate the gradient of input with fixed fa tensor,
            # rather than the "correct" model weight
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight_fa, grad_output, stride, padding, dilation,
                                                    groups)
        if ctx.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = torch.nn.grad.conv2d_weight(input, weight_fa.shape, grad_output, stride, padding, dilation,
                                                      groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias, None, None, None, None


class KPConvFunction(FAConvFunction):
    @staticmethod
    def backward(ctx, grad_output):
        grad_input, grad_weight, grad_weight_fa, grad_bias, stride, padding, dilation, groups = FAConvFunction.backward(
            ctx, grad_output)
        grad_weight_fa = grad_weight

        return grad_input, grad_weight, grad_weight_fa, grad_bias, None, None, None, None


class _FAConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, check_grad=False):
        super(_FAConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode

        if check_grad:
            tensor_constructor = torch.DoubleTensor  # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor  # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            self.weight = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
            self.weight_fa = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
            self.weight_fa = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)


class FAConv2d(_FAConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(FAConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    # @weak_script_method
    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            return FAConvFunction.apply(F.pad(input, expanded_padding, mode='circular'),
                                        self.weight, self.weight_fa, self.bias, self.stride,
                                        _pair(0), self.dilation, self.groups)
        else:
            return FAConvFunction.apply(input, self.weight, self.weight_fa, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)


class KPConv2d(FAConv2d):
    def forward(self, input):
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            return KPConvFunction.apply(F.pad(input, expanded_padding, mode='circular'),
                                        self.weight, self.weight_fa, self.bias, self.stride,
                                        _pair(0), self.dilation, self.groups)
        else:
            return KPConvFunction.apply(input, self.weight, self.weight_fa, self.bias, self.stride,
                                        self.padding, self.dilation, self.groups)
