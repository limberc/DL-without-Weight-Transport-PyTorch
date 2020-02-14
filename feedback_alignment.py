import torch
from torch import nn
from torch.autograd import Function


class LinearFunction(Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_variables
        grad_input = grad_weight = grad_bias = None
        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class LinearFAFunction(Function):
    @staticmethod
    # Same as reference linear function, but with additional feed forward tensor for backward
    def forward(ctx, input, weight, weight_fa, bias=None):
        ctx.save_for_backward(input, weight, weight_fa, bias)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, weight_fa, bias = ctx.saved_variables
        grad_input = grad_weight = grad_weight_fa = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Calculate the gradient of input with fixed feedback alignment tensor,
            # rather than the "correct" model weight
            grad_input = grad_output.mm(weight_fa)
        if ctx.needs_input_grad[1]:
            # grad for weight with FA'ed grad_output from downstream layer
            # it is same with original linear function
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_weight_fa, grad_bias


class LinearFeedbackAlignmentModule(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(LinearFeedbackAlignmentModule, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # weight and bias for forward pass
        # weight has transposed form; more efficient (so i heard) (transposed at forward pass)
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_parameter('bias', None)

        # fixed random weight and bias for FA backward pass
        # does not need gradient
        self.weight_fa = torch.FloatTensor(output_features, input_features)

        # weight initialization
        torch.nn.init.kaiming_uniform(self.weight)
        torch.nn.init.kaiming_uniform(self.weight_fa)
        torch.nn.init.constant(self.bias, 1)

    def forward(self, input):
        return LinearFAFunction.apply(input, self.weight, self.weight_fa, self.bias)


class LinearFeedbackAlignment(nn.Module):
    """
    Linear feed-forward networks with feedback alignment learning
    Does NOT perform non-linear activation after each layer
    """

    def __init__(self, in_features, num_layers, num_hidden_list):
        """
        :param in_features: dimension of input features (784 for MNIST)
        :param num_layers: number of layers for feed-forward net
        :param num_hidden_list: list of integers indicating hidden nodes of each layer
        """
        super(LinearFeedbackAlignment, self).__init__()
        self.in_features = in_features
        self.num_layers = num_layers
        self.num_hidden_list = num_hidden_list

        # create list of linear layers
        # first hidden layer
        self.linear = [LinearFeedbackAlignmentModule(self.in_features, self.num_hidden_list[0])]
        # append additional hidden layers to list
        for idx in range(self.num_layers - 1):
            self.linear.append(LinearFeedbackAlignmentModule(self.num_hidden_list[idx], self.num_hidden_list[idx + 1]))

        # create ModuleList to make list of layers work
        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        """
        forward pass, which is same for conventional feed-forward net
        :param inputs: inputs with shape [batch_size, in_features]
        :return: logit outputs from the network
        """
        # first layer
        linear1 = self.linear[0](inputs)
        # second layer
        linear2 = self.linear[1](linear1)
        return linear2
