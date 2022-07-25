# Inherit from Function
from torch.autograd import Function
import torch
from torch import nn
from torch.nn import functional as F
from math import log2, ceil

def get_dynamic_scale(x, bits, with_grad=False):
    """Calculate dynamic scale for quantization from input by taking the
    maximum absolute value from x and number of bits"""
    with torch.set_grad_enabled(with_grad):
        rex = x.reshape(-1,)
        k = len(rex)
        threshold = torch.topk(rex.abs(), int(0.95*k), largest=False)[0].max()
    return get_scale(bits, threshold)

def get_scale(bits, threshold):
    """Calculate scale for quantization according to some constant and number of bits"""
    try:
        return ceil(log2(threshold))
    except ValueError:
        return 0 
    
    
def quantize(input, bits=16): # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2**(bits-1)-1
    
    scale = 2 ** (bits - get_dynamic_scale(input, bits) - 1)
    #import pdb; pdb.set_trace()
    return input.mul(scale).round().clamp(-thresh, thresh).div(scale)

class LinearFunction(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, weight, bias=None):
        input = quantize(input)
        weight = quantize(weight)
        if bias is not None:
            bias = quantize(bias)
        ctx.save_for_backward(input, weight, bias)
        
        output = input.matmul(weight.t())
        
        #  import pdb; pdb.set_trace()
        return output

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        #import pdb; pdb.set_trace()
        grad_output = quantize(grad_output)
        

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        #import pdb; pdb.set_trace()
        grad_input = grad_output.matmul(weight)
        grad_weight = grad_output.permute(0,2,1).matmul(input)
        
        if bias is not None:
            grad_bias = grad_output.sum(0)
        #import pdb; pdb.set_trace()
        
        return grad_input, grad_weight, grad_bias


class qLinear(nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(qLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.empty(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        # Not a very smart way to initialize weights
        nn.init.uniform_(self.weight, -0.1, 0.1)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -0.1, 0.1)

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )
