from .layers.activations import *
from .layers.affine_layers import *
from .layers.noise_injection import *
from .layers.pooling_layers import *
from .layers.normalization_layers import *


noise_layers = {
    'Dropout': DropoutVarPropagationLayer
}

activation_layers = {
    'linear': LinearActivationVarPropagationLayer,
    'relu': ReLUActivationVarPropagationLayer,
    'softmax': SoftmaxActivationVarPropagationLayer
}

affine_layers = {
    'Dense': DenseVarPropagationLayer,
    'Conv2D': Conv2DVarPropagationLayer
}

pooling_layers = {
    'MaxPooling2D': MaxPooling2DVarPropagationLayer,
    'MemorizedMaxPooling2D': MaxPooling2DVarPropagationLayer,
    'UpSampling2D': UpSampling2DVarPropagationLayer,
    'BatchNormalization': BatchnormVarPropagationLayer
}
