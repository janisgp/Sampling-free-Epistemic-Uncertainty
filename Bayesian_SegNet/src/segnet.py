"""An implementation of SegNet (and Bayesian alternative)."""
from keras.applications.vgg16 import VGG16
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
from .layers import LocalContrastNormalization
from .layers import MemorizedMaxPooling2D
from .layers import MemorizedUpsampling2D
from .losses import build_categorical_crossentropy
from .metrics import build_categorical_accuracy


# static arguments used for all convolution layers in SegNet
_CONV = dict(
    kernel_initializer='he_uniform',
    kernel_regularizer=l2(5e-4),
)


def _conv_bn_relu(x, num_filters: int, bn_train: bool=True):
    """
    Append a convolution + batch normalization + ReLu block to an input tensor.

    Args:
        x: the input tensor to append this dense block to
        num_filters: the number of filters in the convolutional layer

    Returns:
        a tensor with convolution + batch normalization + ReLu block added

    """
    x = Conv2D(num_filters, kernel_size=(3, 3), padding='same', **_CONV)(x)
    x = BatchNormalization()(x, training=bn_train)
    x = Activation('relu')(x)
    return x


def _encode(x, nums_filters: list, bn_train: bool=True):
    """
    Append a encoder block with a given size and number of filters.

    Args:
        x: the input tensor to append this encoder block to
        num_filters: a list of the number of filters for each block
        bn_train: sets batchnormalization to training and prediction mode

    Returns:
        a tuple of:
        - a tensor with convolution blocks followed by max pooling
        - the pooling layer to get indexes from for up-sampling

    """
    # loop over the filters list to apply convolution + BN + ReLu blocks
    for num_filters in nums_filters:
        x = _conv_bn_relu(x, num_filters, bn_train=bn_train)
    # create a max pooling layer to keep indexes from for up-sampling later
    pool = MemorizedMaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    # pass the block output through the special pooling layer
    x = pool(x)
    # return the output tensor and reference to pooling layer to get indexes
    return x, pool


def _decode(x, pool: MemorizedMaxPooling2D, nums_filters: list, bn_train: bool=True):
    """
    Append a decoder block with a given size and number of filters.

    Args:
        x: the input tensor to append this decoder block to
        pool: the corresponding memorized pooling layer to reference indexes
        num_filters: a list of the number of filters for each block
        bn_train: sets batchnormalization to training and prediction mode

    Returns:
        a tensor with up-sampling followed by convolution blocks

    """
    # up-sample using the max pooling indexes
    x = MemorizedUpsampling2D(pool=pool)(x)
    # loop over the filters list to apply convolution + BN + ReLu blocks
    for num_filters in nums_filters:
        x = _conv_bn_relu(x, num_filters, bn_train=bn_train)
    return x


def _classify(x, num_classes: int):
    """
    Add a Softmax classification block to an input CNN.

    Args:
        x: the input tensor to append this classification block to (CNN)
        num_classes: the number of classes to predict with Softmax

    Returns:
        a tensor with dense convolution followed by Softmax activation

    """
    # dense convolution (1 x 1) to filter logits for each class
    x = Conv2D(num_classes, kernel_size=(1, 1), padding='valid', **_CONV)(x)
    # Softmax activation to convert the logits to probability vectors
    x = Activation('softmax')(x)
    return x


def _transfer_vgg16_encoder(model: Model) -> None:
    """
    Transfer trained VGG16 weights (ImageNet) to a SegNet encoder.

    Args:
        model: the SegNet model to transfer encoder weights to

    Returns:
        None

    """
    # load the trained VGG16 model using ImageNet weights
    vgg16 = VGG16(include_top=False)
    # extract all the convolutional layers (encoder layers) from VGG16
    vgg16_conv = [layer for layer in vgg16.layers if isinstance(layer, Conv2D)]
    # extract all convolutional layers from SegNet
    model_conv = [layer for layer in model.layers if isinstance(layer, Conv2D)]
    # iterate over the VGG16 layers to replace the SegNet encoder weights
    for idx, layer in enumerate(vgg16_conv):
        model_conv[idx].set_weights(layer.get_weights())


def segnet(image_shape: tuple, num_classes: int,
    class_weights=None,
    lcn: bool=True,
    dropout_rate: float=None,
    optimizer=SGD(lr=0.1, momentum=0.9),
    pretrain_encoder: bool=True,
    bn_train: bool=True
) -> Model:
    """
    Build a SegNet model for the given image shape.

    Args:
        image_shape: the image shape to create the model for
        num_classes: the number of classes to segment for (e.g. c)
        class_weights: the weights for each class
        lcn: whether to use local contrast normalization on inputs
        dropout_rate: the dropout rate to use for permanent dropout
        optimizer: the optimizer for training the network
        pretrain_encoder: whether to initialize the encoder from VGG16
        bn_train: sets batchnormalization to training and prediction mode

    Returns:
        a compiled model of SegNet

    """
    # ensure the image shape is legal for the architecture
    div = int(2**5)
    for dim in image_shape[:-1]:
        # raise error if the dimension doesn't evenly divide
        if dim % div:
            msg = 'dimension ({}) must be divisible by {}'.format(dim, div)
            raise ValueError(msg)
    # the input block of the network
    inputs = Input(image_shape, name='SegNet_input')
    # assume 8-bit inputs and convert to floats in [0,1]
    x = Lambda(lambda x: x / 255.0, name='pixel_norm')(inputs)
    # apply contrast normalization if set
    if lcn:
        x = LocalContrastNormalization()(x)
    
    # encoder
    x, pool_1 = _encode(x, 2 * [64], bn_train=bn_train)
    x, pool_2 = _encode(x, 2 * [128], bn_train=bn_train)
    x, pool_3 = _encode(x, 3 * [256], bn_train=bn_train)
    x = Dropout(dropout_rate)(x)
    x, pool_4 = _encode(x, 3 * [512], bn_train=bn_train)
    x = Dropout(dropout_rate)(x)
    x, pool_5 = _encode(x, 3 * [512], bn_train=bn_train)
    x = Dropout(dropout_rate)(x)
    
    # decoder
    x = _decode(x, pool_5, 3 * [512], bn_train=bn_train)
    x = Dropout(dropout_rate)(x)
    x = _decode(x, pool_4, [512, 512, 256], bn_train=bn_train)
    x = Dropout(dropout_rate)(x)
    x = _decode(x, pool_3, [256, 256, 128], bn_train=bn_train)
    x = Dropout(dropout_rate)(x)
    x = _decode(x, pool_2, [128, 64], bn_train=bn_train)
    x = _decode(x, pool_1, [64], bn_train=bn_train)
    
    # classifier
    x = _classify(x, num_classes)
    # compile the model
    model = Model(inputs=[inputs], outputs=[x], name='SegNet')
    model.compile(
        optimizer=optimizer,
        loss=build_categorical_crossentropy(class_weights),
        metrics=[build_categorical_accuracy(weights=class_weights)],
    )
    # transfer weights from VGG16
    if pretrain_encoder:
        _transfer_vgg16_encoder(model)

    return model


# explicitly define the outward facing API of this module
__all__ = [segnet.__name__]
