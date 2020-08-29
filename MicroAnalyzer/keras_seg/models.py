from enum import Enum as __Enum
from typing import Callable as __Callable, Union as __Union

import yaml
import segmentation_models as __sm
from keras.layers import Input as __Input, Conv2D as __Conv2D
from keras.models import Model as __Model

# the number of channels required in the images in order to use the
# pretrained imagenet weights
SUPPORTED_IMAGENET_CHANNELS = 3

# the supported activations for the output layer of a model
SUPPORTED_ACTIVATIONS = ['sigmoid', 'softmax']

# an enum containing all the supported model backbones
Backbone = __Enum('Backbone', {b.upper(): b for b in __sm.Backbones.models})


def get_model(config_file_path: str) -> __Model:
    with open(config_file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)  # read config file

    model = globals()[config.pop('model')]  # get model from name

    return model(**config)


def UNet(backbone: Backbone,
         num_channels: int,
         num_classes=1,
         final_activation: str = 'sigmoid',
         use_pretrained_imagenet_weights: bool = False) -> __Model:
    """
    build a UNet segmentation model
    :param final_activation:
    :param backbone: the backbone with which to initialize the base model
    :param num_channels: the number of channels in the input images
    :param num_classes: the number of possible classifications per pixel
    :param use_pretrained_imagenet_weights: if True, the model will use pretrained weights specific to the given
                                            backbone.
                                            NOTE: images must have SUPPORTED_IMAGENET_CHANNELS channels. otherwise
                                            they are transformed to have that number of channels internally.
    :param base_model: a callable that expects a backbone_name (str) an input shape (tuple) and encoder_weights
                       ('imagenet' str or None)
    :return: a compiled keras Model object
    """
    model = __build_segmentation_model(backbone, num_channels, num_classes, final_activation,
                                       use_pretrained_imagenet_weights, __sm.Unet)
    model.name = UNet.__name__
    return model


# TODO add input_shape param instead of num_channels. PSPNet requires full shape without None
# def PSPNet(backbone: Backbone,
#            num_channels: int,
#            num_classes: int = 1,
#            use_pretrained_imagenet_weights: bool = False) -> __Model:
#     """
#     build a Pyramid Scene Parsing Network segmentation model
#     :param backbone: the backbone with which to initialize the base model
#     :param num_channels: the number of channels in the input images
#     :param num_classes: the number of possible classifications per pixel
#     :param use_pretrained_imagenet_weights: if True, the model will use pretrained weights specific to the given
#                                             backbone.
#                                             NOTE: images must have SUPPORTED_IMAGENET_CHANNELS channels. otherwise
#                                             they are transformed to have that number of channels internally.
#     :param base_model: a callable that expects a backbone_name (str) an input shape (tuple) and encoder_weights
#                        ('imagenet' str or None)
#     :return: a compiled keras Model object
#     """
#     model = __build_segmentation_model(backbone, num_channels, num_classes, use_pretrained_imagenet_weights,
#                                        __sm.PSPNet)
#     model.name = PSPNet.__name__
#     return model


def LinkNet(backbone: Backbone,
            num_channels: int,
            num_classes: int = 1,
            final_activation: str = 'sigmoid',
            use_pretrained_imagenet_weights: bool = False) -> __Model:
    """
    build a LinkNet segmentation model
    :param final_activation:
    :param backbone: the backbone with which to initialize the base model
    :param num_channels: the number of channels in the input images
    :param num_classes: the number of possible classifications per pixel
    :param use_pretrained_imagenet_weights: if True, the model will use pretrained weights specific to the given
                                            backbone.
                                            NOTE: images must have SUPPORTED_IMAGENET_CHANNELS channels. otherwise
                                            they are transformed to have that number of channels internally.
    :param base_model: a callable that expects a backbone_name (str) an input shape (tuple) and encoder_weights
                       ('imagenet' str or None)
    :return: a compiled keras Model object
    """
    model = __build_segmentation_model(backbone, num_channels, num_classes, final_activation,
                                       use_pretrained_imagenet_weights, __sm.Linknet)
    model.name = LinkNet.__name__
    return model


def FPN(backbone: Backbone,
        num_channels: int,
        num_classes: int = 1,
        final_activation: str = 'sigmoid',
        use_pretrained_imagenet_weights: bool = False,
        dropout: float = None) -> __Model:
    """
    build a Feature Pyramid Network segmentation model
    :param final_activation:
    :param backbone: the backbone with which to initialize the base model
    :param num_channels: the number of channels in the input images
    :param num_classes: the number of possible classifications per pixel
    :param use_pretrained_imagenet_weights: if True, the model will use pretrained weights specific to the given
                                            backbone.
                                            NOTE: images must have SUPPORTED_IMAGENET_CHANNELS channels. otherwise
                                            they are transformed to have that number of channels internally.
    :param base_model: a callable that expects a backbone_name (str) an input shape (tuple) and encoder_weights
                       ('imagenet' str or None)
    :return: a compiled keras Model object
    """
    model = __build_segmentation_model(backbone, num_channels, num_classes, final_activation,
                                       use_pretrained_imagenet_weights, __sm.FPN, pyramid_dropout=dropout)
    model.name = FPN.__name__
    return model


def __build_segmentation_model(backbone: __Union[Backbone, str], num_channels: int, num_classes: int,
                               final_activation: str, use_pretrained_imagenet_weights: bool,
                               base_model: __Callable[..., __Model], **base_model_args) -> __Model:
    """
    build a model based on the given base_model and backbones
    :param backbone: the backbone with which to initialize the base model
    :param num_channels: the number of channels in the input images
    :param num_classes: the number of possible classifications per pixel
    :param final_activation: the activation for output layer ('softams' or 'sigmoid').
    :param use_pretrained_imagenet_weights: if True, the model will use pretrained weights specific to the given
                                            backbone.
                                            NOTE: images must have SUPPORTED_IMAGENET_CHANNELS channels. otherwise
                                            they are transformed to have that number of channels internally.
    :param base_model: a callable that expects a backbone_name (str) an input shape (tuple) and encoder_weights
                       ('imagenet' str or None)
    :return: a compiled keras Model object
    """
    assert final_activation in SUPPORTED_ACTIVATIONS, (f'unsupported activation: {final_activation}. final activaiton '
                                                       f'must be one of: {SUPPORTED_ACTIVATIONS}')

    if isinstance(backbone, Backbone):
        backbone = backbone.value

    assert isinstance(backbone, str), 'backbone must be a string or a Backbone enum'

    if backbone in [Backbone.VGG16, Backbone.VGG19]:
        assert num_channels == 3, 'VGG16 and VGG19 only support RGB images'

    if use_pretrained_imagenet_weights:
        # imagenet pretrained weights are only available for RGB format.
        # we must map the input to the RGB format shape in order to use them.

        # get base model with RGB input
        model = base_model(backbone_name=backbone,
                           input_shape=(None, None, SUPPORTED_IMAGENET_CHANNELS),
                           encoder_weights='imagenet',
                           classes=num_classes,
                           activation=final_activation,
                           **base_model_args)

        if num_channels != SUPPORTED_IMAGENET_CHANNELS:
            # start model by mapping the image to RGB channels
            inp = __Input(shape=(None, None, num_channels))
            rgb_mapping_layer = __Conv2D(SUPPORTED_IMAGENET_CHANNELS, (1, 1))(inp)

            # apply base_model on rgb mapping layer
            out = model(rgb_mapping_layer)

            # construct model
            model = __Model(inp, out, name=model.name)
    else:
        model = base_model(backbone_name=backbone,
                           input_shape=(None, None, num_channels),
                           encoder_weights=None,
                           classes=num_classes,
                           activation=final_activation,
                           **base_model_args)

    # add backbone value to model
    model.backbone = Backbone[backbone.upper()]

    return model
