import os
import subprocess
import argparse

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
from collections import namedtuple
from collections import OrderedDict
from torch.hub import load_state_dict_from_url

from glob import glob

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import transforms

from torch.autograd import Variable

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse
import copy
import time

import numpy as np
from torch.utils.data import DataLoader
from adabelief_pytorch import AdaBelief

import wandb


class Colors:
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


def create_folder(*paths):
    for directory in paths:
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print("CREATED FOLDER: ", Colors.GREEN, directory, Colors.RESET)
            else:  # TODO: delete already exist folder
                print(Colors.RED, "The directory already exists!!!", Colors.RESET)
        except OSError:
            print(Colors.RED, "Craeting directory failed.", Colors.RESET)


# Hyper parameters
class Config:
    LEARNING_RATE = 1e-3
    BATCH_SIZE = 16
    NUM_EPOCHS = 100


class Swish(nn.Module):
    def __init__(self, name=None):
        super().__init__()
        self.name = name

    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSamePadding(nn.Conv2d):
    """2D Convolutions with same padding"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        name=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2
        self.name = name

    def forward(self, x):
        input_h, input_w = x.size()[2:]
        kernel_h, kernel_w = self.weight.size()[2:]
        stride_h, stride_w = self.stride
        output_h, output_w = (
            math.ceil(input_h / stride_h),
            math.ceil(input_w / stride_w),
        )
        pad_h = max(
            (output_h - 1) * self.stride[0]
            + (kernel_h - 1) * self.dilation[0]
            + 1
            - input_h,
            0,
        )
        pad_w = max(
            (output_w - 1) * self.stride[1]
            + (kernel_w - 1) * self.dilation[1]
            + 1
            - input_w,
            0,
        )
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class BatchNorm2d(nn.BatchNorm2d):
    def __init__(
        self,
        num_features,
        eps=1e-05,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        name=None,
    ):
        super().__init__(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )
        self.name = name


def drop_connect(inputs, drop_connect_rate, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1.0 - drop_connect_rate
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck Block"""

    def __init__(self, block_args, global_params, idx):
        super().__init__()

        block_name = "blocks_" + str(idx) + "_"

        self.block_args = block_args
        self.batch_norm_momentum = 1 - global_params.batch_norm_momentum
        self.batch_norm_epsilon = global_params.batch_norm_epsilon
        self.has_se = (self.block_args.se_ratio is not None) and (
            0 < self.block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        self.swish = Swish(block_name + "_swish")

        # Expansion phase
        in_channels = self.block_args.input_filters
        out_channels = self.block_args.input_filters * self.block_args.expand_ratio
        if self.block_args.expand_ratio != 1:
            self._expand_conv = Conv2dSamePadding(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                bias=False,
                name=block_name + "expansion_conv",
            )
            self._bn0 = BatchNorm2d(
                num_features=out_channels,
                momentum=self.batch_norm_momentum,
                eps=self.batch_norm_epsilon,
                name=block_name + "expansion_batch_norm",
            )

        # Depth-wise convolution phase
        kernel_size = self.block_args.kernel_size
        strides = self.block_args.strides
        self._depthwise_conv = Conv2dSamePadding(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=out_channels,
            kernel_size=kernel_size,
            stride=strides,
            bias=False,
            name=block_name + "depthwise_conv",
        )
        self._bn1 = BatchNorm2d(
            num_features=out_channels,
            momentum=self.batch_norm_momentum,
            eps=self.batch_norm_epsilon,
            name=block_name + "depthwise_batch_norm",
        )

        # Squeeze and Excitation layer
        if self.has_se:
            num_squeezed_channels = max(
                1, int(self.block_args.input_filters * self.block_args.se_ratio)
            )
            self._se_reduce = Conv2dSamePadding(
                in_channels=out_channels,
                out_channels=num_squeezed_channels,
                kernel_size=1,
                name=block_name + "se_reduce",
            )
            self._se_expand = Conv2dSamePadding(
                in_channels=num_squeezed_channels,
                out_channels=out_channels,
                kernel_size=1,
                name=block_name + "se_expand",
            )

        # Output phase
        final_output_channels = self.block_args.output_filters
        self._project_conv = Conv2dSamePadding(
            in_channels=out_channels,
            out_channels=final_output_channels,
            kernel_size=1,
            bias=False,
            name=block_name + "output_conv",
        )
        self._bn2 = BatchNorm2d(
            num_features=final_output_channels,
            momentum=self.batch_norm_momentum,
            eps=self.batch_norm_epsilon,
            name=block_name + "output_batch_norm",
        )

    def forward(self, x, drop_connect_rate=None):
        identity = x
        # Expansion and depth-wise convolution
        if self.block_args.expand_ratio != 1:
            x = self._expand_conv(x)
            x = self._bn0(x)
            x = self.swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self.swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self.swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = (
            self.block_args.input_filters,
            self.block_args.output_filters,
        )
        if (
            self.id_skip
            and self.block_args.strides == 1
            and input_filters == output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(
                    x, drop_connect_rate=drop_connect_rate, training=self.training
                )
            x = x + identity
        return x


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def up_conv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


def custom_head(in_channels, out_channels):
    return nn.Sequential(
        nn.Dropout(),
        nn.Linear(in_channels, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(),
        nn.Linear(512, out_channels),
    )


GlobalParams = namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "num_classes",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
    ],
)
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)

BlockArgs = namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "strides",
        "se_ratio",
    ],
)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def round_filters(filters, global_params):
    """Round number of filters"""
    multiplier = global_params.width_coefficient
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    if not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """Round number of repeats"""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def get_efficientnet_params(model_name, override_params=None):
    """Get efficientnet params based on model name"""
    model_name = model_name[:15]

    params_dict = {
        # (width_coefficient, depth_coefficient, resolution, dropout_rate)
        # Note: the resolution here is just for reference, its values won't be used.
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    }

    if model_name not in params_dict.keys():
        raise KeyError("There is no model named {}.".format(model_name))

    width_coefficient, depth_coefficient, _, dropout_rate = params_dict[model_name]

    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=0.2,
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
    )

    if override_params:
        global_params = global_params._replace(**override_params)

    decoder = BlockDecoder()
    return decoder.decode(blocks_args), global_params


class BlockDecoder(object):
    """Block Decoder for readability"""

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        if "s" not in options or len(options["s"]) != 2:
            raise ValueError("Strides options should be a pair of integers.")

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            strides=[int(options["s"][0]), int(options["s"][1])],
        )

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            "r%d" % block.num_repeat,
            "k%d" % block.kernel_size,
            "s%d%d" % (block.strides[0], block.strides[1]),
            "e%s" % block.expand_ratio,
            "i%d" % block.input_filters,
            "o%d" % block.output_filters,
        ]
        if 0 < block.se_ratio <= 1:
            args.append("se%s" % block.se_ratio)
        if block.id_skip is False:
            args.append("noskip")
        return "_".join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.
        Args:
          string_list: a list of strings, each string is a notation of block.
        Returns:
          A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.
        Args:
          blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
          a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


class EfficientNet(nn.Module):
    def __init__(self, block_args_list, global_params):
        super().__init__()

        self.block_args_list = block_args_list
        self.global_params = global_params

        # Batch norm parameters
        batch_norm_momentum = 1 - self.global_params.batch_norm_momentum
        batch_norm_epsilon = self.global_params.batch_norm_epsilon

        # Stem
        in_channels = 3
        out_channels = round_filters(32, self.global_params)
        self._conv_stem = Conv2dSamePadding(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            bias=False,
            name="stem_conv",
        )
        self._bn0 = BatchNorm2d(
            num_features=out_channels,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon,
            name="stem_batch_norm",
        )

        self._swish = Swish(name="swish")

        # Build _blocks
        idx = 0
        self._blocks = nn.ModuleList([])
        for block_args in self.block_args_list:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self.global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self.global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self.global_params),
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self.global_params, idx=idx))
            idx += 1

            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=1
                )

            # The rest of the _blocks
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    MBConvBlock(block_args, self.global_params, idx=idx)
                )
                idx += 1

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self.global_params)
        self._conv_head = Conv2dSamePadding(
            in_channels, out_channels, kernel_size=1, bias=False, name="head_conv"
        )
        self._bn1 = BatchNorm2d(
            num_features=out_channels,
            momentum=batch_norm_momentum,
            eps=batch_norm_epsilon,
            name="head_batch_norm",
        )

        # Final linear layer
        self.dropout_rate = self.global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self.global_params.num_classes)

    def forward(self, x):
        # Stem
        x = self._conv_stem(x)
        x = self._bn0(x)
        x = self._swish(x)

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self.global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= idx / len(self._blocks)
            x = block(x, drop_connect_rate)

        # Head
        x = self._conv_head(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Pooling and Dropout
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate > 0:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        # Fully-connected layer
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, *, n_classes=1000, pretrained=False):
        return _get_model_by_name(model_name, classes=n_classes, pretrained=pretrained)

    @classmethod
    def encoder(cls, model_name, *, pretrained=False):
        model = cls.from_name(model_name, pretrained=pretrained)

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()

                self.name = model_name[:15]

                self.global_params = model.global_params

                self.stem_conv = model._conv_stem
                self.stem_batch_norm = model._bn0
                self.stem_swish = Swish(name="stem_swish")
                self.blocks = model._blocks
                self.head_conv = model._conv_head
                self.head_batch_norm = model._bn1
                self.head_swish = Swish(name="head_swish")

            def forward(self, x):
                # Stem
                x = self.stem_conv(x)
                x = self.stem_batch_norm(x)
                x = self.stem_swish(x)

                # Blocks
                for idx, block in enumerate(self.blocks):
                    drop_connect_rate = self.global_params.drop_connect_rate
                    if drop_connect_rate:
                        drop_connect_rate *= idx / len(self.blocks)
                    x = block(x, drop_connect_rate)

                # Head
                x = self.head_conv(x)
                x = self.head_batch_norm(x)
                x = self.head_swish(x)
                return x

        return Encoder()

    @classmethod
    def custom_head(cls, model_name, *, n_classes=1000, pretrained=False):
        model_name = model_name[:15]
        if n_classes == 1000:
            return cls.from_name(model_name, n_classes=n_classes, pretrained=pretrained)
        else:

            class CustomHead(nn.Module):
                def __init__(self, out_channels):
                    super().__init__()
                    self.encoder = cls.encoder(model_name, pretrained=pretrained)
                    self.custom_head = custom_head(self.n_channels * 2, out_channels)

                @property
                def n_channels(self):
                    n_channels_dict = {
                        "efficientnet-b0": 1280,
                        "efficientnet-b1": 1280,
                        "efficientnet-b2": 1408,
                        "efficientnet-b3": 1536,
                        "efficientnet-b4": 1792,
                        "efficientnet-b5": 2048,
                        "efficientnet-b6": 2304,
                        "efficientnet-b7": 2560,
                    }
                    return n_channels_dict[self.encoder.name]

                def forward(self, x):
                    x = self.encoder(x)
                    mp = nn.AdaptiveMaxPool2d(output_size=(1, 1))(x)
                    ap = nn.AdaptiveAvgPool2d(output_size=(1, 1))(x)
                    x = torch.cat([mp, ap], dim=1)
                    x = x.view(x.size(0), -1)
                    x = self.custom_head(x)

                    return x

            return CustomHead(n_classes)


def _get_model_by_name(model_name, classes=1000, pretrained=False):
    block_args_list, global_params = get_efficientnet_params(
        model_name[:15], override_params={"num_classes": classes}
    )
    model = EfficientNet(block_args_list, global_params)
    try:
        if pretrained:
            if len(model_name) < 16:
                pretrained_state_dict = load_state_dict_from_url(
                    IMAGENET_WEIGHTS[model_name[:15]]
                )
            else:
                pretrained_state_dict = torch.load(
                    IMAGENET_WEIGHTS[model_name], map_location=device
                )
                model._fc = nn.Linear(1792, 4)

            if classes != 1000:
                random_state_dict = model.state_dict()
                pretrained_state_dict["_fc.weight"] = random_state_dict["_fc.weight"]
                pretrained_state_dict["_fc.bias"] = random_state_dict["_fc.bias"]

            model.load_state_dict(pretrained_state_dict)

    except KeyError as e:
        print(
            f"NOTE: Currently model {e} doesn't have pretrained weights, therefore a model with randomly initialized"
            " weights is returned."
        )

    return model


__all__ = [
    "EfficientUnet",
    "get_efficientunet_b0",
    "get_efficientunet_b1",
    "get_efficientunet_b2",
    "get_efficientunet_b3",
    "get_efficientunet_b4",
    "get_efficientunet_b5",
    "get_efficientunet_b6",
    "get_efficientunet_b7",
    "get_socar_efficientunet_b0",
    "get_socar_efficientunet_b1",
    "get_stanford_efficientunet_b0",
    "get_socar_efficientunet_b4",
    "get_stanford_efficientunet_b4",
]


def get_blocks_to_be_concat(model, x):
    shapes = set()
    blocks = OrderedDict()
    hooks = []
    count = 0

    def register_hook(module):
        def hook(module, input, output):
            try:
                nonlocal count
                if module.name == f"blocks_{count}_output_batch_norm":
                    count += 1
                    shape = output.size()[-2:]
                    if shape not in shapes:
                        shapes.add(shape)
                        blocks[module.name] = output

                elif module.name == "head_swish":
                    # when module.name == 'head_swish', it means the program has already got all necessary blocks for
                    # concatenation. In my dynamic unet implementation, I first upscale the output of the backbone,
                    # (in this case it's the output of 'head_swish') concatenate it with a block which has the same
                    # Height & Width (image size). Therefore, after upscaling, the output of 'head_swish' has bigger
                    # image size. The last block has the same image size as 'head_swish' before upscaling. So we don't
                    # really need the last block for concatenation. That's why I wrote `blocks.popitem()`.
                    blocks.popitem()
                    blocks[module.name] = output

            except AttributeError:
                pass

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    # register hook
    model.apply(register_hook)

    # make a forward pass to trigger the hooks
    model(x)

    # remove these hooks
    for h in hooks:
        h.remove()

    return blocks


class EfficientUnet(nn.Module):
    def __init__(self, encoder, out_channels=2, concat_input=True):
        super().__init__()

        self.encoder = encoder
        self.concat_input = concat_input

        self.up_conv1 = up_conv(self.n_channels, 512)
        self.double_conv1 = double_conv(self.size[0], 512)
        self.up_conv2 = up_conv(512, 256)
        self.double_conv2 = double_conv(self.size[1], 256)
        self.up_conv3 = up_conv(256, 128)
        self.double_conv3 = double_conv(self.size[2], 128)
        self.up_conv4 = up_conv(128, 64)
        self.double_conv4 = double_conv(self.size[3], 64)

        if self.concat_input:
            self.up_conv_input = up_conv(64, 32)
            self.double_conv_input = double_conv(self.size[4], 32)

        self.final_conv = nn.Conv2d(self.size[5], out_channels, kernel_size=1)

    @property
    def n_channels(self):
        n_channels_dict = {
            "efficientnet-b0": 1280,
            "efficientnet-b1": 1280,
            "efficientnet-b2": 1408,
            "efficientnet-b3": 1536,
            "efficientnet-b4": 1792,
            "efficientnet-b5": 2048,
            "efficientnet-b6": 2304,
            "efficientnet-b7": 2560,
        }
        return n_channels_dict[self.encoder.name]

    @property
    def size(self):
        size_dict = {
            "efficientnet-b0": [592, 296, 152, 80, 35, 32],
            "efficientnet-b1": [592, 296, 152, 80, 35, 32],
            "efficientnet-b2": [600, 304, 152, 80, 35, 32],
            "efficientnet-b3": [608, 304, 160, 88, 35, 32],
            "efficientnet-b4": [624, 312, 160, 88, 35, 32],
            "efficientnet-b5": [640, 320, 168, 88, 35, 32],
            "efficientnet-b6": [656, 328, 168, 96, 35, 32],
            "efficientnet-b7": [672, 336, 176, 96, 35, 32],
        }
        return size_dict[self.encoder.name]

    def forward(self, x):
        input_ = x

        blocks = get_blocks_to_be_concat(self.encoder, x)
        _, x = blocks.popitem()

        x = self.up_conv1(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv1(x)

        x = self.up_conv2(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv2(x)

        x = self.up_conv3(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv3(x)

        x = self.up_conv4(x)
        x = torch.cat([x, blocks.popitem()[1]], dim=1)
        x = self.double_conv4(x)

        if self.concat_input:
            x = self.up_conv_input(x)
            x = torch.cat([x, input_], dim=1)
            x = self.double_conv_input(x)

        x = self.final_conv(x)

        return x


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b0", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b1", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b2(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b2", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b3(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b3", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b4", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b5", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b6(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b6", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b7(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b7", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b0", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_socar_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b0-socar", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_socar_efficientunet_b1(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b1-socar", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_stanford_efficientunet_b0(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b0-stanford", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_stanford_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b4-stanford", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


def get_socar_efficientunet_b4(out_channels=2, concat_input=True, pretrained=True):
    encoder = EfficientNet.encoder("efficientnet-b4-socar", pretrained=pretrained)
    model = EfficientUnet(encoder, out_channels=out_channels, concat_input=concat_input)
    return model


__all__ = ["to_numpy", "denormalization", "classify_class", "save_net", "load_net"]


def to_numpy(tensor):
    if tensor.ndim == 3:
        return tensor.to("cpu").detach().numpy()
    return tensor.to("cpu").detach().numpy().transpose(0, 2, 3, 1)  # (Batch, H, W, C)


def denormalization(data, mean, std):
    return (data * std) + mean


def classify_class(x):
    return 1.0 * (x > 0.5)


def save_net(ckpt_dir, net, optim, epoch, is_best=False, best_iou=None):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if is_best == False:
        torch.save(
            {"net": net.state_dict(), "optim": optim.state_dict()},
            os.path.join(ckpt_dir, f"model_epoch_{epoch:04}.pth"),
        )
    elif is_best == True:
        torch.save(
            {"net": net.state_dict(), "optim": optim.state_dict()},
            os.path.join(ckpt_dir, f"best_model_{best_iou:.3f}.pth"),
        )


def load_net(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    ckpt_list = os.listdir(ckpt_dir)
    ckpt_list.sort(key=lambda fname: int("".join(filter(str.isdigit, fname))))

    ckpt_path = os.path.join(ckpt_dir, ckpt_list[-1])
    model_dict = torch.load(ckpt_path, map_location=device)
    print(f"* Load {ckpt_path}")

    net.load_state_dict(model_dict["net"])
    optim.load_state_dict(model_dict["optim"])
    epoch = int("".join(filter(str.isdigit, ckpt_list[-1])))

    return net, optim, ckpt_path


class DatasetV2(Dataset):
    def __init__(self, imgs_dir, mask_dir, transform=None):
        self.img_dir = imgs_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        self.masks = []

        images_path, masks_path = [], []

        for ext in ("*.jpeg", "*.png", "*.jpg"):
            images_path.extend(sorted(glob(os.path.join(imgs_dir, ext))))
            masks_path.extend(sorted(glob(os.path.join(mask_dir, ext))))
        images_path = images_path[:50]
        masks_path = masks_path[:50]

        for i, m in zip(images_path, masks_path):
            self.images.extend([Image.open(i).convert("RGB")])
            self.masks.extend([Image.open(m).convert("L")])

    def __len__(self):
        return len(self.masks)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        np_image = np.array(image)
        np_mask = np.array(mask)

        if self.transform:
            transformed = self.transform(image=np_image, mask=np_mask)
            np_image = transformed["image"]
            np_mask = transformed["mask"]
            np_mask = np_mask.long()

        ret = {
            "img": np_image,
            "label": np_mask,
        }

        return ret


# --------------------------- FOCAL LOSSES ---------------------------


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection + smooth) / (union + smooth)


def train_model(
    dataloaders, batch_num, net, criterion, optim, ckpt_dir, wandb, w_config
):
    wandb.watch(net, criterion, log="all", log_freq=10)

    since = time.time()

    best_model_wts = copy.deepcopy(net.state_dict())
    best_iou = 0
    num_epoch = w_config.epochs

    for epoch in range(1, num_epoch + 1):
        net.train()  # Train Mode
        train_loss_arr = []

        for batch_idx, data in enumerate(dataloaders["train"], 1):
            # Forward Propagation
            img = data["img"].to(device)
            label = data["label"].to(device)

            label = label // 255

            output = net(img)

            # Backward Propagation
            optim.zero_grad()

            loss = criterion(output, label)

            loss.backward()

            optim.step()

            # Calc Loss Function
            train_loss_arr.append(loss.item())
            print_form = "[Train] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f}"
            print(
                print_form.format(
                    epoch, num_epoch, batch_idx, batch_num["train"], train_loss_arr[-1]
                )
            )

        train_loss_avg = np.mean(train_loss_arr)

        # Validation (No Back Propagation)
        with torch.no_grad():
            net.eval()  # Evaluation Mode
            val_loss_arr, val_iou_arr = [], []

            for batch_idx, data in enumerate(dataloaders["val"], 1):
                # Forward Propagation
                img = data["img"].to(device)
                label = data["label"].to(device)

                label = label // 255

                output = net(img)
                output_t = torch.argmax(output, dim=1).float()

                # Calc Loss Function
                loss = criterion(output, label)
                iou = iou_score(output_t, label)

                val_loss_arr.append(loss.item())
                val_iou_arr.append(iou.item())

                print_form = "[Validation] | Epoch: {:0>4d} / {:0>4d} | Batch: {:0>4d} / {:0>4d} | Loss: {:.4f} | IoU: {:.4f}"
                print(
                    print_form.format(
                        epoch,
                        num_epoch,
                        batch_idx,
                        batch_num["val"],
                        val_loss_arr[-1],
                        iou,
                    )
                )

        val_loss_avg = np.mean(val_loss_arr)
        val_iou_avg = np.mean(val_iou_arr)

        if best_iou < val_iou_avg:
            best_iou = val_iou_avg
            best_model_wts = copy.deepcopy(net.state_dict())

        wandb.log(
            {
                "train_epoch_loss": train_loss_avg,
                "val_epoch_loss": val_loss_avg,
                "val_epoch_iou": val_iou_avg,
            },
            step=epoch,
        )

        print_form = "[Epoch {:0>4d}] Training Avg Loss: {:.4f} | Validation Avg Loss: {:.4f} | Validation Avg IoU: {:.4f}"
        print(print_form.format(epoch, train_loss_avg, val_loss_avg, val_iou_avg))

        save_net(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val IoU: {:4f}".format(best_iou))

    wandb.log({"Best val IoU": best_iou}, commit=False)

    net.load_state_dict(best_model_wts)
    save_net(
        ckpt_dir=ckpt_dir,
        net=net,
        optim=optim,
        epoch=epoch,
        is_best=True,
        best_iou=best_iou,
    )


def wandb_setting(sweep_config=None):
    wandb.init(config=sweep_config)
    w_config = wandb.config
    name_str = (
        str(w_config.model)
        + " | "
        + str(w_config.img_size)
        + " | "
        + str(w_config.batch_size)
    )
    wandb.run.name = name_str

    #########Random seed 고정해주기###########
    random_seed = w_config.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    ###########################################

    train_transform = A.Compose(
        [
            A.Resize(w_config.img_size, w_config.img_size),
            A.Normalize(mean=(0.485), std=(0.229)),
            transforms.ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.Resize(w_config.img_size, w_config.img_size),
            A.Normalize(mean=(0.485), std=(0.229)),
            transforms.ToTensorV2(),
        ]
    )

    ##########################################데이터 로드 하기#################################################
    batch_size = w_config.batch_size

    train_dataset = DatasetV2(
        imgs_dir=TRAIN_IMGS_DIR, mask_dir=TRAIN_LABELS_DIR, transform=train_transform
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_dataset = DatasetV2(
        imgs_dir=VAL_IMGS_DIR, mask_dir=VAL_LABELS_DIR, transform=val_transform
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    #############################################################################################################
    train_data_num = len(train_dataset)
    val_data_num = len(val_dataset)

    train_batch_num = int(np.ceil(train_data_num / batch_size))  # np.ceil 반올림
    val_batch_num = int(np.ceil(val_data_num / batch_size))

    batch_num = {"train": train_batch_num, "val": val_batch_num}
    dataloaders = {"train": train_loader, "val": val_loader}

    if w_config.model == "imagenet-b4":
        net = get_efficientunet_b4(
            out_channels=2, concat_input=True, pretrained=True
        ).to(device)
    elif w_config.model == "stfd-ssl-b4":
        net = get_stanford_efficientunet_b4(
            out_channels=2, concat_input=True, pretrained=True
        ).to(device)
    elif w_config.model == "socar-ssl-b4":
        net = get_socar_efficientunet_b4(
            out_channels=2, concat_input=True, pretrained=True
        ).to(device)

    # Loss Function
    if w_config.loss == "CrossEntropy":
        criterion = nn.CrossEntropyLoss().to(device)
    elif w_config.loss == "focal":
        criterion = FocalLoss(gamma=2, alpha=0.5).to(device)

    # Optimizer
    # optimizer 종류 정해주기
    if w_config.optimizer == "sgd":
        optimizer_ft = torch.optim.SGD(
            net.parameters(), lr=w_config.learning_rate, momentum=0.9
        )
    elif w_config.optimizer == "adam":
        optimizer_ft = torch.optim.Adam(
            params=net.parameters(), lr=w_config.learning_rate
        )
    elif w_config.optimizer == "adabelief":
        optimizer_ft = AdaBelief(
            net.parameters(),
            lr=w_config.learning_rate,
            eps=1e-16,
            betas=(0.9, 0.999),
            weight_decouple=True,
            rectify=True,
        )

    ckpt_dir = os.path.join(CKPT_DIR, name_str)

    wandb.watch(net, log="all")
    train_model(
        dataloaders,
        batch_num,
        net,
        criterion,
        optimizer_ft,
        ckpt_dir,
        wandb,
        w_config=w_config,
    )


def set_dir(data_class):
    BUCKET_NAME = "images-annotated"
    global BASE_DIR
    global CKPT_DIR
    global BACKBONE_DIR
    global TRAIN_IMGS_DIR
    global VAL_IMGS_DIR
    global TRAIN_LABELS_DIR
    global VAL_LABELS_DIR
    global ROOT_DIR

    ROOT_DIR = os.getcwd()
    DATA_DIR = os.path.join(ROOT_DIR, "data", data_class)
    BASE_DIR = os.path.join(ROOT_DIR, "checkpoints_dir")

    RESULTS_DIR = os.path.join(ROOT_DIR, "test_results_dir", "test_results_")
    INFER_DIR = os.path.join(ROOT_DIR, "inference_dir", data_class)

    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "valid")

    TRAIN_IMGS_DIR = os.path.join(TRAIN_DIR, "images")
    VAL_IMGS_DIR = os.path.join(VAL_DIR, "images")
    TRAIN_LABELS_DIR = os.path.join(TRAIN_DIR, "masks")
    VAL_LABELS_DIR = os.path.join(VAL_DIR, "masks")

    create_folder(
        BASE_DIR,
        RESULTS_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TRAIN_IMGS_DIR,
        VAL_IMGS_DIR,
        TRAIN_LABELS_DIR,
        VAL_LABELS_DIR,
    )

    dir_arc = {
        RESULTS_DIR: f"data/{data_class}/",
        TRAIN_IMGS_DIR: f"data/{data_class}/train/images",
        VAL_IMGS_DIR: f"data/{data_class}/valid/images",
        TRAIN_LABELS_DIR: f"data/{data_class}/train/masks",
        VAL_LABELS_DIR: f"data/{data_class}/valid/masks",
    }
    # mount
    for path, dir_name in dir_arc.items():
        subprocess.run(["gcsfuse", "--only-dir", dir_name, BUCKET_NAME, path])
    subprocess.run(["gcsfuse", "model-cpt", BASE_DIR])
    # BACKBONE_DIR = os.path.join(CKPT_DIR, "pretrained")
    # subprocess.run(["gcsfuse", "--only-dir", "pretrained", "model-cpt", BACKBONE_DIR])
    BACKBONE_DIR = os.path.join(
        BASE_DIR, "pretrained", "pretrained_effb4_orignal_stanford_16000.pt"
    )
    CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
    # subprocess.run(["gcsfuse", "--only-dir", "checkpoints", "model-cpt", CKPT_DIR])
    return dir_arc


def umount(dir_arc):
    # unmount
    def unmount_fuse(folder):
        ret_code = 1
        while ret_code == 1:
            ret = subprocess.run(["fusermount", "-u", folder])
            ret_code = ret.returncode
            time.sleep(1)  # wait
        print(Colors.GREEN, f"Unmount f{folder} done!", Colors.RESET)

    for f in dir_arc.keys():
        unmount_fuse(f)
    unmount_fuse(BASE_DIR)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="Define metadata")
    parser.add_argument(
        "-t",
        "--type",
        default="scratch",
        help="Choose Data Type - dent, scratch, spacing",
    )
    parser.add_argument(
        "-p", "--project-name", default="[VIAI] retrain test", help="WandB Project Name"
    )
    parser.add_argument("-e", "--epochs", default=30, help="Setting train epochs")
    parser.add_argument(
        "-b", "--batch-size", default=16, help="Setting dataset batch size"
    )

    args = parser.parse_args()

    project_name = str(args.project_name)
    entity_name = "viai"
    data_class = str(args.type)
    dir_arc = set_dir(data_class)
    IMAGENET_WEIGHTS = {
        "efficientnet-b0": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b0-355c32eb.pth",
        "efficientnet-b1": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b1-f1951068.pth",
        "efficientnet-b2": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b2-8bb594d6.pth",
        "efficientnet-b3": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b3-5fb5a3c3.pth",
        "efficientnet-b4": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b4-6ed6700e.pth",
        "efficientnet-b5": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b5-b6417697.pth",
        "efficientnet-b6": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b6-c76e70fd.pth",
        "efficientnet-b7": "https://github.com/lukemelas/EfficientNet-PyTorch/releases/download/1.0/efficientnet-b7-dcc49843.pth",
        "efficientnet-b4-stanford": BACKBONE_DIR,
    }

    # Config
    sweep_config = {
        "method": "grid",
        "name": f"grid-stfd_sweep_v14_{data_class}",
        "metric": {"name": "Best val IoU", "goal": "maximize"},
        "parameters": {
            "epochs": {"value": int(args.epochs)},
            "batch_size": {"value": int(args.batch_size)},
            "optimizer": {"value": "adabelief"},
            "model": {"value": "stfd-ssl-b4"},
            "loss": {"value": "focal"},
            "img_size": {"value": 512},
            "seed": {"value": 0},
            "learning_rate": {"value": 1e-3},
        },
    }
    sweep_id = wandb.sweep(sweep_config, project=project_name, entity=entity_name)

    wandb.agent(sweep_id, wandb_setting, count=20)
    umount(dir_arc)
