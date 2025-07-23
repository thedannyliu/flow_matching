# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
import torch
from torchvision.transforms.v2 import (
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToDtype,
    ToImage,
)


def get_train_transform():
    transform_list = [
        ToImage(),
        Resize((64, 64)),
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ]
    return Compose(transform_list)
