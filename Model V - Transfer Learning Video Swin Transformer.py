#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os

base_path = "dataset"

folders = ["2_crawling","3_crying","4_dancing","5_eating","7_falling_slide","8_falling","9_playing_w_animals","10_playing_w_toy","12_smiling","15_walking"]

video_label_dict = {}

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(folder_path, file)
            video_label_dict[video_path] = folder

print(video_label_dict)


# In[ ]:


video_label_list = list(video_label_dict.items())
video_paths = list(video_label_dict.keys())
video_labels = list(video_label_dict.values())


# In[ ]:


from sklearn.preprocessing import LabelEncoder
label_mapping = {
    "2_crawling": 0,
    "3_crying": 1,
    "4_dancing": 2,
    "5_eating": 3,
    "7_falling_slide": 4,
    "8_falling": 5,
    "9_playing_w_animals": 6,
    "10_playing_w_toy": 7,
    "12_smiling": 8,
    "15_walking": 9,
}
label_encoder = LabelEncoder()
label_encoder.fit([label_mapping[label] for label in video_labels])
labels_encoder = label_encoder.transform([label_mapping[label] for label in video_labels])


# In[ ]:


from sklearn.model_selection import train_test_split

# Split the data into training (70%) and temporary (30%) sets
train_video_paths, test_video_paths, train_video_labels, test_video_labels = train_test_split(
    video_paths, labels_encoder, test_size=0.3, stratify=video_labels, random_state=42
)

# Split the temporary data into testing (20% of original data) and validation (10% of original data) sets
val_video_paths, test_video_paths, val_video_labels, test_video_labels = train_test_split(
    test_video_paths, test_video_labels, test_size=2/3, stratify=test_video_labels, random_state=42
)


# In[ ]:


len(train_video_paths)


# In[ ]:


len(test_video_paths)


# In[ ]:


len(val_video_paths)


# In[ ]:


from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.transforms._presets import VideoClassification

from torchvision.utils import _log_api_usage_once

from torchvision.models._api import register_model, Weights, WeightsEnum

from torchvision.models._meta import _KINETICS400_CATEGORIES
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.swin_transformer import PatchMerging, SwinTransformerBlock


# In[ ]:


class ShiftedWindowAttention3d(nn.Module):
    """
    See :func:`shifted_window_attention_3d`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if len(window_size) != 3 or len(shift_size) != 3:
            raise ValueError("window_size and shift_size must be of length 2")

        self.window_size = window_size  # Wd, Wh, Ww
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self) -> None:
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                self.num_heads,
            )
        )  # 2*Wd-1 * 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self) -> None:
        # get pair-wise relative position index for each token inside the window
        coords_dhw = [torch.arange(self.window_size[i]) for i in range(3)]
        coords = torch.stack(
            torch.meshgrid(coords_dhw[0], coords_dhw[1], coords_dhw[2], indexing="ij")
        )  # 3, Wd, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 3, Wd*Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, Wd*Wh*Ww, Wd*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wd*Wh*Ww, Wd*Wh*Ww, 3
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1

        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        # We don't flatten the relative_position_index here in 3d case.
        relative_position_index = relative_coords.sum(-1)  # Wd*Wh*Ww, Wd*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self, window_size: List[int]) -> torch.Tensor:
        return _get_relative_position_bias(self.relative_position_bias_table, self.relative_position_index, window_size)  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        _, t, h, w, _ = x.shape
        size_dhw = [t, h, w]
        window_size, shift_size = self.window_size.copy(), self.shift_size.copy()
        # Handle case where window_size is larger than the input tensor
        window_size, shift_size = _get_window_and_shift_size(shift_size, size_dhw, window_size)

        relative_position_bias = self.get_relative_position_bias(window_size)

        return shifted_window_attention_3d(
            x,
            self.qkv.weight,
            self.proj.weight,
            relative_position_bias,
            window_size,
            self.num_heads,
            shift_size=shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            qkv_bias=self.qkv.bias,
            proj_bias=self.proj.bias,
            training=self.training,
        )


# Modified from:
# https://github.com/SwinTransformer/Video-Swin-Transformer/blob/master/mmaction/models/backbones/swin_transformer.py
class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: List[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)  # B C T Wh Ww
        x = x.permute(0, 2, 3, 4, 1)  # B T Wh Ww C
        if self.norm is not None:
            x = self.norm(x)
        return x


# In[ ]:


class SwinTransformer3d(nn.Module):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 400.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """

    def __init__(
        self,
        patch_size: List[int],
        embed_dim: int,
        depths: List[int],
        num_heads: List[int],
        window_size: List[int],
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        stochastic_depth_prob: float = 0.1,
        num_classes: int = 400,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        block: Optional[Callable[..., nn.Module]] = None,
        downsample_layer: Callable[..., nn.Module] = PatchMerging,
        patch_embed: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes

        if block is None:
            block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-5)

        if patch_embed is None:
            patch_embed = PatchEmbed3d

        # split image into non-overlapping patches
        self.patch_embed = patch_embed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=dropout)

        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        # build SwinTransformer blocks
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2**i_stage
            for i_layer in range(depths[i_stage]):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(
                    block(
                        dim,
                        num_heads[i_stage],
                        window_size=window_size,
                        shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
                        mlp_ratio=mlp_ratio,
                        dropout=dropout,
                        attention_dropout=attention_dropout,
                        stochastic_depth_prob=sd_prob,
                        norm_layer=norm_layer,
                        attn_layer=ShiftedWindowAttention3d,
                    )
                )
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            # add patch merging layer
            if i_stage < (len(depths) - 1):
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)

        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        # x: B C T H W
        x = self.patch_embed(x)  # B _T _H _W C
        x = self.pos_drop(x)
        x = self.features(x)  # B _T _H _W C
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)  # B, C, _T, _H, _W
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def _swin_transformer3d(
    patch_size: List[int],
    embed_dim: int,
    depths: List[int],
    num_heads: List[int],
    window_size: List[int],
    stochastic_depth_prob: float,
    weights: Optional[WeightsEnum],
    progress: bool,
    **kwargs: Any,
) -> SwinTransformer3d:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))

    model = SwinTransformer3d(
        patch_size=patch_size,
        embed_dim=embed_dim,
        depths=depths,
        num_heads=num_heads,
        window_size=window_size,
        stochastic_depth_prob=stochastic_depth_prob,
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


_COMMON_META = {
    "categories": _KINETICS400_CATEGORIES,
    "min_size": (1, 1),
    "min_temporal_size": 1,
}


# In[ ]:


class PatchEmbed3d(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (List[int]): Patch token size.
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size: List[int],
        in_channels: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.tuple_patch_size = (patch_size[0], patch_size[1], patch_size[2])

        self.proj = nn.Conv3d(
            in_channels,
            embed_dim,
            kernel_size=self.tuple_patch_size,
            stride=self.tuple_patch_size,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward function."""
        # padding
        _, _, t, h, w = x.size()
        pad_size = _compute_pad_size_3d((t, h, w), self.tuple_patch_size)
        x = F.pad(x, (0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
        x = self.proj(x)  # B C T Wh Ww
        x = x.permute(0, 2, 3, 4, 1)  # B T Wh Ww C
        if self.norm is not None:
            x = self.norm(x)
        return x



# In[ ]:


import cv2
import torch
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

class MyCustomDataset(Dataset):
    def __init__(self, video_paths, video_labels, num_frames):
        self.video_paths = video_paths
        self.video_labels = video_labels
        self.num_frames = num_frames
        self.patch_embed = PatchEmbed3d(patch_size=[2, 4, 4])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.video_labels[idx]

        # Read video frames
        video_frames, _, _ = torchvision.io.read_video(video_path)

        # Resize video frames as a batch
        video_frames = video_frames.float()
        
        resized_frames = F.interpolate(video_frames.permute(0, 3, 1, 2), size=(32, 32), mode='bilinear')
        resized_frames_tensor = resized_frames.permute(1, 0, 2, 3)

        video_tensor = resized_frames_tensor.float() / 255.0
        
        total_frames = video_tensor.shape[1]
        indices = np.linspace(0, total_frames-1, 16, dtype=int)  # Generate evenly spaced indices
        video_tensor = video_tensor[:, indices, :, :]

        

        # Apply patch embedding
  #     patch_embeddings = self.patch_embed(video_tensor)

        return video_tensor, label, video_path



# In[ ]:


train_dataset = MyCustomDataset(train_video_paths, train_video_labels, num_frames=16)
test_dataset = MyCustomDataset(test_video_paths, test_video_labels, num_frames=16)
val_dataset = MyCustomDataset(val_video_paths, val_video_labels, num_frames=16)


# In[ ]:


data_loader_train = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
data_loader_val = torch.utils.data.DataLoader(val_dataset, batch_size=8, shuffle=True)


# In[ ]:


class Swin3D_T_Weights(WeightsEnum):
    KINETICS400_V1 = Weights(
        url="https://download.pytorch.org/models/swin3d_t-7615ae03.pth",
        transforms=partial(
            VideoClassification,
            crop_size=(224, 224),
            resize_size=(256,),
            mean=(0.4850, 0.4560, 0.4060),
            std=(0.2290, 0.2240, 0.2250),
        ),
        meta={
            **_COMMON_META,
            "recipe": "https://github.com/SwinTransformer/Video-Swin-Transformer#kinetics-400",
            "_docs": (
                "The weights were ported from the paper. The accuracies are estimated on video-level "
                "with parameters `frame_rate=15`, `clips_per_video=12`, and `clip_len=32`"
            ),
            "num_params": 28158070,
            "_metrics": {
                "Kinetics-400": {
                    "acc@1": 77.715,
                    "acc@5": 93.519,
                }
            },
            "_ops": 43.882,
            "_file_size": 121.543,
        },
    )
    DEFAULT = KINETICS400_V1


# In[ ]:


@register_model()
@handle_legacy_interface(weights=("pretrained", Swin3D_T_Weights.KINETICS400_V1))
def swin3d_t_1(*, weights: Optional[Swin3D_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer3d:
    """
    Constructs a swin_tiny architecture from
    `Video Swin Transformer <https://arxiv.org/abs/2106.13230>`_.

    Args:
        weights (:class:`~torchvision.models.video.Swin3D_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.video.Swin3D_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.video.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.video.Swin3D_T_Weights
        :members:
    """
    weights = Swin3D_T_Weights.verify(weights)

    return _swin_transformer3d(
        patch_size=[2, 4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[8, 7, 7],
        stochastic_depth_prob=0.1,
        weights=weights,
        progress=progress,
        **kwargs,
    )


# In[ ]:


def _get_window_and_shift_size(
    shift_size: List[int], size_dhw: List[int], window_size: List[int]
) -> Tuple[List[int], List[int]]:
    for i in range(3):
        if size_dhw[i] <= window_size[i]:
            # In this case, window_size will adapt to the input size, and no need to shift
            window_size[i] = size_dhw[i]
            shift_size[i] = 0

    return window_size, shift_size


torch.fx.wrap("_get_window_and_shift_size")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> Tensor:
    window_vol = window_size[0] * window_size[1] * window_size[2]
    # In 3d case we flatten the relative_position_bias
    relative_position_bias = relative_position_bias_table[
        relative_position_index[:window_vol, :window_vol].flatten()  # type: ignore[index]
    ]
    relative_position_bias = relative_position_bias.view(window_vol, window_vol, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")


def _compute_pad_size_3d(size_dhw: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> Tuple[int, int, int]:
    pad_size = [(patch_size[i] - size_dhw[i] % patch_size[i]) % patch_size[i] for i in range(3)]
    return pad_size[0], pad_size[1], pad_size[2]


torch.fx.wrap("_compute_pad_size_3d")


def _compute_attention_mask_3d(
    x: Tensor,
    size_dhw: Tuple[int, int, int],
    window_size: Tuple[int, int, int],
    shift_size: Tuple[int, int, int],
) -> Tensor:
    # generate attention mask
    attn_mask = x.new_zeros(*size_dhw)
    num_windows = (size_dhw[0] // window_size[0]) * (size_dhw[1] // window_size[1]) * (size_dhw[2] // window_size[2])
    slices = [
        (
            (0, -window_size[i]),
            (-window_size[i], -shift_size[i]),
            (-shift_size[i], None),
        )
        for i in range(3)
    ]
    count = 0
    for d in slices[0]:
        for h in slices[1]:
            for w in slices[2]:
                attn_mask[d[0] : d[1], h[0] : h[1], w[0] : w[1]] = count
                count += 1

    # Partition window on attn_mask
    attn_mask = attn_mask.view(
        size_dhw[0] // window_size[0],
        window_size[0],
        size_dhw[1] // window_size[1],
        window_size[1],
        size_dhw[2] // window_size[2],
        window_size[2],
    )
    attn_mask = attn_mask.permute(0, 2, 4, 1, 3, 5).reshape(
        num_windows, window_size[0] * window_size[1] * window_size[2]
    )
    attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


torch.fx.wrap("_compute_attention_mask_3d")


def shifted_window_attention_3d(
    input: Tensor,
    qkv_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    qkv_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    training: bool = True,
) -> Tensor:
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[B, T, H, W, C]): The input tensor, 5-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): 3-dimensions window size, T, H, W .
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention (T, H, W).
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        training (bool, optional): Training flag used by the dropout parameters. Default: True.
    Returns:
        Tensor[B, T, H, W, C]: The output tensor after shifted window attention.
    """
    b, t, h, w, c = input.shape
    # pad feature maps to multiples of window size
    pad_size = _compute_pad_size_3d((t, h, w), (window_size[0], window_size[1], window_size[2]))
    x = F.pad(input, (0, 0, 0, pad_size[2], 0, pad_size[1], 0, pad_size[0]))
    _, tp, hp, wp, _ = x.shape
    padded_size = (tp, hp, wp)

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))

    # partition windows
    num_windows = (
        (padded_size[0] // window_size[0]) * (padded_size[1] // window_size[1]) * (padded_size[2] // window_size[2])
    )
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        window_size[0],
        padded_size[1] // window_size[1],
        window_size[1],
        padded_size[2] // window_size[2],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(
        b * num_windows, window_size[0] * window_size[1] * window_size[2], c
    )  # B*nW, Wd*Wh*Ww, C

    # multi-head attention
    qkv = F.linear(x, qkv_weight, qkv_bias)
    qkv = qkv.reshape(x.size(0), x.size(1), 3, num_heads, c // num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    q = q * (c // num_heads) ** -0.5
    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask to handle shifted windows with varying size
        attn_mask = _compute_attention_mask_3d(
            x,
            (padded_size[0], padded_size[1], padded_size[2]),
            (window_size[0], window_size[1], window_size[2]),
            (shift_size[0], shift_size[1], shift_size[2]),
        )
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout, training=training)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), c)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout, training=training)

    # reverse windows
    x = x.view(
        b,
        padded_size[0] // window_size[0],
        padded_size[1] // window_size[1],
        padded_size[2] // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        c,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(b, tp, hp, wp, c)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))

    # unpad features
    x = x[:, :t, :h, :w, :].contiguous()
    return x


torch.fx.wrap("shifted_window_attention_3d")


# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the SwinTransformer3d model
model = swin3d_t_1()
model.head = nn.Linear(in_features=768, out_features=10)


# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Move the model and loss function to the appropriate device
model.to(device)
criterion.to(device)




# In[ ]:


print(model)


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
# Create lists to store the loss and accuracy values
train_loss_values = []
train_acc_values = []
test_loss_values = []
test_acc_values = []

# Set the number of training epochs
num_epochs = 100


# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    for inputs, labels, _ in data_loader_train:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels)

        running_loss += loss.item()

    train_loss = running_loss / len(data_loader_train.dataset)
    train_acc = correct_predictions.double() / len(data_loader_train.dataset) *100
    
    train_loss_values.append(train_loss)
    train_acc_values.append(train_acc/100)
    
    
    # Validation phase
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    with torch.no_grad():
        for inputs, labels, _ in data_loader_test:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            running_loss += loss.item() * inputs.size(0)
            correct_predictions += torch.sum(preds == labels)

    test_loss = running_loss / len(data_loader_test.dataset)
    test_acc = correct_predictions.double() / len(data_loader_test.dataset) *100
    
    test_loss_values.append(test_loss)
    test_acc_values.append(test_acc/100)
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")


# In[ ]:


# Plotting the curves
epochs = range(1, num_epochs+1)


# Plotting loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss_values, label='Train')
plt.plot(epochs, test_loss_values, label='Test')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Test Loss')
plt.legend()

# Plotting accuracy curves
plt.subplot(1, 2, 2)
plt.plot(epochs, [x.item() for x in train_acc_values], label='Train')
plt.plot(epochs, [x.item() for x in test_acc_values], label='Test')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Test Accuracy')
plt.legend()

# Display the plot
plt.tight_layout()
plt.show()


# In[ ]:


torch.save(model, 'model_video_swintransform.pth')


# In[ ]:


model.eval()
running_loss = 0.0
correct_predictions_top1 = 0
correct_predictions_top5 = 0

with torch.no_grad():
    for inputs, labels, _ in data_loader_val:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)
        correct_predictions_top1 += torch.sum(preds == labels)
        
        # Calculate top-5 accuracy
        _, pred_labels_top5 = torch.topk(outputs, k=5, dim=1)
        correct_predictions_top5 += torch.sum(pred_labels_top5 == labels.view(-1, 1))

epoch_loss = running_loss / len(data_loader_test.dataset)
epoch_acc_top1 = correct_predictions_top1.double() / len(data_loader_val.dataset)
epoch_acc_top5 = correct_predictions_top5.double() / len(data_loader_val.dataset)

print(f"Top-1 Accuracy: {epoch_acc_top1.item() * 100:.2f}%")
print(f"Top-5 Accuracy: {epoch_acc_top5.item() * 100:.2f}%")


# In[ ]:


model.eval()
running_loss = 0.0
correct_counts = {}
total_counts = {}

with torch.no_grad():
    for inputs, labels, _ in data_loader_val:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)

        for true_label, pred_label in zip(labels, preds):
            if true_label.item() in total_counts:
                total_counts[true_label.item()] += 1
            else:
                total_counts[true_label.item()] = 1

            if true_label.item() == pred_label.item():
                if true_label.item() in correct_counts:
                    correct_counts[true_label.item()] += 1
                else:
                    correct_counts[true_label.item()] = 1

epoch_loss = running_loss / len(data_loader_val.dataset)

print("Top-1 Accuracy by True Label:")
for label in sorted(total_counts.keys()):
    accuracy = correct_counts.get(label, 0) / total_counts[label]
    print(f"True Label: {label}, Accuracy: {accuracy * 100:.2f}%")


# In[ ]:


model.eval()
running_loss = 0.0
correct_counts = {}
total_counts = {}

# Lists to store the video paths, true labels, and predicted labels
video_paths = []
true_labels = []
predicted_labels = []

with torch.no_grad():
    for inputs, labels, video_path in data_loader_val:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        running_loss += loss.item() * inputs.size(0)

        for video_path, true_label, pred_label in zip(video_path, labels, preds):
            # Append the video path, true label, and predicted label to the respective lists
            video_paths.append(video_path)
            true_labels.append(true_label.item())
            predicted_labels.append(pred_label.item())

print("Videos with All Labels:")
for video_path, true_label, predicted_label in zip(video_paths, true_labels, predicted_labels):
    print(f"Video Path: {video_path}, True Label: {true_label}, Predicted Label: {predicted_label}")

