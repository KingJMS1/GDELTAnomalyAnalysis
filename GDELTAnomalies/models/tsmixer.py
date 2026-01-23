from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# This TSMixer Implementation taken from https://github.com/ditschuk/pytorch-tsmixer

# License:
# Copyright 2023 Konstantin Ditschuneit
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

class TimeBatchNorm2d(nn.BatchNorm1d):
    """This class extends nn.BatchNorm1d to apply batch normalization across time and
    feature dimensions.

    Attributes:
        num_time_steps (int): Number of time steps in the input.
        num_series (int): Number of time series in the input.
    """

    def __init__(self, normalized_shape: tuple[int, int]):
        """Initializes the TimeBatchNorm2d module.

        Args:
            normalized_shape (tuple[int, int]): A tuple (num_time_steps, num_series)
                representing the shape of the time and feature dimensions to normalize.
        """
        num_time_steps, num_series = normalized_shape
        super().__init__(num_series * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_series = num_series

    def forward(self, x: Tensor) -> Tensor:
        """Applies the batch normalization over the last two dimensions of the input tensor.

        Args:
            x (Tensor): A 3D tensor with shape (N, S, C), where N is the batch size,
                S is the number of time steps, and C is the number of channels.

        Returns:
            Tensor: A 3D tensor with batch normalization applied over the last two dims.

        Raises:
            ValueError: If the input tensor is not 3D.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")

        # Reshaping input to combine time and feature dimensions for normalization
        x = x.reshape(x.shape[0], -1, 1)

        # Applying batch normalization
        x = super().forward(x)

        # Reshaping back to original dimensions (N, S, C)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_series)

        return x


class FeatureMixing(nn.Module):
    """A module for feature mixing with flexibility in normalization and activation.

    This module provides options for batch normalization before or after mixing features,
    uses dropout for regularization, and allows for different activation functions.

    Args:
        lookback: The length of the sequences to be transformed.
        input_series: The number of input channels to the module.
        output_series: The number of output channels from the module.
        ff_dim: The dimension of the feed-forward network internal to the module.
        activation_fn: The activation function used within the feed-forward network.
        dropout_rate: The dropout probability used for regularization.
            the rest of the operations.
    """

    def __init__(
        self,
        lookback: int,
        input_series: int,
        output_series: int,
        ff_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the FeatureMixing module with the provided parameters."""
        super().__init__()

        self.norm = norm_type((lookback, output_series))
        

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_series, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_series)

        self.projection = (
            nn.Linear(input_series, output_series)
            if input_series != output_series
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FeatureMixing module.

        Args:
            x: A 3D tensor with shape (N, C, L) where C is the channel dimension.

        Returns:
            The output tensor after feature mixing.
        """
        x_proj = self.projection(x)

        x = self.fc1(x)  # Apply the first linear transformation.
        x = self.activation_fn(x)  # Apply the activation function.
        x = self.dropout(x)  # Apply dropout for regularization.
        x = self.fc2(x)  # Apply the second linear transformation.
        x = self.dropout(x)  # Apply dropout again if needed.

        x = x_proj + x  # Add the projection shortcut to the transformed features.

        return self.norm(x)



class TimeMixing(nn.Module):
    """Applies a transformation over the time dimension of a sequence.

    This module applies a linear transformation followed by an activation function
    and dropout over the sequence length of the input feature tensor after converting
    feature maps to the time dimension and then back.

    Args:
        input_series: The number of input channels to the module.
        lookback: The length of the sequences to be transformed.
        activation_fn: The activation function to be used after the linear transformation.
        dropout_rate: The dropout probability to be used after the activation function.
    """

    def __init__(
        self,
        lookback: int,
        input_series: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        """Initializes the TimeMixing module with the specified parameters."""
        super().__init__()
        self.norm = norm_type((lookback, input_series))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(lookback, lookback)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the time mixing operations on the input tensor.

        Args:
            x: A 3D tensor with shape (N, C, L), where C = channel dimension and
                L = sequence length.

        Returns:
            The normalized output tensor after time mixing transformations.
        """
        x_temp = feature_to_time(
            x
        )  # Convert feature maps to time dimension. Assumes definition elsewhere.
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)  # Convert back from time to feature maps.

        return self.norm(x + x_res)  # Apply normalization and combine with original input.


class MixerLayer(nn.Module):
    """A residual block that combines time and feature mixing for sequence data.

    This module sequentially applies time mixing and feature mixing, which are forms
    of data augmentation and feature transformation that can help in learning temporal
    dependencies and feature interactions respectively.

    Args:
        lookback: The length of the input sequences.
        input_series: The number of input channels to the module.
        output_series: The number of output channels from the module.
        ff_dim: The inner dimension of the feedforward network used in feature mixing.
        activation_fn: The activation function used in both time and feature mixing.
        dropout_rate: The dropout probability used in both mixing operations.
    """

    def __init__(
        self,
        lookback: int,
        input_series: int,
        output_series: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        """Initializes the MixLayer with time and feature mixing modules."""
        super().__init__()

        self.time_mixing = TimeMixing(
            lookback,
            input_series,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            lookback,
            input_series,
            output_series,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MixLayer module.

        Args:
            x: A 3D tensor with shape (N, C, L) to be processed by the mixing layers.

        Returns:
            The output tensor after applying time and feature mixing operations.
        """
        x = self.time_mixing(x)  # Apply time mixing first.
        x = self.feature_mixing(x)  # Then apply feature mixing.

        return x

def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    """Converts a time series tensor to a feature tensor."""
    return x.permute(0, 2, 1)


feature_to_time = time_to_feature


class TSMixer(nn.Module):
    """TSMixer model for time series forecasting.

    This model uses a series of mixer layers to process time series data,
    followed by a linear transformation to project the output to the desired
    prediction length.

    Attributes:
        mixer_layers: Sequential container of mixer layers.
        temporal_projection: Linear layer for temporal projection.

    Args:
        lookback: Length of the input time series sequence.
        horizon: Desired length of the output prediction sequence.
        input_series: Number of input series.
        output_series: Number of output channels. Defaults to input_series.
        activation_fn: Activation function to use. Defaults to "relu".
        num_blocks: Number of mixer blocks. Defaults to 2.
        dropout_rate: Dropout rate for regularization. Defaults to 0.1.
        ff_dim: Dimension of feedforward network inside mixer layer. Defaults to 64.
        norm_type: Type of normalization to use. "batch" or "layer". Defaults to "batch".
    """

    def __init__(
        self,
        lookback: int,
        horizon: int,
        input_series: int,
        output_series: int = None,
        activation_fn: str = "relu",
        num_blocks: int = 2,
        dropout_rate: float = 0.1,
        ff_dim: int = 64,
        norm_type: str = "batch",
    ):
        super().__init__()

        # Transform activation_fn to callable
        activation_fn = getattr(F, activation_fn)

        # Transform norm_type to callable
        assert norm_type in {
            "batch",
            "layer",
        }, f"Invalid norm_type: {norm_type}, must be one of batch, layer."
        norm_type = TimeBatchNorm2d if norm_type == "batch" else nn.LayerNorm

        # Build mixer layers
        self.mixer_layers = self._build_mixer(
            num_blocks,
            input_series,
            output_series,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            lookback=lookback,
            norm_type=norm_type,
        )

        # Temporal projection layer
        self.temporal_projection = nn.Linear(lookback, horizon)

    def _build_mixer(
        self, num_blocks: int, input_series: int, output_series: int, **kwargs
    ):
        """Build the mixer blocks for the model.

        Args:
            num_blocks (int): Number of mixer blocks to be built.
            input_series (int): Number of input channels for the first block.
            output_series (int): Number of output channels for the last block.
            **kwargs: Additional keyword arguments for mixer layer configuration.

        Returns:
            nn.Sequential: Sequential container of mixer layers.
        """
        output_series = output_series if output_series is not None else input_series
        channels = [input_series] * (num_blocks - 1) + [output_series]

        return nn.Sequential(
            *[
                MixerLayer(input_series=in_ch, output_series=out_ch, **kwargs)
                for in_ch, out_ch in zip(channels[:-1], channels[1:])
            ]
        )

    def forward(self, x_hist: torch.Tensor) -> torch.Tensor:
        """Forward pass of the TSMixer model.

        Args:
            x_hist (torch.Tensor): Input time series tensor.

        Returns:
            torch.Tensor: The output tensor after processing by the model.
        """
        x = self.mixer_layers(x_hist)

        x_temp = feature_to_time(x)
        x_temp = self.temporal_projection(x_temp)
        x = time_to_feature(x_temp)

        return x
