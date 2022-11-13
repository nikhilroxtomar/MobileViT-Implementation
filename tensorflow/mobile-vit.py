
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers as L


def inverted_residual_block(inputs, num_filters, strides=1, expansion_ratio=1):
    ## Point-Wise Convolution
    x = L.Conv2D(
        filters=expansion_ratio*inputs.shape[-1],
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Depth-Wise Convolution
    x = L.DepthwiseConv2D(
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Point-Wise Convolution
    x = L.Conv2D(
        filters=num_filters,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)

    ## Residual Connection
    if strides == 1 and (inputs.shape == x.shape):
        return L.Add()([inputs, x])
    return x


def mlp(x, mlp_dim, dim, dropout_rate=0.1):
    x = L.Dense(mlp_dim, activation="swish")(x)
    x = L.Dropout(dropout_rate)(x)
    x = L.Dense(dim)(x)
    x = L.Dropout(dropout_rate)(x)
    return x

def transformer_encoder(x, num_heads, dim, mlp_dim):
    skip_1 = x
    x = L.LayerNormalization()(x)
    x = L.MultiHeadAttention(
        num_heads=num_heads, key_dim=dim
    )(x, x)
    x = L.Add()([x, skip_1])

    skip_2 = x
    x = L.LayerNormalization()(x)
    x = mlp(x, mlp_dim, dim)
    x = L.Add()([x, skip_2])

    return x

def mobile_vit_block(inputs, num_filters, dim, patch_size=2, num_layers=1):
    B, H, W, C = inputs.shape

    ## 3x3 conv
    x = L.Conv2D(
        filters=C,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## 1x1 conv: d-dimension
    x = L.Conv2D(
        filters=dim,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Reshape x to flattened patches
    P = patch_size*patch_size
    N = int(H*W//P)
    x = L.Reshape((P, N, dim))(x)

    ## Transformr Encoder
    for _ in range(num_layers):
        x = transformer_encoder(x, 1, dim, dim*2)

    ## Reshape
    x = L.Reshape((H, W, dim))(x)

    ## 1x1 conv: C-dimension
    x = L.Conv2D(
        filters=C,
        kernel_size=1,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Concatenation
    x = L.Concatenate()([x, inputs])

    ## 3x3 conv
    x = L.Conv2D(
        filters=num_filters,
        kernel_size=3,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    return x

def MobileViT(input_shape, num_channels, dim, expansion_ratio, num_layers=[2, 4, 3], num_classes=1000):
    ## Input layer
    inputs = L.Input(input_shape)

    ## Stem
    x = L.Conv2D(
        filters=num_channels[0],
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False
    )(inputs)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)
    x = inverted_residual_block(x, num_channels[1], strides=1, expansion_ratio=expansion_ratio)

    ## Stage 1
    x = inverted_residual_block(x, num_channels[2], strides=2, expansion_ratio=expansion_ratio)
    x = inverted_residual_block(x, num_channels[3], strides=1, expansion_ratio=expansion_ratio)
    x = inverted_residual_block(x, num_channels[4], strides=1, expansion_ratio=expansion_ratio)

    ## Stage 2
    x = inverted_residual_block(x, num_channels[5], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[6], dim[0], num_layers=num_layers[0])

    ## Stage 3
    x = inverted_residual_block(x, num_channels[7], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[8], dim[1], num_layers=num_layers[1])

    ## Stage 4
    x = inverted_residual_block(x, num_channels[9], strides=2, expansion_ratio=expansion_ratio)
    x = mobile_vit_block(x, num_channels[10], dim[2], num_layers=num_layers[2])
    x = L.Conv2D(
        filters=num_channels[11],
        kernel_size=1,
        padding="same",
        use_bias=False
    )(x)
    x = L.BatchNormalization()(x)
    x = L.Activation("swish")(x)

    ## Classifier
    x = L.GlobalAveragePooling2D()(x)
    outputs = L.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.models.Model(inputs, outputs)
    return model


def MobileViT_S(input_shape, num_classes):
    num_channels = [16, 32, 64, 64, 64, 96, 144, 128, 192, 160, 240, 640]
    dim = [144, 192, 240]
    expansion_ratio = 4

    return MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=num_classes
    )

def MobileViT_XS(input_shape, num_classes):
    num_channels = [16, 32, 48, 48, 48, 64, 96, 80, 120, 96, 144, 384]
    dim = [96, 120, 144]
    expansion_ratio = 4

    return MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=num_classes
    )

def MobileViT_XXS(input_shape, num_classes):
    num_channels = [16, 16, 24, 24, 24, 48, 64, 64, 80, 80, 96, 320]
    dim = [64, 80, 96]
    expansion_ratio = 2

    return MobileViT(
        input_shape,
        num_channels,
        dim,
        expansion_ratio,
        num_classes=num_classes
    )


if __name__ == "__main__":


    model = MobileViT_XXS((256, 256, 3), 1000)
    model.summary()
