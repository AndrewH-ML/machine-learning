# model/u_net.py

import tf_keras as keras
from tf_keras import layers

def conv_block(inputs=None, n_filters=32, dropout_prob=0, max_pooling=True):
    """
    Builds a convolutional downsampling block consisting of two Conv2D layers,
    optional Dropout, and optional MaxPooling2D.

    Args:
        inputs: Input tensor.
        n_filters: Number of filters for the convolutional layers.
        dropout_prob: Dropout probability.
        max_pooling: Whether to apply MaxPooling2D.
    Returns:
        next_layer, skip_connection: Downsampled tensor and skip connection tensor.
    """
    conv = layers.Conv2D(n_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_initializer='he_normal')(inputs)
    conv = layers.Conv2D(n_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_initializer='he_normal')(conv)
    if dropout_prob > 0:
        conv = layers.Dropout(dropout_prob)(conv)
    if max_pooling:
        next_layer = layers.MaxPooling2D((2, 2))(conv)
    else:
        next_layer = conv
    skip_connection = conv
    return next_layer, skip_connection


def upsampling_block(expansive_input, contractive_input, n_filters=32):
    """
    Builds a convolutional upsampling block using Conv2DTranspose for upsampling,
    followed by concatenation and two Conv2D layers.

    Args:
    
        expansive_input: Input tensor from the previous expansive layer.
        contractive_input: Corresponding tensor from the contracting path (skip connection).
        n_filters: Number of filters for the convolutional layers.
    Returns:
        conv: Output tensor after upsampling and convolution.
    """
    up = layers.Conv2DTranspose(n_filters, kernel_size=3, strides=(2, 2),
                         padding='same')(expansive_input)
    merge = layers.concatenate([up, contractive_input], axis=3)
    conv = layers.Conv2D(n_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_initializer='he_normal')(merge)
    conv = layers.Conv2D(n_filters, kernel_size=3, activation='relu',
                  padding='same', kernel_initializer='he_normal')(conv)
    return conv

def unet_model(input_size=(96, 128, 3), n_filters=32, n_classes=23):
    """
    Unet model
    
    Arguments:
        input_size -- Input shape 
        n_filters -- Number of filters for the convolutional layers
        n_classes -- Number of output classes
    Returns: 
        model -- tf.keras.Model
    """
    inputs = layers.Input(input_size)
    
    # Contractive path (Encoder)

    cblock1 = conv_block(inputs, n_filters=n_filters)
    # We chain the first element of the output of each block to be the input of the next conv_block. 
    # number of filters doubled at each step 
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3) 
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Expanding Path (decoding)

    ublock6 = upsampling_block(cblock5[0], cblock4[1], n_filters*8)
    # We chain the output of the previous block as expansive_input and the corresponding contractive block output.
    ublock7 = upsampling_block(ublock6, cblock3[1],  n_filters*4)
    ublock8 = upsampling_block(ublock7, cblock2[1],  n_filters*2)
    ublock9 = upsampling_block(ublock8, cblock1[1],  n_filters)

    conv9 = layers.Conv2D(n_filters,
                 kernel_size = 3,
                 activation='relu',
                 padding='same',
                 # set 'kernel_initializer' same as above exercises
                 kernel_initializer='he_normal')(ublock9)

    conv10 = layers.Conv2D(n_classes, 
                    kernel_size = 1,
                    padding='same')(conv9)
    model = keras.Model(inputs=inputs, outputs=conv10)

    return model

