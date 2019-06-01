#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import segmentation_models as sm
import keras
import keras.backend as K
import numpy as np
from seg_net.mobilenet_v2 import MobileNetV2

def handle_block_names(stage):
    conv_name = 'decoder_stage{}_conv'.format(stage)
    bn_name = 'decoder_stage{}_bn'.format(stage)
    relu_name = 'decoder_stage{}_relu'.format(stage)
    up_name = 'decoder_stage{}_upsample'.format(stage)
    return conv_name, bn_name, relu_name, up_name

def SepConv_BN(filters, prefix, stride=1, kernel_size=3, rate=1, use_batchnorm=True, depth_activation=True, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
    """
    def layer(x):
        if stride == 1:
            depth_padding = 'same'
        else:
            kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
            pad_total = kernel_size_effective - 1
            pad_beg = pad_total // 2
            pad_end = pad_total - pad_beg
            x = keras.layers.ZeroPadding2D((pad_beg, pad_end))(x)
            depth_padding = 'valid'
    
        if not depth_activation:
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                            padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = keras.layers.Activation('relu')(x)
        x = keras.layers.Conv2D(filters, (1, 1), padding='same',
                   use_bias=False, name=prefix + '_pointwise')(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
        if depth_activation:
            x = keras.layers.Activation('relu')(x)
    
        return x
    
    return layer

def ConvRelu(filters, kernel_size = 3, stride = 1, use_batchnorm=False, conv_name='conv', bn_name='bn', relu_name='relu'):
    def layer(x):
        x = keras.layers.Conv2D(filters, (kernel_size,kernel_size), strides = (stride,stride), padding="same", name=conv_name, use_bias=not(use_batchnorm))(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=bn_name)(x)
        x = keras.layers.Activation('relu', name=relu_name)(x)
        return x
    return layer

def densenet_matte(filters, classes, kernel_size = 3, stride = 1, use_batchnorm=False, sep_conv = True, n_layers = 3):
    
    def layer(input_tensor):
        x = input_tensor
        
        for i in range(1,n_layers+1):
            if not sep_conv:
                x = keras.layers.Conv2D(filters, (kernel_size,kernel_size) , strides = (stride,stride),
                                                 padding='same', name='densenet_matt{}'.format(i))(x)
            else:
                x = SepConv_BN(filters, 'densenet_matt{}'.format(i), stride=stride, kernel_size=kernel_size, rate=1,
                           use_batchnorm=use_batchnorm, depth_activation=True, epsilon=1e-3)(x)
        
        return x
    
    return layer

def Upsample2D_Block(filters, stage, kernel_size=3, upsample=(256,256), 
                     use_batchnorm=True, depth_activation=True, epsilon=1e-3, skip=None, sep_conv = True):

    def layer(input_tensor):
        
        if not sep_conv:
            conv_name, bn_name, relu_name, up_name = handle_block_names(stage)
        else:
            layer_name = 'decoder_stage{}_'.format(stage)

        x = keras.layers.Lambda(lambda xx: K.tf.image.resize_bilinear(xx,
                                                upsample,
                                                align_corners=True))(input_tensor)
        
        if skip is not None:
            x = keras.layers.Concatenate()([x, skip])
        

        if not sep_conv:  
            x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                     conv_name=conv_name + '1', bn_name=bn_name + '1', relu_name=relu_name + '1')(x)
            x = ConvRelu(filters, kernel_size, use_batchnorm=use_batchnorm,
                         conv_name=conv_name + '2', bn_name=bn_name + '2', relu_name=relu_name + '2')(x)
        else:            
            x = SepConv_BN(filters, layer_name+ '1', stride=1, kernel_size=3, rate=1,
                           use_batchnorm=True, depth_activation=True, epsilon=1e-3)(x)
            
            x = SepConv_BN(filters, layer_name+ '2', stride=1, kernel_size=3, rate=1,
                           use_batchnorm=True, depth_activation=True, epsilon=1e-3)(x)

        return x
    return layer

def Unet(pretrained = None,
		 backbone = 'mobilenetv2',
         weights='imagenet',
         include_top=False,
         input_shape = (256,256,3),
         input_tensor=None,
         alpha=1.0,
         pooling=None,
         classes = 1,
         decoder_filters = (256,128,64,32,16),
         n_upsample_blocks=5,
         activation = 'sigmoid',
         densenet_matting_filters = 16,
         densenet_matting_layers = 3,
         sep_conv = False,
         matt_sep_conv = False,
         freeze_encoder = False):
    
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        img_input = input_tensor
    
    if backbone == 'mobilenetv2':
	    encoder = MobileNetV2(input_shape=None,
                               alpha=alpha,
                               include_top=False,
                               weights='imagenet',
                               input_tensor=img_input,
                               pooling=pooling)
    else:
        raise ValueError('The `backbone` argument should be `mobilenetv2`')

    encoder_features = sm.backbones.get_feature_layers(backbone, n=4)
    
    skip_connection_idx = ([sm.utils.get_layer_number(encoder, l) if isinstance(l, str) else l
                                        for l in encoder_features])

    if freeze_encoder:
        for layer in encoder.layers:
            if not isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False
    
    x = encoder.output

    for i in range(n_upsample_blocks):
            idx = (len(skip_connection_idx)-i)
            # check if there is a skip connection
            skip_connection = None
            if i < len(skip_connection_idx):
                skip_connection = encoder.layers[skip_connection_idx[i]].output
    
            upsample = sm.utils.to_tuple((int(np.ceil(input_shape[0]/2**idx)),int(np.ceil(input_shape[1]/2**idx))))
    
            x = Upsample2D_Block(decoder_filters[i], i, upsample=upsample,
                                 skip=skip_connection, use_batchnorm=True, sep_conv = sep_conv)(x)
    
    if densenet_matting_layers:
        x = densenet_matte(densenet_matting_filters, classes = classes, kernel_size = 3, stride = 1, 
                           use_batchnorm=True, sep_conv = matt_sep_conv, 
                           n_layers = densenet_matting_layers)(x)
                           
    if not sep_conv:
        x = keras.layers.Conv2D(classes, (3,3), padding='same', name='final_conv')(x)
    else:
        x = keras.layers.SeparableConv2D(classes, (3,3), padding='same', name='final_conv')(x)
            
    if activation in {'softmax','sigmoid'}:
        x = keras.layers.Activation(activation, name=activation)(x)
	    
    model = keras.Model(img_input, x)

    if pretrained:
    	print("LOADING ",pretrained)
    	model.load_weights(pretrained, by_name=True)
    
    return model