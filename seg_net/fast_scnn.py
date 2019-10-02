#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
sys.path.append("..")
import segmentation_models as sm
import keras
import keras.backend as K
from seg_net.mobilenet_v2 import MobileNetV2
# 1,259,685

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

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

def ConvRelu(filters, kernel_size = 3, stride = 1, use_batchnorm=False, use_relu = False, conv_name='conv', bn_name='bn', relu_name='relu', padding="same"):
    def layer(x):
        x = keras.layers.Conv2D(filters, (kernel_size,kernel_size), strides = (stride,stride), padding=padding, name=conv_name, use_bias=False)(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=bn_name)(x)
        if use_relu:
            x = keras.layers.Activation('relu', name=relu_name)(x)
        return x
    return layer

def conv_block(inputs, name, conv_type, kernel, kernel_size, strides, padding='same', relu=True, batchnorm = True):
    '''Custom function for conv2d --> conv_block'''
    
    if(conv_type == 'ds'):        
        x = SepConv_BN(kernel, name+'_ds', stride=strides, kernel_size=kernel_size, rate=1, use_batchnorm=batchnorm, depth_activation=relu, epsilon=1e-3)(inputs)
    else:
        x = ConvRelu(kernel, kernel_size = kernel_size, stride = strides, use_batchnorm=batchnorm, use_relu=relu,
                     conv_name=name+'_conv', bn_name=name+'_bn', relu_name=name+'_relu', padding=padding)(inputs)

    return x

def _res_bottleneck(inputs, name, filters, kernel, t, strides, residue=False):
    '''Residual custom method'''
    
    in_channels = K.int_shape(inputs)[-1]
    tchannel = in_channels * t
    pointwise_filters = _make_divisible(filters, 8) # Referenced from keras_application mobilenet_v2
    x = inputs

    x = conv_block(x, name + '_res_1', 'conv', tchannel, 1, strides=1, relu=False)
    x = keras.layers.ReLU(6.0)(x)
    x = keras.layers.DepthwiseConv2D(kernel, strides=strides, padding = 'same', depth_multiplier=1)(x)    
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU(6.0)(x)
    x = conv_block(x, name + '_res_2', 'conv', pointwise_filters, 1, strides=1, relu=False)

    if residue and in_channels == pointwise_filters and strides == 1 :
        x = keras.layers.add([x, inputs])
    
    return x

def bottleneck_block(inputs, prefix, filters, kernel, t, strides, n):
    '''Bottleneck custom method'''
    x = _res_bottleneck(inputs, prefix+'_bottelneck_0', filters, kernel, t, strides, False)
    for i in range(1,n):
        x = _res_bottleneck(x, prefix+'_bottelneck_{}'.format(i), filters, kernel, t, (1,1), True)
    return x

def pyramid_pooling_block(input_tensor, channel, bin_sizes):
    '''Pyramid pooling block Method'''

    b,w,h,c = keras.backend.int_shape(input_tensor)
    concat_list = [input_tensor]

    for bin_size in bin_sizes:
        x = keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
        x = conv_block(x, 'PPM_bin_{}'.format(bin_size), 'conv', channel//len(bin_sizes), 1, strides=1, relu=True)
        x = keras.layers.Lambda(lambda x: K.tf.image.resize_images(x, (w,h)))(x)
#        x = keras.layers.UpSampling2D((w//bin_size, h//bin_size))(x)
        concat_list.append(x)
        
    x = keras.layers.Concatenate()(concat_list)
    
#    x = conv_block(x, 'PPM_final','conv', channel, 1, strides=1, relu=True)
    
    return x

#input_shape = (1024, 2048, 3)
'''
pretrained = None
backbone = 'mobilenetv2'
weights='imagenet'
include_top=False
input_shape = (224, 224, 3)
input_tensor=None
alpha=1.0
pooling=None
classes = 1
activation = 'sigmoid'
freeze_encoder = False
dropout_rate = 0
'''
def fast_scnn(pretrained = None,
    		 backbone = 'mobilenetv2',
             weights='imagenet',
             include_top=False,
             input_shape = (1024, 2048, 3),
             input_tensor=None,
             alpha=1.0,
             pooling=None,
             classes = 1,
             activation = 'sigmoid',
             freeze_encoder = False,
             dropout_rate = 0.3):
    '''
    Fast SCNN Model

    Args:
        pretrained (str): Path to pretrained model.
        input_shape (tuple(int)): Target image size.
        num_classes (int): Number of classes.
        activation (str): Last layer activation function.

    Returns:
        Keras Model of fast_scnn.
    '''

#    if backbone in {'mobilenetv2'}:
#        assert input_shape[0] == input_shape[1]
#        assert input_shape[0]%48 == 0 and input_shape[1]%48 == 0
        
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        img_input = input_tensor
        
    # Input Layer
    if backbone in {'mobilenetv2'}:
        encoder = MobileNetV2(input_shape=None,
                               alpha=alpha,
                               include_top=False,
                               weights='imagenet',
                               input_tensor=img_input,
                               pooling=pooling)
        
        encoder_features = sm.backbones.get_feature_layers(backbone, n=4)
    
        skip_connections = ([sm.utils.get_layer_number(encoder, l) if isinstance(l, str) else l
                                                                        for l in encoder_features])
    
    
        if freeze_encoder:
            for layer in encoder.layers:
                if not isinstance(layer, keras.layers.BatchNormalization):
                    layer.trainable = False
            
        lds_layer = encoder.layers[skip_connections[-1]].output
        gfe_layer = encoder.layers[-1].output
    else:
        lds_layer = conv_block(img_input, 'lds_1', 'conv', 32, 3, strides = 2)
        lds_layer = conv_block(lds_layer, 'lds_2','ds', 48, 3, strides = 2)
        lds_layer = conv_block(lds_layer, 'lds_3','ds', 64, 3, strides = 2)
        
        """#### Assembling all the methods"""        
        gfe_layer = bottleneck_block(lds_layer, 'Block_1', 64, (3, 3), t = 6, strides = (2, 2), n = 3)
        gfe_layer = bottleneck_block(gfe_layer, 'Block_2', 96, (3, 3), t = 6, strides = (2, 2), n = 3)
        gfe_layer = bottleneck_block(gfe_layer, 'Block_3', 128, (3, 3), t = 6, strides = (1, 1), n = 3)
        
    gfe_layer = pyramid_pooling_block(gfe_layer, 128, [1,2,3,6])

    """## Step 3: Feature Fusion"""

    ff_layer1 = conv_block(lds_layer, 'ff_1', 'conv', 128, 1, padding = 'same', strides = 1, relu=False)
    
    if backbone in {'mobilenetv2'}:
        ff_layer2 = keras.layers.UpSampling2D((16, 16))(gfe_layer)
    else:
        ff_layer2 = keras.layers.UpSampling2D((4, 4))(gfe_layer)
        
    ff_layer2 = keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    ff_layer2 = keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = keras.layers.Activation('relu')(ff_layer2)
    ff_layer2 = keras.layers.Conv2D(128, 1, strides = (1, 1), padding='same', activation=None)(ff_layer2)

    ff_final = keras.layers.add([ff_layer1, ff_layer2])
    ff_final = keras.layers.BatchNormalization()(ff_final)
    ff_final = keras.layers.Activation('relu')(ff_final)

    """## Step 4: Classifier"""
    classifier =  conv_block(ff_final, 'cl_1', 'ds', 128, 3, strides = 1, relu=True, batchnorm = True)
    classifier =  conv_block(classifier, 'cl_2', 'ds', 128, 3, strides = 1, relu=True, batchnorm = True)

    classifier = conv_block(classifier, 'cl_3', 'conv', classes, 1, strides=1, relu=True, batchnorm = True)
    
    if dropout_rate:
        classifier = keras.layers.Dropout( rate = dropout_rate )(classifier)
    
    if backbone in {'mobilenetv2'}:
        classifier = keras.layers.UpSampling2D((2, 2))(classifier)
    else:
        classifier = keras.layers.UpSampling2D((8, 8))(classifier)
        
    classifier = keras.layers.Activation(activation)(classifier)

    """## Model Compilation"""
    if backbone in {'mobilenetv2'}:
        fast_scnn = keras.Model(inputs = encoder.input , outputs = classifier, name = 'Fast_SCNN')
    else:
        fast_scnn = keras.Model(inputs = img_input , outputs = classifier, name = 'Fast_SCNN')
    
    if pretrained:
        fast_scnn.load_weights(pretrained, by_name=True)
    
    return fast_scnn
    
    
    