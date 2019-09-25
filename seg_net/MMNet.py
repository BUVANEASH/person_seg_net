#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 18:28:25 2019

@author: avantariml
"""
import keras
import keras.backend as K

def alpha_loss(y_true,y_pred):
    return K.mean(K.abs(y_pred - y_true))
 
def comp_loss(y_true,y_pred,images):
    reconst_fg = K.tf.multiply(images, y_pred)
    true_fg = K.tf.multiply(images, y_true)
    return K.tf.losses.absolute_difference(reconst_fg, true_fg)

def grad_loss(y_true,y_pred):
    filter_x = K.tf.constant([[-1/8, 0, 1/8],
                            [-2/8, 0, 2/8],
                            [-1/8, 0, 1/8]],
                           name="sober_x", dtype=K.tf.float32, shape=[3, 3, 1, 1])
    filter_y = K.tf.constant([[-1/8, -2/8, -1/8],
                            [0, 0, 0],
                            [1/8, 2/8, 1/8]],
                           name="sober_y", dtype=K.tf.float32, shape=[3, 3, 1, 1])
    filter_xy = K.tf.concat([filter_x, filter_y], axis=-1)

    grad_alpha = K.tf.nn.conv2d(y_pred, filter_xy, strides=[1, 1, 1, 1], padding="SAME")
    grad_masks = K.tf.nn.conv2d(y_true, filter_xy, strides=[1, 1, 1, 1], padding="SAME")

    return K.tf.losses.absolute_difference(grad_alpha, grad_masks)

def kl_loss(y_true,y_pred):
    return keras.losses.kullback_leibler_divergence(y_true,y_pred)

def custom_loss(input_layer, aux_layer):
    def loss(y_true,y_pred):
        return alpha_loss(y_true,y_pred) + comp_loss(y_true,y_pred,input_layer) + grad_loss(y_true,y_pred) + kl_loss(y_true,y_pred) + kl_loss(y_true,aux_layer)
    return loss

def multiply_depth(depth, depth_multiplier, min_depth=8, divisor=8):
    multiplied_depth = round(depth * depth_multiplier)
    divisible_depth = (multiplied_depth + divisor // 2) // divisor * divisor
    return max(min_depth, divisible_depth)

def DW_Sep_Conv_BN(filters, prefix, stride=1, kernel_size=3, rate=1, use_batchnorm=True, relu=True, relu6 = True, epsilon=1e-3):
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
    
        x = keras.layers.DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate),
                                         padding=depth_padding, use_bias=False, name=prefix + '_depthwise')(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name=prefix + '_depthwise_BN', epsilon=epsilon)(x)
        if relu:
            x = keras.layers.ReLU(max_value=6.0 if relu6 else None, name= prefix+'_1_Relu'+('6' if relu6 else ''))(x)
        
        if filters:
            x = keras.layers.Conv2D(filters, (1, 1), padding='same',
                                    use_bias=False, name=prefix + '_pointwise')(x)
            if use_batchnorm:
                x = keras.layers.BatchNormalization(name=prefix + '_pointwise_BN', epsilon=epsilon)(x)
            if relu:
                x = keras.layers.ReLU(max_value=6.0 if relu6 else None, name= prefix+'_2_Relu'+('6' if relu6 else ''))(x)
    
        return x
    
    return layer

def ConvRelu(filters, kernel_size = 3, stride = 1, use_batchnorm=False, use_relu = False, relu6 = True, prefix='conv', padding="same"):
    def layer(x):
        x = keras.layers.Conv2D(filters, (kernel_size,kernel_size), strides = (stride,stride), 
                                padding=padding, name= prefix+'_pointwise_conv', use_bias=False)(x)
        if use_batchnorm:
            x = keras.layers.BatchNormalization(name= prefix+'_pointwise_BN')(x)
        if use_relu:
            x = keras.layers.ReLU(max_value=6.0 if relu6 else None, name= prefix+'_pointwise_Relu'+('6' if relu6 else ''))(x)
        return x
    return layer


def encoder_block(inputs, expanded_depth, output_depth, depth_multiplier, rates, stride, name):
    
    expanded_depth = multiply_depth(expanded_depth, depth_multiplier)
    output_depth = multiply_depth(output_depth, depth_multiplier)
    
    convs = []
    for i, rate in enumerate(rates):
        x = ConvRelu(expanded_depth, kernel_size = 1, stride = 1, 
                     use_batchnorm=True, use_relu = True, relu6 = True, prefix=name+"_branch_{}".format(i), padding="same")(inputs)
        if stride > 1:
            x = DW_Sep_Conv_BN(filters=None, prefix=name+"_branch_{}_strided".format(i), stride=stride, kernel_size=3, rate=1, 
                               use_batchnorm=True, relu=True, relu6 = True, epsilon=1e-3)(x)
        
        x = DW_Sep_Conv_BN(filters=None, prefix=name+"_branch_{}_dialation".format(i), stride=1, kernel_size=3, rate=rate, 
                               use_batchnorm=True, relu=True, relu6 = True, epsilon=1e-3)(x)
                
        convs.append(x)
        
    if len(convs) > 1:
        x = keras.layers.Concatenate()(convs)
    else:
        x = convs[0]
        
    x = ConvRelu(output_depth, kernel_size = 1, stride = 1, 
                 use_batchnorm=True, use_relu = False, relu6 = False, prefix=name+"_merge", padding="same")(x)
    
#    x = keras.layers.Dropout(0.3)(x)
    
    return x

def decoder_block(inputs, shortcut_input, compressed_depth, shortcut_depth, depth_multiplier, num_of_resize, name):
    
    compressed_depth = multiply_depth(compressed_depth, depth_multiplier)
    
    if shortcut_depth is not None:
        shortcut_depth = multiply_depth(shortcut_depth, depth_multiplier)
    
    x = inputs
    x = ConvRelu(compressed_depth, kernel_size = 1, stride = 1, 
                 use_batchnorm=True, use_relu = True, relu6 = True, prefix=name, padding="same")(x)
    
    for i in range(num_of_resize):
        resize_shape = [v * 2**(i+1) for v in K.int_shape(inputs)[1:3]]
        x = keras.layers.Lambda(lambda xx: K.tf.image.resize_bilinear(xx,
                                                resize_shape,
                                                align_corners=True))(x)
        
    if shortcut_input is not None:
        shortcut = DW_Sep_Conv_BN(filters=shortcut_depth, prefix=name+"refine", stride=1, kernel_size=3, rate=1, 
                               use_batchnorm=True, relu=True, relu6 = True, epsilon=1e-3)(shortcut_input)
        
        x = keras.layers.Concatenate()([x,shortcut])
        
    return x

def init_block(inputs, depth, depth_multiplier, name):
    depth = multiply_depth(depth, depth_multiplier)
    x = ConvRelu(depth, kernel_size = 3, stride = 2, 
                 use_batchnorm=False, use_relu = True, relu6 = True, prefix=name, padding="same")(inputs)
    return x

def final_block(inputs, num_outputs, name):
    x = ConvRelu(num_outputs, kernel_size = 1, stride = 1, 
                 use_batchnorm=False, use_relu = True, relu6 = True, prefix=name, padding="same")(inputs)
    return x

def MMNet(pretrained = None,
          input_shape = (256,256,3),
          input_tensor=None,
          depth_multiplier = 1.0,
          classes = 1,
          activation = 'sigmoid'):
    
    if input_tensor is None:
        img_input = keras.layers.Input(shape=input_shape)
    else:
        img_input = input_tensor
        
    endpoints = {}
        
    endpoints["init_block"] = init_block(img_input, 32, depth_multiplier, "init_block")
    
    endpoints["enc_block1"] = encoder_block(endpoints["init_block"], 16, 16, depth_multiplier, [1, 2, 4, 8], 2, "enc_block1")
    endpoints["enc_block2"] = encoder_block(endpoints["enc_block1"], 16, 24, depth_multiplier, [1, 2, 4, 8], 1, "enc_block2")
    endpoints["enc_block3"] = encoder_block(endpoints["enc_block2"], 24, 24, depth_multiplier, [1, 2, 4, 8], 1, "enc_block3")
    endpoints["enc_block4"] = encoder_block(endpoints["enc_block3"], 24, 24, depth_multiplier, [1, 2, 4, 8], 1, "enc_block4")
    endpoints["enc_block5"] = encoder_block(endpoints["enc_block4"], 32, 40, depth_multiplier, [1, 2, 4], 2, "enc_block5")
    endpoints["enc_block6"] = encoder_block(endpoints["enc_block5"], 64, 40, depth_multiplier, [1, 2, 4], 1, "enc_block6")
    endpoints["enc_block7"] = encoder_block(endpoints["enc_block6"], 64, 40, depth_multiplier, [1, 2, 4], 1, "enc_block7")
    endpoints["enc_block8"] = encoder_block(endpoints["enc_block7"], 64, 40, depth_multiplier, [1, 2, 4], 1, "enc_block8")
    endpoints["enc_block9"] = encoder_block(endpoints["enc_block8"], 80, 80, depth_multiplier, [1, 2], 2, "enc_block9")
    endpoints["enc_block10"] = encoder_block(endpoints["enc_block9"], 120, 80, depth_multiplier, [1, 2], 1, "enc_block10")

    endpoints["dec_block1"] = decoder_block(endpoints["enc_block10"], endpoints["enc_block5"], 64, 64, depth_multiplier, 1, "dec_block0")
    endpoints["dec_block2"] = decoder_block(endpoints["dec_block1"], endpoints["enc_block1"], 40, 40, depth_multiplier, 1, "dec_block1")

    endpoints["enhancement_block1"] = encoder_block(endpoints["dec_block2"], 40, 40, depth_multiplier, [1, 2, 4], 1, "enhancement_block1")
    endpoints["enhancement_block2"] = encoder_block(endpoints["enhancement_block1"], 40, 40, depth_multiplier, [1, 2, 4], 1, "enhancement_block2")

    endpoints["dec_block3"] = decoder_block(endpoints["enhancement_block2"], None, 16, None, depth_multiplier, 2, "dec_block3")

    # Final Deconvolution
    endpoints["final_block"] = final_block(endpoints["dec_block3"], num_outputs=classes, name="final_block")
    endpoints["classifier"] = keras.layers.Activation(activation)(endpoints["final_block"])

    # aux output
    endpoints["aux_block0"] = final_block(endpoints["enc_block10"], num_outputs=classes, name="aux_block0")
    endpoints["aux_block"] = keras.layers.Lambda(lambda x: K.tf.image.resize_images(x, input_shape[:2]))(endpoints["aux_block0"])
    
    mmnet = keras.Model(inputs = img_input , outputs = endpoints["classifier"], name = 'MMNet')
    
    if pretrained:
        mmnet.load_weights(pretrained, by_name=True)
    
    return mmnet, endpoints 