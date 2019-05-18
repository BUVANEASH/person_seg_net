import tensorflow as tf
import tensorflow.keras.backend as K
# 1,259,685

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def conv_block(inputs, conv_type, kernel, kernel_size, strides, padding='same', relu=True, batchnorm = True):
    '''Custom function for conv2d --> conv_block'''
    
    if(conv_type == 'ds'):
        x = tf.keras.layers.SeparableConv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)
    else:
        x = tf.keras.layers.Conv2D(kernel, kernel_size, padding=padding, strides = strides)(inputs)  

    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)

    if relu:
        x = tf.keras.layers.Activation('relu')(x)

    return x

def _res_bottleneck(inputs, filters, kernel, t, strides, residue=False):
    '''Residual custom method'''
    
    in_channels = K.int_shape(inputs)[-1]
    tchannel = in_channels * t
    pointwise_filters = _make_divisible(filters, 8) # Referenced from keras_application mobilenet_v2
    x = inputs

    x = conv_block(x, 'conv', tchannel, (1, 1), strides=(1, 1), relu=False)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = tf.keras.layers.DepthwiseConv2D(kernel, strides=strides, depth_multiplier=1, padding='same')(x)    
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU(6.0)(x)
    x = conv_block(x, 'conv', pointwise_filters, (1, 1), strides=(1, 1), padding='same', relu=False)

    if residue and in_channels == pointwise_filters and strides == (1,1) :
        x = tf.keras.layers.add([x, inputs])
    
    return x

def bottleneck_block(inputs, filters, kernel, t, strides, n):
    '''Bottleneck custom method'''
    x = _res_bottleneck(inputs, filters, kernel, t, strides, False)
    for i in range(1,n):
        x = _res_bottleneck(x, filters, kernel, t, (1,1), True)
    return x

def pyramid_pooling_block(input_tensor, bin_sizes):
    '''Pyramid pooling block Method'''

    b,w,h,c = tf.keras.backend.int_shape(input_tensor)
    concat_list = [input_tensor]

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w//bin_size, h//bin_size), strides=(w//bin_size, h//bin_size))(input_tensor)
        x = conv_block(x, 'conv', 128, (1, 1), strides=(1, 1), relu=True)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize_images(x, (w,h)))(x)
        concat_list.append(x)

    return tf.keras.layers.Concatenate()(concat_list)

def fast_scnn(pretrained = None, input_shape = (1024, 2048, 3), num_classes = 19, activation = 'softmax', dropout_rate = 0.3):
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
    
    # Input Layer
    input_layer = tf.keras.layers.Input(shape = input_shape, name = 'input_layer')

    lds_layer = conv_block(input_layer, 'conv', 32, (3, 3), strides = (2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 48, (3, 3), strides = (2, 2))
    lds_layer = conv_block(lds_layer, 'ds', 64, (3, 3), strides = (2, 2))
    
    """#### Assembling all the methods"""
    gfe_layer = bottleneck_block(lds_layer, 64, (3, 3), t = 6, strides = (2, 2), n = 3)
    gfe_layer = bottleneck_block(gfe_layer, 96, (3, 3), t = 6, strides = (2, 2), n = 3)
    gfe_layer = bottleneck_block(gfe_layer, 128, (3, 3), t = 6, strides = (1, 1), n = 3)
    gfe_layer = pyramid_pooling_block(gfe_layer, [1,2,3,6])

    """## Step 3: Feature Fusion"""

    ff_layer1 = conv_block(lds_layer, 'conv', 128, (1, 1), padding = 'same', strides = (1, 1), relu=False)

    ff_layer2 = tf.keras.layers.UpSampling2D((4, 4))(gfe_layer)
    ff_layer2 = tf.keras.layers.DepthwiseConv2D((3,3), strides=(1, 1), depth_multiplier=1, padding='same')(ff_layer2)
    ff_layer2 = tf.keras.layers.BatchNormalization()(ff_layer2)
    ff_layer2 = tf.keras.layers.Activation('relu')(ff_layer2)
    ff_layer2 = tf.keras.layers.Conv2D(128, 1, strides = (1, 1), padding='same', activation=None)(ff_layer2)

    ff_final = tf.keras.layers.add([ff_layer1, ff_layer2])
    ff_final = tf.keras.layers.BatchNormalization()(ff_final)
    ff_final = tf.keras.layers.Activation('relu')(ff_final)

    """## Step 4: Classifier"""
    classifier =  conv_block(ff_final, 'ds', 128, (3, 3), strides = (1, 1), relu=True, batchnorm = True)
    classifier =  conv_block(classifier, 'ds', 128, (3, 3), strides = (1, 1), relu=True, batchnorm = True)

    classifier = conv_block(classifier, 'conv', num_classes, (1, 1), strides=(1, 1), padding='same', relu=True, batchnorm = True)
    if dropout_rate:
        classifier = tf.keras.layers.Dropout( rate = dropout_rate )(classifier)
    classifier = tf.keras.layers.UpSampling2D((8, 8))(classifier)
    classifier = tf.keras.layers.Activation(activation, name = 'output_layer')(classifier)

    """## Model Compilation"""

    fast_scnn = tf.keras.Model(inputs = input_layer , outputs = classifier, name = 'Fast_SCNN')
    
    if pretrained:
        fast_scnn.load_weights(pretrained)
    
    return fast_scnn
    
    
    