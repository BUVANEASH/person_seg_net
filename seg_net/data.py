from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from keras.applications import imagenet_utils

def weighted_cross_entropy(beta):
    '''
    Weighted Cross Entropy loss function

    Args:
        beta (float): beta weight value

    Returns:
        A loss function that accepts (y_true, y_pred)
    '''
  
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())

        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
        
        return tf.reduce_mean(loss)
    
    return loss

def lr_scheduler(epoch, lr):
    '''
    Learning Rate Scheduler that reduces the learning according to the epoch and decay rate.

    Args:
        epoch (int): Epoch Number.
        lr (float): Learning Rate

    Returns:
        Deacayed Learining Rate.
    '''
    decay_rate = 0.1
    decay_step = 50
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def preprocess_input(img, imgNorm="sub_mean"):
    '''
    Preprocess the Input.

    Args:
        img (uint8): Input image.
        imgNorm (str): Normalization Type
            - "sub_and_divide"
            - "sub_mean"
            - "divide"

    Returns:
        Normalized Image ``float32``

    '''
    if imgNorm == "sub_and_divide":
        img = np.float32(img) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = img.astype(np.float32)
        img[:,:,0] -= 103.939
        img[:,:,1] -= 116.779
        img[:,:,2] -= 123.68
    elif imgNorm == "divide":
        img = img.astype(np.float32)
        img = img/255.0
    elif imgNorm == "imagenet":
        img = imagenet_utils.preprocess_input(img, mode='tf')
    else:
        img = img

    return img
        
def adjustData(img,mask, imgNorm="sub_mean" , binary = False, multiclass = True, alpha = True, num_class=2):
    '''
    Adjusts Image and Annotation before feeding to model training.

    Args:
        img (uint8): Input Image.
        mask (uint8): Mask Annotation Image.
        imgNorm (str): Image Normalization type.
             "sub_and_divide"
            - "sub_mean"
            - "divide"
        binary (bool): Binary Class or Not.
        multiclass (bool): Multiclass or Not.
        alpha (bool): To have two class layers for background (1-mask) and foreground (mask).
        num_class (int): Number of classes.

    Returns:
        Tuple of Image and Mask Annotation Pair of type ``float32``.
    '''
    
    img = preprocess_input(img,imgNorm)
    dnum = len(mask.shape)

    if binary:
        mask[mask>0] = 1
    
    if multiclass:        
        mask = mask[:,:,:,0] if (dnum == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        if alpha and (num_class == 2):            
            if (dnum == 4):
                new_mask[:,:,:,0] = (1 - mask) # Background
                new_mask[:,:,:,1] = mask # Foreground
            else:
                new_mask[:,:,0] = (1 - mask) # Background
                new_mask[:,:,1] = mask # Foreground
        else:      
            for i in range(num_class):
                new_mask[mask == i,i] = 1       
            new_mask = new_mask.astype(np.float32)  
    else:
        new_mask = np.array( mask / np.max(mask) ,dtype=np.float32)
    
    return (img,new_mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    imgNorm="divide" , binary = False, multiclass = True, num_class=2, alpha =False, save_to_dir = None, 
                   target_size = (240,240),seed = 1):
    '''
    Generates Train image and mask pairs.
    
    Notes:
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same \
        if you want to visualize the results of generator, set save_to_dir = "your path".

    Args:
        batch_size (int): Batch Size.
        train_path (str): Tarining images directory.
        image_folder (str): Train image folder name.
        mask_folder (str): Train mask folder name.
        image_color_mode (str): Color mode to read input images.
        mask_color_mode (str): Color mode to read mask images.
        image_save_prefix (str): prefix to save input images.
        mask_save_prefix (str): prefix to save mask images.
        imgNorm (str): Image Normalization type.
             "sub_and_divide"
            - "sub_mean"
            - "divide" 
        binary (bool): Binary Class or Not.
        multiclass (bool): Multiclass or Not.
        num_class (int): Number of classes.
        save_to_dir (str): Directory to save the augmented image and mask pair.
        target_size (tuple(int)): Target image size to resize.
        seed (float): Random seed.

    Yeilds:
        Image and Mask Annotation Pairs.
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img, mask, imgNorm, binary, multiclass, alpha, num_class)
        yield (img,mask)

def validationGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    imgNorm="sub_mean" , binary = False, multiclass = True, num_class=2, alpha =True, save_to_dir = None,
                    target_size = (240,240),seed = 1):
    '''
    Generates Validation image and mask pairs.
    
    Notes:
        use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same \
        if you want to visualize the results of generator, set save_to_dir = "your path".

    Args:
        batch_size (int): Batch Size.
        train_path (str): Tarining images directory.
        image_folder (str): Train image folder name.
        mask_folder (str): Train mask folder name.
        image_color_mode (str): Color mode to read input images.
        mask_color_mode (str): Color mode to read mask images.
        image_save_prefix (str): prefix to save input images.
        mask_save_prefix (str): prefix to save mask images.
        imgNorm (str): Image Normalization type.
             "sub_and_divide"
            - "sub_mean"
            - "divide" 
        binary (bool): Binary Class or Not.
        multiclass (bool): Multiclass or Not.
        num_class (int): Number of classes.
        save_to_dir (str): Directory to save the augmented image and mask pair.
        target_size (tuple(int)): Target image size to resize.
        seed (float): Random seed.

    Yeilds:
        Image and Mask Annotation Pairs.
    '''
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img, mask, imgNorm, binary, multiclass, alpha, num_class)
        yield (img,mask)

def testGenerator(test_path,target_size = (256,256),as_gray = False):
    '''
    Generates Test Image and mask pairs.

    Args:
        test_path (str): Test images path directory.
        target_size (tupple(int)) : Target image size to resize.
        as_gray (bool): Whether to read image as gray scale.
    '''
    for im in sorted(os.listdir(test_path)):
        img = io.imread(os.path.join(test_path,im),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img