from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans

def lr_scheduler(epoch, lr):
    decay_rate = 0.1
    decay_step = 50
    if epoch % decay_step == 0 and epoch:
        return lr * decay_rate
    return lr

def preprocess_input(img, imgNorm="sub_mean"):
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
    return img
        
def adjustData(img,mask, imgNorm="sub_mean" , binary = False, multiclass = True,num_class=2):
    
    img = preprocess_input(img,imgNorm)
    
    if multiclass:           
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            new_mask[mask == i,i] = 1       
        new_mask = new_mask.astype(np.uint8)  
    else:
        if binary:
            new_mask = mask
            new_mask[mask>0] = 1
            new_mask = new_mask.astype(np.float32)
        else:
            new_mask = mask / np.max(mask)
            new_mask = new_mask.astype(np.float32)
    
    return (img,new_mask)

def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    imgNorm="sub_mean" , binary = False, multiclass = True, num_class=2, save_to_dir = None, 
                   target_size = (240,240),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
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
        img,mask = adjustData(img, mask, imgNorm, binary, multiclass, num_class)
        yield (img,mask)

def validationGenerator(batch_size,train_path,image_folder,mask_folder,image_color_mode = "rgb",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    imgNorm="sub_mean" , binary = False, multiclass = True, num_class=2, save_to_dir = None,
                    target_size = (240,240),seed = 1):
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
        img,mask = adjustData(img, mask, imgNorm, binary, multiclass, num_class)
        yield (img,mask)

def testGenerator(test_path,num_image = 30,target_size = (256,256),as_gray = False):
    for im in sorted(os.listdir(test_path)):
        img = io.imread(os.path.join(test_path,im),as_gray = as_gray)
        img = img / 255
        img = trans.resize(img,target_size)
        img = np.reshape(img,(1,)+img.shape)
        yield img