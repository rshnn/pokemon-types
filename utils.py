"""utilities for data pipelining   
"""

import numpy as np  
import matplotlib.pyplot as plt 
import json 
import tensorflow as tf 


# Global variables 
IMG_HEIGHT = 120
IMG_WIDTH = 120

CHANNELS = 3

## type encodings for plotting  
with open('data/munged/labels.json') as json_file:
    type_encoding = json.load(json_file)

type_encoding = {int(k):v for k, v in type_encoding.items()}
type_encoding


def augment(image):
    image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1] 
    image = tf.image.flip_left_right(image)  
    image = tf.image.flip_up_down(image)  

    return image 


def parse_function_augment(filename, label): 
    """ Returns a tuple of (normalized image array, label)  
    Apply transformation as well (this function is for training images).  
    
    filename:  string representing path to image 
    label: multi-hot encoded array of size N_LABELS 
    """
    
    # Read image from file 
    img_string = tf.io.read_file(filename)
    
    # Decode it into a dense vector 
    img_decoded = tf.image.decode_png(img_string, channels=CHANNELS)
    
    # Resize it to a fixed shape 
    img_resized = tf.image.resize(img_decoded, [IMG_HEIGHT, IMG_WIDTH])
    
    # Normalize it to 0-1 scale 
    img_normed = img_resized / 255.0
    
    img_aug = augment(img_normed)  
        
    return img_aug, label 



def parse_function(filename, label): 
    """ Returns a tuple of (normalized image array, label)  
    
    filename:  string representing path to image 
    label: multi-hot encoded array of size N_LABELS 
    """
    
    # Read image from file 
    img_string = tf.io.read_file(filename)
    
    # Decode it into a dense vector 
    img_decoded = tf.image.decode_png(img_string, channels=CHANNELS)
    
    # Resize it to a fixed shape 
    img_resized = tf.image.resize(img_decoded, [IMG_HEIGHT, IMG_WIDTH])
    
    # Normalize it to 0-1 scale 
    img_normed = img_resized / 255.0
           
    return img_normed, label 




def parse_function_mobilenet_augment(filename, label): 
    """ Returns a tuple of (normalized image array, label)  
    Apply transformation as well (this function is for training images).  
    
    filename:  string representing path to image 
    label: multi-hot encoded array of size N_LABELS 
    """
    
    # Read image from file 
    img_string = tf.io.read_file(filename)
    
    # Decode it into a dense vector 
    img_decoded = tf.image.decode_png(img_string, channels=CHANNELS)
    
    # Resize it to a fixed shape 
    img_resized = tf.image.resize(img_decoded, [160, 160])
    
    # Normalize it to 0-1 scale 
    img_normed = img_resized / 255.0
    
    img_aug = augment(img_normed)  
        
    return img_aug, label 



def parse_function_mobilenet(filename, label): 
    """ Returns a tuple of (normalized image array, label)  
    
    filename:  string representing path to image 
    label: multi-hot encoded array of size N_LABELS 
    """
    
    # Read image from file 
    img_string = tf.io.read_file(filename)
    
    # Decode it into a dense vector 
    img_decoded = tf.image.decode_png(img_string, channels=CHANNELS)
    
    # Resize it to a fixed shape 
    img_resized = tf.image.resize(img_decoded, [160, 160])
    
    # Normalize it to 0-1 scale 
    img_normed = img_resized / 255.0
           
    return img_normed, label 




def create_dataset_mobilenet(filenames, labels, SHUFFLE_BUFFER_SIZE, 
                    AUTOTUNE, BATCH_SIZE, augment=True): 
    """ Load and parse a tf.data.Dataset.  
    
    filenames: list of image paths 
    labels: numpy array of shape (BATCH_SIZE, N_LABELS)
    """

    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if augment is True:  
        # Parse and preprocess observations in parallel
        dataset = dataset.map(parse_function_mobilenet_augment, num_parallel_calls=AUTOTUNE)      
    else: 
        dataset = dataset.map(parse_function_mobilenet, num_parallel_calls=AUTOTUNE)

    
    # This is a small dataset, only load it once, and keep it in memory.
    dataset = dataset.cache()
    # Shuffle the data each buffer size
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset





def create_dataset(filenames, labels, SHUFFLE_BUFFER_SIZE, 
                    AUTOTUNE, BATCH_SIZE, augment=True): 
    """ Load and parse a tf.data.Dataset.  
    
    filenames: list of image paths 
    labels: numpy array of shape (BATCH_SIZE, N_LABELS)
    """
        
    # Create a first dataset of file paths and labels
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    if augment is True:  
        # Parse and preprocess observations in parallel
        dataset = dataset.map(parse_function_augment, num_parallel_calls=AUTOTUNE)      
    else: 

        dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    
    # This is a small dataset, only load it once, and keep it in memory.
    dataset = dataset.cache()
    # Shuffle the data each buffer size
    dataset = dataset.shuffle(buffer_size=SHUFFLE_BUFFER_SIZE)
        
    # Batch the data for multiple steps
    dataset = dataset.batch(BATCH_SIZE)
    # Fetch batches in the background while the model is training.
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    
    return dataset


def plot_prediction_grid(model, dataset, dims=(3, 3),print_types=True):
    """ Plots a prediction grid given an model and dataset 
    
    model:   tf model 
    dataset:  tf.data.Dataset 
    print_types:  If True, will lookup string names for types 
    """
    i, j = dims

    fig, axes = plt.subplots(i, j, figsize=(3*i, 3*j))
    axes.reshape(-1, 1)

    for f, l in dataset.take(1):


        # loop through one batch for N times 
        for i, ax in enumerate(axes.reshape(-1)):

            ax.imshow(f[i])
            ax.axis(False)

            # Make prediction on sample image from dataset 
            sample_img = np.expand_dims(f[i], axis=0)      
            pred_raw = model.predict(sample_img)
            pred_soft = tf.keras.layers.Softmax()(pred_raw)  
            pred_idx = pred_soft[0].numpy().argsort()[-2:][::-1]       

            # Get actual label 
            idx_labels = np.where(l[i].numpy() == 1)[0] 

            # Write actual and prediction 
            if print_types: 
                act_types_str = [] 
                pred_types_str = [] 

                for idx in idx_labels: 
                    act_types_str.append(type_encoding[idx])

                for idx in pred_idx: 
                    pred_types_str.append(type_encoding[idx])

                idx_labels = act_types_str
                pred_idx = pred_types_str


            ax.set_title("A: {}, P: {}".format(idx_labels, pred_idx), size=10)
