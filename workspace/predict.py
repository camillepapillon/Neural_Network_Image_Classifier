import argparse 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import time 
import json
from PIL import Image
import matplotlib.pyplot as plt

image_size = 224 


def process_image(image):  
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image


def predict(image_path, model, top_k): 
    im = Image.open(image_path)
    test_image = np.asarray(im)
    processed_test_image = process_image(test_image)
    expanded_test_image = np.expand_dims(processed_test_image, axis=0)
    ps =  reloaded_model.predict(expanded_test_image)
    top_values, top_indices = tf.nn.top_k(ps, top_k)
    
    top_values = top_values.numpy()
    top_indices = top_indices.numpy()
    return top_values, top_indices, im 


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description= 'Predict flower app')
    parser.add_argument('path')
    parser.add_argument('model')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')


    args = parser.parse_args()
    print(args)

    path_image = args.path
    model = args.model 
    top_k = args.top_k
    category_names = args.category_names 
    
#     set top k value if none : 
    if top_k is None : 
        top_k = 5

   
    #  Load the model 
    reloaded_model = tf.keras.models.load_model(model, custom_objects={'KerasLayer': hub.KerasLayer})
    print(reloaded_model.summary())

    # Predict the image 
    flower_names = [] 
    top_values, top_indices, im = predict('./test_images/' + path_image, reloaded_model, int(top_k))
     #  Load json file for class name 
    
    if category_names is not None: 
        with open(category_names, 'r') as f:
            class_names = json.load(f) 
        for fl in top_indices[0] :
            flower_names.append(class_names[str(fl+1)])
    
    print(top_values)
    if flower_names is not None : 
        for flower in flower_names: 
            print(flower)
            
    print(top_indices)
    
    