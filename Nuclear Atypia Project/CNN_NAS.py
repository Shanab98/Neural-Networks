# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 15:15:48 2021

@author: sbeniami
"""


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import SGD

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import metrics 
import matplotlib.pyplot as plt
print(tf.__version__)

#%%

def main(df, data_dir, save_path, num_epochs, batch_sz, lr, lower_contrast, upper_contrast, hue, img_shape):
    model_name = 'B' + str(batch_sz) + '-CON' + str(lower_contrast) + '-LR' + str(lr) + '-HUE' + str(hue) 
    save_path = save_path + model_name 
        
    def read_image(image_file, label): 
        image = tf.io.read_file(data_dir + image_file + '.png')
        image = tf.image.decode_image(image, channels=3, dtype = 'uint8')
        image = tf.cast(image, 'float32')
        image = image/255.
        image.set_shape(img_shape)
    
        return image, label
    
    def augment(image, label): 
        image = tf.image.random_contrast(image, lower_contrast,  upper_contrast)
        image = tf.image.random_hue(image, hue)
        image = tf.image.random_flip_left_right(image) #50%
        return image, label
    
    # Data Loader
    image_names = df.patch_name.to_list()
    labels = (df.score -1).to_list() #making classes 0,1,2 instead of 1,2,3 
    
    train_images, test_images, train_labels, test_labels = train_test_split(image_names, labels, test_size = 0.20, random_state=42)
    ds_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
   
    ds_train = ds_train.map(read_image).map(augment).batch(batch_sz)
    ds_test = ds_test.map(read_image).batch(batch_sz)
    
    """
    # Visualize Images
    for x,y, in ds_train:
        x= (x.numpy()*255).astype('uint8')   
        plt.imshow(x[0,:,:,:])
        plt.show()
        
    #Visualize data split
    plt.hist(train_labels)
    plt.show()
    plt.hist(test_labels)
    plt.show()
    """
    
    # Model 
    model = models.Sequential()
    
    model.add(layers.Conv2D(16, (5, 5), activation='relu', input_shape=(256,256,3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3)) #, activation = 'softmax')) #three classes 
    model.add(layers.Softmax())
    model.summary()
    
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    opt = SGD(learning_rate=lr)
    
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    
    checkpoint_filepath = save_path + '/checkpoints/'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)
    
    
    history = model.fit(ds_train, epochs=num_epochs, 
                        validation_data=(ds_test),
                        callbacks=[model_checkpoint_callback])
    #Plot Accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.savefig(save_path + '/accuracy.png', bbox_inches='tight')
    plt.show()
  
    test_loss, test_acc = model.evaluate(ds_test, verbose=2)
    print(test_acc)
    
    
    #Saving results 
    models.save_model(model, save_path)
    
    predictions = model.predict(ds_test)
    preds = np.zeros((predictions.shape[0]))
    for test in range(predictions.shape[0]): 
        preds[test] = np.argmax(predictions[test]) 
    results = pd.DataFrame()
    results['names'] = test_images
    results['label'] = test_labels
    results['prediction'] = preds 
    results.to_csv(save_path + '/results.csv')
    
################################## CALL MAIN ####################################
# Initialize Variables
save_dir = 'C:/Users/sbeniami/Documents/GitHub/Shana.Beniamin/Nuclear Atypia Project/Models/' 
data_dir = 'C:/Users/sbeniami/Documents/GitHub/Shana.Beniamin/Nuclear Atypia Project/DATA/Patches 40x/'
df= pd.read_csv('C:/Users/sbeniami/Documents/GitHub/Shana.Beniamin/Nuclear Atypia Project/DATA/dataset.csv')

num_epochs = 25
batch_sz = 8
lr = 0.001
lower_contrast = 0.6
upper_contrast = 1
hue = 0.2
img_shape = [256,256,3]

main(df, data_dir, save_dir, num_epochs, batch_sz, lr, lower_contrast, upper_contrast, hue, img_shape)
    
