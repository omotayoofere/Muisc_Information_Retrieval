# Music Information Retrieval
The massive consumption of audio and video contents in this era across multiple media for streaming has necessitated the improvement of information retrieval processes. An automated rather than a manual approach highly necessary to improves users’ experience in accessing contents in a massive content library. 

Music classification, a technique that enhances users’ music experience through recommendation, curation, and analysis of listening behavior.
- Recommendation: Once musical attributes have labeled a system can recommend music to users based on frequently consumed musical attributes of the users. 
- Curation: Music curation replaces human’s manual effort in browsing enormous music libraries efficiently. 
- Listening behavior analysis: Most modern streaming services provide annual reports of personal listening trends for generic view as to what genre/form of music caught most attention

The motivation behind this study is to achieve a better score for the accuracy metric in the classification of music data by exploring handful for machine learning models. For a given song, the music classifier predicts its genre based on relevant musical attributes


### Problem Definition
* In this project, we will be exploring multi-class classification, that is, categorizing each music sample into either of the ten (10) labels available. 
### What is our data
* I have choosen the famous GTZAN dataset which is available on [kaggle](/https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
### Libraries and Dependencies
* I will be utilizing google collab on this project and mounting the dataset on Google drive. Also, I have listed all necessary libraries and dependencies needed for this project.
#### Mounting drive
* The spectogram of the music data (visual representation of the spectrum of frequencies of sound or other signals as they vary with time) which is also present in the dataset on [kaggle](/https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) was saved to a folder in the google drive and loaded from there
``` python
from google.colab import drive
drive.mount('/content/gdrive')
```

#### Importing necessary libraries
``` python
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

import os
import cv2
from PIL import Image
from numpy import asarray
import glob
import random

#from tensorflow import keras
from sklearn.model_selection import train_test_split, StratifiedKFold

from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Activation, Flatten, MaxPooling2D, Dropout
from sklearn.model_selection import cross_val_score

#importing splitfolders for use after install
import splitfolders

#EarlyStopping
from keras.callbacks import EarlyStopping

import kerastuner
from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
```

### 1. Loading the Data
* The image dataset used for this CNN model is gotten by extracting the spectogram of each audio data using [librosa](/https://librosa.org/doc/latest/index.html) - a is a python package for music and audio analysis.in the dataset and saving each data in a genre to a different folder
  ``` python
  X = librosa.stft(x)
  Xdb = librosa.amplitude_to_db(abs(X))
  ```
* The function below perfomrsa the follow task:
  - Loads the data from the storage 
  - Resizes each image and converts each data to greyscale - to reduce computational cost
  - Converts images to an array and appends to a container 
  - Converts the categorical labels to numerical labels and assign to respective array
  ``` python
    def structure_dataset(gdrive_path):
      categories = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
      data = []
      label = []

      for x in categories:
        path = gdrive_path + f'/{x}' + '/*.png' 
        #used to check for extensions in folders
        for file in glob.glob(path):
          #reading the image and converting to greyscale
          img = cv.imread(file, cv.IMREAD_GRAYSCALE)

          #Resizing images
          IMG_SIZE = 350
          image = cv.resize(img, (IMG_SIZE, IMG_SIZE))

          #Appends the image to the container holding the newly sized images
          data.append(image)
        
          #Converts image to an array
          X = np.asarray(data)
        
          #Appends array for each image to a container
          label.append(x)
        
          #Giving a numeric label to categories of image dataset
          label_dict = {
            'blues': 0,
            'classical': 1,
            'country': 2,
            'disco': 3,
            'hiphop': 4,
            'jazz': 5,
            'metal': 6,
            'pop': 7,
            'reggae': 8,
            'rock': 9,
          }
        
          #mapping the image labels and the numeric labels created
          y = np.array(list(map(label_dict.get, label)))
      return X, y
  ```

### 2. Model Preparation
  The preparation of the model makes sure the data is model ready and all processes needed to make this happen occurs in this step. Actions including;
  - Reshaping the data to ensure uniformity across all dataset to reduce image dataset to a size that prevents a high computational cost in a case where we have a large frames of images
  - Splitting the data set into chunks to have a set for training the model and another for testing evalutaing the tested model: The approach used in this project is the [k-fold cross validation](/https://scikit-learn.org/stable/modules/cross_validation.html) and a subsequent extraction of 20% of the training data for validation. This approach was used considering the small size of the dataset. In a case of large dataset, the conventional [k-fold cross validation](/https://scikit-learn.org/stable/modules/cross_validation.html) would have been preferred
  - [Early stopping](https://keras.io/api/callbacks/early_stopping/), a monitor, was introduced to check and stop the training process of every epoch to get the best model based on preset parameters.
  - Using the [k-fold cross validation](/https://scikit-learn.org/stable/modules/cross_validation.html), the model was 
     - Trained on 64% of the dataset
     - Evaluated to adjust parameters on 16% of the dataset
     - Tested on 20% of the dataset
  ```python
     X = X.reshape(len(X), 350, 350, 1)
     
     scores = []
     actual = []
     preds = []
     
     def evaluate_model(X, y):
        kfold = StratifiedKFold(n_splits=10, random_state=random.seed(101), shuffle=True)
        current_fold = 0
        for train, test in kfold.split(X,y):
          current_fold += 1
          print('Training fold %d' % current_fold)
          
          model = build_model()
      
          train_X, train_y, test_X, test_y = X[train], y[train], X[test], y[test]

          #Extract a 20% slot from training set for validation
          tr, val = next(StratifiedKFold(n_splits=5, shuffle=True).split(train_X, train_y))
          tr_X, tr_y, val_X, val_y = train_X[tr], train_y[tr], train_X[val], train_y[val]

          Es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

          history = model.fit(tr_X, tr_y, epochs=100, batch_size=50, validation_data=(val_X, val_y), verbose=0, callbacks=[Es])
      
          _, acc = model.evaluate(test_X, test_y, verbose=0)

          print('>> %.3f' % (acc * 100.0))

          scores.append(acc)
          preds.append(history)
        
        print("%.2f%% (+/- %.2f%%)" % (np.mean(scores), np.std(scores)))
        return scores, preds
  ```
## 3. Model Plots
  Visual representation of every epoch of the training process gives a view as to how each [metric](/https://www.baeldung.com/cs/ml-loss-accuracy) fairs during the training of the model.
``` python
  def summarize(histories):
    for i in range(len(histories)):
      plt.figure()
      plt.subplot(211)
      plt.title('Cross Entropy Loss')
      plt.plot(histories[i].history['loss'], color='blue', label='train')
      plt.plot(histories[i].history['val_loss'], color='red', label='val')

      plt.subplot(212)
      plt.title('Classification Accuracy')
      plt.plot(histories[i].history['accuracy'], color='blue', label='train')
      plt.plot(histories[i].history['val_loss'], color='red', label='val')
      plt.show()
```

## 4. Run Model
  This function runs the model and summarizes the scores and the histories
```python
  def run():
    scores, histories = evaluate_model(X, y)
    summarize(histories)
    summarize_performance(scores)
```
## 5. Model Performance
  A summary of the model's performance is shown below where it could be seen that the mean of accuracies of each epoch is recorded to be 38%, this is a weak score and thus requires adjustment. Additional approach to mitigate this poor score is discussed in the next section.
```python
  def summarize_performance(scores):
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (np.mean(scores)*100, np.std(scores)*100, len(scores)))
    plt.boxplot(scores)
    plt.show()
```
  ![alt text](https://github.com/omotayoofere/Muisc_Information_Retrieval/blob/main/Capture.PNG "Model Summary")
  
## 6. Further tasks
* Augumenting the image data by creating various shades of the original images thus, adding more variation to the image dataset.
  ```python
  #An approach to augument data by presenting different form of the data to the model
  datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        #rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        #horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

  datagen.fit(X)
  ```
* Tuning HyperParameter: This approach runs and simulates the best model (one that gets the best measurement metric) doing the heavy lifting, making the remaining tweaks easy to perform during model training.
  ```python
  ```
