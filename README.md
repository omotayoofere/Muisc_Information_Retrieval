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
* I have choosen the famous GTZAN dataset which is available on kaggle at https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
### Libraries and Dependencies
* I have listed all necessary libraries and dependencies needed for this project.

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

### Getting the Data
* Stuff used
## 1. Exploratory Data Analysis
* Stuff used
### Data Engineering; Ensuring data is ready for training
* Stuff used
## 2. Model Training
* Separating Labels from Features
* Splitting data
## 3. Model Comparison
* Separating Labels from Features
* Splitting data
## 4. Model Tuning
* Separating Labels from Features
* Splitting data
## 5. Model Evaluation
* Separating Labels from Features
* Splitting data
## 6. Further tasks
* Separating Labels from Features
* Splitting data

