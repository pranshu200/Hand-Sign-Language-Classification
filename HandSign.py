#%% 
#Import required libraries:
try:
    import sys,os
    import tensorflow as tf
    import numpy as np
    import pandas as pd
    import warnings
    import matplotlib.pyplot as plt
    print("----Libraries Loaded----")
except:
    print("----Libraries Not Loaded----")
#%%
#Loading, Reading, and Pre-Processing Dataset

os.chdir(r'C:\PRANSHU\PROJECTS\Hand-Sign-Classification')
os.listdir()  #path where file is present
warnings.filterwarnings("ignore")   #remove warnings
print("----Folder Loaded----")
# %%
os.listdir()
#%%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_path= 'C:/PRANSHU\PROJECTS/Hand-Sign-Classification\Dataset/asl_alphabet_train/asl_alphabet_train'
testing_path= 'C:/PRANSHU\PROJECTS/Hand-Sign-Classification\Dataset/asl_alphabet_test/asl_alphabet_test'
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(training_path,target_size=(64,64),batch_size=10,class_mode='categorical')
test_set = val_datagen.flow_from_directory(testing_path,target_size=(64,64),batch_size=10,class_mode='categorical')
print("Dataset Loaded")
#%%

#Building a model:
from Model import classifier_model
model = classifier_model()
model.summary()
# %%
#Compiling and Training Model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='CNN_Layers.png')
#%%