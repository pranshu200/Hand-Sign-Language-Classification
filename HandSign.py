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

os.chdir(r'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification')
os.listdir()  #path where file is present
warnings.filterwarnings("ignore")   #remove warnings
print("----Folder Loaded----")
# %%
os.listdir()
#%%
#Loading the Dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_path= 'C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/DATASET/train'
validation_path= 'C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/DATASET/validation'
train_datagen =ImageDataGenerator(
        rescale=1.0 / 255.0, 
        rotation_range=0,  
        zoom_range = 0.15,  
        width_shift_range=0.10,  
        height_shift_range=0.10,  
        horizontal_flip=False,  
        vertical_flip=False) 
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0, 
        rotation_range=0,  
        zoom_range = 0.15,  
        width_shift_range=0.10,  
        height_shift_range=0.10,  
        horizontal_flip=False,  
        vertical_flip=False)
training_set = train_datagen.flow_from_directory(training_path,target_size=(64,64),batch_size=10,class_mode='categorical')
validation_set = val_datagen.flow_from_directory(validation_path,target_size=(64,64),batch_size=10,class_mode='categorical')
print("Dataset Loaded")
#%%

#Building a model:
from Model import classifier_model
model = classifier_model()
model.summary()
# %%
'''#Compiling and Training Model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='CNN_Layers.png')
#%%'''
#Training the model
with tf.device('/GPU:0'):
    history = model.fit(
        training_set,
        steps_per_epoch=14400//10,
        epochs=100,
        verbose=1,
        validation_data = validation_set,
        validation_steps=3600//10
        #callbacks=[earlystopping]
    )
#%%
#Accuracy
plt.plot(history.history['accuracy'][0:220])
plt.plot(history.history['val_accuracy'][0:220])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training_accuracy', 'validation_accuracy'])
plt.show()
plt.savefig('Accuracy.jpg')
#%%
#Loss

plt.plot(history.history['loss'][0:11])
plt.plot(history.history['val_loss'][0:11])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'validation_loss'])
plt.show()
plt.savefig('Loss.jpg')
#%%
#Save Model
m_json = model.to_json() 
with open("HandSignModel1.json", "w") as json_file:  
    json_file.write(m_json)  
model.save_weights("HandSignModel1_Weights.h5")
# %%
#Loading the Saved Model
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

load_model = model_from_json(open('C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/Saved Model/HandSignModel1.json','r').read())  
load_model.load_weights('C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/Saved Model/HandSignModel1_Weights.h5')
print("------Saved Model Loaded------")
#%%
#Testing
def process_image(path):
    img = load_img(path, target_size = (64,64))
    img_tensor = img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis = 0)
    img_tensor/=255.0
    return img_tensor
#%%
#Testing
path1 = 'C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/TESTING_DATA/Bye200.jpg'
img_original1 = load_img(path1)
imgplot = plt.imshow(img_original1)
plt.show()
#%%
pred1 = load_model.predict(process_image(path1))

print(pred1)

a=np.argmax(pred1,axis=1)
print(a)
#%%
classes= {1: 'Bye', 2: 'CallMe', 3: 'God', 4: 'GoodLuck', 5: 'Hello', 6: 'I', 7:'ILoveYou', 8: 'No', 9: 'Ok', 10: 'Peace', 11: 'Smile', 12: 'ThumbsDown', 13: 'ThumbsUp', 14: 'Water', 15: 'Yes'}
#print(classes)
print("\n*****************************")
print("Prediction: ",classes[np.argmax(pred1)+1])
print("*****************************")
#%%
