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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
training_path= 'C:/PRANSHU\PROJECTS/Hand-Sign-Language-Classification\Dataset/asl_alphabet_train'
testing_path= 'C:/PRANSHU\PROJECTS/Hand-Sign-Language-Classification\Dataset/asl_alphabet_test'
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
'''#Compiling and Training Model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='CNN_Layers.png')
#%%'''
#Training the model
with tf.device('/GPU:0'):
    history = model.fit(
        training_set,
        steps_per_epoch=87000//50,
        epochs=10, 
        verbose=1,
        validation_data = test_set,
        validation_steps=29//1
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
#%%
#Loss

plt.plot(history.history['loss'][0:11])
plt.plot(history.history['val_loss'][0:11])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training_loss', 'validation_loss'])
plt.show()
#%%
#Save Model
m_json = model.to_json() 
with open("CNN_model.json", "w") as json_file:  
    json_file.write(m_json)  
model.save_weights("CNN_model_Weights.h5")
# %%
#Loading the Saved Model
from tensorflow.keras.models import load_model
from keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array

cnnmodel = model_from_json(open('C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/Saved Model/CNN_model.json','r').read())  
cnnmodel.load_weights('C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/Saved Model/CNN_model_Weights.h5')
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

path1 = 'C:/PRANSHU/PROJECTS/Hand-Sign-Language-Classification/images.jpg'
img_original1 = load_img(path1)
#%%
pred1 = model.predict(process_image(path1))

print(pred1)

a=np.argmax(pred1,axis=1)
print(a)
#%%
