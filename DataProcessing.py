#%%
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
os.chdir(r'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification')
os.listdir()  #path where file is present
warnings.filterwarnings("ignore")   #remove warnings
print("----Folder Loaded----")

#%%
l=[]
path = 'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification/NewDataset\Train-Dataset'
files = os.listdir(path)
for f in files:
    l.append(f)
print("\nNames of Classes")
print(l)

#%%
#TRAIN DATA
totalFiles = 0
totalDir = 0
filesinfolder=0
for base, dirs, files in os.walk('C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification/NewDataset\Train-Dataset'):
    print('Searching in : ',base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        filesinfolder+=1
        totalFiles += 1
    print(base,"=",filesinfolder)
    filesinfolder=0


print('Total number of files',totalFiles)
print('Total Number of directories',totalDir)
print('Total:',(totalDir + totalFiles))

#%%
#VALIDATION DATA
totalFiles = 0
totalDir = 0
filesinfolder=0
for base, dirs, files in os.walk('C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification/NewDataset\Validation-Dataset'):
    print('Searching in : ',base)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        filesinfolder+=1
        totalFiles += 1
    print(base,"=",filesinfolder)
    filesinfolder=0


print('Total number of files',totalFiles)
print('Total Number of directories',totalDir)
print('Total:',(totalDir + totalFiles))
# %%
