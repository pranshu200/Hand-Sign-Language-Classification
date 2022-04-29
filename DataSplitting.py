#%%
#Import Required Libraries
try:
    import sys,os
    import cv2
    import time
    from PIL import Image
    import numpy as np
    print("Libraries Loaded")
except:
    print("Libraries not Loaded")

#%%
#RAW DATASET
Path= 'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET2'
l=[]
files = os.listdir(Path)
for f in files:
    l.append(f)
print("\nNames of Classes")
print(l)
print()
print('Total Number of images per label in Dataset')
totalFiles = 0
totalDir = 0
filesinfolder=0
for base, dirs, files in os.walk(Path):
    print('Searching in : ',base[35::])
    for directories in dirs:
        totalDir += 1
    for Files in files:
        filesinfolder+=1
        totalFiles += 1
    print(base,"=",filesinfolder)
    filesinfolder=0

print('Total number of files',totalFiles)
print('Total Number of directories',totalDir)




# %%
#Create new Directories
new_dataset='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET'
os.mkdir(new_dataset)

#Create new folders
#train
train_dir=os.path.join(new_dataset,'train')
os.mkdir(train_dir)
#validation
validation_dir=os.path.join(new_dataset,'validation')
os.mkdir(validation_dir)

#Under the two folders create folders for all the classes

for i in l:
    temp=os.path.join(train_dir,i)
    os.mkdir(temp)
    temp=os.path.join(validation_dir,i)
    os.mkdir(temp)

print("Folders Created")

#%%
#Splitting Data into Train and Validation
train_dir='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET/train'
for label in l:
    count=1
    input_path =Path + '//' +label
    for imgnum in range(1,961,1):
        image_path=input_path+'//'+label+str(imgnum)+'.jpg'
        img=cv2.imread(image_path)
        temp= os.path.join(train_dir,label,label+'{}.jpg'.format("_Train"+str(count)))
        count+=1
        cv2.imwrite(temp,img)
#%%
#creating Validation Dataset
validation_dir='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET/validation'
for label in l:
    count=1
    input_path =Path + '//' +label
    for imgnum in range(961,1201,1):
        image_path=input_path+'//'+label+str(imgnum)+'.jpg'
        img=cv2.imread(image_path)
        temp= os.path.join(validation_dir,label,label+'{}.jpg'.format("_Validation"+str(count)))
        count+=1
        cv2.imwrite(temp,img)

# %%
#Checking the Data in Train Folder:
Path='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET/train'
l=[]
files = os.listdir(Path)
for f in files:
    l.append(f)
print("\nNames of Classes")
print(l)
print()
print('Total Number of images per label in Train Dataset')
totalFiles = 0
totalDir = 0
filesinfolder=0
for base, dirs, files in os.walk(Path):
    print('Searching in : ',base[35::])
    for directories in dirs:
        totalDir += 1
    for Files in files:
        filesinfolder+=1
        totalFiles += 1
    print(base,"=",filesinfolder)
    filesinfolder=0

print('Total number of files',totalFiles)
print('Total Number of directories',totalDir)

# %%
#Checking the Data in Validation Folder:
Path='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET/validation'
l=[]
files = os.listdir(Path)
for f in files:
    l.append(f)
print("\nNames of Classes")
print(l)
print()
print('Total Number of images per label in Train Dataset')
totalFiles = 0
totalDir = 0
filesinfolder=0
for base, dirs, files in os.walk(Path):
    print('Searching in : ',base[35::])
    for directories in dirs:
        totalDir += 1
    for Files in files:
        filesinfolder+=1
        totalFiles += 1
    print(base,"=",filesinfolder)
    filesinfolder=0

print('Total number of files',totalFiles)
print('Total Number of directories',totalDir)

# %%
