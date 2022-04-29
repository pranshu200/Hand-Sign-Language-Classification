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
#RAW DATASET1 
Path= 'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET1'
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
new_dataset='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification/DATASET2'
os.mkdir(new_dataset)

#Under the Dataset2 folders create folders for all the classes
for i in l:
  temp=os.path.join(new_dataset,i)
  os.mkdir(temp)

print("Folders Created")

#%%
#Working of Data Processing
print("Working of Data Processing taking example ThumbsUp")
img=cv2.imread('C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\ThumbsUp.jpg')
cv2.imshow("INPUT",img)
print(img.shape)
cropped_img=img[100:380,180:450]
cv2.imshow("Cropped Imaage",cropped_img)
resized_img=cv2.resize(cropped_img,(200,200))
cv2.imshow("Resized Image",resized_img)
#blackwhite_img=cv2.cvtColor(resized_img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("Black&White Image",blackwhite_img)
mat1=np.ones(resized_img.shape,dtype="uint8")*30
mat2=np.ones(resized_img.shape,dtype="uint8")*50
dark_img1=cv2.subtract(resized_img,mat1)
cv2.imshow("Dark Resized Image",dark_img1)
dark_img2=cv2.subtract(resized_img,mat2)
cv2.imshow("Darker Resized Image",dark_img2)
flipped_img = cv2.flip(resized_img, 1)
cv2.imshow("Flipped Image",flipped_img)
dark_img3=cv2.subtract(flipped_img,mat1)
cv2.imshow("Dark Flipped Image",dark_img3)
dark_img4=cv2.subtract(flipped_img,mat2)
cv2.imshow("Darker Flipped Image",dark_img4)
cv2.waitKey(0)


# %%
#Increasing the Dataset
new_dir='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET2'
for label in l:
    input_path =Path + '//' +label
    count=1
    for imgnum in range(200):
        image_path=input_path+'//'+label+str(imgnum)+'.jpg'
        img=cv2.imread(image_path)
        cropped_img=img[100:380,180:450]
        resized_img=cv2.resize(cropped_img,(200,200)) #Resized Image
        mat1=np.ones(resized_img.shape,dtype="uint8")*30
        mat2=np.ones(resized_img.shape,dtype="uint8")*50
        dark_img1=cv2.subtract(resized_img,mat1) #Dark Resized Image
        dark_img2=cv2.subtract(resized_img,mat2) #Darker Resized Image
        flipped_img = cv2.flip(resized_img, 1) #Flipped Image
        dark_img3=cv2.subtract(flipped_img,mat1) #Dark Flipped Image
        dark_img4=cv2.subtract(flipped_img,mat2) #Darker Flipped Image
        temp= os.path.join(new_dir,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp, resized_img)
        temp=os.path.join(new_dir,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp,dark_img1)
        temp=os.path.join(new_dir,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp,dark_img2)
        temp=os.path.join(new_dir,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp, flipped_img)
        temp=os.path.join(new_dir,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp,dark_img3)
        temp=os.path.join(new_dir,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp,dark_img4)

#%%
#TestSet
test_dataset='C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification/TESTSET'
os.mkdir(test_dataset)
for i in l:
    temp=os.path.join(test_dataset,i)
    os.mkdir(temp)
#%%
count=200
for label in l:
    input_path =Path + '//' +label
    for imgnum in range(200,210,1):
        image_path=input_path+'//'+label+str(imgnum)+'.jpg'
        img=cv2.imread(image_path)
        cropped_img=img[80:400,140:460]
        resized_img=cv2.resize(cropped_img,(200,200)) #Resized Image
        mat1=np.ones(resized_img.shape,dtype="uint8")*30
        mat2=np.ones(resized_img.shape,dtype="uint8")*50
        dark_img1=cv2.subtract(resized_img,mat1) #Dark Resized Image
        dark_img2=cv2.subtract(resized_img,mat2) #Darker Resized Image
        flipped_img = cv2.flip(resized_img, 1) #Flipped Image
        dark_img3=cv2.subtract(flipped_img,mat1) #Dark Flipped Image
        dark_img4=cv2.subtract(flipped_img,mat2) #Darker Flipped Image
        temp= os.path.join(test_dataset,label,label+'{}.jpg'.format(str(count)))
        count+=1
        cv2.imwrite(temp, resized_img)
# %%
