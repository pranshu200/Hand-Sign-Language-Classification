#%%
#Import Required Libraries
try:
    import os
    import cv2
    import time
    from PIL import Image
    print("Libraries Loaded")
except:
    print("Libraries not Loaded")

#%%
#Define the Path for the Dataset
if not os.path.exists("DATASET1"):
    os.makedirs("DATASET1")
Img_PATH = 'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET1'

labels = ['ThumbsUp','Hello','ThumbsDown','Thanks','ILoveYou','GoodLuck','Ok','CallMe','Yes','No','Peace','Heart','Bye','I','Smile']
print(labels)
img_count = 210

#%%
#Collection of Images for all the labels 
for label in labels:
    !mkdir {'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET1\\'+label}
    cap = cv2.VideoCapture(0)
    #cap.set(3,240)
    #cap.set(4,240)
    print('Collecting images for {}'.format(label))
    time.sleep(10)
    for imgnum in range(img_count): 
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(Img_PATH,label,label+'{}.jpg'.format(str(imgnum)))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# %%
#Classes
l=[]
files = os.listdir(Img_PATH)
for f in files:
    l.append(f)
print("\nNames of Classes")
print(l)

#%%
#Total Number of images per label in Dataset
totalFiles = 0
totalDir = 0
filesinfolder=0
for base, dirs, files in os.walk('C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\DATASET1'):
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