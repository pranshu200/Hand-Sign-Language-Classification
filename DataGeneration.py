#%%
try:
    import os
    import cv2
    import time
    import uuid
    from PIL import Image
    print("Libraries Loaded")
except:
    print("Libraries not Loaded")

#%%
Img_PATH = 'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification/CHECK'

labels = ['hello', 'thanks','yes', 'no', 'food', 'drink', 'bye', 'ok', 'what', 'is', 'your', 'name','smile','I','want']
print(labels)
img_count = 1

#%%
for label in labels:
    !mkdir {'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\NewDataset\Test-Dataset\\'+label}
    cap = cv2.VideoCapture(0)
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    for imgnum in range(img_count): 
        print('Collecting image {}'.format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(Img_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imgname, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# %%
for label in labels:
    os.mkdir {'C:\PRANSHU\PROJECTS\Hand-Sign-Language-Classification\NewDataset\Test_Dataset\\'+label}
# %%
