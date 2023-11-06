#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
#%%
nc=(8,8)
framesize=(1920,1080)
criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
objp=np.zeros((nc[0]*nc[1],3),np.float32)
objp[:,:2]=np.mgrid[0:nc[0],0:nc[1]].T.reshape(-1,2)
objPoints=[]
imgPoints=[]
paths=["Camera_CAL//1.mp4","Camera_CAL//2.mp4","Camera_CAL//3.mp4","Camera_CAL//4.mp4","Camera_CAL//5.mp4","Camera_CAL//6.mp4"]
for i in paths:
    video=cv2.VideoCapture(i);
    _,row=video.read();
    print(row.shape);
    row=cv2.cvtColor(row,cv2.COLOR_BGR2RGB);
    plt.imshow(row)
    plt.show()

# %%
