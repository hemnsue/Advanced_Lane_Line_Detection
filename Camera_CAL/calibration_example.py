#%%
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#%%
nc=(7,7);
framesize=(1920,1080);
criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,30,0.001);
objp=np.zeros((nc[0]*nc[1],3),np.float32)
objp[:,:2]=np.mgrid[0:nc[0],0:nc[1]].T.reshape(-1,2);
objPoints=[];
imgPoints=[];
paths=["11.jpg","12.jpg","14.jpg","15.jpg","16.jpg"];
for image in paths:
    row=cv.imread(image);
    print(row);
    gray=cv.cvtColor(row,cv.COLOR_BGR2GRAY);
    ret,corners=cv.findChessboardCorners(gray,nc,None);
    if ret==True:
        objPoints.append(objp);
        corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria);
        imgPoints.append(corners);
        cv.drawChessboardCorners(row,nc,corners2,ret);
        #plt.imshow(row);
        #plt.show();
ret,cameraMatrix,dist,rvecs,tvecs=cv.calibrateCamera(objPoints,imgPoints,gray.shape[::-1],None,None);
print("Camera Calibrated: ",ret);
print("\nCamera Matrix \n",cameraMatrix);
print("\nDistortion Parameters :\n",dist);
print("\nRotation Vectors:\n",rvecs);
print("\nTranslation Vectors:\n",tvecs);
img=cv.imread("13.jpg");
h,w=img.shape[:2];
newCameraMatrix,roi=cv.getOptimalNewCameraMatrix(cameraMatrix,dist,(w,h),1,(w,h));
dst=cv.undistort(img,cameraMatrix,dist,None,newCameraMatrix);
x,y,w,h=roi;
dst=dst[y:y+h,x:x+w];
cv.imwrite("Calibration.jpg",dst);
# %%
