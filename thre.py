#%%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
#%%
def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    hls = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    if orient == 'x':
        abs_sobel = np.absolute(cv.Sobel(l_channel, cv.CV_64F, 1, 0,ksize=3))
    if orient == 'y':
        abs_sobel = np.absolute(cv.Sobel(l_channel, cv.CV_64F, 0, 1,ksize=3))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def camera_calibrate():
    nc=(7,7);
    framesize=(1920,1080);
    criteria=(cv.TERM_CRITERIA_EPS+cv.TERM_CRITERIA_MAX_ITER,30,0.001);
    objp=np.zeros((nc[0]*nc[1],3),np.float32)
    objp[:,:2]=np.mgrid[0:nc[0],0:nc[1]].T.reshape(-1,2);
    objPoints=[];
    imgPoints=[];
    paths=["Camera_CAL//11.jpg","Camera_CAL//12.jpg","Camera_CAL//13.jpg","Camera_CAL//14.jpg","Camera_CAL//15.jpg","Camera_CAL//16.jpg"];
    for image in paths:
        row=cv.imread(image);
        print(row);
        gray=cv.cvtColor(row,cv.COLOR_BGR2GRAY);
        ret,corners=cv.findChessboardCorners(gray,nc,None);
        if ret==True:
            objPoints.append(objp);
            imgPoints.append(corners);
        ret,cameraMatrix,dist,rvecs,tvecs=cv.calibrateCamera(objPoints,imgPoints,framesize,None,None);
    return cameraMatrix,dist;

def thresholding(img):
    #gray_img =cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    sobelx=abs_sobel_thresh(img, orient='x', thresh_min=20,thresh_max=100)

    # set the spacing between axes.
    cv.imshow(img,"og")

    cv.imshow(sobelx, "sob")

    cv.waitkey(0)


video=cv.VideoCapture("20231016_072727.mp4");
_,row=video.read();
print(row.shape);
row=cv.cvtColor(row,cv.COLOR_BGR2RGB)
mtx, dist = camera_calibrate();
undist = cv.undistort(dist, mtx, dist, None, mtx);
thresholding(undist)
#pts = np.array([[350,750],[770,500],[1110,500],[1750,750]], np.int32)
#pts = pts.reshape((-1,1,2))
#cv.polylines(row,[pts],True,(0,255,255))
#plt.imshow(row);
#plt.show();

# %%
