#%%
import cv2
import numpy as np
import os
#%%
def camera_calibrate():
    nc=(7,7)
    framesize=(1920,1080)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)
    objp=np.zeros((nc[0]*nc[1],3),np.float32)
    objp[:,:2]=np.mgrid[0:nc[0],0:nc[1]].T.reshape(-1,2)
    objPoints=[]
    imgPoints=[]
    base_path = r"D:\vid\Camera_CAL"
    paths = [
        os.path.join(base_path, "11.jpg"),
        os.path.join(base_path, "12.jpg"),
        os.path.join(base_path, "13.jpg"),
        os.path.join(base_path, "14.jpg"),
        os.path.join(base_path, "15.jpg"),
        os.path.join(base_path, "16.jpg"),
    ]
    for image in paths:
        row=cv2.imread(image)
        print(row)
        gray=cv2.cvtColor(row,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,nc,None)
        if ret==True:
            objPoints.append(objp)
            imgPoints.append(corners)
        ret,cameraMatrix,dist,rvecs,tvecs=cv2.calibrateCamera(objPoints,imgPoints,framesize,None,None)
    return cameraMatrix,dist

#%%
def hls_sobel_edge_detection(img, threshold1=100, threshold2=200):
    
    # Convert the image to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    
    # Extract the Lightness channel
    lightness = hls[:,:,1]
    
    # Apply Sobel edge detection on the Lightness channel
    sobelx = cv2.Sobel(lightness, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(lightness, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine the x and y Sobel gradients
    sobel_combined = np.sqrt(sobelx**2 + sobely**2)
    
    # Convert the gradient values to uint8
    sobel_combined = np.uint8(sobel_combined)
    
    # Apply thresholding to get binary edge image
    _,edges = cv2.threshold(sobel_combined, threshold1, threshold2, cv2.THRESH_BINARY)
    
    return edges
#%%
# Path to your image
mtx,dist=camera_calibrate()
video=cv2.VideoCapture("20231016_072727.mp4")
_,row=video.read()
print(row.shape)
row=cv2.cvtColor(row,cv2.COLOR_BGR2RGB)
undist=cv2.undistort(row,mtx,dist,None,mtx)
print(row.shape)
#%%

# Set threshold values
threshold1 = 125
threshold2 = 255

# Perform Sobel edge detection in HLS color space
edges_sobel_hls = hls_sobel_edge_detection(undist, threshold1, threshold2)

# Display the original image and edges detected using Sobel in HLS color space
cv2.imshow('Original Image', row)
cv2.imshow('Sobel Edge Detection in HLS', edges_sobel_hls)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
