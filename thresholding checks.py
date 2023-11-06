def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

def color_threshold(image, sthresh=(0,255), vthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    # Return the combined s_channel & v_channel binary image
    return output

def s_channel_threshold(image, sthresh=(0,255)):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]  # use S channel

    # create a copy and apply the threshold
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= sthresh[0]) & (s_channel <= sthresh[1])] = 1
    return binary_output

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height), max(0,int(center-width)):min(int(center+width),img_ref.shape[1])] = 1
    return output
#%% ###UNDISTORTING THE IMAGE
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
            #corners2=cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria);
            imgPoints.append(corners);
            #cv.drawChessboardCorners(row,nc,corners2,ret);
            #plt.imshow(row);
            #plt.show();
        ret,cameraMatrix,dist,rvecs,tvecs=cv.calibrateCamera(objPoints,imgPoints,framesize,None,None);
    return cameraMatrix,dist;


video=cv.VideoCapture("20231016_072727.mp4");
_,row=video.read();
print(row.shape);
unchanged=row;
row=cv.cvtColor(row,cv.COLOR_BGR2RGB);
pts = np.array([[500,750],[770,545],[1110,545],[1600,750]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(row,[pts],True,(0,255,255))
plt.imshow(row);
#plt.imshow(row);
plt.show();

mtx, dist = camera_calibrate();
undist = cv.undistort(unchanged, mtx, dist, None, mtx);

#Apply Sobel operator in X-direction to experiment with gradient thresholds

gradx = abs_sobel_thresh(unchanged, orient='x', thresh_min=20, thresh_max=100)

#Visualize the results before/after absolute sobel operator is applied on a test image in x direction to find the
#vertical lines, since the lane lines are close to being vertical
plt.figure(figsize = (18,12))
grid = gridspec.GridSpec(1,2)

# set the spacing between axes.
grid.update(wspace=0.1, hspace=0.1)  

plt.subplot(grid[0])
plt.imshow(undistTest, cmap="gray")
plt.title('Undistorted Image')

plt.subplot(grid[1])
plt.imshow(gradx, cmap="gray")
plt.title('Absolute sobel threshold in X direction')

plt.show()