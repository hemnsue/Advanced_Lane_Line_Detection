#%% Initialise
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
from moviepy.editor import VideoFileClip
%matplotlib inline
#%%Camera Calibration
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

#%% Region of Interest
video=cv.VideoCapture("20231016_072727.mp4");
_,row=video.read();
print(row.shape);
row=cv.cvtColor(row,cv.COLOR_BGR2RGB);
pts = np.array([[500,750],[770,545],[1110,545],[1600,750]], np.int32)
pts = pts.reshape((-1,1,2))
cv.polylines(row,[pts],True,(0,255,255))
plt.imshow(row);
#plt.imshow(row);
plt.show();


 #%%Birds Eye View
mtx, dist = camera_calibrate();
def warp(img):
    undist = cv.undistort(img, mtx, dist, None, mtx);
    img_size = (img.shape[1], img.shape[0]);
    offset = 500; #300
    
    # Source points taken from images with straight lane lines, these are to become parallel after the warp transform
    src = np.float32([
        (500,750), # bottom-left corner
        (770,545), # top-left corner
        (1110,540), # top-right corner
        (1600,750) # bottom-right corner
    ]);
    # Destination points are to be parallel, taken into account the image size
    dst = np.float32([
        [offset, img_size[1]],             # bottom-left corner
        [offset, 0],                       # top-left corner
        [img_size[0]-offset, 0],           # top-right corner
        [img_size[0]-offset, img_size[1]]  # bottom-right corner
    ]);
    # Calculate the transformation matrix and it's inverse transformation
    M = cv.getPerspectiveTransform(src, dst);
    M_inv = cv.getPerspectiveTransform(dst, src);
    warped = cv.warpPerspective(undist, M, img_size);
   
    return warped, M_inv;

# %%Binary Image Process

def binary_thresholded(img):
    # Transform image to gray scale
    gray_img =cv.cvtColor(img, cv.COLOR_BGR2GRAY);
    # Apply sobel (derivative) in x direction, this is usefull to detect lines that tend to be vertical
    sobelx = cv.Sobel(gray_img, cv.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    # Scale result to 0-255
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sx_binary = np.zeros_like(scaled_sobel)
    # Keep only derivative values that are in the margin of interest
    sx_binary[(scaled_sobel >= 30) & (scaled_sobel <= 255)] = 1

    # Detect pixels that are white in the grayscale image
    white_binary = np.zeros_like(gray_img)
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1

    # Convert image to HLS
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS)
    H = hls[:,:,0]
    S = hls[:,:,2]
    sat_binary = np.zeros_like(S)
    # Detect pixels that have a high saturation value
    sat_binary[(S > 90) & (S <= 255)] = 1

    hue_binary =  np.zeros_like(H)
    # Detect pixels that are yellow using the hue component
    hue_binary[(H > 10) & (H <= 25)] = 1

    # Combine all pixels detected above
    binary_1 = cv.bitwise_or(sx_binary, white_binary)
    binary_2 = cv.bitwise_or(hue_binary, sat_binary)
    binary = cv.bitwise_or(binary_1, binary_2)

    return binary

#%%
video=cv.VideoCapture("20231016_072727.mp4");
_,row=video.read();
cow=cv.cvtColor(row,cv.COLOR_BGR2RGB);
binary_thresh = binary_thresholded(cow)
out_img = np.dstack((binary_thresh, binary_thresh, binary_thresh))*255
binary_warped, M_inv = warp(binary_thresh)
plt.imshow(out_img, cmap='gray')

#%%Detection of Lane Lines Using Histogram
def find_lane_pixels_using_histogram(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0);
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2);
    leftx_base = np.argmax(histogram[:midpoint]);
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint;

    # Choose the number of sliding windows
    nwindows = 12
    # Set the width of the windows +/- margin
    margin = 120
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_poly(binary_warped,leftx, lefty, rightx, righty):
    ### Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)   
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
    
    return left_fit, right_fit, left_fitx, right_fitx, ploty

def draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty):     
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
        
    margin = 100
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv.fillPoly(window_img, np.int_([left_line_pts]), (100, 100, 0))
    cv.fillPoly(window_img, np.int_([right_line_pts]), (100, 100, 0))
    result = cv.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='green')
    plt.plot(right_fitx, ploty, color='blue')
    ## End visualization steps ##
    return result

#%% Detection of lane lines based on previous step

    ### STEP 5: Detection of Lane Lines Based on Previous Step ###

#img = cv2.imread('test_images/test5.jpg')
#binary_thresh = binary_thresholded(img)
#binary_warped, M_inv = warp(binary_thresh)

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
#left_fit = np.array([ 2.13935315e-04, -3.77507980e-01,  4.76902175e+02])
#right_fit = np.array([4.17622148e-04, -4.93848953e-01,  1.11806170e+03])

def find_lane_pixels_using_prev_poly(binary_warped):
    global prev_left_fit
    global prev_right_fit
    # width of the margin around the previous polynomial to search
    margin = 100
    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])    
    ### Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                    prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                    prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))).nonzero()[0]
    right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                    prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                    prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin))).nonzero()[0]
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


#leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped)
#left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
#out_img = draw_poly_lines(binary_warped, left_fitx, right_fitx, ploty)
#plt.imshow(out_img)

#%%
### STEP 6: Calculate Vehicle Position and Curve Radius ###

def measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/1080 # meters per pixel in y dimension
    xm_per_pix = 3.7/960 # meters per pixel in x dimension
    
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return left_curverad, right_curverad

def measure_position_meters(binary_warped, left_fit, right_fit):
    # Define conversion in x from pixels space to meters
    xm_per_pix = 3.7/960 # meters per pixel in x dimension
    # Choose the y value corresponding to the bottom of the image
    y_max = binary_warped.shape[0]
    # Calculate left and right line positions at the bottom of the image
    left_x_pos = left_fit[0]*y_max**2 + left_fit[1]*y_max + left_fit[2]
    right_x_pos = right_fit[0]*y_max**2 + right_fit[1]*y_max + right_fit[2] 
    # Calculate the x position of the center of the lane 
    center_lanes_x_pos = (left_x_pos + right_x_pos)//2
    # Calculate the deviation between the center of the lane and the center of the picture
    # The car is assumed to be placed in the center of the picture
    # If the deviation is negative, the car is on the felt hand side of the center of the lane
    veh_pos = ((binary_warped.shape[1]//2) - center_lanes_x_pos) * xm_per_pix 
    return veh_pos

#left_curverad, right_curverad =  measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
#print('left curve radius in meters  = ', left_curverad)
#print('right curve radius in meters = ', right_curverad)
#veh_pos = measure_position_meters(binary_warped, left_fit, right_fit)
#print('vehicle position relative to center  = ', veh_pos)

#%%
### STEP 7: Project Lane Delimitations Back on Image Plane and Add Text for Lane Info ###

def project_lane_info(img, binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv.warpPerspective(color_warp, M_inv, (img.shape[1], img.shape[0]))
    
    # Combine the result with the original image
    out_img = cv.addWeighted(img, 1, newwarp, 0.3, 0)
    
    cv.putText(out_img,'Curve Radius [m]: '+str((left_curverad+right_curverad)/2)[:7],(40,70), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.6, (255,255,255),2,cv.LINE_AA)
    cv.putText(out_img,'Center Offset [m]: '+str(veh_pos)[:7],(40,150), cv.FONT_HERSHEY_COMPLEX_SMALL, 1.6,(255,255,255),2,cv.LINE_AA)
    
    return out_img

#new_img = project_lane_info(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), binary_warped, ploty, left_fitx, right_fitx, M_inv, left_curverad, right_curverad, veh_pos)

# Plot the result
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
#f.tight_layout()
#ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#ax1.set_title('Original Image', fontsize=20)
#ax2.imshow(new_img, cmap='gray')
#ax2.set_title('Image With Lane Marked', fontsize=20)

#%%
global left_fit_hist
left_fit_hist=np.array([])

global right_fit_hist
right_fit_hist=np.array([])

#%%
#STEP 8: Lane Finding Pipeline on Video#

def lane_finding_pipeline(img):
    global left_fit_hist
    global right_fit_hist
    global prev_left_fit
    global prev_right_fit
    binary_thresh=binary_thresholded(img)
    binary_warped,M_inv=warp(binary_thresh)
    if (len(left_fit_hist)==0):
        leftx, lefty, rightx, righty = find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        # Store fit in history
        left_fit_hist=np.array(left_fit)
        new_left_fit=np.array(left_fit)
        left_fit_hist=np.vstack([left_fit_hist,new_left_fit])
        right_fit_hist=np.array(right_fit)
        new_right_fit=np.array(right_fit)
        right_fit_hist=np.vstack([right_fit_hist,new_right_fit])
    else:
        prev_left_fit = [np.mean(left_fit_hist[:,0]), np.mean(left_fit_hist[:,1]), np.mean(left_fit_hist[:,2])]
        prev_right_fit = [np.mean(right_fit_hist[:,0]), np.mean(right_fit_hist[:,1]), np.mean(right_fit_hist[:,2])]
        leftx, lefty, rightx, righty = find_lane_pixels_using_prev_poly(binary_warped)
        if(len(lefty)==0 or len(righty)==0):
            leftx,lefty,rightx,righty=find_lane_pixels_using_histogram(binary_warped)
        left_fit, right_fit, left_fitx, right_fitx, ploty = fit_poly(binary_warped,leftx, lefty, rightx, righty)
        #add new values in history
        new_left_fit=np.array(left_fit)
        left_fit_hist=np.vstack([left_fit_hist,new_left_fit])
        new_right_fit=np.array(right_fit)
        right_fit_hist=np.vstack([right_fit_hist,new_right_fit])
        #remove old values 
        if(len(left_fit_hist)>10):
            left_fit_hist=np.delete(left_fit_hist,0,0)
            right_fit_hist=np.delete(right_fit_hist,0,0)
    left_curverad,right_curverad=measure_curvature_meters(binary_warped, left_fitx, right_fitx, ploty)
    veh_pos=measure_position_meters(binary_warped,left_fit,right_fit)
    out_img=project_lane_info(img,binary_warped,ploty,left_fitx,right_fitx,M_inv, left_curverad, right_curverad, veh_pos)
    return out_img

video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("20231016_072727.mp4")
output_clip = clip1.fl_image(lane_finding_pipeline)
%time output_clip.write_videofile(video_output, audio=False)

# %%
