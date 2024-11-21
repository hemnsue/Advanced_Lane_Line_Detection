# Advanced_Lane_Line_Detection
Output of the code in reallife dataset with real situations on road:
https://drive.google.com/file/d/1cnCpZcO7MLx6u-gNxVcD2jFN99WGMtvb/view?usp=sharing
# For detection of lane lines first we calibrate the camera
The camera is calibrated using chessboard images taken from the same camera present in the Camera_CAL directory.Camera calibration is done to find the distortions in the image, the image is then undistorted after finding the distortion matrix.
* First the image is converted to gray scale
* After gray scale conversion the dsitrotion matrices are found using calibrateCameara() function
* The undistortion matrix found using the calibrateCamera function is used to undistort the frames
# The Lane Detection Pipeline is as follows:
* Camera calibration is done to find the distortion matrix
* The frame is read from the recorded video
* The frame is then undistorted
* The frames are now blurred to soften up the edges
* Now colour grading and binary thresholding is done for extraction of data using different colour spaces like HLS,grayscale and then edge detection using sobel x and sobel y , hence using bitwise or and and merging the binary data
* Now a specific region of interest is defined on the frame in terms of pixel values(x,y)//predefined
* The data inside the region of interest is only considered for lane line detection
