# Advanced_Lane_Line_Detection
Output of the code in reallife dataset with real situations on road:
https://drive.google.com/file/d/1cnCpZcO7MLx6u-gNxVcD2jFN99WGMtvb/view?usp=sharing
# For detection of lane lines first we calibrate the camera
The camera is calibrated using chessboard images taken from the same camera present in the Camera_CAL directory.Camera calibration is done to find the distortions in the image, the image is then undistorted after finding the distortion matrix.
* First the image is converted to gray scale
* After gray scale conversion the dsitrotion matrices are found using calibrateCameara() function
* The undistortion matrix found using the calibrateCamera function is used to undistort the frames

 
