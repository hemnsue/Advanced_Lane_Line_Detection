import cv2
def binary_thresholded(img):
    gray_img =cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv2.imshow("grayscale",gray_img)
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
    white_binary[(gray_img > 200) & (gray_img <= 255)] = 1-
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