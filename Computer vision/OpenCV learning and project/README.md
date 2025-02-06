# Basics of Image Processing

## Image Reading and Display
- Use `cv2.imread()` to read an image.
- Use `cv2.imshow()` to display an image.

## Image Saving
- Use `cv2.imwrite()` to save an image.

## Image Attributes
- Retrieve image properties such as dimensions and number of channels.

## Pixel Manipulation
- Access and modify pixel values of an image.

# Image Preprocessing

## Grayscale Conversion
- Use `cv2.cvtColor()` to convert an image to grayscale.

## Image Resizing
- Use `cv2.resize()` to adjust the size of an image.

## Image Rotation
- Use `cv2.getRotationMatrix2D()` and `cv2.warpAffine()` to rotate an image.

## Image Smoothing
- Use `cv2.GaussianBlur()`, `cv2.medianBlur()`, etc., for image smoothing.

## Edge Detection
- Use `cv2.Canny()` for edge detection.

# Image Transformations

## Affine Transformation
- Use `cv2.getAffineTransform()` and `cv2.warpAffine()` for affine transformations.

## Perspective Transformation
- Use `cv2.getPerspectiveTransform()` and `cv2.warpPerspective()` for perspective transformations.

# Feature Detection and Description

## Corner Detection
- Use `cv2.cornerHarris()` and `cv2.goodFeaturesToTrack()` to detect corners.

## SIFT/SURF
- Use `cv2.xfeatures2d.SIFT_create()` and `cv2.xfeatures2d.SURF_create()` for feature detection and description.

## ORB
- Use `cv2.ORB_create()` for feature detection and description.

# Object Detection and Tracking

## Template Matching
- Use `cv2.matchTemplate()` for template matching.

## Haar Cascade Classifier
- Use `cv2.CascadeClassifier()` for tasks like face detection.

## Optical Flow
- Use `cv2.calcOpticalFlowPyrLK()` for optical flow tracking.

# Image Segmentation

## Thresholding
- Use `cv2.threshold()` for
