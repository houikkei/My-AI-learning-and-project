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
- Use `cv2.threshold()` for threshold-based segmentation.

## Contour Detection
- Use `cv2.findContours()` to detect contours in an image.

## Watershed Algorithm
- Use `cv2.watershed()` for image segmentation.

# Machine Learning and Deep Learning

## K-Means Clustering
- Use `cv2.kmeans()` for image clustering.

## SVM
- Use `cv2.ml.SVM_create()` for Support Vector Machine classification.

## Deep Learning Models
- Use OpenCV’s DNN module to load and run deep learning models.

# Video Processing

## Video Reading and Display
- Use `cv2.VideoCapture()` to read a video and `cv2.VideoWriter()` to save it.

## Frame Processing
- Process video frames for tasks like object detection and tracking.

# Camera Calibration and 3D Reconstruction

## Camera Calibration
- Use `cv2.calibrateCamera()` for camera calibration.

## Stereo Vision
- Use `cv2.StereoBM_create()` for stereo matching.

## 3D Reconstruction
- Use `cv2.reprojectImageTo3D()` for 3D reconstruction.

# Utility Tools

## Drawing Functions
- Use `cv2.line()`, `cv2.rectangle()`, `cv2.circle()`, etc., to draw shapes on images.

## Mouse Events
- Use `cv2.setMouseCallback()` to handle mouse events.

## Image Pyramids
- Use `cv2.pyrUp()` and `cv2.pyrDown()` for image pyramid operations.
