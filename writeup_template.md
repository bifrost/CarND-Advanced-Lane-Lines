## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1a]: ./camera_cal/calibration17.jpg "Distorted"
[image1b]: ./output_images/dist_correction_calibration.jpg "Undistorted"
[image2]: ./test_images/test2.jpg "Road"
[image2]: ./output_images/dist_correction_test2.jpg "Road Transformed"
[image3]: ./output_images/binary_test2.jpg "Binary Example"
[image4a]: ./output_images/perspective_image.jpg "Perspective Example"
[image4b]: ./output_images/perspective_warped.jpg "Warp Example"
[image5a]: ./examples/color_fit_lines.jpg "Fit Visual"
[image5b]: ./output_images/lane_warped_test2.jpg "Warped Output"
[image5c]: ./output_images/lane_image_test2.jpg "Output"
[image6]: ./output_images/annotate_test2.jpg "Annotated"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in code cell [3] of the IPython notebook located in "./AdvancedLaneLines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world, see `getCalibrationPoints`. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

*Distorted Image*
![alt text][image1a]

*Undistorted Image*
![alt text][image1b]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one, see code cell [4]:

*Distorted Image*
![alt text][image2a]

```pytthon
    undistort_image = cv2.undistort(image, mtx, dist, None, mtx)
```
where `mtx` is the camera matrix and `dist` distortion coefficients from the `cv2.calibrateCamera()` function.

*Undistorted Image*
![alt text][image2b]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, see code cell [5]. The function `combining_thresholds` combine different thresholds methods. I have tried several combination and found that following combination did a great job for the test images:

```python
    gradx = abs_sobel_thresh(l_channel, orient='x', sobel_kernel=kernel, thresh=x_thresh)
    grady = abs_sobel_thresh(l_channel, orient='y', sobel_kernel=kernel, thresh=y_thresh)
    s_binary = thresholds(s_channel, kernel=kernel, thresh=s_thresh)

    combined = np.zeros_like(gradx)
    combined[
        ((gradx == 1) & (grady == 1)) |
        s_binary == 1
    ] = 1
```
where `l_channel` and `s_channel` is the L and S channel from the HLS color space. 

The code for creating the binary image looke as followind:

```python
    binary_image = combining_thresholds(
        image, 
        kernel=5, 
        x_thresh=(30, 255),
        y_thresh=(30, 255),
        mag_thresh=(50, 255),
        dir_thresh=(0.1, 1.4),
        s_thresh=(140, 255)
    )
    
    binary_image = region_of_interest(binary_image, vertices_roi)
```
Notice that the function `region_of_interest` afterwards has been applied on the binary image to zero all pixels outside region of interest, see code cell [6].

The region of interest is:

| ROI           |
|:-------------:|
| 170, 700      |
| 560, 450      |
| 720, 450      |
| 1140, 700     |

Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform can be found in code cell [7]. The following lines finds the perspective transform matrix `M` and inverse perspective transform matrix `Minv`.

```python
    M = cv2.getPerspectiveTransform(src_vertices, dst_vertices)
    Minv = cv2.getPerspectiveTransform(dst_vertices, src_vertices)
```

The image is warped by following call:

```python
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
```

I chose the hardcode the source and destination points in the following manner:

```python
src_vertices = np.float32([
        [590, 452],
        [690, 452], 
        [1120, 720], 
        [194, 720]
    ])

dst_vertices = np.float32([
        [294, 0],
        [1020, 0], 
        [1020, 720], 
        [294, 720]
    ])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 452      | 294, 0        | 
| 690, 452      | 1020, 0       |
| 1120, 720     | 1020, 720     |
| 194, 720      | 294, 720      |

*Image*
![alt text][image4a]

*Warped Image*
![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To  identified lane-line pixels and fit their positions with a polynomial I used following two methods `fitSlidingWindow` and `fitAroundPoly`.

*fitSlidingWindow*
This function fits the lane-line pixels to a polynomial by using a sliding window approach.
The function `find_lane_pixels` is called to find lane pixels of interest. It use a sliding window to find lane pixels, see Lesson 7, 4. "Finding the Lines: Sliding Window". Next the line pixels are used to initialize a `Line` object that fits a polynomial, see code cell [8].

```python
def fitSlidingWindow(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty = find_lane_pixels(binary_warped)
    
    shape = binary_warped.shape
    leftLane = Line(shape, xm_per_pix, ym_per_pix)
    leftLane.init(leftx, lefty)
    
    rightLane = Line(shape, xm_per_pix, ym_per_pix)
    rightLane.init(rightx, righty)

    return leftLane, rightLane
```

where `xm_per_pix` is meters per pixel in x dimension and `ym_per_pix` is meters per pixel in y dimension. 

*fitAroundPoly*
This function fits the lane-line pixels to a polynomial by using prior knowledge from the previous lane-line polynomial.
The function `search_around_poly` is called to find lane pixels of interest. It search for the pixels around the lane-line polynomial, see Lesson 7, 5. "Finding the Lines: Search from Prior". Next the line pixels are used to initialize a `Line` object that fits a polynomial, see code cell [10].

```python
def fitAroundPoly(binary_warped, leftLane, rightLane):
    leftx, lefty, rightx, righty = search_around_poly(binary_warped, leftLane.best_fit, rightLane.best_fit)
    
    shape = binary_warped.shape
    leftLane = Line(shape, xm_per_pix, ym_per_pix)
    leftLane.init(leftx, lefty)
    
    rightLane = Line(shape, xm_per_pix, ym_per_pix)
    rightLane.init(rightx, righty)

    return leftLane, rightLane
```

The `Line` object fits the lane lines with a 2nd order polynomial kinda like this, in the function `fitPolynomial`:

![alt text][image5a]

This image shows the lane area on warped image.

*Lane area on warped image*
![alt text][image5b]

This image shows the lane area on un image.

*Lane area on undistort image*
![alt text][image5c]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The following methods implemented in the `Line` class, is used to calculate the radius of curvature of the lane and the position of the vehicle with respect to center, see code cell [8]:

Presumption:
- the projection of the lane is about 30 meters long
- the lane is about 3.7 meters wide

Conversions of x and y from pixels space to meters
- `xm_per_pix = 3.7/726` meters per pixel in x dimension (1020 - 294 = 726) 
- `ym_per_pix = 30.0/720` meters per pixel in y dimension 

```python        
# use best_fit to estimate radius_of_curvature
def updateRadiusOfCurvature(self):
    if self.best_fit is None:
        return

    # scale polynomial coefficients A and B to fit in meters
    a_scale = self.xm_per_pix/(self.ym_per_pix**2)
    b_scale = self.xm_per_pix/self.ym_per_pix

    # calculate radius_of_curvature
    y_eval = self.bottom_y
    numerator = ((1 + (2*a_scale*self.best_fit[0]*y_eval*self.ym_per_pix + b_scale*self.best_fit[1])**2)**1.5)
    denominator = np.absolute(2*a_scale*self.best_fit[0])

    self.radius_of_curvature = numerator/denominator
```

where `a_scale` and `b_scale` is used to scale polynomial coefficients A and B to fit in meters. 

```python
# use best_fit to estimate line_base_pos
def updateLineBasePos(self):
    if self.best_fit is None:
        return

    # calculate line_base_pos
    y_eval = self.bottom_y
    fitx = self.fitX(y_eval)
    self.line_base_pos = abs(fitx - self.shape[1]/2)*self.xm_per_pix
```

Note that in both methods `best_fit` is used for estimation.

The lane `radius_of_curvature` and `line_base_pos` is calculated in the `annotate` function. The function is used to annotate an image with `radius_of_curvature` and `line_base_pos`.

```python
# annotate an image with radius_of_curvature and line_base_pos for a lane
def annotate(img, left_line, right_line):
    if not left_line.detected or not right_line.detected:
        return
    
    radius_of_curvature = (left_line.radius_of_curvature + right_line.radius_of_curvature)/2
    putText(img, 'Radius of Curvature {0:.0f}(m)'.format(radius_of_curvature), (50, 60))
    
    center = (left_line.line_base_pos + right_line.line_base_pos)/2
    line_base_pos = center - left_line.line_base_pos
    leftRight = 'left' if line_base_pos > 0 else 'right' 
    putText(img, 'Vehicle is {0:.2f}m {1} of center'.format(abs(line_base_pos), leftRight), (50, 120))
```

where `radius_of_curvature` is defined as `(left_line.radius_of_curvature + right_line.radius_of_curvature)/2` and `line_base_pos` is defined as `(left_line.line_base_pos + right_line.line_base_pos)/2 - left_line.line_base_pos`.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The following is the lane area image annotated with `radius_of_curvature` and `line_base_pos`, see code cell [12]. 

*Annotated image*
![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The all is put together in `find_lane` and consist of following steps:

```
1. resize to (1280, 720)
2. undistort image
3. make binary image
4. keep region of interest
5. warp image to bird's eye view perspective
6. find lane lines
```

The las step `6. find lane lines` use both `fitSlidingWindow` and `fitAroundPoly` dependen on if prior polynomial fit exists, see code cell [14]. 

The function `process_image` is used to process the movie images and do the annotation, see code cell [15]. 


Here's a [link to my video result](./videos_output/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Choosing the right area for perspective mapping seems to be important for the polynomial fit, data in horisont is very noisy.

The greatest weakness of the algorithm seems to be the binary image detection that is not strong enough for the challenge video. The current pipeline has been optimized for the test images, so further improvements should be applied to images from the challenge video.

Adjustments of meta parameters in `6. find lane lines` and `sanityCheck` might also improve result. 

### Appendix

#### Line Class
```python    
# Presumption 
# - the projection of the lane is about 30 meters long
# - the lane is about 3.7 meters wide
# Define conversions in x and y from pixels space to meters
xm_per_pix = 3.7/726 # meters per pixel in x dimension (1020 - 294 = 726) 
ym_per_pix = 30.0/720 # meters per pixel in y dimension 

class Line():
    def __init__(self, shape, xm_per_pix, ym_per_pix):
        
        # max last n lines saved
        self.max_n_lines = 5
        
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None 
        # the last n fits of the line
        self.recent_fitted = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None  
        
        # the shape of the line image
        self.shape = shape
        # meters per pixel in y dimension
        self.ym_per_pix = ym_per_pix
        # meters per pixel in x dimension
        self.xm_per_pix = xm_per_pix
        # bottom of your image
        self.bottom_y = self.shape[0] - 1
        
    def props(self):
        return vars(self)
        
    def init(self, allx, ally):
        self.fitPolynomial(allx, ally)
        self.updateLineBasePos()
        self.updateRadiusOfCurvature()
        
    # calculate current_fit
    def fitPolynomial(self, allx, ally):
        line_detected = len(allx) != 0
        
        # Fit new polynomials
        fit = None
        bestx = None
        recent_xfitted = []
        recent_fitted = []
        if line_detected == True:    
            fit = np.polyfit(ally, allx, 2)
            y_eval = self.bottom_y
            bestx = fit[0]*y_eval**2 + fit[1]*y_eval + fit[2]
            recent_xfitted = [bestx]
            recent_fitted = [fit]
            
        self.detected = line_detected
        self.allx = allx
        self.ally = ally
        self.current_fit = fit
        self.best_fit = fit
        self.bestx = bestx
        self.recent_xfitted = recent_xfitted
        self.recent_fitted = recent_fitted
           
    def append(self, line):
        if line.detected is False:
            raise Exception('do not append an undetected line')
        
        newLine = Line(self.shape, self.xm_per_pix, self.ym_per_pix)
      
        newLine.__dict__.update(self.__dict__)
        
        newLine.recent_xfitted = list(newLine.recent_xfitted)
        newLine.recent_xfitted.append(line.bestx)
        if len(newLine.recent_xfitted) > self.max_n_lines:
            newLine.recent_xfitted = newLine.recent_xfitted[1:]
        
        newLine.bestx = np.average(newLine.recent_xfitted)
        
        newLine.recent_fitted = list(newLine.recent_fitted)
        newLine.recent_fitted.append(line.best_fit)
        if len(newLine.recent_fitted) > self.max_n_lines:
            newLine.recent_fitted = newLine.recent_fitted[1:]
            
        newLine.current_fit = line.current_fit
        newLine.best_fit = np.average(newLine.recent_fitted, axis=0)
        newLine.diffs = newLine.best_fit - line.best_fit
   
        newLine.allx = line.allx
        newLine.ally = line.ally
        
        newLine.updateLineBasePos()
        newLine.updateRadiusOfCurvature()
        
        return newLine
        
    # use best_fit to estimate line_base_pos
    def updateLineBasePos(self):
        if self.best_fit is None:
            return
        
        # calculate line_base_pos
        y_eval = self.bottom_y
        fitx = self.fitX(y_eval)
        self.line_base_pos = abs(fitx - self.shape[1]/2)*self.xm_per_pix
        
    # use best_fit to estimate radius_of_curvature
    def updateRadiusOfCurvature(self):
        if self.best_fit is None:
            return
        
        # scale polynomial coefficients A and B to fit in meters
        a_scale = self.xm_per_pix/(self.ym_per_pix**2)
        b_scale = self.xm_per_pix/self.ym_per_pix
        
        # calculate radius_of_curvature
        y_eval = self.bottom_y
        numerator = ((1 + (2*a_scale*self.best_fit[0]*y_eval*self.ym_per_pix + b_scale*self.best_fit[1])**2)**1.5)
        denominator = np.absolute(2*a_scale*self.best_fit[0])
        
        self.radius_of_curvature = numerator/denominator
    
    # Generate x and y values for plotting
    def generateXY(self, shape):
        ploty = np.linspace(0, shape[0]-1, shape[0])
            
        return self.fitX(ploty), ploty
    
    # use best_fit to fit x from y
    def fitX(self, ploty):
        try:
            fitx = self.best_fit[0]*ploty**2 + self.best_fit[1]*ploty + self.best_fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            # print('The function failed to fit a line!')
            fitx = 1*ploty**2 + 1*ploty
            
        return fitx
```

