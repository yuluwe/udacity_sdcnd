## Advanced Lane Finding Project

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

[image1]: ./Testout/distortion.jpg "Distortion Correction"
[image2]: ./Testout/undistort_demo.jpg "Road Transformed"
[image3]: ./Testout/thres_mask.jpg "Binary Example"
[image4]: ./Testout/fit.jpg "Warp Example"
[image5]: ./Testout/final.jpg "Fit Visual"
[image6]: ./Testout/final_pipeline.jpg "Output"
[video1]: ./project_video.mp4 "Project Video Result"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

Basically I just used the sample code from the coding practice on course website. The following result can be found in the "P4.ipynb" file

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The first step of the pipeline is the distortion correction. Again, the following result can be found in the "P4.ipynb" file
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (Detailed steps can be found in the "P4.ipynb file and"). 

For the color thresholding, I tested two different color space(HLS and LAB) and used the "H" and "S" channels from HLS and "L" and "B" channels from LAB.

For the gradient thresholding, I combined the direction and magnitude threhold as well as the sobel x and y gradient threshold. 

The result after apply all above is shown below and it can be found in the "P4.ipynb" file in the section of "Demostrate the pipeline". I also did a simple masking to choose the region of interest.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In stead of using "varying" nodes for the transformation that is used in the demo, I simply defined 8 fixed points, but the final result is still good.

```python
src = np.float32([[560, 485],
                  [770, 485],
                  [1100, 700],
                  [240, 700]])
dst = np.float32([[300, 250],
                  [1000, 250],
                  [1000, 700],
                  [300, 700]])
```
The left image below shows the result of persective tranformation from the "masked and thresholded" result.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After obtaining the persective transformation result, I applied the sliding window search as is taught in the course vedios and fitted a 2nd-order polynomial based on the points found (the red and blue points in the right image above). The basic idea is using the histogram of binary map in the vertical direction. If the previous steps are accomplished properly, we will see two peaks in the histogram which will be the positions of the lanes. Then we can choose the points around the two positions and update the postions based on the centroids of the chosen points. We begin the search from the bottom of the image and repeat this process upwards. In this way we will find all the points we need to fit the lanes.

To do this, I also defined a class call "Line", which contained the coefficients of the fitting result (in the member "current_fit"). This class will also be used to implement the moving average filter in the later step. I stored 8 latest fittings in the class. The best fitting will be the average of the 8 lastest results. Every time when a new fitting is generated, I will compare the current fitting with the best fitting. If the difference between them is acceptable, the current result will be push into the list and the oldest fitting will be pop out. The best fit will be returned to draw lanes in the later step.

The following code is part of the class and the full definition can be found in the "P4.ipynb". 
```python
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False 
        
        # x values of the last 10 fits of the line
        self.recent_xfitted = [] 
        
        #average x values of the fitted line over the last 10 iterations
        self.bestx = None     
        
        #polynomial coefficients averaged over the last 10 iterations
        self.best_fit = None  
        
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])] 
    .
    .
    .
```

The following image shows the final result of the image processing pipeline.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The detail process to compute radius of curvature of the lane is clearly shown in the course website. We can use the lane points at the bottom of the image to calculate the result. That is, the y-axis we need is always the image height and we can simply apply the equation to the coefficients we found for the best fitting results to get the radius. It is important to note that a unit conversion is needed to convert pixels into meters.

Similarly, to calculate the deviation from the center of the road, we just need to calculate the x-coordinates of lanes at the bottom of the image. Again in this case, the y-coordinate is always the image height and what we need to do is applying the fitting model to find x-coordinates of the two lanes. The difference between average of the x-coordinates and half of the image width will be car's deviation from the road center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code for the above process can be found in the "P4.ipynb" file and the following image is a demo for the final result.

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](./Testout/project.mp4) and 
here's a [link to the challenge video result](./Testout/challenge.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The color and gradient thresholding function is not perfect and can be improved. If the image is affected by strong light like the "harder_challenge.mp4", the thresholding result will be very terrible. The moving average filter I wrote need a good initialization to ensure a good performance, making the pipeline unstable with respect to the various and complicated real world situations.

For me, the most critical part of this project is the color and gradient thresholding. If the result of this part is good, all steps after this will be very simple and robust. If I could have more time, I would try some more color space and thresholds.
