# **Finding Lane Lines on the Road** 

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


---

## Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First of all, I applied color selection to the original image. I refered to method described in https://medium.com/@liamondrop/joshua-owoyemi-nice-writeup-ac6a7caef035 that combined both RGB and HSV color to increase accuracy. I defined the range of white color in RBG and yellow color in RGB&HSV, and then created a mask to get yellow and white color in the original image. Next, I turned the filtered image to gray scale and apply the "region_of_interst()" function. Then I used the "canny()" functon to detect the edges in the thrid step and used "hough_line()" function in the fourth step to draw the lane line in the original image. The last step is was using the "weighted_img" function retuen the original image with lane lines.

In order to draw lane lines, I defined a new funtion called "draw_line_new()". I kept the original function since it was extremely helpful when tuning the hough function parameters and color selection thresholds. The basic idea of my new function is to seperate the points on the lines found by "hough_lines()" into two groups based on the sign of their slope. Then I could use "numpy.polyfit()" to fit two lane lines based on the two group of points.


### 2. Identify potential shortcomings with your current pipeline


Since I did not smooth my lane drawing functon, the lane lines were still kind of jumpy. Also, my currently code did not work well in indentifying white color, especially when the white lane segments are extremely short. This could result in large leaps of the slope of the lane lines.


### 3. Suggest possible improvements to your pipeline

I will keep working on tuning the parameters of "hough_line" function and thresholds of color selection. Meanwhile, I can also smooth the drawing function to make the lane lines more stable. I was also thinking a way to save the slope of previous frames and pass it to the next frame so that I could use the value filtering out some "obviously wrong slope calculation" due to outliers.
