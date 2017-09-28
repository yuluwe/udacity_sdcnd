# **Build a Traffic Sign Recognition Project**
---
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report
---

[//]: # (Image References)

[image1]: ./writeup/testset.png "Visualization"
[image4]: ./writeup/img1.png "Traffic Sign 1"
[image5]: ./writeup/img2.png "Traffic Sign 2"
[image6]: ./writeup/img3.png "Traffic Sign 3"
[image7]: ./writeup/img4.png "Traffic Sign 4"
[image8]: ./writeup/img5.png "Traffic Sign 5"
[image9]: ./writeup/gray.png "Grayscale"
[image10]: ./writeup/pandr.png "Rotaion and Projection"
[image11]: ./writeup/proj.png "Projection"
[image12]: ./writeup/cut.png "Random Cutting"


## Reflections
---
#### In this section, the detailed procedures of the goals of this peoject will be discussed and I will show how they meets the requirements from the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually. Note: I referred to this [website](https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python) for grayscale conversion. I used this equation because I cannot import cv2 module in the jupyter notebook. As a result, I could not use the built-in function in cv2 and had to do the conversion manually.
---

### 1. Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data was distributed with respect to the classes.

![alt text][image1]

#### 3. Introduce the preprocess methods

In order to augment the dataset. I used the following five methods:

##### Convert all image data to grayscale.
This is because the pattern of traffic signs already contains enough infomation for us to do the classification. Here is an example of the grayscale image.

![alt text][image9]


##### Perform projective transformation to all image data.
With projective transformation, I could easily enlarge the dataset without collecting new traffic sign pictures. I wrote the function [img_project()] to do the transformation. I first defined four 6x6 regions in every corners of the original 32x32 image. Then I used a random function to pick four random points in these regions and used the [skimage.transform.ProjectiveTransform()] to map the old image into the tetragon defined by the four points. Here is an example of the image after projection.

![alt text][image11]

##### Perform rotation transformation to all image data.
Similar to projective transformation, I used rotation transformation to generating more data. I wrote the function [img_ratate()] to perform a rotation with angle between -45 and 45. Since the new image could be similar to the original one when the angle is small, I combined rotation and projection together to enlarge the difference. Here is an example of the image after rotation and projection.

![alt text][image10]

##### Perform random cutting to all image data.
This is just another way to create new data. I wrote the function [img_hole] to randomly remove a 6x6 block from the original image. Since the traffic signs are usually  in the center of the image. The 6x6 block will be chosen from the center 14x14 region. Here is an example of image after random cutting.

![alt text][image12]

#### In conclusion, I used grayscale, projective transformation, rotation, and random cutting to enlarge the dataset by three times.Since I apply the transformation to all image data, the proportion of every class still remains the same. The new training set, validation set, and test set have a size of 139196, 17640, and 50520 respectively.


### 2. Design and Test a Model Architecture

#### 1. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale normalized image   			| 
| Convolution1 6x6     	| 1x1 stride, same padding, outputs 32x32x64 	|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| RELU					|												|
| Dropout				| keep_prob = 0.8								|
|       				|												|
| Convolution2 6x6     	| 1x1 stride, same padding, outputs 16x16x128 	|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128 		     		|
| RELU					|												|
| Dropout				| keep_prob = 0.6								|
|       				|												|
| Convolution3 6x6     	| 1x1 stride, same padding, outputs 8x8x256  	|
| Max pooling	      	| 2x2 stride,  outputs 4x4x256 	            	|
| RELU					|												|
| Dropout				| keep_prob = 0.4								|
| Flatten				| outputs 4096									|
|     					|												|
| Fully connected 1		| outputs 1024        							|
| RELU					|												|
| Dropout				| keep_prob = 0.1								|
|    					|												|
| Fully connected 2		| outputs 43        							|
| Softmax				|         							    		|
| L2 Regulization		| beta = 0.0001									|
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a batch size 128 and epoch size 100. When the validation accuracy is lower than 0.93, I used a learning rate of 0.0005, and when validation accuracy reached 0.95, I dropped the learning rate to 0.0001 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 

My final model results were:

* training set accuracy of 1.000000
* validation set accuracy of 0.973696
* test set accuracy of 0.973317 for the original test set and 0.964727 for the augmented test set

My network was built based on the LeNet. I first tried the same architecture as the "LeNet Lab" and found the accuracy was much lower than 0.93. I followed the instructions from the rubric and augmented the dataset, but the test accuracy was still relatively low. As a result, I thought my current model was "under-fit" and added one more convolutional layer and two more fully connected layers to the net. Then I found the training accuracy quickly exceeded the validation accuracy and I known I overfitted the training set, so I added dropout to all the layers. After some adjustment of the parameters, I found 3 convolutional layers and two fully connected layers gave me the best test accuracy. The highest test accuracy of the original test set I ever reached was 0.979121. However, since each time I ran my notebook, the dataset would be different and hence the accuracy would also be different.

The keep_probs of each layers did not affect accuracy severely. The reason I chose these values is because I wanted to keep more information from the opriginal image and less information between layer and layer. Also, when all the values of keep_probs were too low, the validation and test accuracy would stop at a value around 0.93. When their values were too high, I would get a overfitting model. 

For the stddev value in weight initialization, I tried values of 0.1, 0.001, and 0.0001. It turned out 0.001 gave me a better performance. 

For learning rate, I chose 0.0005 and 0.0001. I tried some values larger than 0.001, but but their performance were bad. I got both low training accuracy and validation accuracy. As a result, I chose the current two values to keep a balance between accuracy and time.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it has unusual brightness.

The second image might be difficult to classify because the color of lower half of the image is very close to that of its surrounding.

The third image might be difficult to classify because there are actually two signs in the image.

The fourth image might be difficult to classify because it is very obscure.

The fifth image might be difficult to classify because part of the symbol has a different color.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield                             		| Yield   									| 
| 80 km/h                       			| 80 km/h									|
| Road work	                     			| Road work									|
| Vehicles over 3.5 metric tons prohibited	| Vehicles over 3.5 metric tons prohibited	|
| Ahead only		                     	| Ahead only            					|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. As a result, there is no need to calculate F1-score. Also, I am not very clear about how to classify the result as "True Positive", "True Negative", "False Positive", and "False Negative". 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The softmax probability calculation can be found in the cell [189] of my jupyter notebook. As you can see from the result of [Tranffic Sign 1]. My model performs quite accurate. The probability for [Yield] sign is almost 1 and the rest are all close to 0. Also, the result of the rest four pictures are also similar to this one, hence I only give picture 1 as an example.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.000        			| Yield   			     						| 
| 1.199e-13     		| Priority road 								|
| 4.208e-15				| No vehicles									|
| 1.755e-15	      		| Bumpy Road					 				|
| 1.488e-18			    | End of all speed and passing limits      		|
