# **Behavioral Cloning** 

## Writeup
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The final model was a convolutional neural network with alternating convolution and pooling layers, a dropout layer for regularisation and a few fully connected layers at the end of the network. I used RELU activations for each layer since they are known to perform well and also do not have the vanishing gradients problem unlike sigmoid functions. They also make the network sparse and help in reducing the complexity of the network in some cases.

#### 2. Attempts to reduce overfitting in the model

The model contains a dropout layer in order to reduce overfitting (model.py line 71). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 86). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and also flipping the images to train the model.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to design a simple convolution neural network model with 2 convolution and pooling layers and 2 fully connected layers. I also converted my image to grayscale and normalised it before passing it to the neural network for training. I thought this model might be appropriate because I felt that colors would not be so important to identify the road boundaries in the neural network. Since I was using a grayscale image, I also felt that a small network with less parameters would suffice. Since the network was small, I did not use any kind of regularisation.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set as well as the validation set. This implied that the model was underfitting. 

To combat the overfitting, I added 1 convolution and pooling layer and 3 fully connected layer. I also increased the number of filters in each convolutional layer from 16 to 32 and decided to use normalised RGB images as input in case they provide useful information that is not available in grayscale images. Although the input size was much larger, it turned out that the network was even more complex and so, it was overfitting the training set.

As a result, for the final model, I added a dropout layer and also cropped the image before passing it to the network, so that the network only gets useful parts of the image and does not get affected by the surroundings and other kinds of noise in the image.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 image   							| 
| Cropping2D     	| crop top 50 and bottom 20 pixels in image 	|
| Lambda					|	normalize between -0.5 to 0.5											|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 88x318x32     									|
| Max pooling	      	| 2x2 stride,  outputs 44x159x32 				|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 42x157x32     									|
| Max pooling	      	| 2x2 stride,  outputs 21x78x32 				|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 19x76x32     									|
| Max pooling	      	| 2x2 stride,  outputs 9x38x32 				|
| Dropout		| 0.5 for training        									|
| RELU					|												|
| Flatten					|										|
| Fully connected		| 5000 neurons        									|
| RELU					|												|
| Fully connected		| 1000 neurons        									|
| RELU					|												|
| Fully connected		| 200 neurons        									|
| RELU					|												|
| Fully connected		| 50 neurons        									|
| RELU					|												|
| Fully connected				| 1 neuron       									|

#### 3. Creation of the Training Set & Training Process

To capture good driving behaviour, I also used the images from the left and right camera along with the center camera images. For the left and right camera images, I added or subtracted a correction of 0.2 to the steering angle so that the vehicle gets trained to recover to the center of the road.

Since the entire track 1 only has turns in one direction, it was possible that the vehicle would only learn to turn in a single direction and not generalise well if just trained on this data. Hence, I also flipped each image and augmented the dataset with these flipped images and the negative of the steering angle of the corresponding original images.

After the collection process, I had 48210 number of data points. I then preprocessed this data by cropping away the top 50 and bottom 20 pixels of each image and then normalizing it in the range of -0.5 to 0.5. This was done during the training and validation process itself by adding Cropping and Lambda layers in the beginning of the neural network for cropping and normalization.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the mean absolute error for the training and validation set. I used an adam optimizer so that manually training the learning rate wasn't necessary.
