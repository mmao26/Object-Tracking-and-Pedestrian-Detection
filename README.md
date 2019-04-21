## Object Tracking and Pedestrian Detection

 Several tracking methods, namely, Kalman and Particle Filters, are implemented for image sequences and videos.

#### Kalman Filter

* The Kalman Filter (KF) is a method used to track linear dynamical systems with Gaussian noise. Correct the state and the covariances using the Kalman gain and the measurements obtained from the sensor.

* In order to find and track pedestrians in a video, the sensor has to be updated with a function that can locate people in an image. A function in OpenCV called Histogram of Gradients (HoG) descriptors can be directly used.


#### Particle Filter

*  In implement particle filter, we need to track an image patch template taken from the first frame of the video. We define the image patch as the model, and the state is only the 2D center location of the patch. Thus each particle will be a (u, v) pixel location representing a proposed location for the center of the template window.

#### Changes in Appearance

*  In order to track the object with changing shape, we use **Infinite Impulse Response (IIR)** filter. We first find the best tracking window for the current particle distribution and then we update the current window model to be a weighted sum of the last model and the current best estimate.


#### Particle Filters and Occlusions
* In order to track target, which might be occluded, we need some way of relying less on updating our appearance model from previous frames and more on a more sophisticated model of the dynamics of the figure we want to track. A scale factor ‘s’, representing the particle state for each particle. The movement of particle (x, y)  depends on the distance from the “the strongest particle” (x, y) in the prior frame.


### Running the Tests
All results for each project are shown in **Report.pdf**.

### Authors
* **Manqing Mao,** maomanqing@gmail.com

<!-- See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project. -->
