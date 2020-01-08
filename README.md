# MQR Tracker 

Python implementation of a robust multy-object tracker based on optical flow and a model of color, the tracker was specifically taylored for vehicles. To test the tracker see import the library ```MQR.py``` and use the tracker as you would if it as an OpenCV-like tracker object https://docs.opencv.org/3.4/d2/d0a/tutorial_introduction_to_tracker.html

## Use in traffic Estimation. 

The tracker was a key part in a traffic estimation software developed in the city of Bogotá. The software is built upon a single shot mutibox detector (https://arxiv.org/pdf/1512.02325.pdf). fine tuned over a particular dataset that is not available to the public. For a demo of the functionality of the software see ```demo0.avi``` and ```demo1.avi```, for the code see ```main.py``` and for a detailed explanation of the methods see the draft ```OF_and_color_Tracker.pdf```   

## Results.

The tracker was compared against the other implementations available in OpenCV using the database provided by the colombian government and the following results were obtained. 



