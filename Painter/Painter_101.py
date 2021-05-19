#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 11:21:48 2021

@author: orram

What are we doing here?!

Well, if you can't do, teach...

I find painting, and drewing an impossible and amazing talent, ever saince 
I was a teenager I was drown to paintings but unfotunatly never had a huge talent
to it. What I do know, or am learning to, is to teach machines to do staff. 
Combining the two I had this thought at my mined for a long time, why not build
a robot that can paint art works? Starting from simulations and simple shapes 
on the computer and hopefully moving from the virtual to the real world and 
painting with real colors! 
During this task I also want to ask questions about art and neuroscience, 
questions about representations of the object as we paint it, about how
this representations can change and what can we learn from these to our lives. 

I will write it as a blog post so hopefully it will be organized and readable, 
but probebly not as orgenized nicely as a finished project. But I hope the final
stages of each part will be compiled neatly to a folder showing the final projects
from each test. 

As I see it today, this project has four main stages:
    1) Learn simple drewing in python - nothing too fancy, drewing pixle by pixle
        in a numpy array. First trying to copy and then to pass a classifier
        but keeping the output as far from the original as possible.
    2) Move to a drewing platform - learn to control an app with python and learn 
        to drew using brushes. Again, teach it to replicate and change a given 
        image.
    3) Connect to a camera - drew from the real world. 
    4) Build a robot - do everything with a real robotic arm and real brushes!
    
Every steps is diffecult but I guess the first one will be the easiest and the 
later stages will take more and more time. 

Let's have fun!

"""
#an example to drew a shape using PIL

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

'''
#Task one - paint the given image (one color)
#Task two - paint the given image (few colors, make the same color)
#task three - paint a given image, reward is given from a classifier and 
             #the aim is to recreate the shape class but not duplicate
#Task four - paint from a commend (not from example)

##################################### TASK ONE ################################################
########################## Paint the given image (one color) ##################################

Description of the task:
    Given a 100X100 image, duplicate the shape that is shown in the image.
    Do do so, the leaner has N strokes, defined as the action.
    
    State space  - the given image to replecate, the 'paper' on which to drew
    Action space - Let's start with as simple as we can -
                   starting point, end point, only strate lines are allowed.
                   (Later we can look on controlling drewing software or adding
                   abbillities such as speed, z axis, curve etc...)
                 
    Dataset      - 1,000 shapes to train on, 100 to test.
    
                        
'''

################################## Define the Dataset ########################################

shape_dataset = []
num_shapes = 4
for n in range(num_shapes):
    '''
    Drew a rendom rectangle by drewing 4 random numbers:
        x_coor, y_coor - starting coordinates
        w,h            - width and highet
        
    From thes extrapulate all the coordinates
    '''
    
    start_x = np.random.randint(15, 55)
    start_y = np.random.randint(5, 45)
    w = np.random.randint(5,15)
    h = np.random.randint(5,15)
    end_x = start_x - w
    end_y = start_y - h
    #plt.figure()
    
    temp_image = Image.new('RGB', (64, 64), 'white')  # could also open an existing image here to draw shapes over it
    
    draw = ImageDraw.Draw(temp_image)
    
    draw.rectangle((start_x, start_y, end_x, end_y), outline='red')  # can vary this bit to draw different shapes in different positions
    
    #image.save('01_rectangle.png')
    #plt.imshow(temp_image)
    shape_dataset.append(temp_image)

################################ Define Enviorment #########################################
'''
Define the enviorment 
* that drews a random image from a given dataset.
* recieves an action from the actor and drews it
* outputs the next state and reward
*

'''





















