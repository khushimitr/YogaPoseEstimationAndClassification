#!/usr/bin/env python
# coding: utf-8

# '''Helper Drawing Functions'''

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd 
import cv2
import os


# In[3]:


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

## Drawing the keypoints
def draw_keypoints(frame,keypoints,confidence_thresh,score,match):
    color = (0,255,255)
    if(match and score > 85):
        color = (255,255,255)
    for kp in keypoints:
        kx,ky,kp_conf = kp
        if kp_conf > confidence_thresh:
            cv2.circle(frame,(int(kx),int(ky)), 4, color, -2)

def draw_connections(frame,keypoints,edges, confidence_thresh,score,match):
    color = (0,0,255)
    if(match and score > 85):
        color = (0,255,0)
    shaped = np.squeeze(keypoints)
    
    for edge,c in edges.items():
      p1,p2 = edge
      x1,y1,c1 = shaped[p1]
      x2,y2,c2 = shaped[p2]
      if(c1 > confidence_thresh) & (c2 > confidence_thresh):
        cv2.line(frame,(int(x1),int(y1)),(int(x2),int(y2)),color,3)

def image_resize(image,width = None,height = None,inter = cv2.INTER_AREA):
    (h,w,_) = image.shape
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

