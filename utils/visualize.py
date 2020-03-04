import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2
import sys, os

if(os.path.isdir("build/output/output_images/") == False):
    print("Directory not found. Please check you ran display_samples.py file before running visualize.py")
    print("Exiting...")
    sys.exit(0)

if(os.path.isdir("build/output/output_animation/") == False):
    os.mkdir("build/output/output_animation/")

# img_list = [cv2.imread("build/output/output_images/out_" + str(i) + ".png") for i in range(0, 159)]

img_index = 0
img_list = []
while(True):
   temp_img = cv2.imread("build/output/output_images/out_" + str(img_index) + ".png", 1)
   if(temp_img is None):
       print("Finished reading all ({}) images".format(img_index))
       print("Now creating animation")
       break
   img_list.append(temp_img)
   img_index += 1

fig = plt.figure(figsize=(8, 8))
ims = [[plt.imshow(i, animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

# Change mencoder to PillowWriter
ani.save("build/output/output_animation/animation.gif", writer='mencoder')
