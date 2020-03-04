import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import cv2

img_list = [cv2.imread("output/color_cuda/out_" + str(i) + ".png") for i in range(0, 159)]
fig = plt.figure(figsize=(8, 8))
ims = [[plt.imshow(i, animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True)

ani.save("animation.gif", writer='mencoder')
