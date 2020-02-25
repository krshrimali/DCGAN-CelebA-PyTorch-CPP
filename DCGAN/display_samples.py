import argparse
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np

j = 0
for i in range(2, 1591, 10):
    filename = "build/color/dcgan-sample-" + str(i) + ".pt"
    module = torch.jit.load(filename)
    images = list(module.parameters())[0]

    for index in range(64):
        # image = images[index].detach().cpu().reshape(64, 64, 3).mul(255).to(torch.uint8)
        image = images[index].detach().cpu().mul(255).to(torch.uint8).numpy()
        array = np.transpose(image, (1, 2, 0))
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        # cv2.imwrite("out-" + str(index) + ".png", array)
        axis = plt.subplot(8, 8, 1+index)
        plt.imshow(array)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    plt.savefig("output/color_cuda/out_" + str(j) + ".png")
    print("Saved ", "output/color_cuda/out_" + str(j) + ".png")
    j += 1
