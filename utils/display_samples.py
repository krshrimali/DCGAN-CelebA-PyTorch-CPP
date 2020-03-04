import argparse
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import os

if(os.path.isdir("build/output/output_images") == False):
    os.mkdir("build/output/output_images/")

if(os.path.isdir("build/output/output_animation") == False):
    os.mkdir("build/output/output_animation/")

fig_number = 0
checkpoint_number = 0
while(True):
    filename = "build/output/dcgan-sample-" + str(checkpoint_number) + ".pt"
    try:
        module = torch.jit.load(filename)
    except ValueError:
        print("Loading all (till dcgan-sample-{}.pt) checkpoints finished...".format(checkpoint_number))
        print("Please check build/output/output_images/ for images")
        break
    except:
        print("Error occured.")
        break
    images = list(module.parameters())[0]

    for index in range(64):
        image = images[index].detach().cpu().mul(255).to(torch.uint8).numpy()
        array = np.transpose(image, (1, 2, 0))
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        axis = plt.subplot(8, 8, 1+index)
        plt.imshow(array)
        axis.get_xaxis().set_visible(False)
        axis.get_yaxis().set_visible(False)
    plt.savefig("build/output/output_images/out_" + str(fig_number) + ".png")
    print("Saved figure {} at: {}".format(fig_number, "build/output/output_images/out_" + str(fig_number) + ".png"))
    fig_number += 1
    checkpoint_number += 2
