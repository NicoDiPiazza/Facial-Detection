import pygame
import cv2
import math
from PIL import Image
import numpy
import time
import sys

start_time = time.perf_counter()


def sigmoid(a):
    return(1/(1+math.exp(-a)))

def getLumninance(pixel: list, chosen_image):
    #returns a value between 0 and 1 of the percent luminance of the pixel
    (r, g, b) = Image.open(chosen_image).load()[pixel[0], pixel[1]]
    averageRGB = (r + g + b)/3
    return(averageRGB/255)

N_Images = 18
imageWidth = 640
imageHeight = 480
inputWidth = 48
inputHeight = 32
imageLayerSize = inputWidth * inputHeight + 1
hiddenLayerSize = 10 + 1
numpy.set_printoptions(threshold=sys.maxsize)

scalingX = imageWidth / inputWidth
scalingY = imageHeight / inputHeight

# for every image
saveFile = open(r'facial detection\luminanceFile.txt', 'w')
saveFile.close()
saveFile = open(r'facial detection\luminanceFile.txt', 'a')

for i in range(N_Images):
    filename = r'C:\Users\ThinkPad\OneDrive - Oregon State University\Documents\python\facial detection\datasetClean\webcam_image' + str(i) + '.jpg'
    luminances = numpy.zeros(imageLayerSize)
    for j in range(imageLayerSize - 1):
        pixX = (j%inputWidth) * scalingX - scalingX/2
        pixY = math.floor(j / inputWidth) * scalingY - scalingY/2
        # getting the luminances takes about 4-5 sec per image, so moving it outside the loop means that it is WAY faster after when 
        # actually training.
        if True:
            luminances[j] = getLumninance([pixX, pixY], filename)

    saveFile.write(str(luminances))
    saveFile.write('\n')
saveFile.close()

end_time = time.perf_counter()

run_time = end_time - start_time
print(f"The program ran in {run_time:.4f} seconds")
