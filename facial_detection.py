import pygame
import cv2
import math
from PIL import Image
import numpy


#vars
stop = False
dt = 100

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise IOError("Cannot open webcam")

result, frame = cam.read()
if not result:
    raise IOError("Cannot read frame from webcam")

cv2.imwrite("webcam_image.jpg", frame)
#cam.release()
chosen_image = "webcam_image.jpg"
currentWebcamImage = pygame.image.load("webcam_image.jpg")

tester = Image.open(chosen_image)
(width, height) = tester.size
screen = pygame.display.set_mode((width + 20, height + 20))

while ( stop != True):
    
    for event in pygame.event.get():
        #key inputs
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                #stops the loop
                stop = True
    
    result, frame = cam.read()
    cv2.imwrite("webcam_image.jpg", frame)
    chosen_image = "webcam_image.jpg"
    currentWebcamImage = pygame.image.load("webcam_image.jpg")
    print(pygame.mouse.get_pos())
    #graphics
    screen.blit(currentWebcamImage, (10, 10))

    #time between each frame
    pygame.time.wait(dt)
    #updates the frame
    pygame.display.update()


