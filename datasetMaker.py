import pygame
import cv2
from PIL import Image
import os

#vars
stop = False
dt = 5
i = 0

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

while stop!= True and i < 20:
    for event in pygame.event.get():
        #key inputs
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                #stops the loop
                stop = True
    
    result, frame = cam.read()
    cv2.imwrite("webcam_image.jpg", frame)
    currentWebcamImage = pygame.image.load("webcam_image.jpg")
    print(pygame.mouse.get_pos(), str(i))

    #The literal r is required for specifying the correct path
    if pygame.mouse.get_pressed()[0]:
        folder_path = r'C:\Users\ThinkPad\OneDrive - Oregon State University\Documents\python\facial detection\datasetClean'
        file_name = "webcam_image" + str(i) + ".jpg"
        full_path = os.path.join(folder_path, file_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        cv2.imwrite(full_path, frame)
        pygame.time.wait(1000)
        i = i + 1

    #graphics
    screen.blit(currentWebcamImage, (10, 10))

    #time between each frame
    pygame.time.wait(dt)
    #updates the frame
    pygame.display.update()
    


