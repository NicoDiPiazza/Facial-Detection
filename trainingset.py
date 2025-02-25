import pygame
from PIL import Image
import numpy


class TrainingFrame():
    def __init__(self, eyeOne: list, eyeTwo: list, mouth: list, filename: str):
        self.eyeOne = eyeOne
        self.eyeTwo = eyeTwo
        self.mouth = mouth
        self.filename = filename

def boxIt(pixel: list, img):
    for i in range(10):

        img[pixel[0] - 5, pixel[1] -5 + i] = 255
        img[pixel[0] + 5, pixel[1] -5 + i] = 255
        img[pixel[0] - 5 + i, pixel[1] -5] = 255
        img[pixel[0] - 5 + i, pixel[1] +5] = 255



#vars
stop = False
dt = 1
trainingData = numpy.zeros(20, TrainingFrame)
N_images = 18

saveFile = open(r'facial detection\training_data_save_file.txt', 'w')
saveFile.close()


chosen_image = "webcam_image.jpg"
tester = Image.open(chosen_image)
(width, height) = tester.size
screen = pygame.display.set_mode((width + 20, height + 20))

for i in range(N_images):
    file_path = r'C:\Users\ThinkPad\OneDrive - Oregon State University\Documents\python\facial detection\datasetClean\webcam_image' + str(i) + '.jpg'
    trainingData[i] = TrainingFrame([-1, -1], [-1, -1], [-1, -1], file_path)
    currentWebcamImage = pygame.image.load(file_path)
    screen.blit(currentWebcamImage, (10, 10))
    pygame.display.update()
    for j in range(3):
        stop = False
        
        while stop != True:
            
            if pygame.event.wait().type == pygame.MOUSEBUTTONDOWN:
                stop = True

            if j == 0:
                trainingData[i].eyeOne = pygame.mouse.get_pos()
            if j == 1:
                trainingData[i].eyeTwo = pygame.mouse.get_pos()
            if j == 2:
                trainingData[i].mouth = pygame.mouse.get_pos()


                
    print(pygame.mouse.get_pos())
    saveFile = open(r'facial detection\training_data_save_file.txt', 'a')
    saveFile.write(str(trainingData[i].eyeOne))
    saveFile.write(str(trainingData[i].eyeTwo))
    saveFile.write(str(trainingData[i].mouth))
    saveFile.write('\n')
    saveFile.close()

for i in range(N_images):
    
    currentWebcamImage = pygame.image.load(trainingData[i].filename)
    screen.blit(currentWebcamImage, (10, 10))

    chosen_image = trainingData[i].filename

    tester = Image.open(chosen_image)
    img = tester.load()

    boxIt(trainingData[i].eyeOne, img)
    boxIt(trainingData[i].eyeTwo, img)
    boxIt(trainingData[i].mouth, img)
    print(trainingData[i].eyeOne, trainingData[i].eyeTwo, trainingData[i].mouth)

    tester.save('newTestImage.jpg')
    trainingTester = pygame.image.load('newTestImage.jpg')

    stop = False
    while stop != True:
        screen.blit(trainingTester, (10, 10))
        #updates the frame
        pygame.display.update()

        if pygame.event.wait().type == pygame.MOUSEBUTTONDOWN:
            stop = True
