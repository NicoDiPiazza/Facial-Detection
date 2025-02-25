import pygame
import cv2
import math
from PIL import Image
import numpy
import time
import sys


def biasDatThang(layer):
    length = len(layer)
    layer[length -1] = 1

def boxIt(pixel: list, img):
    for i in range(10):
        img[pixel[0] - 5, pixel[1] -5 + i] = 255
        img[pixel[0] + 5, pixel[1] -5 + i] = 255
        img[pixel[0] - 5 + i, pixel[1] -5] = 255
        img[pixel[0] - 5 + i, pixel[1] +5] = 255

def sigmoid(a):
    return(1/(1+math.exp(-a)))

def ReLu(a):
    return(max(0, a))

def getLumninance(pixel: list, chosen_image):
    #returns a value between 0 and 1 of the percent luminance of the pixel
    (r, g, b) = Image.open(chosen_image).load()[pixel[0], pixel[1]]
    averageRGB = (r + g + b)/3
    return(averageRGB/255)



# read in all of the weights

# Initialize all of the layers

imageWidth = 640
imageHeight = 480
inputWidth = 48
inputHeight = 32
imageLayerSize = inputWidth * inputHeight + 1
hiddenLayerSize = 10 + 1

scalingX = imageWidth / inputWidth
scalingY = imageHeight / inputHeight

    # these are all node layers
inputLayer = numpy.zeros(imageLayerSize)
hiddenLayerOne = numpy.zeros(hiddenLayerSize)
hiddenLayerTwo = numpy.zeros(hiddenLayerSize)
hiddenLayerThree = numpy.zeros(hiddenLayerSize)
# we don't subtract 1 from the size of the output layer because our loops and functions all act as though it has a bias neuron, 
# and so we'll end up ignoring a pixel output if we subtract one.
outputLayer = numpy.zeros(imageLayerSize)

biasDatThang(inputLayer)
biasDatThang(hiddenLayerOne)
biasDatThang(hiddenLayerTwo)
biasDatThang(hiddenLayerThree)
biasDatThang(outputLayer)

weightsInputOne = numpy.zeros((imageLayerSize, hiddenLayerSize))
weightsOneTwo = numpy.zeros((hiddenLayerSize, hiddenLayerSize))
weightsTwoThree = numpy.zeros((hiddenLayerSize, hiddenLayerSize))
weightsThreeOutput = numpy.zeros((hiddenLayerSize, imageLayerSize))

layers = [inputLayer, hiddenLayerOne, hiddenLayerTwo, hiddenLayerThree, outputLayer]
weightSets = [weightsInputOne, weightsOneTwo, weightsTwoThree, weightsThreeOutput]

saveFile = open(r'facial detection\networkSaveFile.txt', 'r')
weightsText = saveFile.readlines()
saveFile.close()
numpy.set_printoptions(threshold=sys.maxsize)


currentWeight = -1
currentNode = -1
currentWeightSet = -1
decimal = 1
for j in range(len(weightsText)):
    listNumbers = [list(char) for char in weightsText[j]]
    for i in range(len(listNumbers)):
        if listNumbers[i][0] == 'y':
            currentWeightSet = currentWeightSet + 1
            currentNode = -2
            currentWeight = -1
        if listNumbers[i][0] == '[':
            currentNode = currentNode + 1
            currentWeight = 0
        if listNumbers[i][0] == ',':
            currentWeight = currentWeight + 1
            decimal = 1

        if listNumbers[i][0] == '0':
            decimal = decimal / 10
        if listNumbers[i][0] == '1':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (1 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '2':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (2 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '3':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (3 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '4':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (4 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '5':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (5 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '6':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (6 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '7':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (7 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '8':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (8 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '9':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] + (9 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '-' and listNumbers[i -1][0] == 'e':
            orderOfMag = 10**(10*int(listNumbers[i +1][0]) + int(listNumbers[i +2][0]))
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] / orderOfMag
        if listNumbers[i][0] == '+' and listNumbers[i -1][0] == 'e':
            orderOfMag = 10**(10*int(listNumbers[i +1][0]) + int(listNumbers[i +2][0]))
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] * orderOfMag
        if listNumbers[i-10][0] == '-' and listNumbers[i-8][0] == '.':
            weightSets[currentWeightSet][currentNode][currentWeight] = weightSets[currentWeightSet][currentNode][currentWeight] * (-1)
        

start_time = time.perf_counter()
# feed it an image
N_images = 4
filename = r'C:\Users\ThinkPad\OneDrive - Oregon State University\Documents\python\facial detection\datasetClean\webcam_image' + str(numpy.random.randint(0, N_images)) + '.jpg'
luminances = numpy.zeros(imageLayerSize)
for j in range(imageLayerSize - 1):
    pixX = (j%inputWidth) * scalingX - scalingX/2
    pixY = math.floor(j / inputWidth) * scalingY - scalingY/2

    luminances[j] = getLumninance([pixX, pixY], filename)

end_time = time.perf_counter()

run_time = end_time - start_time
print(f"The program ran in {run_time:.4f} seconds")



# have it calculate where it thinks the eyes are

for j in range(imageLayerSize - 1):
    inputLayer[j] = luminances[j]


for prevLayer in range(len(layers)-1):
    currentWeightSet = weightSets[prevLayer]
    #we subtract one so that we never overwrite the bias neuron
    for currentNode in range(len(layers[prevLayer+1]) -1):
        currentSum = 0
        for prevNode in range(len(layers[prevLayer])):
            currentSum = currentSum + layers[prevLayer][prevNode] * currentWeightSet[prevNode][currentNode]
        #assigning the node the appropriate value
        layers[prevLayer+1][currentNode] = ReLu(currentSum)



# box those locations
eyeOneIndex = 0
eyeTwoIndex = 0

# we subtract 1 so that the vestigial bias neuron isn't mistaken for an eye. It wouldn't even fit on the image
for j in range(len(outputLayer) -1):
    if outputLayer[j] > outputLayer[eyeOneIndex]:
        eyeOneIndex = j
    elif outputLayer[j] > outputLayer[eyeTwoIndex]:
        eyeTwoIndex = j
    
eyes = [eyeOneIndex, eyeTwoIndex]

tester = Image.open(filename)
img = tester.load()

for j in eyes:
    pixX = (j%inputWidth) * scalingX - scalingX/2
    pixY = math.floor(j / inputWidth) * scalingY - scalingY/2
    boxIt([pixX, pixY], img)


#display the image
tester.save('newTestImage.jpg')
trainingTester = pygame.image.load('newTestImage.jpg')
(width, height) = tester.size
screen = pygame.display.set_mode((width + 20, height + 20))


stop = False
while stop != True:
    screen.blit(trainingTester, (10, 10))
    #updates the frame
    pygame.display.update()

    if pygame.event.wait().type == pygame.MOUSEBUTTONDOWN:
        stop = True
