import pygame
import cv2
import math
from PIL import Image
import numpy
import time
import sys


class TrainingFrame():
    def __init__(self, eyeOne: list, eyeTwo: list, mouth: list, filename: str):
        self.eyeOne = eyeOne
        self.eyeTwo = eyeTwo
        self.mouth = mouth
        self.filename = filename


def getLumninance(pixel: list, chosen_image):
    #returns a value between 0 and 1 of the percent luminance of the pixel
    (r, g, b) = Image.open(chosen_image).load()[pixel[0], pixel[1]]
    averageRGB = (r + g + b)/3
    return(averageRGB/255)

def randomizeWeights(weights: list):
    ballparkWeightMagnitude = 10
    length = len(weights)
    width = len(weights[0])
    for i in range(length):
        for j in range(width):
            weights[i][j] = numpy.random.random() * (2*ballparkWeightMagnitude) - (ballparkWeightMagnitude)

def biasDatThang(layer):
    length = len(layer)
    layer[length -1] = 1



def eyeScore(eyeOne: list, eyeTwo: list, pixel: list):
    #gives any pixel a score based on how close the nearest eye. A pixel near an eye will return close to one, 
    # and a pixel far from an eye will return near zero.

    # we use a reversed(and slightly augmented) sigmoid to produce the score as a function of distance

    desiredSpan = 200
    m = 1/(math.floor(desiredSpan/8))
    #distances
    eyeOneDistance = math.sqrt((eyeOne[0] - pixel[0])*(eyeOne[0] - pixel[0]) + (eyeOne[1] - pixel[1])*(eyeOne[1] - pixel[1]))
    eyeTwoDistance = math.sqrt((eyeTwo[0] - pixel[0])*(eyeTwo[0] - pixel[0]) + (eyeTwo[1] - pixel[1])*(eyeTwo[1] - pixel[1]))

    if eyeOneDistance <= eyeTwoDistance:
        #note that e is not raised to a negative power
        return (1/(1+math.exp(m*eyeOneDistance - 4)))
    else:
        return (1/(1+math.exp(m*eyeTwoDistance - 4)))

def dSigmoid(a):
    return(math.exp(-a)/((1+math.exp(-a))*(1+math.exp(-a))))

def sigmoid(a):
    return(1/(1+math.exp(-a)))




#vars
saveFile = open(r'facial detection\training_data_save_file.txt', 'r')
firstLine = saveFile.readlines()
saveFile.close()
numpy.set_printoptions(threshold=sys.maxsize)


N_images = 4

imageWidth = 640
imageHeight = 480

#reconstituted numbers
recon_numbers = numpy.zeros(N_images* 6)
trainingData = numpy.zeros(20, TrainingFrame)
trainingIndexOrder = numpy.zeros(1000, int)
trainingOrder = numpy.zeros(1000, list)



#this turns the data in the text file into an intelligible list of ints
x = 0
for j in range(N_images):
    list_numbers = [list(char) for char in firstLine[j]]
    for i in list_numbers:
        if i[0] == '0':
            recon_numbers[x] = recon_numbers[x] * 10
        if i[0] == '1':
            recon_numbers[x] = recon_numbers[x] * 10 + 1
        if i[0] == '2':
            recon_numbers[x] = recon_numbers[x] * 10 + 2
        if i[0] == '3':
            recon_numbers[x] = recon_numbers[x] * 10 + 3
        if i[0] == '4':
            recon_numbers[x] = recon_numbers[x] * 10 + 4
        if i[0] == '5':
            recon_numbers[x] = recon_numbers[x] * 10 + 5
        if i[0] == '6':
            recon_numbers[x] = recon_numbers[x] * 10 + 6
        if i[0] == '7':
            recon_numbers[x] = recon_numbers[x] * 10 + 7
        if i[0] == '8':
            recon_numbers[x] = recon_numbers[x] * 10 + 8
        if i[0] == '9':
            recon_numbers[x] = recon_numbers[x] * 10 + 9
        if i[0] == ',' or i[0] == ')':
            x = x+1


#this arranges the list of ints into their respective coordinate pairs, and assigns them to the various training data images
for i in range(N_images):
    filename = r'C:\Users\ThinkPad\OneDrive - Oregon State University\Documents\python\facial detection\datasetClean\webcam_image' + str(i) + '.jpg'
    eyeOne = [recon_numbers[i*6], recon_numbers[i*6 + 1]]
    eyeTwo = [recon_numbers[i*6+2], recon_numbers[i*6+3]]
    mouth = [recon_numbers[i*6+4], recon_numbers[i*6+5]]
    
    trainingData[i] = TrainingFrame(eyeOne, eyeTwo, mouth, filename)



## initialize the neural network
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
outputLayer = numpy.zeros(imageLayerSize -1)

biasDatThang(inputLayer)
biasDatThang(hiddenLayerOne)
biasDatThang(hiddenLayerTwo)
biasDatThang(hiddenLayerThree)



    #these are all weight layers
    # note that increases in the amount of hidden layers are much less impactful for calculation time than increases in layer size
    # which is why I have chosen to increase hidden layers instead of nodes per hidden layer.
weightsInputOne = numpy.zeros((imageLayerSize, hiddenLayerSize))
weightsOneTwo = numpy.zeros((hiddenLayerSize, hiddenLayerSize))
weightsTwoThree = numpy.zeros((hiddenLayerSize, hiddenLayerSize))
weightsThreeOutput = numpy.zeros((hiddenLayerSize, imageLayerSize -1))

randomizeWeights(weightsInputOne)
randomizeWeights(weightsOneTwo)
randomizeWeights(weightsTwoThree)
randomizeWeights(weightsThreeOutput)

layers = [inputLayer, hiddenLayerOne, hiddenLayerTwo, hiddenLayerThree, outputLayer]
weightSets = [weightsInputOne, weightsOneTwo, weightsTwoThree, weightsThreeOutput]


##create a list of 1000 non-repeating random integers between 1 and 1200

for i in range(1000):
    repeats = True
    while repeats:
        trainingIndexOrder[i] = numpy.random.randint(1200)
        repeats = False
        for j in range(i):
            if trainingIndexOrder[j] == trainingIndexOrder[i] and i != j:
                repeats = True


start_time = time.perf_counter()

for i in range(N_images):
    saveFile = open(r'facial detection\networkSaveFile.txt', 'w')
    saveFile.close()
    saveFile = open(r'facial detection\networkSaveFile.txt', 'a')
    currentImage = trainingData[i].filename
    currentEyeOne = trainingData[i].eyeOne
    currentEyeTwo = trainingData[i].eyeTwo
    currentMouth = trainingData[i].mouth

    for j in range(imageLayerSize - 1):
        pixX = (j%inputWidth) * scalingX - scalingX/2
        pixY = math.floor(j / inputWidth) * scalingY - scalingY/2
        inputLayer[j] = getLumninance([pixX, pixY], currentImage)


    for prevLayer in range(len(layers)-1):
        currentWeightSet = weightSets[prevLayer]
        #we subtract one so that we never overwrite the bias neuron
        for currentNode in range(len(layers[prevLayer+1]) -1):
            currentSum = 0
            for prevNode in range(len(layers[prevLayer])):
                currentSum = currentSum + layers[prevLayer][prevNode] * currentWeightSet[prevNode]
            


    for j in range(imageLayerSize - 1):
        pixX = (j%inputWidth) * scalingX - scalingX/2
        pixY = math.floor(j / inputWidth) * scalingY - scalingY/2
        loss = eyeScore(currentEyeOne, currentEyeTwo, [pixX, pixY]) - outputLayer[j]

        #print(getLumninance([pixX, pixY], currentImage))
    saveFile.write(str(weightSets))
    saveFile.close()
end_time = time.perf_counter()

run_time = end_time - start_time
print(f"The program ran in {run_time:.4f} seconds")
print(inputLayer)

