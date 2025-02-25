import math
from PIL import Image
import numpy
import time
import sys


start_time = time.perf_counter()

class TrainingFrame():
    def __init__(self, eyeOne: list, eyeTwo: list, mouth: list, filename: str, luminances: list):
        self.eyeOne = eyeOne
        self.eyeTwo = eyeTwo
        self.mouth = mouth
        self.filename = filename
        self.luminances = luminances


def getLumninance(pixel: list, chosen_image):
    #returns a value between 0 and 1 of the percent luminance of the pixel
    (r, g, b) = Image.open(chosen_image).load()[pixel[0], pixel[1]]
    averageRGB = (r + g + b)/3
    return(averageRGB/255)

def randomizeWeights(weights: list):
    ballparkWeightMagnitude = 1
    length = len(weights)
    width = len(weights[0])
    for i in range(length):
        for j in range(width):
            weights[i][j] = numpy.random.random() * (2*ballparkWeightMagnitude) - (ballparkWeightMagnitude)

def biasDatThang(layer):
    length = len(layer)
    layer[length -1] = 1



def eyeScore(eyeOne: list, eyeTwo: list, pixel: list, generation: int):
    #gives any pixel a score based on how close the nearest eye. A pixel near an eye will return close to one, 
    # and a pixel far from an eye will return near zero.

    # we use a reversed(and slightly augmented) sigmoid to produce the score as a function of distance
    # by basing the distance dropoff off of the generation, we encourage the network to get closer over time, by us being 
    # more specific about where the eyes are
    m = (sigmoid((generation/1200)-4) * (generation**(1/6)) *600 + 200)/5000
    #distances
    eyeOneDistance = math.sqrt((eyeOne[0] - pixel[0])*(eyeOne[0] - pixel[0]) + (eyeOne[1] - pixel[1])*(eyeOne[1] - pixel[1]))
    eyeTwoDistance = math.sqrt((eyeTwo[0] - pixel[0])*(eyeTwo[0] - pixel[0]) + (eyeTwo[1] - pixel[1])*(eyeTwo[1] - pixel[1]))

    if eyeOneDistance <= eyeTwoDistance:
        #note that e is not raised to a negative power, this is an altered sigmoid function
        
        return (1/(1+math.exp(m*eyeOneDistance - 4)))
    else:
        return (1/(1+math.exp(m*eyeTwoDistance - 4)))

def dSigmoid(a):
    return(math.exp(-a)/((1+math.exp(-a))*(1+math.exp(-a))))

def sigmoid(a):
    try:
        return(1/(1+math.exp(-a)))
    except:
        # occasionally we hit a math overflow error, I'm not sure if a gets to negative or too positive
        print(str(a) + ' is too much.')

# i think that using relu may eliminate the math overflow error we were hitting
def ReLu(a):
    return(max(0, a))

def dReLu(a):
    #if a is negative, then 0 will be max. 0 / a = 0. If a is positive, then a will be max, a/a = 1
    return(ReLu(a)/(a + 0.000000000000000000001))

def learningRate(x):
    return 1/(100*(x**(1/3)))






#vars
saveFile = open(r'facial detection\training_data_save_file.txt', 'r')
firstLine = saveFile.readlines()
saveFile.close()
numpy.set_printoptions(threshold=sys.maxsize)


N_images = 18

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
# we don't subtract 1 from the size of the output layer because our loops and functions all act as though it has a bias neuron, 
# and so we'll end up ignoring a pixel output if we subtract one.
outputLayer = numpy.zeros(imageLayerSize)

biasDatThang(inputLayer)
biasDatThang(hiddenLayerOne)
biasDatThang(hiddenLayerTwo)
biasDatThang(hiddenLayerThree)
biasDatThang(outputLayer)



    #these are all weight layers
    # note that increases in the amount of hidden layers are much less impactful for calculation time than increases in layer size
    # which is why I have chosen to increase hidden layers instead of nodes per hidden layer.
weightsInputOne = numpy.zeros((imageLayerSize, hiddenLayerSize))
weightsOneTwo = numpy.zeros((hiddenLayerSize, hiddenLayerSize))
weightsTwoThree = numpy.zeros((hiddenLayerSize, hiddenLayerSize))
weightsThreeOutput = numpy.zeros((hiddenLayerSize, imageLayerSize))

adjustWeightsOne = numpy.zeros(imageLayerSize)
adjustWeightsTwo = numpy.zeros(hiddenLayerSize)
adjustWeightsThree = numpy.zeros(hiddenLayerSize)
adjustWeightsOutput = numpy.zeros(hiddenLayerSize)
cost = numpy.zeros(imageLayerSize)

saveFile = open(r'facial detection\networkSaveFile.txt', 'r')
weightsText = saveFile.readlines()
saveFile.close()

layers = [inputLayer, hiddenLayerOne, hiddenLayerTwo, hiddenLayerThree, outputLayer]
weightSets = [weightsInputOne, weightsOneTwo, weightsTwoThree, weightsThreeOutput]
adjustWeightSets = [adjustWeightsOne, adjustWeightsTwo, adjustWeightsThree, adjustWeightsOutput, cost]


startFromScratch = True
if startFromScratch:
    randomizeWeights(weightsInputOne)
    randomizeWeights(weightsOneTwo)
    randomizeWeights(weightsTwoThree)
    randomizeWeights(weightsThreeOutput)
else:
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


recordedLuminances = numpy.zeros((N_images, imageLayerSize - 1))

saveFile = open(r'facial detection\luminanceFile.txt', 'r')
luminancesText = saveFile.readlines()
saveFile.close()

currentPixelIndex = -1
currentLuminanceImage = 0
decimal = 0.1
for j in range(len(luminancesText)):
    listNumbers = [list(char) for char in luminancesText[j]]
    for i in range(len(listNumbers)):
        if listNumbers[i][0] == ']':
            currentLuminanceImage = currentLuminanceImage + 1
            currentPixelIndex = -1
            
        if listNumbers[i][0] == '.':
            currentPixelIndex = currentPixelIndex + 1
            decimal = 0.1

        if listNumbers[i][0] == '0':
            decimal = decimal / 10
        if listNumbers[i][0] == '1':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (1 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '2':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (2 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '3':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (3 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '4':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (4 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '5':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (5 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '6':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (6 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '7':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (7 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '8':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (8 * decimal)
            decimal = decimal / 10
        if listNumbers[i][0] == '9':
            recordedLuminances[currentLuminanceImage][currentPixelIndex] = recordedLuminances[currentLuminanceImage][currentPixelIndex] + (9 * decimal)
            decimal = decimal / 10
       




##create a list of N_images non-repeating random integers between 1 and 1200

for i in range(18):
    repeats = True
    while repeats:
        trainingIndexOrder[i] = numpy.random.randint(18)
        repeats = False
        for j in range(i):
            if trainingIndexOrder[j] == trainingIndexOrder[i] and i != j:
                repeats = True

#this arranges the list of ints into their respective coordinate pairs, and assigns them to the various training data images
for i in range(N_images):
    filename = r'C:\Users\ThinkPad\OneDrive - Oregon State University\Documents\python\facial detection\datasetClean\webcam_image' + str(i) + '.jpg'
    eyeOne = [recon_numbers[i*6], recon_numbers[i*6 + 1]]
    eyeTwo = [recon_numbers[i*6+2], recon_numbers[i*6+3]]
    mouth = [recon_numbers[i*6+4], recon_numbers[i*6+5]]
    luminances = recordedLuminances[i]
    for j in range(imageLayerSize - 1):
        pixX = (j%inputWidth) * scalingX - scalingX/2
        pixY = math.floor(j / inputWidth) * scalingY - scalingY/2

    
    trainingData[i] = TrainingFrame(eyeOne, eyeTwo, mouth, filename, luminances)


#start_time = time.perf_counter()

#advise using a number around 5000 for loops
f = 0
while f < 5000:
    i = trainingIndexOrder[f%N_images]
    saveFile = open(r'facial detection\networkSaveFile.txt', 'w')
    saveFile.close()
    saveFile = open(r'facial detection\networkSaveFile.txt', 'a')
    currentImage = trainingData[i].filename
    currentEyeOne = trainingData[i].eyeOne
    currentEyeTwo = trainingData[i].eyeTwo
    currentMouth = trainingData[i].mouth
    currentLuminances = trainingData[i].luminances
    currentLearningRate = learningRate(f + 1)

    for j in range(imageLayerSize - 1):
        inputLayer[j] = currentLuminances[j]


    for prevLayer in range(len(layers)-1):
        currentWeightSet = weightSets[prevLayer]
        #we subtract one so that we never overwrite the bias neuron
        for currentNode in range(len(layers[prevLayer+1]) -1):
            currentSum = 0
            for prevNode in range(len(layers[prevLayer])):
                currentSum = currentSum + layers[prevLayer][prevNode] * currentWeightSet[prevNode][currentNode]
            #assigning the node the appropriate value
            layers[prevLayer+1][currentNode] = ReLu(currentSum)
            

#Finding loss
    for j in range(imageLayerSize - 1):
        pixX = (j%inputWidth) * scalingX - scalingX/2
        pixY = math.floor(j / inputWidth) * scalingY - scalingY/2
        loss = (eyeScore(currentEyeOne, currentEyeTwo, [pixX, pixY], i) - outputLayer[j]) ** 2
        adjustWeightSets[len(adjustWeightSets) -1][j] = 2 * (outputLayer[j] - eyeScore(currentEyeOne, currentEyeTwo, [pixX, pixY], i)) * currentLearningRate




#backpropogating
    
# for each node, we want to adjust each of the weights that feed into it accordingly, starting with the last layer of nodes

#we should save the partials from each layer that we do so that we don't have to recalculate them every layer


# for each layer, starting from the end
    for j in range(len(layers)-1):
        
        currentLayer = layers[len(layers) -j -2] # this starts on the last hidden layer and ends on the input layer
        prevLayer = layers[len(layers) - j -1] #this starts at the output layer and ends on the first hidden layer
        currentAdjustWeightSet = adjustWeightSets[len(layers) - j -1]
        
        for currentNodeIndex in range(len(currentLayer)):
            currentNode = currentLayer[currentNodeIndex]

            
            for prevNodeIndex in range(len(prevLayer) -1):
                # we skip the last neuron, because the weights that go into the bias neuron don't matter, it will always be 1
                prevNode = prevLayer[prevNodeIndex]
                currentWeight = weightSets[len(layers) -j -2][currentNodeIndex][prevNodeIndex] # the weight that goes from the current 
                # node in the current layer to the prev node from the prev layer
                dCda = adjustWeightSets[len(layers) - j -1][prevNodeIndex]
                dadw = dReLu(prevNode)*currentNode
                dadb = dReLu(prevNode) * currentWeight
                adjustWeightSets[len(layers) - j -2][currentNodeIndex] = adjustWeightSets[len(layers) - j -2][currentNodeIndex] + dCda
                
                weightSets[len(layers) -j -2][currentNodeIndex][prevNodeIndex] = currentWeight - (adjustWeightSets[len(layers) - j -2][currentNodeIndex] * dadw)
                adjustWeightSets[len(layers) - j -2][currentNodeIndex] = adjustWeightSets[len(layers) - j -2][currentNodeIndex] * dadb


                #apply the relevant backpropogation
                backProp = 0



    #reset all the adjustments
    adjustWeightsOne = numpy.zeros(imageLayerSize)
    adjustWeightsTwo = numpy.zeros(hiddenLayerSize)
    adjustWeightsThree = numpy.zeros(hiddenLayerSize)
    adjustWeightsOutput = numpy.zeros(hiddenLayerSize)
    cost = numpy.zeros(imageLayerSize)

    saveFile.write(str(weightSets))
    saveFile.close()

    f = f + 1
    print(f)

end_time = time.perf_counter()

run_time = end_time - start_time
print(f"The program ran in {run_time:.4f} seconds")
