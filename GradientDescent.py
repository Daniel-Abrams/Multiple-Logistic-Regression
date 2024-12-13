import pandas as pd
from data import *
import math

rows = 10
columns = 11
step_size = 0.0001
max_steps = 400
epsilon = 0.000001
def performGradientDescent():
    done = False
    i = 0

    while done == False:
        done = True
        gradientList = []
        for j in range (columns) :
            gradient = []
            for k in range (rows):
                gradient.append(calculatePartialDerivative(TrainingData[k],socialMediaTrainingData[j],j))
            if computeMagnitude(gradient) * step_size > epsilon:
                gradientList.append(gradient)
                done = False
            else:
                
                gradientList.append([0]*rows)
    
        for j in range (columns):
            for k in range (rows):
                weights[k][j] =  weights[k][j] - gradientList[j][k] * step_size
        
        i += 1

    


 # function for computing the dot product of the weights and parameters
def weightedSum(index,platformIndex):
    return ( weights[0][platformIndex] * genderTraining[index] +
            weights[1][platformIndex] * ageTraining[index] +
            weights[2][platformIndex] * relationshipTraining[index] +
            weights[3][platformIndex] * parentTraining[index] +
            weights[4][platformIndex] * educationTraining[index] +
            weights[5][platformIndex] * employmentTraining[index] +
            weights[6][platformIndex] * raceTraining[index] +
            weights[7][platformIndex] * incomeTraining[index] +
            weights[8][platformIndex] * politicalTraining[index] +
            weights[9][platformIndex])
 # same function, but draws from the test value arrays
def weightedSumTest(index,platformIndex):
    return ( weights[0][platformIndex] * genderTest[index] +
            weights[1][platformIndex] * ageTest[index] +
            weights[2][platformIndex] * relationshipTest[index] +
            weights[3][platformIndex] * parentTest[index] +
            weights[4][platformIndex] * educationTest[index] +
            weights[5][platformIndex] * employmentTest[index] +
            weights[6][platformIndex] * raceTest[index] +
            weights[7][platformIndex] * incomeTest[index] +
            weights[8][platformIndex] * politicalTest[index] +
            weights[9][platformIndex])



def linkFunction(t):
    return 1/(1 + math.e**(-t))

def calculatePartialDerivative(feature_value,platform,platformIndex):
    derivative_sum = 0

    # Take the partial derivative of the sum of the errors
    for i in range(0,1349):
        weighted_sum = weightedSum(i, platformIndex)
        sigmoid_output = linkFunction(weighted_sum)  # Sigmoid function
        sigmoid_derivative = sigmoid_output * (1 - sigmoid_output)  # Derivative of sigmoid
        error = 2 * (sigmoid_output - platform[i])  # Error term from loss function
        
        derivative_sum += error * sigmoid_derivative * feature_value[i]

    derivative_sum = derivative_sum / 1350.0
    return derivative_sum

def testModel():
    matches_array = []
    model_output_array = []
    numCorrect = 0
    
    for j in range(0,columns):
        model_output = []
        matches = []
        for i in range(0,150):
            output = round(linkFunction(weightedSumTest(i,j)))
            model_output.append(output)
            if output == socialMediaTestData[j][i]:
                matches.append(1)
                numCorrect += 1
            else:
                matches.append(0)

            matches_array.append(matches)
        model_output_array.append(model_output)

    print("Matches: ")
   # print(matches_array)
   
    print("outputs:")
    print(model_output_array)
    print("Percentage: ")
    print(numCorrect/1650.0)


def computeMagnitude(gradient):
    sum = 0

    for element in gradient:
        sum = sum + element**2

    return sum**2
    

def calculateError():
  sum = 0
  for i in range(0,1349):
        sum = sum + (linkFunction( weightedSum(i,0)) - socialMediaTrainingData[0][i])**2
       

normalizeData()
performGradientDescent()
testModel()
