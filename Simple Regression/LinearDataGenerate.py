import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def GetGeneID(WeightMatrixDF, Gene):

    GeneIndex = WeightMatrixDF.index.tolist().index(Gene)
    return GeneIndex
def ReLU(x):
    return np.maximum(0, x)

def Cycle (WeightMatrix, InputArray, Biases):
    #output as dot product of input array and weight matrix plus biases
    OutputArray = np.dot(InputArray, WeightMatrix) + Biases
    #apply ReLU
    OutputArray = ReLU(OutputArray)

    return OutputArray

def CreateBiasArray(Cols, BiasDF):
    BiasArray = [0]*len(Cols)

    # print(BiasDF)
    BiasDict = BiasDF.to_dict()
    BiasDict = BiasDict['0']
    # print(BiasDict)

    i=0
    for col in Cols:
        if col in BiasDF.index:
            BiasArray[i] = BiasDict[col]
        i+=1

    return BiasArray, BiasDict

def CompleteCycles(WeightMatrix, InputArray, BiasArray, Cycles):
    # print('InputArray', InputArray)
    Memory = [0] * len(InputArray)
    for Ite in range(Cycles):
        # print('Cycle', Ite,)
        OutputArray = Cycle(WeightMatrix, InputArray, BiasArray)
        Memory += OutputArray
        # create positive probability distribution of decay
        Decay = abs(np.random.normal(0.5, 0.007, len(OutputArray)))
        # print(Decay)
        # Decay = 1-Decay

        InputArray = Memory * Decay
        # print(np.round(InputArray, 2))
        Memory = Memory - InputArray
        for i in range(len(InputArray)):
            if InputArray[i] < 0:
                InputArray[i] = 0
    return OutputArray

def GetInputGenes(seed):
    InputList = pd.read_csv('OutputGEs/Seed'+str(seed)+'/InputGenes.csv', index_col=0, header=0)

    InputGenes = InputList['InputGenes'].tolist()
    return InputGenes

def GenerateInputCocktail(FullGeneList, InGeneList,  MinCons, MaxCons):


    PixelArray = []
    Cocktail = []

    for gene in FullGeneList:
        if gene in InGeneList:
            Cocktail.append(MaxCons)
        else:
            Cocktail.append(MinCons)

    return Cocktail


# Read in the data
WeightMatrixDF = pd.read_csv('../../../../Data/WeightMatrix.csv', header=0, index_col=0)

BiasesDF = pd.read_csv('../../../../Data/NormaizedBias.csv', index_col=0)

FullInputGenes = WeightMatrixDF.index.tolist()

WeightMatrix = WeightMatrixDF.to_numpy()

BiasArray, BiasDict = CreateBiasArray(WeightMatrixDF.columns.tolist(), BiasesDF)


InputGeneList = ['b3067']

FinalDF = pd.DataFrame(index=FullInputGenes)

for Cons in np.arange(0, 5.1, 0.2):
    for ite in range(10):
        print('Cons', Cons, 'ite', ite)
        InputCocktail = GenerateInputCocktail(FullInputGenes, InputGeneList,  2, Cons)
        # print(FullInputGenes)
        # print(InputCocktail)

        Output = CompleteCycles(WeightMatrix, InputCocktail, BiasArray, 10)

        FinalDF['Cons'+str(round(Cons,1))+'Ite'+str(ite)] = Output

print(FinalDF.head())
FinalDF.to_csv('Results/LinReg.csv')
