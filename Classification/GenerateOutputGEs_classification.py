import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Digits
from multiprocessing import Pool


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
        Decay = abs(np.random.normal(0.5, 0.001, len(OutputArray)))
        # print(Decay)

        # Decay = 1-Decay

        InputArray = Memory * Decay
        Memory = Memory - InputArray
        for i in range(len(InputArray)):
            if InputArray[i] < 0:
                InputArray[i] = 0
    return OutputArray

def GetInputGenes():
    FullGRNN = pd.read_csv('../../Adrian/EcoliNetworkV3.csv')

    EffectorDF = FullGRNN[FullGRNN['SourceType'] == 'effector']
    EffectorGeneList = EffectorDF['Target'].unique()

    WeightMatrixDF = pd.read_csv('../../Data/WeightMatrix.csv', header=0, index_col=0)
    Inputs = WeightMatrixDF.index.tolist()

    #common genes between EffectorGeneList and Inputs
    CommonGenes = list(set(EffectorGeneList) & set(HighestOccuringList))

    # np.random.seed(seed)
    InputGenes = np.random.choice(CommonGenes, 16, replace=False)
    return InputGenes


def GenerateInputCocktail(FullGeneList, InGeneList, Image, MinCons):
    InterpolatedData = pd.read_csv('../../Data/InterpolateData.csv', header=0, index_col=0)
    # print(InterpolatedData\)

    # print(FullGeneList)
    # print(InGeneList)
    InGeneList = InGeneList.tolist()
    PixelArray = []
    Cocktail = []
    for row in range(4):
        # print(Image[row])
        PixelArray += Image[row].tolist()


    for gene in FullGeneList:
        # print(gene)
        # print(InterpolatedData.loc[0, gene])
        if gene in InGeneList:
            IngeneID = InGeneList.index(gene)
            if PixelArray[IngeneID] == 1:
                # print(gene, IngeneID, PixelArray[IngeneID], FullGeneList.index(gene))
                Cocktail.append(InterpolatedData[gene].max())
            else:
                Cocktail.append(InterpolatedData[gene].min())
        else:
            Cocktail.append(InterpolatedData.loc[0, gene])
    # print([[gene,[Cocktail[FullGeneList.index(gene)]]] for gene in InGeneList])
    return Cocktail


# Read in the data
WeightMatrixDF = pd.read_csv('../../Data/WeightMatrix.csv', header=0, index_col=0)

BiasesDF = pd.read_csv('../../Data/NormaizedBias.csv', index_col=0)

FullInputGenes = WeightMatrixDF.index.tolist()

WeightMatrix = WeightMatrixDF.to_numpy()

BiasArray, BiasDict = CreateBiasArray(WeightMatrixDF.columns.tolist(), BiasesDF)

def Parallelize(Seed):
    print('Seed', Seed)
    # if directory does not exist, create it
    if not os.path.exists('OutputGEs/Seed' + str(Seed)):
        os.makedirs('OutputGEs/Seed' + str(Seed))

    InputGeneList = GetInputGenes()
    InputGenesDF = pd.DataFrame(InputGeneList, columns=['InputGenes'])
    InputGenesDF.to_csv('OutputGEs/Seed' + str(Seed) + '/InputGenes.csv')

    for D in range(len(Digits.Digits)):
        # print('Digit', D)

        OutputDF = pd.DataFrame(index=WeightMatrixDF.index.tolist())
        for inst in range(len(Digits.Digits[D])):
            # print('Digit', D, 'Instance', inst)
            # print(Digits.Digits[D][inst])

            InputCocktail = GenerateInputCocktail(FullInputGenes, InputGeneList, Digits.Digits[D][inst], 0)

            for ite in range(5):
                Output = CompleteCycles(WeightMatrix, InputCocktail, BiasArray, 10)
                OutputDF['Digit' + str(D) + 'Instance' + str(inst) + 'Ite' + str(ite)] = Output

        OutputDF.to_csv('OutputGEs/Seed' + str(Seed) + '/Digit' + str(D) + 'Output.csv')


if __name__ == '__main__':
    SimArray = []
    with Pool(18) as p:
        for i in range(100,500):
            SimArray.append(p.apply_async(Parallelize, args=(i,)))
        p.close()
        p.join()


# for Seed in range(150, 700):
#     Parallelize(Seed)
#         # exit()
#         # OutputDF['Val'] = Output

