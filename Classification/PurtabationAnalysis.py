import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import Digits

Seed = 76
Seed = 29
Seed = 29
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

def GenerateInputCocktail(FullGeneList, InputeGenes, PertGene, NewCons, BaseCons):
    Cocktail = []
    for gene in FullGeneList:
        if gene == PertGene:
            # print(gene, InputGene)
            Cocktail.append(NewCons)
        elif gene in InputeGenes:
            Cocktail.append(BaseCons)
        else:
            Cocktail.append(0)

    return Cocktail


# Read in the data
WeightMatrixDF = pd.read_csv('../../Data/WeightMatrix.csv', header=0, index_col=0)

BiasesDF = pd.read_csv('../../Data/NormaizedBias.csv', index_col=0)

FullInputGenes = WeightMatrixDF.index.tolist()

WeightMatrix = WeightMatrixDF.to_numpy()

BiasArray, BiasDict = CreateBiasArray(WeightMatrixDF.columns.tolist(), BiasesDF)

FinalIndices = OutputGenes = pd.read_csv('../GRNNGEDataAnalsis/GEAnalysisResults/Maps/'+str(Seed)+'OutputGenes.csv', index_col=0, header=0)['OutputGenes'].tolist()


# FinalDF = pd.DataFrame(index=FinalIndices)
# for Digit in range(5):
#     for Inst in range(10):
#         for ite in range(12):
#             print('Digit', Digit, 'Inst', Inst, 'ite', ite, int(ite/2))
#             InputCocktail = GenerateInputCocktail(FullInputGenes, GetInputGenes(Seed), Digits.Digits[Digit][Inst], 0, int(ite/2))
#             # print(GetInputGenes(Seed))
#             # print(InputCocktail)
#             # exit()
#             Output = CompleteCycles(WeightMatrix, InputCocktail, BiasArray, 10)
#             # print(Output)
#             OutputArray = [Output[GetGeneID(WeightMatrixDF, FinalIndices[0])], Output[GetGeneID(WeightMatrixDF, FinalIndices[1])], Output[GetGeneID(WeightMatrixDF, FinalIndices[2])], Output[GetGeneID(WeightMatrixDF, FinalIndices[3])], Output[GetGeneID(WeightMatrixDF, FinalIndices[4])]]
#             FinalDF = pd.concat([FinalDF, pd.DataFrame(OutputArray, index=FinalIndices, columns=[str(Digit)+str(Inst)+str(ite)])], axis=1)
#             # print(Output[GetGeneID(WeightMatrixDF, 'b0124')], Output[GetGeneID(WeightMatrixDF, 'b1806')], Output[GetGeneID(WeightMatrixDF, 'b2029')], Output[GetGeneID(WeightMatrixDF, 'b0976')], Output[GetGeneID(WeightMatrixDF, 'b4302')])
#             # print(GetInputGenes(0))
# FinalDF.to_csv('PerturbationAnalysis/Seed'+str(Seed)+'Output.csv')

FinalDF = pd.DataFrame(index=FinalIndices)

print(len(GetInputGenes(Seed)))
for InputGene in GetInputGenes(Seed):
    for cons in range(10):
        for ite in range(20):

            print('InputGene', InputGene, 'Cons', cons, 'ite', ite)
            InputCocktail = GenerateInputCocktail(FullInputGenes, GetInputGenes(Seed), InputGene, cons, 5)
            Output = CompleteCycles(WeightMatrix, InputCocktail, BiasArray, 10)

            OutputArray = [Output[GetGeneID(WeightMatrixDF, FinalIndices[0])], Output[GetGeneID(WeightMatrixDF, FinalIndices[1])], Output[GetGeneID(WeightMatrixDF, FinalIndices[2])],
                           Output[GetGeneID(WeightMatrixDF, FinalIndices[3])], Output[GetGeneID(WeightMatrixDF, FinalIndices[4])]]
            print(OutputArray)
            FinalDF = pd.concat([FinalDF, pd.DataFrame(OutputArray, index=FinalIndices, columns=[InputGene+'_'+str(cons)+'_'+str(ite)])], axis=1)

FinalDF.to_csv('PerturbationAnalysis/Seed'+str(Seed)+'Output.csv')
            # print(InputGene
            # print(InputCocktail)
            # exit()

