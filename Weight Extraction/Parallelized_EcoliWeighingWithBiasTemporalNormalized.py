import time
import scipy
import pandas as pd
import numpy as np
import csv
import os
from numba import jit, njit
import concurrent.futures
from multiprocessing import Pool
import shutil

#start clock
start = time.time()
Lrate = 0.00001
Epochs = 100000000

def unique(list1):
    unique_list = []

    for x in list1:
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

@jit(nopython=True)
# @njit(parallel=True)
def Train (Inputs, Outputs, epochs, Lrate, MinMSEDiff):
# def Train(Constraints, Inputs, Outputs, epochs, Lrate, MinMSEDiff):
    print('Train Started')
    # print(Constraints)

    #With Biases
    weights = np.random.random((Inputs.shape[1]+1, 1))
    # print('Initial Weights', weights, weights.shape)
    # print('Inputs', Inputs.shape)
    Inputs = np.concatenate((Inputs, np.ones((Inputs.shape[0], 1))), axis=1)


    OldMSE = 0
    for i in range(epochs):

        Toutputs = np.dot(Inputs, weights)

        error = (Outputs - Toutputs)

        MSE = np.sum([err ** 2 for err in error.T][0]) / len(Outputs)


        adj = error * Lrate
        weights += np.dot(Inputs.T, adj)
        # print('weightsP', weights)
        # for j in range(len(Constraints)):
        #     if Constraints[j] == 1:
        #         if weights[j] < 0:
        #             weights[j] = 0
        #     elif Constraints[j] == -1:
        #         if weights[j] > 0:
        #             weights[j] = 0
        # print('weightsA', weights)

        if i > 2:
            if (OldMSE-MSE) < MinMSEDiff or OldMSE<MSE:
                return weights, i, MSE
                break

        OldMSE = MSE
        # print("Epoch:{}, MSE:{}, Lrate{}".format(i, MSE, Lrate), end="\n")
    # print("")
    # print('Max epoch', i, 'MSE', MSE, 'Weights', weights)
    return weights, i, MSE

def Test (Inputs, Outputs, weights):
    # print(Inputs)
    Inputs = np.concatenate((Inputs, np.ones((Inputs.shape[0], 1))), axis=1)
    Toutputs = np.dot(Inputs, weights)
    error = (Outputs - Toutputs)
    MSE = np.sum([pow(err, 2) for err in error.T])/len(Outputs)

    return MSE, Toutputs

def PerceptGeneWeightExtraction (gene):
    TrainInputs = []
    TrainOutputs = []

    TestInputs = []
    TestOutputs = []

    Proceed = True

    # print(gene)
    motifDF = GRNDF[GRNDF['Target'] == gene]
    Types = list(motifDF['Type'])
    # Constrains = []
    # for Type in Types:
    #     if Type == '+':
    #         Constrains.append(1)
    #     elif Type == '-':
    #         Constrains.append(-1)
    # Constrains = np.array(Constrains)
    # print(motifDF)

    GenesList = list(motifDF['Source'])

    GenesList = GenesList + [gene]

    for Tempgene in GenesList:
        if Tempgene not in TrainSet.columns:
            print('Not in TrainSet', Tempgene)
            Proceed = False
            GenesList.remove(Tempgene)
            DroppedGenes.append(Tempgene)

    if Proceed:
        # print(GenesList)
        TraintmpDF = TrainSet[GenesList]
        # print('TraintmpDF', TraintmpDF)
        for row in TraintmpDF.iterrows():
            # print('test', np.array(row[1].values))
            if not np.isnan(np.array(row[1].values)).any():
                TrainInputs.append(np.array(row[1].values[:-1]))
                TrainOutputs.append(np.array([row[1].values[-1]]))

        TesttmpDF = TestSet[GenesList]
        for row in TesttmpDF.iterrows():
            if not np.isnan(np.array(row[1].values)).any():
                TestInputs.append(np.array(row[1].values[:-1]))
                TestOutputs.append(np.array([row[1].values[-1]]))


        del TrainOutputs[0]
        del TrainInputs[len(TrainInputs) - 1]

        del TestOutputs[0]
        del TestInputs[len(TestInputs) - 1]

        # print('TrainInputs', TrainInputs)
        # print('TrainOutputs', TrainOutputs)

        TrainInputs = np.array(TrainInputs)
        TrainOutputs = np.array(TrainOutputs)

        TestInputs = np.array(TestInputs)
        TestOutputs = np.array(TestOutputs)

        TempOutputVsPredictedDF = pd.DataFrame(columns=['Output', 'Predicted'], index=range(len(TestOutputs)))

        Weights, Epoch, FinalMSE = Train(TrainInputs, TrainOutputs, Epochs, Lrate, 0.00000000000001)
        Accuracy, Predicts = Test(TestInputs, TestOutputs, Weights)

        TempOutputVsPredictedDF['Output'] = TestOutputs
        TempOutputVsPredictedDF['Predicted'] = Predicts

        TempOutputVsPredictedDF.to_csv(Path + '/EachGeneComparison/' + gene + '.csv')
        #
        with open(Path + '/WeightedFilteredGRNV/' + gene + '.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Source', 'Target', 'Weight', 'Epoch', 'Accuracy', 'TrainInputs'])
            for Ngene in range(len(GenesList)):
                # WeightedFullGRN['']
                writer.writerow([GenesList[Ngene], GenesList[-1], Weights[Ngene][0], Epoch, Accuracy, len(TrainInputs)])
                print('writing',
                      [GenesList[Ngene], GenesList[-1], Weights[Ngene][0], Epoch, Accuracy, len(TrainInputs)])

        end = time.time()
        print("\033[94m {}\033[00m".format('Done'), gene, round(TargetGenes.index(gene) * 100 / len(TargetGenes)), '%',
              TargetGenes.index(gene), 'out of', len(TargetGenes), 'genes')


# Read TrainSet
DataSet = pd.read_csv('../Data/TranscriptomicData/TrainingData/GSE65244/GSE65244_time_course_wt_120_interpolNameCorrenctedNormalized.csv', index_col=0)
# DataSet = DataSet.drop('log-TPM', axis=1)
print(DataSet)
# exit()
# DataSet = DataSet.mul(10000)

TrainSet = DataSet
# print(TrainSet)
# print(TrainSet)
TestSet =  DataSet
TrainSet = TrainSet.reset_index(drop=True)
TestSet = TestSet.reset_index(drop=True)



newDF = pd.DataFrame(columns=TrainSet.columns, index=TrainSet.index)

#Convert TrainSet to float
TrainSet = TrainSet.astype(float)

#Convert TestSet to float
TestSet = TestSet.astype(float)



GRNDF = pd.read_csv('../PathExtraction/Results/EcoliNetworkV3.csv')

#Remove SelfSource
GRNDF = GRNDF[GRNDF['Source'] != GRNDF['Target']]

#Remove effectors
GRNDF = GRNDF[GRNDF['SourceType'] != 'effector']

TargetGenes = unique(list(GRNDF['Target']))

#Create a folder to save the results
Path = 'WeightResults/TemporalWithBiasConstrainedNormalized_LRate'+str(Lrate)+'_Epochs'+str(Epochs)
if not os.path.exists('WeightResults'):
    os.makedirs('WeightResults')

if not os.path.exists(Path+'/EachGeneComparison'):
    os.makedirs(Path+ '/EachGeneComparison')

if os.path.exists(Path+'/WeightedFilteredGRNV'):
    shutil.rmtree(Path+'/WeightedFilteredGRNV', ignore_errors=True)
os.makedirs(Path+'/WeightedFilteredGRNV')


DroppedGenes = []
processes = []
# for gene in TargetGenes:

if __name__ == '__main__':
    for gene in TargetGenes:
        print(gene)
        PerceptGeneWeightExtraction(gene)
        break


    SimArray = []
    with Pool(19) as p:
        # for gene in ['b0114', 'b0115', 'b0116', 'b1037', 'b1038', 'b1039', 'b1040', 'b2570', 'b2571', 'b2572', 'b3212', 'b3213', 'b3214', 'b3515', 'b3516', 'b3517']:
        for gene in TargetGenes:
            SimArray.append(p.apply_async(PerceptGeneWeightExtraction, args=(gene,)))
        p.close()
        p.join()

    end = time.time()
    print('Total time', end - start)


    #DroppedGenes to csv
    # with open(r'WeightResults/Path/DroppedGenes.csv', 'w') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(DroppedGenes)

    files = os.listdir(Path+'/WeightedFilteredGRNV')
    cols = pd.read_csv(Path+'/WeightedFilteredGRNV' + '/' + files[0]).columns

    FullGRNDF = pd.DataFrame(columns=cols)

    for file in files:
        tempDF = pd.read_csv(Path+'/WeightedFilteredGRNV' + '/' + file)
        print(file, files.index(file), len(files))
        FullGRNDF = pd.concat([FullGRNDF, tempDF])

    FullGRNDF.to_csv(Path + '/WeightedGRN.csv', index=False)

