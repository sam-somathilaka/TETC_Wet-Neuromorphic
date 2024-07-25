import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def Unique (list):
    UniqueList = []
    for item in list:
        if item not in UniqueList:
            UniqueList.append(item)
    return UniqueList

Savepath = 'GEAnalysisResults/FullVarPlots/'
for seed in range(29,30):
    path = 'GEAnalysisResults/FilteredByVar/Seed' + str(seed) + '/'

    Digit0 = pd.read_csv(path + 'Digit0Output.csv', index_col=0)
    Digit1 = pd.read_csv(path + 'Digit1Output.csv', index_col=0)
    Digit2 = pd.read_csv(path + 'Digit2Output.csv', index_col=0)
    Digit3 = pd.read_csv(path + 'Digit3Output.csv', index_col=0)
    Digit4 = pd.read_csv(path + 'Digit4Output.csv', index_col=0)

    ZeroAvg = Digit0['Avg'].tolist()

    print(Digit0.head())
    UniqueGenes = Unique(Digit0.index.tolist()+Digit1.index.tolist()+Digit2.index.tolist()+Digit3.index.tolist()+Digit4.index.tolist())

    CommonGenes = []
    for gene in UniqueGenes:
        if gene in Digit0.index.tolist() and gene in Digit1.index.tolist() and gene in Digit2.index.tolist() and gene in Digit3.index.tolist() and gene in Digit4.index.tolist():
            CommonGenes.append(gene)

    NewVarDF = pd.DataFrame(columns=['Gene', 'DigitVar', 'RecordVar'], index=CommonGenes)

    for gene in CommonGenes:
        AvgArray = [Digit0.loc[gene]['Avg'], Digit1.loc[gene]['Avg'], Digit2.loc[gene]['Avg'], Digit3.loc[gene]['Avg'], Digit4.loc[gene]['Avg']]
        VarArray = Digit0.loc[gene]['Var']+ Digit1.loc[gene]['Var']+ Digit2.loc[gene]['Var']+ Digit3.loc[gene]['Var']+ Digit4.loc[gene]['Var']

        NewVarDF.loc[gene] = [gene, np.var(AvgArray), VarArray/5]

    #scatter plot
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.scatter(NewVarDF['DigitVar'], NewVarDF['RecordVar'], s=3)
    ax.set_title('DigitVar vs RecordVar_' + str(seed))
    ax.set_xlabel('DigitVar')
    ax.set_ylabel('RecordVar')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig(Savepath + 'DigitVarVsRecordVar_' + str(seed) + '.png', dpi=300)
    plt.show()




