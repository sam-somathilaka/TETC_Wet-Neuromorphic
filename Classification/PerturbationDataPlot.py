import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KernelDensity

def CalculateMI(x, y):
    xy = np.column_stack([x, y])
    kde_xy = KernelDensity(bandwidth=0.2).fit(xy)
    kde_x = KernelDensity(bandwidth=0.2).fit(x[:, np.newaxis])
    kde_y = KernelDensity(bandwidth=0.2).fit(y[:, np.newaxis])
    log_denom = np.log(np.exp(kde_x.score_samples(x[:, np.newaxis])) *
                       np.exp(kde_y.score_samples(y[:, np.newaxis])))
    log_num = kde_xy.score_samples(xy)
    mi_xy = np.mean(log_num - log_denom) / np.log(2)
    return mi_xy

#set sns style
sns.set_style("whitegrid")

Seed = 29
FinalDF = pd.read_csv('PerturbationAnalysis/Seed'+str(Seed)+'Output.csv', index_col=0, header=0)
InputGenes = pd.read_csv('OutputGEs/Seed'+str(Seed)+'/InputGenes.csv', index_col=0, header=0)['InputGenes'].tolist()
print(FinalDF.columns.tolist())
# print(InputGenes.head())


NewFinalDF = pd.DataFrame(index=FinalDF.index.tolist())
Cols= []

MaxCons = 10
Iterations = 20

for gene in InputGenes:
    for cons in range(MaxCons):
        NewFinalDF[gene+'_'+str(cons)+'AVG'] = [0]*len(FinalDF.index.tolist())
        NewFinalDF[gene+'_'+str(cons)+'VAR'] = [0]*len(FinalDF.index.tolist())
        tempDF = pd.DataFrame(index=FinalDF.index.tolist())

        for ite in range(Iterations):
            label = str(gene)+'_'+str(cons)+'_'+str(ite)
            tempDF[str(gene)+str(ite)] = FinalDF[label]

        NewFinalDF[gene+'_'+str(cons)+'AVG'] = np.mean(tempDF, axis=1)
        NewFinalDF[gene+'_'+str(cons)+'VAR'] = np.var(tempDF, axis=1)

print(NewFinalDF.head())


Digit = 1
Inst = 0

Cols = []
fig = plt.figure(figsize=(7, 5))

MIDF = pd.DataFrame(index=InputGenes, columns=FinalDF.index.tolist())

# DigitDimDict = {0: [0, 1.2], 1: [0, 3], 2: [0, 6], 3: [0, 0.8], 4: [0, 2]} #29
# DigitDimDict = {0: [0, 3.9], 1: [0, 4.5], 2: [0, 6], 3: [0, 0.8], 4: [0, 2]} #68
for InputGene in InputGenes:
    IGIdx = InputGenes.index(InputGene)
    ax = fig.add_subplot(4, 4, IGIdx+1,)
    for OutputGene in FinalDF.index.tolist():
        OGIdx = FinalDF.index.tolist().index(OutputGene)
        tempAVGs = []
        tempVars = []
        for cons in range(MaxCons):
            tempAVGs.append(NewFinalDF.loc[OutputGene, InputGene+'_'+str(cons)+'AVG'])
            tempVars.append(NewFinalDF.loc[OutputGene, InputGene+'_'+str(cons)+'VAR'])
        print(InputGene, OutputGene, tempAVGs)
        MI = CalculateMI(np.array(range(0, MaxCons)), np.array(tempAVGs))
        MIDF[OutputGene][InputGene] = MI
        ax = sns.lineplot(x=range(0, MaxCons), y=tempAVGs, ax=ax)
        ax.fill_between(range(0, MaxCons), np.array(tempAVGs)-np.array(tempVars), np.array(tempAVGs)+np.array(tempVars), alpha=0.5)
        ax.set_ylabel(InputGene)
        # if OGIdx == 0:
        #     ax.set_ylabel(InputGene)
        #     ax.set_ylim(0, 1.1)
        # if OGIdx == 1:
        #     ax.set_ylim(0, 5)
        # if OGIdx == 2:
        #     ax.set_ylim(0, 3)
        # if OGIdx == 3:
        #     ax.set_ylim(0, 2)
        # if OGIdx == 4:
        #     ax.set_ylim(0, 1.75)
        #
        # if IGIdx == 15:
        #     ax.set_xlabel(OutputGene)



        # print('Digit', digit, 'Gene', InputGene)
        # print(NewFinalDF.loc[OutputGene, InputGene+'_'+str(cons)+'AVG'])

    # min = 0
    # max = 0
    # tempMinMax = []
    #
    # gene = NewFinalDF.index.tolist()[Digit]
    # print('Digit', Digit, 'Gene', gene)
    # ax = fig.add_subplot(1, 5, Digit+1, )
    #
    # tempAVGs = []
    # tempVars = []
    #
    # for Cons in range(6):
    #     tempAVGs.append(NewFinalDF.loc[gene, str(Digit)+str(Cons)+'AVG'])
    #     tempVars.append(NewFinalDF.loc[gene, str(Digit)+str(Cons)+'VAR'])
    #
    # ax = sns.lineplot(x=range(0, 6), y=tempAVGs, ax=ax)
    # ax = sns.lineplot(x=range(0, 6), y=tempVars, ax=ax)
    # # ax.fill_between(range(0, 6), np.array(tempAVGs)-np.array(tempVars), np.array(tempAVGs)+np.array(tempVars), alpha=0.2)
    #
    # ax.set_title('Digit ' + str(Digit) + ' Gene ' + gene)
    # # ax.set_ylim(DigitDimDict[Digit][0], DigitDimDict[Digit][1])
    #
    # # print(subDF.head(10))


# ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
# exit()
# set font to times new roman
plt.rcParams["font.family"] = "Times New Roman"
MIDF = MIDF.astype(float)
fig = plt.figure(figsize=(3, 4))
ax = fig.add_subplot(1, 1, 1)
ax = sns.heatmap(MIDF, cmap='YlGnBu', ax=ax, annot=True, fmt='.2f')
# ax.set_title('Information Gain')
#rotate x labels
ax.set_xticklabels(ax.get_xticklabels(), rotation=35)

#have only one legend




plt.tight_layout()
# plt.savefig('PerturbationAnalysis/Seed'+str(Seed)+'InformationGain.png', dpi=300)
plt.show()

