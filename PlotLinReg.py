import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

path = 'Results/LinReg.csv'

Data = pd.read_csv(path, header=0, index_col=0)
# print(Data.head().to_string())

Cols = []
# for Col in np.arange(0, 5.1, 0.2):
#     Cols.append('Cons'+str(Col))
#     Cols.append('Cons'+str(Col)+'_std')
#
# print(Cols)
NewDF = pd.DataFrame()
NewTemp = pd.DataFrame()

for Cons in np.arange(0, 5.1, 0.2):
    tempDF = pd.DataFrame()
    Cons = round(Cons, 1)
    for i in range(10):
        tempDF[i] = Data['Cons'+str(Cons)+'Ite'+str(i)]

    NewDF['Cons'+str(Cons)] = tempDF.mean(axis=1)
    NewTemp['Cons'+str(Cons)] = tempDF.mean(axis=1)
    NewDF['Cons'+str(Cons)+'_std'] = tempDF.std(axis=1)

DFMax = NewTemp.max().max()
DFMin = NewTemp.min().min()

NewDF = (NewDF - DFMin) / (DFMax - DFMin)

LinDF = NewDF.copy()
NonLin2DF = NewDF.copy()
NonLin3DF = NewDF.copy()


# Calculate Coefficients
for row in NewDF.iterrows():
    Vals = []
    for Cons in np.arange(0, 5.1, 0.2):
        Cons = round(Cons, 1)
        Vals.append(row[1]['Cons'+str(Cons)])

    # Vals = (Vals - np.min(Vals)) / (np.max(Vals) - np.min(Vals))


    #if all values are 0
    if sum(Vals) != 0:
    #linear correlation coefficient
        X = np.arange(0, 5.1, 0.2)
        X = X/10
        # X = (X - np.min(X)) / (np.max(X) - np.min(X))
        # Pearson correlation coefficient
        LinDF.loc[row[0], 'PearsonCoeff'] = abs(np.corrcoef(X, Vals)[0][1])


        #Deg 1 nonlinear correlation coefficient
        Line = np.polyfit(X, Vals, 1)
        LinDF.loc[row[0], 'Deg1Coef1'] = Line[0]
        LinDF.loc[row[0], 'Deg1Coef2'] = Line[1]

        #Deg 2 nonlinear correlation coefficient
        Coeff = np.polyfit(X, Vals, 2)
        NonLin2DF.loc[row[0], 'Deg2Coef1'] = Coeff[0]
        NonLin2DF.loc[row[0], 'Deg2Coef2'] = Coeff[1]
        NonLin2DF.loc[row[0], 'Deg2Coef3'] = Coeff[2]

        #Deg 3 nonlinear correlation coefficient
        Coeff = np.polyfit(X, Vals, 3)
        NonLin3DF.loc[row[0], 'Deg3Coef1'] = Coeff[0]
        NonLin3DF.loc[row[0], 'Deg3Coef2'] = Coeff[1]
        NonLin3DF.loc[row[0], 'Deg3Coef3'] = Coeff[2]
        NonLin3DF.loc[row[0], 'Deg3Coef4'] = Coeff[3]

#linear regression plots
LinDF = LinDF.sort_values(by=['PearsonCoeff'], ascending=False)
print(LinDF[['Deg1Coef1', 'Deg1Coef2', 'PearsonCoeff']].head(100).to_string())
# exit()
# print(LinDF.head(20).to_string())

LinGeneList = ['b1380', ]
# RowLoc = [5,7,1]
fig = plt.figure(figsize=(5,3))
ax = plt.subplot(111)

# LinDF = LinDF.head(30)

# LinDF = LinDF.sort_values(by=['Deg1Coef1'], ascending=False)
PlotGenes = ['b1380', 'b3293','b4435']
LinDF = LinDF.loc[PlotGenes]

print(LinDF[['Deg1Coef1', 'Deg1Coef2']].head().to_string())

for row in LinDF.iterrows():
    Gene = row[0]
    # print(row[1]['Deg1Coef1'], row[1]['Deg1Coef2'])

    GeneAVG = []
    GeneSTD = []

    for Cons in np.arange(0, 5.1, 0.2):
        Cons = round(Cons, 1)
        GeneAVG.append(LinDF.loc[Gene]['Cons'+str(Cons)])
        GeneSTD.append(LinDF.loc[Gene]['Cons'+str(Cons)+'_std'])

    X = np.arange(0, 5.1, 0.2)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X = X / 10
    # GeneAVGMax = np.max(GeneAVG)
    # GeneAVGMin = np.min(GeneAVG)
    # GeneSTDMin = np.min(GeneSTD)

    # GeneAVG = (GeneAVG - GeneAVGMin) / (GeneAVGMax - GeneAVGMin)
    # GeneSTD = (GeneSTD - GeneSTDMin) / (GeneAVGMax - GeneAVGMin)
    # GeneAVG = np.divide(GeneAVG, 6.3)
    # GeneSTD = np.divide(GeneSTD, 6.3)

    ax.plot(X, GeneAVG,  linewidth=1,  label=Gene)
    ax.fill_between(X, np.array(GeneAVG) - np.array(GeneSTD), np.array(GeneAVG) + np.array(GeneSTD),  alpha=0.25)

    # plot linear regression line with Deg1Coef1  Deg1Coef2 with the same color as the line
    ax.plot(X, row[1]['Deg1Coef1']*X + row[1]['Deg1Coef2'], color=ax.lines[-1].get_color(), linewidth=3, linestyle='--', alpha=0.2)


    #place equation on the plot
    # ax.text(0.7, LinDF.index.to_list().index(Gene)/10, 'y = '+str(round(row[1]['Deg1Coef1'], 2))+'x + '+str(round(row[1]['Deg1Coef2'], 2)), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

# ax.set_yticks(np.arange(0, 6, 1))
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0, xmax=0.5)
#legend
# ax.legend()
plt.tight_layout()
plt.savefig('Results/Plots/LinReg.png', dpi=300)
plt.show()
# LinDF.to_csv('Results/Plots/LinRegCoefs.csv')
# exit()

# print(NonLin2DF)
#
# #Nonlinear regression plots 2degree
# PlotGenes = []
# Deg2Coef1IndexList = NonLin2DF.sort_values(by=['Deg2Coef1'], ascending=False).index.tolist()
# Deg2Coef2IndexList = NonLin2DF.sort_values(by=['Deg2Coef2'], ascending=False).index.tolist()
# Deg2Coef3IndexList = NonLin2DF.sort_values(by=['Deg2Coef3'], ascending=False).index.tolist()
#
# elementIds = range(10,20)
#
PlotGenes = ['b0124', 'b2487', 'b3751']
# # # for i in elementIds:
# # #     PlotGenes.append(Deg2Coef1IndexList[i])
# # #     PlotGenes.append(Deg2Coef2IndexList[i])
# # #     PlotGenes.append(Deg2Coef3IndexList[i])
# #
NonLin2DF = NonLin2DF.loc[PlotGenes]
# # print(NonLin2DF.head().to_string())


# NonLin2DF.to_csv('Results/Plots/NonLinReg2Coeffs.csv')

fig = plt.figure(figsize=(5, 3))
ax = plt.subplot(111)
for Cons in np.arange(0, 5.1, 0.2):
    Cons = round(Cons, 1)
    NonLin2DF = NonLin2DF[NonLin2DF['Cons'+str(Cons)] < 20]

print(NonLin2DF[['Deg2Coef1', 'Deg2Coef2', 'Deg2Coef3']])
# exit()
for row in NonLin2DF.iterrows():
    Gene = row[0]


    GeneAVG = []
    GeneSTD = []

    for Cons in np.arange(0, 5.1, 0.2):
        Cons = round(Cons, 1)
        GeneAVG.append(NonLin2DF.loc[Gene]['Cons'+str(Cons)])
        GeneSTD.append(NonLin2DF.loc[Gene]['Cons'+str(Cons)+'_std'])

        # Devide GeneAVG by max
    # GeneAVG = np.array(GeneAVG)/np.max(GeneAVG)
    # GeneSTD = np.array(GeneSTD)/np.max(GeneAVG)
    # GeneAVGMax = np.max(GeneAVG)
    # GeneAVGMin = np.min(GeneAVG)
    # GeneAVG = np.divide(GeneAVG, 15)
    # GeneSTD = np.divide(GeneSTD, 15)

    X= np.arange(0, 5.1, 0.2)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X = X/10
    ax.plot(X, GeneAVG,  linewidth=1,  label=Gene)
    ax.fill_between(X, np.array(GeneAVG) - np.array(GeneSTD), np.array(GeneAVG) + np.array(GeneSTD),  alpha=0.25)

    # plot qudrating regression line with Deg2Coef1  Deg2Coef2  Deg2Coef3 with the same color as the line
    ax.plot(X, row[1]['Deg2Coef1']*X**2 + row[1]['Deg2Coef2']*X + row[1]['Deg2Coef3'], color=ax.lines[-1].get_color(), linewidth=3, linestyle='--', alpha=0.5)

# ax.set_ylim(ymin=0)
ax.set_ylim(ymin=0)
ax.set_xlim(xmin=0, xmax=0.5)
# ax.set_ylim(0,1)
# only 5 ticks on y axis
# ax.set_yticks(np.arange(0, 20, 4))
#legend
# ax.legend()
plt.tight_layout()
# plt.savefig('Results/Plots/NonLinReg2.png', dpi=300)
plt.show()
# exit()

#Nonlinear regression plots 3degree

NonLin3DF['Diff'] = abs(NonLin3DF['Deg3Coef1'] - NonLin3DF['Deg3Coef2'])*abs(NonLin3DF['Deg3Coef2'] - NonLin3DF['Deg3Coef3'])
NonLin3DF = NonLin3DF.sort_values(by=['Diff'], ascending=False)



#if cons = 0, remove record from the dataframe
for Cons in np.arange(0, 5.1, 0.2):
    Cons = round(Cons, 1)
    # NonLin3DF = NonLin3DF[NonLin3DF['Cons'+str(Cons)] != 0]


PlotGenes = ['b0400', 'b2156', 'b0367']
# Deg3Coef1IndexList = NonLin3DF.sort_values(by=['Deg3Coef1'], ascending=False).index.tolist()
# Deg3Coef2IndexList = NonLin3DF.sort_values(by=['Deg3Coef2'], ascending=False).index.tolist()
# Deg3Coef3IndexList = NonLin3DF.sort_values(by=['Deg3Coef3'], ascending=False).index.tolist()
# Deg3Coef4IndexList = NonLin3DF.sort_values(by=['Deg3Coef4'], ascending=False).index.tolist()



elementIds = range(0,10)

NonLin3DF = NonLin3DF.loc[PlotGenes]
print(NonLin3DF[['Deg3Coef1', 'Deg3Coef2', 'Deg3Coef3', 'Deg3Coef4']].head().to_string())
# print(NonLin3DF.head().to_string())
fig = plt.figure(figsize=(5, 3))
ax = plt.subplot(111)

# NonLinDF = NonLinDF.head(5)
for row in NonLin3DF.iterrows():
    Gene = row[0]

    GeneAVG = []
    GeneSTD = []

    for Cons in np.arange(0, 5.1, 0.2):
        Cons = round(Cons, 1)
        GeneAVG.append(NonLin3DF.loc[Gene]['Cons'+str(Cons)])
        GeneSTD.append(NonLin3DF.loc[Gene]['Cons'+str(Cons)+'_std'])

        # Devide GeneAVG by max
    # GeneAVG = np.array(GeneAVG)/np.max(GeneAVG)
    # GeneSTD = np.array(GeneSTD)/np.max(GeneAVG)
    # GeneAVG = np.divide(GeneAVG, 42)
    # GeneSTD = np.divide(GeneSTD, 42)

    X = np.arange(0, 5.1, 0.2)
    # X = (X - np.min(X)) / (np.max(X) - np.min(X))
    X = X/10
    ax.plot(X, GeneAVG,  linewidth=1,  label=Gene)
    ax.fill_between(X, np.array(GeneAVG) - np.array(GeneSTD), np.array(GeneAVG) + np.array(GeneSTD),  alpha=0.25)

    # plot cubic regression line with Deg3Coef1  Deg3Coef2  Deg3Coef3  Deg3Coef4 with the same color as the line
    ax.plot(X, row[1]['Deg3Coef1']*X**3 + row[1]['Deg3Coef2']*X**2 + row[1]['Deg3Coef3']*X + row[1]['Deg3Coef4'], color=ax.lines[-1].get_color(), linewidth=3, linestyle='--', alpha=0.5)


# ax.set_ylim(0,1)
ax.set_xlim(xmin=0, xmax=0.5)
ax.set_ylim(ymin=0)

#legend
# ax.legend()
plt.tight_layout()
# plt.savefig('Results/Plots/NonLinReg3.png', dpi=300)
plt.show()


# NonLin3DF.to_csv('Results/Plots/NonLinReg3.csv')