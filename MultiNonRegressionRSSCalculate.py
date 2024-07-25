import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

sns.set_style("whitegrid")

path = 'Results/MultiRegR2.csv'

Data = pd.read_csv(path, header=0, index_col=0)
#Drop C1 and C2 and C3
Data = Data.drop(['C1', 'C2', 'C3', 'r2'], axis=1)
# Data = Data.head(500)
#Clear data
# Data = Data[Data['r2'] != 0]
# Data = Data[(Data['C1'] != 0) | (Data['C2'] != 0)]

# print(Data.head())



# PlotGene = ['b0659'] #NonLinear2
# PlotGene = ['b3242'] #Linear
# PlotGene = ['b3090'] #Linear
# PlotGene = ['b3261'] #Linear
# PlotGene = ['b3357']

CoefDF = pd.DataFrame(columns=['Gene', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'RSS'])

# 3d scatter plot
for PlotGene in Data.index:
    print('Gene', Data.index.tolist().index(PlotGene)+1, 'of', len(Data.index))

    # rotate the axes and update
    # ax.view_init(30, -100)


    TempDF = pd.DataFrame(columns=['Cons1', 'Cons2', 'Output', 'OuputStd'])
    for G1 in np.arange(0, 5.1, 0.2):
        for G2 in np.arange(0, 5.1, 0.2):
            G1 = round(G1, 1)
            G2 = round(G2, 1)
            TempDF = pd.concat([TempDF, pd.DataFrame({'Cons1': G1/10, 'Cons2': G2/10, 'Output': [Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)]], 'OutputStd': [Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)+'_std']]})], ignore_index=True)

    #normilize
    TempDF['Output'] = (TempDF['Output'] - np.min(TempDF['Output'])) / (np.max(TempDF['Output']) - np.min(TempDF['Output']))
    # print(TempDF)
    # exit()
    # ax.scatter(TempDF['Cons1'], TempDF['Cons2'], TempDF['Output'], s=5, c=TempDF['Output'], cmap='viridis')

    #if TempDF['Output'] doesn't have any NaN

    if TempDF['Output'].isnull().values.any() == False:

        def func(x, a, b, c, d, e, f):
            return a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1] + d*x[0] + e*x[1] + f

        # def func1(x, a, b, c, d, e, f, g, h, i, j):
        #     return a*x[0]**3+ b*x[1]**3 + c*x[0]**2*x[1] + d*x[0]**2 + e*x[0]*x[1]**2 + f*x[1]**2 + g*x[0]*x[1] + h*x[0] + i*x[1] + j

        popt, pcov = curve_fit(func, [TempDF['Cons1'], TempDF['Cons2']], TempDF['Output'])


        #RSS
        RSS = 0
        for G1 in np.arange(0, 0.51, 0.02):
            for G2 in np.arange(0, 0.51, 0.02):
                G1 = round(G1, 1)
                G2 = round(G2, 1)
                TempOut = TempDF[(TempDF['Cons1'] == G1) & (TempDF['Cons2'] == G2)]['Output'].tolist()[0]
                Predicted = func([G1, G2], *popt)

                RSS += (TempOut - Predicted)**2

        CoefDF = pd.concat([CoefDF, pd.DataFrame([[PlotGene, popt[0], popt[1], popt[2], popt[3], popt[4], popt[5], RSS]], columns=['Gene', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'RSS'])], axis=0)

print(CoefDF.head())

#sort
CoefDF = CoefDF.sort_values(by=['RSS'], ascending=True)
CoefDF.to_csv('Results/MultiRegCoefRSS.csv')
exit()
#only if RSS>0
# CoefDF = CoefDF[CoefDF['RSS'] > 0.1]
