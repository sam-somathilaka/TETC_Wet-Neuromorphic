import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

path = 'Results/MultiRegR2.csv'

Data = pd.read_csv(path, header=0, index_col=0)
# print(Data.head())

#Clear data
# Data = Data[Data['r2'] != 0]
# Data = Data[(Data['C1'] != 0) | (Data['C2'] != 0)]

# print(Data.head())


#randomly select 5 genes from Data.index
PlotGenes = Data.sample(n=5).index.tolist()
PlotGene = 'b0904' #NonLinear2
PlotGene = 'b4406' #NonLinear2




fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
# rotate the axes and update
# ax.view_init(30, -100)


TempDF = pd.DataFrame(columns=['Cons1', 'Cons2', 'Output', 'OuputStd'])
for G1 in np.arange(0, 5.1, 0.2):
    for G2 in np.arange(0, 5.1, 0.2):
        G1 = round(G1, 1)
        G2 = round(G2, 1)
        TempDF = pd.concat([TempDF, pd.DataFrame({'Cons1': [G1], 'Cons2': [G2], 'Output': [Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)]], 'OutputStd': [Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)+'_std']]})], ignore_index=True)


print(TempDF.head())
# exit()
# ax.scatter(TempDF['Cons1'], TempDF['Cons2'], TempDF['Output'], s=5, c=TempDF['Output'], cmap='viridis')

#fit curved surface
from scipy.optimize import curve_fit
def func(x, a, b, c, d, e, f):
    return a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1] + d*x[0] + e*x[1] + f

# def func1(x, a, b, c, d, e, f, g, h, i, j):
#     return a*x[0]**3+ b*x[1]**3 + c*x[0]**2*x[1] + d*x[0]**2 + e*x[0]*x[1]**2 + f*x[1]**2 + g*x[0]*x[1] + h*x[0] + i*x[1] + j

popt, pcov = curve_fit(func, [TempDF['Cons1'], TempDF['Cons2']], TempDF['Output'])

#RSS
RSS = 0
for G1 in np.arange(0, 5.1, 0.2):
    for G2 in np.arange(0, 5.1, 0.2):
        G1 = round(G1, 1)
        G2 = round(G2, 1)
        TempOut = TempDF[(TempDF['Cons1'] == G1) & (TempDF['Cons2'] == G2)]['Output'].tolist()[0]
        Predicted = func([G1, G2], *popt)

        RSS += (TempOut - Predicted)**2

print(PlotGene+ ' RSS ', RSS)

# print('Curve '+ str(popt[0])+'*x[0]**2 + '+str(popt[1])+'*x[1]**2 + '+str(popt[2])+'*x[0]*x[1] + '+str(popt[3])+'*x[0] + '+str(popt[4])+'*x[1] + '+str(popt[5]))
#draw curved surface with cmap
x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)
Z = func([X, Y], *popt)


ax.plot_surface(X, Y, Z, alpha=0.7, edgecolor='none', cmap='viridis')
ax.scatter(TempDF['Cons1'], TempDF['Cons2'], TempDF['Output'], s=10,  color='blue', alpha=0.2)



#write plane equation in the plot
# ax.text(0, 0, 0, 'y = '+str(round(C1, 2))+'x1 + '+str(round(C2, 2))+'x2 + '+str(round(C3, 2)), color='black', fontsize=10)


#
# #draw plane
# x = np.linspace(0, 5, 100)
# y = np.linspace(0, 5, 100)
# X, Y = np.meshgrid(x, y)
# Z = C1*X + C2*Y + C3
# ax.plot_surface(X, Y, Z, alpha=0.1, color='blue', edgecolor='none')


x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
X, Y = np.meshgrid(x, y)


# error bars
# for i in range(len(TempDF)):
#     ax.vlines(x=TempDF.iloc[i]['Cons1'], ymin=TempDF.iloc[i]['Output']-TempDF.iloc[i]['OutputStd'], ymax=TempDF.iloc[i]['Output']+TempDF.iloc[i]['OutputStd'], color='black', alpha=0.5, linewidth=0.5, )
ax.set_xlabel('b3067')
ax.set_ylabel('b3357')
ax.set_zlabel('Expression Level')


plt.title('Gene '+PlotGene)
plt.tight_layout()
# plt.savefig('Results/MultiReg3D_'+PlotGene[0]+'.png', dpi=300)
plt.show()


