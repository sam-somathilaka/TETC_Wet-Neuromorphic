import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

path = 'Results/MultiRegR2.csv'

Data = pd.read_csv(path, header=0, index_col=0)
# print(Data.head())

#Clear data
Data = Data[Data['r2'] != 0]
Data = Data[(Data['C1'] != 0) | (Data['C2'] != 0)]

# print(Data.head())

PlotGenes = ['b3090'] #Linear
# PlotGene = ['b1411'] #Linear

# PlotGenes = ['b3261'] #NonLinear2


# 3d scatter plot

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')
# rotate the axes and update
# ax.view_init(30, -100)

for PlotGene in PlotGenes:
    ValArray = []
    StdArray = []

    # ConsArray = []
    # for G1 in np.arange(0, 5.1, 0.2):
    #     for G2 in np.arange(0, 5.1, 0.2):
    #         G1 = round(G1, 1)
    #         G2 = round(G2, 1)
    #         ValArray.append(Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)])
    #         StdArray.append(Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)+'_std'])
    #         ConsArray.append(G1)
    #
    # #normalize
    # # ValArray = (ValArray - np.min(ValArray)) / (np.max(ValArray) - np.min(ValArray))
    # StdArray = (StdArray - np.min(StdArray)) / (np.max(ValArray) - np.min(ValArray))

    TempDF = pd.DataFrame(columns=['Cons1', 'Cons2', 'Output', 'OuputStd'])
    for G1 in np.arange(0, 5.1, 0.2):
        for G2 in np.arange(0, 5.1, 0.2):
            G1 = round(G1, 1)
            G2 = round(G2, 1)
            TempDF = pd.concat([TempDF, pd.DataFrame({'Cons1': [G1], 'Cons2': [G2], 'Output': [Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)]], 'OutputStd': [Data.loc[PlotGene, 'Cons'+str(G1)+'_'+str(G2)+'_std']]})], ignore_index=True)


    ax.scatter(TempDF['Cons1'], TempDF['Cons2'], TempDF['Output'], s=5)
    #draw plane using C1, C2, and C3
    C1 = Data.loc[PlotGene, 'C1']
    C2 = Data.loc[PlotGene, 'C2']
    C3 = Data.loc[PlotGene, 'C3']

    print( C1, C2, C3)

    #write plane equation in the plot
    # ax.text(0, 0, 0, 'y = '+str(round(C1, 2))+'x1 + '+str(round(C2, 2))+'x2 + '+str(round(C3, 2)), color='black', fontsize=10)



    #draw plane
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z1 = C1*X + C2*Y + C3
    ax.plot_surface(X, Y, Z1, alpha=0.1, color='blue', edgecolor='none')


    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    X, Y = np.meshgrid(x, y)




# error bars
# for i in range(len(TempDF)):
#     ax.vlines(x=TempDF.iloc[i]['Cons1'], ymin=TempDF.iloc[i]['Output']-TempDF.iloc[i]['OutputStd'], ymax=TempDF.iloc[i]['Output']+TempDF.iloc[i]['OutputStd'], color='black', alpha=0.5, linewidth=0.5, )
ax.set_xlabel('b3067')
ax.set_ylabel('b3357')
ax.set_zlabel('Expression Level')


plt.title('Gene '+PlotGene+' ' +str(round(C1,3))+' '+str(round(C2,3))+' '+str(round(C3,3)))
plt.tight_layout()
#dim the background
ax.grid(alpha=0.25)
#white background

# plt.savefig('MultiReg3D_'+PlotGene[0]+'.png', dpi=300)
plt.show()



