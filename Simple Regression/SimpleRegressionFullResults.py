import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

path = 'Results/LinReg.csv'

Data = pd.read_csv(path, header=0, index_col=0)
print(Data.head().to_string())

Cols = []

NewDF = pd.DataFrame()
NewTemp = pd.DataFrame()

# print(NewDF.head())
for Cons in np.arange(0, 5.1, 0.2):
    tempDF = pd.DataFrame()
    Cons = round(Cons, 1)
    for i in range(10):
        tempDF[i] = Data['Cons'+str(Cons)+'Ite'+str(i)]

    NewDF['Cons'+str(Cons)] = tempDF.mean(axis=1)
    NewTemp['Cons' + str(Cons)] = tempDF.mean(axis=1)
    NewDF['Cons'+str(Cons)+'_std'] = tempDF.std(axis=1)

#Calculate total std
DFMax = NewTemp.max().max()
DFMin = NewTemp.min().min()
# print('DFMax', 'DFMin', DFMax, DFMin)
NewDF = (NewDF - DFMin) / (DFMax - DFMin)

NewDF['TotalStd'] = [0]*len(NewDF)

for Cons in np.arange(0, 5.1, 0.2):
    Cons = round(Cons, 1)
    NewDF['TotalStd'] = NewDF['TotalStd'] + NewDF['Cons'+str(Cons)+'_std']

#Drop records with 0 total std
NewDF = NewDF[NewDF['TotalStd'] != 0]
NewDF['TotalStd'] = NewDF['TotalStd']/len(np.arange(0, 5.1, 0.2))
NewDF = NewDF.sort_values(by=['TotalStd'], ascending=True)
print(NewDF.head().to_string())
# NewDF = NewDF[NewDF['TotalStd'] < 1]


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
        # Pearson correlation coefficient
        LinDF.loc[row[0], 'PearsonCoeff'] = abs(np.corrcoef(X, Vals)[0][1])

        # R^2 of NonLin2DF
        NonLin2DF.loc[row[0], 'R2'] = np.corrcoef(X, Vals)[0][1] ** 2


        # R^2 of NonLin3DF
        NonLin3DF.loc[row[0], 'R2'] = np.corrcoef(X, Vals)[0][1] ** 2

        #Deg 1 nonlinear correlation coefficient
        Line = np.polyfit(X, Vals, 1)
        LinDF.loc[row[0], 'Deg1Coef1'] = Line[0]
        LinDF.loc[row[0], 'Deg1Coef2'] = Line[1]

        #Deg 2 nonlinear correlation coefficient
        Coeff = np.polyfit(X, Vals, 2)
        NonLin2DF.loc[row[0], 'Deg2Coef1'] = Coeff[0]
        NonLin2DF.loc[row[0], 'Deg2Coef2'] = Coeff[1]
        NonLin2DF.loc[row[0], 'Deg2Coef3'] = Coeff[2]

        NonLin2DFRSS = 0
        for i in range(len(X)):
            NonLin2DFRSS += (Vals[i] - (Coeff[0]*X[i]**2 + Coeff[1]*X[i] + Coeff[2]))**2
        NonLin2DF.loc[row[0], 'RSS'] = NonLin2DFRSS

        #Deg 3 nonlinear correlation coefficient
        Coeff = np.polyfit(X, Vals, 3)
        NonLin3DF.loc[row[0], 'Deg3Coef1'] = Coeff[0]
        NonLin3DF.loc[row[0], 'Deg3Coef2'] = Coeff[1]
        NonLin3DF.loc[row[0], 'Deg3Coef3'] = Coeff[2]
        NonLin3DF.loc[row[0], 'Deg3Coef4'] = Coeff[3]

        NonLin3DFRSS = 0
        for i in range(len(X)):
            NonLin3DFRSS += (Vals[i] - (Coeff[0]*X[i]**3 + Coeff[1]*X[i]**2 + Coeff[2]*X[i] + Coeff[3]))**2
        NonLin3DF.loc[row[0], 'RSS'] = NonLin3DFRSS
print(LinDF['Deg1Coef1'].max(), LinDF['Deg1Coef1'].min(), 'LinDF Max', 'Min')
print(NonLin2DF['Deg2Coef1'].max(), NonLin2DF['Deg2Coef1'].min(), 'NonLin2DF Deg2Coef1 Max', 'Min')
print(NonLin2DF['Deg2Coef2'].max(), NonLin2DF['Deg2Coef2'].min(), 'NonLin2DF Deg2Coef2 Max', 'Min')
print(NonLin3DF['Deg3Coef1'].max(), NonLin3DF['Deg3Coef1'].min(), 'NonLin3DF Deg3Coef1 Max', 'Min')
print(NonLin3DF['Deg3Coef2'].max(), NonLin3DF['Deg3Coef2'].min(), 'NonLin3DF Deg3Coef2 Max', 'Min')
print(NonLin3DF['Deg3Coef3'].max(), NonLin3DF['Deg3Coef3'].min(), 'NonLin3DF Deg3Coef3 Max', 'Min')
#linear regression plots
LinDF = LinDF.sort_values(by=['PearsonCoeff'], ascending=False)
# remove records with pearson coeff is nan
LinDF = LinDF[~LinDF['PearsonCoeff'].isna()]
print(LinDF)
fig,ax = plt.subplots(figsize=(4, 4))


ax = sns.scatterplot(x="PearsonCoeff", y="Deg1Coef1", data=LinDF, s=15, linewidth=0.25, alpha=1)
ax.set_xlabel('Pearson Coefficient')
ax.set_ylabel('Deg1Coef1')
ax.set_ylim(ymin=-0.6)
ax.grid(alpha=0.25)
ax.set_xlabel('Pearson Coefficient against b3067')
plt.tight_layout()
# plt.savefig('Results/Plots/PearsonCoeffvsDeg1Coef1.png', dpi=300)
plt.show()
# exit()

print(NonLin2DF[['Deg2Coef1', 'Deg2Coef2']])

# NonLin2DF = NonLin2DF[(NonLin2DF['Deg2Coef1'] <4) & (NonLin2DF['Deg2Coef1'] > -2)]
# NonLin2DF = NonLin2DF[(NonLin2DF['Deg2Coef2'] <10) & (NonLin2DF['Deg2Coef2'] > -20)]
fig,ax = plt.subplots(figsize=(4, 4))

# bar plot
g = ax.scatter(x="Deg2Coef1", y="Deg2Coef2", c="R2", cmap='viridis', data=NonLin2DF, s=15, linewidth=0.25, alpha=1)
# ax.set_xlabel('Pearson Coefficient')
ax.set_ylabel('Deg1Coef1')
ax.set_ylim(-1,1)
ax.set_xlim(-2,3)
ax.grid(alpha=0.25)
# ax.set_xlabel('Pearson Coefficient against b3067')
plt.tight_layout()

# color bar
# plt.colorbar(g)

# plt.savefig('Results/Plots/Deg2Coef1vsDeg2Coef2.png', dpi=300)
plt.show()

print (NonLin3DF)
NonLin3DF = NonLin3DF[(NonLin3DF['Deg3Coef1'] <10) & (NonLin3DF['Deg3Coef1'] > 0)]
NonLin3DF = NonLin3DF[(NonLin3DF['Deg3Coef2'] <5) & (NonLin3DF['Deg3Coef2'] > -10)]
NonLin3DF = NonLin3DF[(NonLin3DF['Deg3Coef3'] <1) & (NonLin3DF['Deg3Coef3'] > -1)]
#3d plot for NonLin3DF

fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
# rotate the axes and update
ax.view_init(10, 15)

# Data for three-dimensional scattered points
xdata = NonLin3DF['Deg3Coef1']
ydata = NonLin3DF['Deg3Coef2']
zdata = NonLin3DF['Deg3Coef3']
Cdata = NonLin3DF['R2']

ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis', s=5, linewidth=0.25, alpha=0.8);

ax.set_xlabel('Deg3Coef1')
ax.set_ylabel('Deg3Coef2')
ax.set_zlabel('Deg3Coef3')

ax.set_xlim(0, 10)
ax.set_ylim(-12, 2.5)
ax.set_zlim(-1, 1)
#dim the grid
ax.grid(alpha=0.25)
#dim the background
ax.xaxis.pane.fill = False
plt.tight_layout()
# plt.savefig('Results/Plots/Deg3Coef1vsDeg3Coef2vsDeg3Coef3.png', dpi=300)
plt.show()
