import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

sns.set_style("whitegrid")

path = 'Results/MultiReg.csv'

Data = pd.read_csv(path, header=0, index_col=0)
print(Data)

NewDF = pd.DataFrame()

for G1 in np.arange(0, 5.1, 0.2):
    for G2 in np.arange(0, 5.1, 0.2):
        G1 = round(G1, 1)
        G2 = round(G2, 1)
        tempDF = pd.DataFrame()
        for i in range(10):
            tempDF[i] = Data['Cons'+str(G1)+'_'+str(G2)+'Ite'+str(i)]

        NewDF['Cons'+str(G1)+'_'+str(G2)] = tempDF.mean(axis=1)
        NewDF['Cons'+str(G1)+'_'+str(G2)+'_std'] = tempDF.std(axis=1)

#Calculate total std
NewDF['TotalStd'] = [0]*len(NewDF)

for G1 in np.arange(0, 5.1, 0.2):
    for G2 in np.arange(0, 5.1, 0.2):
        G1 = round(G1, 1)
        G2 = round(G2, 1)
        NewDF['TotalStd'] = NewDF['TotalStd'] + NewDF['Cons'+str(G1)+'_'+str(G2)+'_std']

#Drop records with 0 total std
NewDF = NewDF[NewDF['TotalStd'] != 0]
NewDF['TotalStd'] = NewDF['TotalStd']/len(np.arange(0, 5.1, 0.2)**2)
NewDF = NewDF.sort_values(by=['TotalStd'], ascending=True)
NewDF = NewDF[NewDF['TotalStd'] < 1]

NewDF = NewDF.drop(columns=['TotalStd'])

# calculate r2 score
from sklearn.metrics import r2_score

for row in NewDF.iterrows():
    print('Gene ', NewDF.index.get_loc(row[0]), 'out of ', len(NewDF))
    TempDF = pd.DataFrame(columns=['C1', 'C2', 'Output'])
    # print(row[1])

    for G1 in np.arange(0, 5.1, 0.2):
        for G2 in np.arange(0, 5.1, 0.2):
            G1 = round(G1, 1)
            G2 = round(G2, 1)
            TempDF = pd.concat([TempDF, pd.DataFrame({'C1': [G1], 'C2': [G2], 'Output': [row[1]['Cons'+str(G1)+'_'+str(G2)]]})], ignore_index=True)
    #
    model = LinearRegression()
    X, y = TempDF[['C1', 'C2']], TempDF['Output']
    model.fit(X, y)
    RegScore = model.score(X, y)
    #Plane equation
    print('y = ', model.coef_[0], 'x1 + ', model.coef_[1], 'x2 + ', model.intercept_)
    NewDF.loc[row[0], 'r2'] = RegScore
    NewDF.loc[row[0], 'C1'] = model.coef_[0]
    NewDF.loc[row[0], 'C2'] = model.coef_[1]
    NewDF.loc[row[0], 'C3'] = model.intercept_
    #     df[['NumberofEmployees','ValueofContract']], df.AverageNumberofTickets)
    # model.fit(X, y)
    # print(TempDF.head())
    # exit()
    # Output = [0]*len(Vals)
    # r2 = r2_score(Vals, Output)
    # NewDF.loc[row[0], 'r2'] = r2
NewDF = NewDF.sort_values(by=['r2'], ascending=False)
NewDF.to_csv('Results/MultiRegR2.csv')