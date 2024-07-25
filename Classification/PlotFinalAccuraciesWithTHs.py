import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#set sns style
sns.set_style("whitegrid")


def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

    # return string
    return str1

Seed = 29
# Seed = 449
# Seed = 436
FinalDF = pd.read_csv('AccuracyTesting/Seed'+str(Seed)+'Output.csv', index_col=0, header=0)
FinalDFMax = FinalDF.max().max()
FinalDFMin = FinalDF.min().min()
FinalDF = FinalDF.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)

FinalDF1 = pd.read_csv('AccuracyTesting/Seed'+str(Seed)+'Output_Filtered.csv', index_col=0, header=0)
FinalDFMax1 = FinalDF1.max().max()
FinalDFMin1 = FinalDF1.min().min()
FinalDF1 = FinalDF1.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=1)

THs = np.arange(0, 1.01, 0.05)

Cols = ['Gene']+list(THs)
print(Cols)
FullF1DF = pd.DataFrame()
FullAccDF = pd.DataFrame()


# TH = {0:0.3, 1:0.3, 2:3, 3:0.125, 4:0.5}
THs = {0:0.1, 1:0.5, 2:0.85, 3:0.1, 4:0.35}

i = 0


CM = pd.DataFrame()
CM1 = pd.DataFrame()

for row in FinalDF.iterrows():
    TH = THs[i]
    gene = row[0]
    # print(gene, i)
    tempDF = pd.DataFrame(columns=['Val', 'Digit'])
    Vals = row[1].tolist()
    # print(len(Vals))
    # exit()
    j=0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for val in Vals:
        if j >= i*100 and j < ((i+1)*100)-1:
            tempDF = pd.concat([tempDF, pd.DataFrame([[val, i]], columns=['Val', 'Digit'])], axis=0)
            if val > TH:
                TP += 1
            else:
                FN += 1
        else:
            tempDF = pd.concat([tempDF, pd.DataFrame([[val, 5]], columns=['Val', 'Digit'])], axis=0)
            if val > TH:
                FP += 1
            else:
                TN += 1
        j+=1
    # print([gene, TP, TN, FP, FN])
    CM = pd.concat([CM, pd.DataFrame([[gene, TP, TN, FP, FN]])], axis=0)


    i += 1


CM.columns = ['Gene', 'TP', 'TN', 'FP', 'FN']

# CM['FN'] = CM['FN']*0.75
# CM['FN'] = CM['FN'].astype(int)
#
# CM['FP'] = CM['FP']*0.75
# CM['FP'] = CM['FP'].astype(int)

CM['Precision'] = CM['TP'] / (CM['TP'] + CM['FP'])
CM['Recall'] = CM['TP'] / (CM['TP'] + CM['FN'])
CM['F1'] = 2 * ((CM['Precision'] * CM['Recall']) / (CM['Precision'] + CM['Recall']))
CM['Accuracy'] = (CM['TP'] + CM['TN']) / (CM['TP'] + CM['TN'] + CM['FP'] + CM['FN'])


print(CM)

i=0
for row in FinalDF1.iterrows():
    TH = THs[i]
    gene = row[0]
    # print(gene, i)
    tempDF = pd.DataFrame(columns=['Val', 'Digit'])
    Vals = row[1].tolist()
    # print(len(Vals))
    # exit()
    j=0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for val in Vals:
        if j >= i*100 and j < ((i+1)*100)-1:
            tempDF = pd.concat([tempDF, pd.DataFrame([[val, i]], columns=['Val', 'Digit'])], axis=0)
            if val > TH:
                TP += 1
            else:
                FN += 1
        else:
            tempDF = pd.concat([tempDF, pd.DataFrame([[val, 5]], columns=['Val', 'Digit'])], axis=0)
            if val > TH:
                FP += 1
            else:
                TN += 1
        j+=1
    # print([gene, TP, TN, FP, FN])
    CM1 = pd.concat([CM1, pd.DataFrame([[gene, TP, TN, FP, FN]])], axis=0)


    i += 1


CM1.columns = ['Gene', 'TP', 'TN', 'FP', 'FN']

# CM['FN'] = CM['FN']*0.75
# CM['FN'] = CM['FN'].astype(int)
#
# CM['FP'] = CM['FP']*0.75
# CM['FP'] = CM['FP'].astype(int)

CM1['Precision'] = CM1['TP'] / (CM1['TP'] + CM1['FP'])
CM1['Recall'] = CM1['TP'] / (CM1['TP'] + CM1['FN'])
CM1['F1'] = 2 * ((CM1['Precision'] * CM1['Recall']) / (CM1['Precision'] + CM1['Recall']))
CM1['Accuracy'] = (CM1['TP'] + CM1['TN']) / (CM1['TP'] + CM1['TN'] + CM1['FP'] + CM1['FN'])


print(CM1)



# exit()
# to float
# THCol = listToString(list(str(TH))[0:4])
# print(THCol)
# FullF1DF[THCol] = CM['F1']
# FullAccDF[THCol] = CM['Accuracy']



# FullF1DF['Gene'] = CM['Gene']
# FullAccDF['Gene'] = CM['Gene']
#
# FullF1DF.index = FullF1DF['Gene']
# FullAccDF.index = FullAccDF['Gene']
#
# FullF1DF = FullF1DF.drop(columns=['Gene'])
# FullAccDF = FullAccDF.drop(columns=['Gene'])
# print(FullAccDF)
# exit()

AccuracyComparison = pd.DataFrame(columns=['Gene', 'Accuracy', 'Accuracy_Filtered'])
AccuracyComparison['Gene'] = CM['Gene']
AccuracyComparison['Accuracy'] = CM['Accuracy']
AccuracyComparison['Accuracy_Filtered'] = CM1['Accuracy']

print(AccuracyComparison)

#grouped barplot with values
fig, ax = plt.subplots(figsize=(4, 2))
AccuracyComparison.plot(x='Gene', y=['Accuracy', 'Accuracy_Filtered'], kind='bar', ax=ax, rot=0, width=0.8, color=['#5E7FB8', '#C9D8FF'])

# for p in ax.patches:
#     ax.annotate(str(round(p.get_height(), 2)), (p.get_x() * 1.005, p.get_height() * 1.005))

ax.set_xlabel('Gene')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.set_title('Accuracy Comparison')
ax.legend().remove()
# plt.savefig('AccuracyTesting/AccuracyComparison', dpi=300)
plt.tight_layout()
plt.show()