import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

path = 'Results/MultiRegR2.csv'

Data = pd.read_csv(path, header=0, index_col=0)


#Clear data
Data = Data[Data['r2'] != 0]
Data = Data[(Data['C1'] != 0) | (Data['C2'] != 0)]

#scatter plot using C1 and C2, while the hue is the r2
fig = plt.figure(figsize=(5, 5))
sns.scatterplot(x='C1', y='C2', data=Data, hue='r2', s=15, legend=False, palette='viridis', )
plt.xlabel('C1')
plt.ylabel('C2')
plt.xlim(-0.4, 0.4)
plt.tight_layout()
# plt.savefig('MultiRegR2.png', dpi=300)
plt.show()

