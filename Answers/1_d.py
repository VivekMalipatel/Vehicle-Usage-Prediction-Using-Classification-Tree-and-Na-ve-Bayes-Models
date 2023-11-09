import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
trainData = pd.read_excel('claim_history.xlsx')

def set_leaf (row):
   if (np.isin(row['OCCUPATION'], ['Blue Collar', 'Student', 'Unknown'])):
      if (np.isin(row['EDUCATION'], ['Below High Sc'])):
         Leaf = 0 
      else:
         Leaf = 1
   else:
      if (np.isin(row['CAR_TYPE'], ['Sports Car', 'SUV', 'Minivan'])):
         Leaf = 2
      else:
         Leaf = 3

   return(Leaf)

trainData = trainData.assign(Leaf = trainData.apply(set_leaf, axis = 1))

countTable = pd.crosstab(index = trainData['Leaf'], columns = trainData['CAR_USE'],
                             margins = False, dropna = True)

predProbCAR_USE = countTable.div(countTable.sum(1), axis = 'index')
print(predProbCAR_USE)

testData = trainData[['CAR_TYPE', 'CAR_USE','EDUCATION','OCCUPATION']]
df_x=[]
for index,row in testData.iterrows():
    if ((row['OCCUPATION'] == 'Blue Collar') |  (row['OCCUPATION'] == 'Student') | (row['OCCUPATION'] == 'Unknown')):
        if (row['EDUCATION'] == 'Below High Sc'):
            df_x.append(predProbCAR_USE.iloc[0][1])
        else:
            df_x.append(predProbCAR_USE.iloc[1][1])
    else:
        if ((row['CAR_TYPE'] == 'Sports Car') | (row['CAR_TYPE'] == 'SUV') | (row['CAR_TYPE'] == 'Minivan')):
            df_x.append(predProbCAR_USE.iloc[2][1])
        else:
            df_x.append(predProbCAR_USE.iloc[3][1])

df_x=pd.DataFrame(df_x)
plt.figure(figsize = (10,6), dpi = 200)
plt.hist(df_x, bins=20)
plt.title('Histogram of predicted probabilities of CAR_USE = Private')
plt.xlabel('predicted probabilities of CAR_USE = Private')
plt.ylabel('Proportion of Observations')
#plt.xticks(ticks=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1))
plt.grid(axis = 'y')
plt.show() 
