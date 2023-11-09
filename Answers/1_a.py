import numpy
import pandas
import itertools

def plog2p (proportion):
   if (0.0 < proportion and proportion < 1.0):
      result = proportion * numpy.log2(proportion)
   elif (proportion == 0.0 or proportion == 1.0):
      result = 0.0
   else:
      result = numpy.nan

   return (result)

def NodeEntropy (nodeCount):
   nodeTotal = numpy.sum(nodeCount)
   nodeProportion = nodeCount / nodeTotal
   nodeEntropy = - numpy.sum(nodeProportion.apply(plog2p))

   return (nodeTotal, nodeEntropy)

def EntropyNominalSplit (target, catPredictor, splitList, debug = 'N'):
   branch_indicator = numpy.where(catPredictor.isin(splitList), 'LEFT', 'RIGHT')
   xtab = pandas.crosstab(index = branch_indicator, columns = target, margins = False, dropna = True)
   if (debug == 'Y'):
      print('Target: ', target)
      print('Predictor: ', catPredictor)
      print('Split Value: ', splitList)
      print('Split Crosstabulation:')
      print(xtab)

   nNode = 0
   splitEntropy = 0.0
   tableTotal = 0.0
   for idx, row in xtab.iterrows():
      nNode = nNode + 1
      rowTotal, rowEntropy = NodeEntropy(row)
      tableTotal = tableTotal + rowTotal
      splitEntropy = splitEntropy + rowTotal * rowEntropy

   splitEntropy = splitEntropy / tableTotal
  
   return(nNode, splitEntropy)

def takeEntropy(s):
      return s[2]

def GetNominalSplit (inData) :

   ufreq_A = inData.iloc[:,0].astype('category').value_counts()

   # Extract the columns for easier use
   A = inData.iloc[:,0]
   B = inData.iloc[:,1]

   crossTable = pandas.crosstab(index = A, columns = B, margins = True, dropna = True)   
   print(crossTable)

   # Calculate the entropy of all the possible splits
   category_A = set(ufreq_A.index)
   n_category = len(category_A)

   split_summary = []
   for size in range(1, n_category):
      comb_size = itertools.combinations(category_A, size)
      for item in list(comb_size):
         left_branch = list(item)
         right_branch = list(category_A.difference(left_branch))
         nNode, splitEntropy = EntropyNominalSplit (B, A, left_branch, debug = 'N')
         if (nNode > 1):
               split_summary.append([left_branch, right_branch, splitEntropy])

   # Determine the split that yields the lowest splitEntropy
   split_summary.sort(key = takeEntropy, reverse = False)
   print('=== Optimal Split ===')
   print(' Left Branch Set: ', split_summary[0][0])
   print('Right Branch Set: ', split_summary[0][1])
   print('   Split Entropy: ', split_summary[0][2])
   return split_summary

def GetOrdinalSplit (inData,predValue) :

   ufreq_A = inData.iloc[:,0].astype('category').value_counts()

   # Extract the columns for easier use
   A = inData.iloc[:,0]
   B = inData.iloc[:,1]

   crossTable = pandas.crosstab(index = A, columns = B, margins = True, dropna = True)   
   print(crossTable)

   # Calculate the entropy of all the possible splits
   category_A = set(ufreq_A.index)
   n_category = len(category_A)

   split_summary = []
   for i in range(1, n_category):
         left_branch = list(predValue[0:i])
         right_branch = list(predValue[i:n_category])
         nNode, splitEntropy = EntropyNominalSplit (B, A, left_branch, debug = 'N')
         if (nNode > 1):
               split_summary.append([left_branch, right_branch, splitEntropy])

   # Determine the split that yields the lowest splitEntropy
   split_summary.sort(key = takeEntropy, reverse = False)
   print('=== Optimal Split ===')
   print(' Left Branch Set: ', split_summary[0][0])
   print('Right Branch Set: ', split_summary[0][1])
   print('   Split Entropy: ', split_summary[0][2])
   return split_summary


trainData = pandas.read_excel('claim_history.xlsx')

print("/////////////////////////FIRST LEVEL/////////////////////////////////")

print()
print("CarType :")
inData = trainData[['CAR_TYPE', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData)
print()
print("Occupation :")
inData = trainData[['OCCUPATION', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData)
print()
print("Education :")
inData = trainData[['EDUCATION', 'CAR_USE']]
tree_CAR_TYPE = GetOrdinalSplit(inData,['Below High Sc', 'High School', 'Bachelors', 'Masters', 'PhD'])


print("/////////////////////////SECOND LEVEL/////////////////////////////////")

print()
print()
print("leftBranch :")
leftBranch = trainData[trainData['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
print()
print("CarType :")
inData = leftBranch[['CAR_TYPE', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData)
print()
print("Occupation :")
inData = leftBranch[['OCCUPATION', 'CAR_USE']]
tree_OCCUPATION = GetNominalSplit(inData)
print()
print("Education :")
inData = leftBranch[['EDUCATION', 'CAR_USE']]
tree_EDUCATION = GetOrdinalSplit(inData, ['Below High Sc', 'High School', 'Bachelors', 'Masters', 'PhD'])



print()
print()
print()
print("RightBranch :")
rightBranch = trainData[~trainData['OCCUPATION'].isin(['Blue Collar', 'Student', 'Unknown'])]
print()
print("CarType :")
inData = rightBranch[['CAR_TYPE', 'CAR_USE']]
tree_CAR_TYPE = GetNominalSplit(inData)
print()
print("Occupation :")
inData = rightBranch[['OCCUPATION', 'CAR_USE']]
tree_OCCUPATION = GetNominalSplit(inData)
print()
print("Education :")
inData = rightBranch[['EDUCATION', 'CAR_USE']]
tree_EDUCATION = GetOrdinalSplit(inData, ['Below High School', 'High School', 'Bachelors', 'Masters', 'PhD'])




print("\n/////////////////////////Number of Observations and Probabilities/////////////////////////////////\n")
def set_leaf (row):
   if (numpy.isin(row['OCCUPATION'], ['Blue Collar', 'Student', 'Unknown'])):
      if (numpy.isin(row['EDUCATION'], ['Below High Sc'])):
         Leaf = 0 
      else:
         Leaf = 1
   else:
      if (numpy.isin(row['CAR_TYPE'], ['Sports Car', 'SUV', 'Minivan'])):
         Leaf = 2
      else:
         Leaf = 3

   return(Leaf)

trainData = trainData.assign(Leaf = trainData.apply(set_leaf, axis = 1))

countTable = pandas.crosstab(index = trainData['Leaf'], columns = trainData['CAR_USE'],
                             margins = False, dropna = True)
print(countTable)

predProbCAR_USE = countTable.div(countTable.sum(1), axis = 'index')
print(predProbCAR_USE)