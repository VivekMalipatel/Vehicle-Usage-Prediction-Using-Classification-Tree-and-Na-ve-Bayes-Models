import pandas

trainData = pandas.read_excel('claim_history.xlsx')

count = 0
testData = trainData[['CAR_TYPE', 'CAR_USE','EDUCATION','OCCUPATION']].dropna()

for index,row in testData.iterrows():
    if ((row['OCCUPATION'] == 'Blue Collar') |  (row['OCCUPATION'] == 'Student') | (row['OCCUPATION'] == 'Unknown')):
        if (row['EDUCATION'] == 'Below High Sc'):
            if(row['CAR_USE']=='Commercial'):
                count+=1
        else:
            if(row['CAR_USE']=='Private'):
                count+=1
    else:
        if ((row['CAR_TYPE'] == 'Sports Car') | (row['CAR_TYPE'] == 'SUV') | (row['CAR_TYPE'] == 'Minivan')):
            if(row['CAR_USE']=='Commercial'):
                count+=1
        else:
            if(row['CAR_USE']=='Private'):
                count+=1

print("Misclassification Rate is :")
mis = count/len(testData)
print(mis*100)