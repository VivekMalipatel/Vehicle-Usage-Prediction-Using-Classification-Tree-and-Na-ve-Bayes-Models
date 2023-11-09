import numpy
import pandas
from sklearn import preprocessing, naive_bayes

inputData = pandas.read_excel('claim_history.xlsx')

# EBilling -> CreditCard, Gender, JobCategory
subData = inputData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()

subData = subData.astype('category')
xTrain = pandas.get_dummies(subData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']])

# Correctly Use sklearn.naive_bayes.CategoricalNB
feature = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(subData['CAR_USE'])
yLabel = labelEnc.inverse_transform([0, 1])

uCarType = numpy.unique(subData['CAR_TYPE'])
uOccupation = numpy.unique(subData['OCCUPATION'])
uEducation = numpy.unique(subData['EDUCATION'])

featureCategory = [uCarType , uOccupation, uEducation]

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(subData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']])

_objNB = naive_bayes.CategoricalNB(alpha = 0.01)
thisModel = _objNB.fit(xTrain, yTrain)

y_predProb = thisModel.predict_proba(xTrain)
nbrs_pred = numpy.argmax(y_predProb, axis = 1)
mis = numpy.mean(numpy.where(nbrs_pred == yTrain, 0, 1))
print("Misclassification Rate :")
print(mis)



