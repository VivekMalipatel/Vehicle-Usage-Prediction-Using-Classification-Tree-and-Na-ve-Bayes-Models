import numpy
import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing, naive_bayes

inputData = pandas.read_excel('claim_history.xlsx')

# EBilling -> CreditCard, Gender, JobCategory
subData = inputData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION', 'CAR_USE']].dropna()

subData = subData.astype('category')
xTrain = pandas.get_dummies(subData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']])

feature = ['CAR_TYPE', 'OCCUPATION', 'EDUCATION']

labelEnc = preprocessing.LabelEncoder()
yTrain = labelEnc.fit_transform(subData['CAR_USE'])
yLabel = labelEnc.inverse_transform([0, 1])

uCarType = numpy.unique(subData['CAR_TYPE'])
uOccupation = numpy.unique(subData['OCCUPATION'])
uEducation = numpy.unique(subData['EDUCATION'])

featureCategory = [uCarType, uOccupation, uEducation]

featureEnc = preprocessing.OrdinalEncoder(categories = featureCategory)
xTrain = featureEnc.fit_transform(subData[['CAR_TYPE', 'OCCUPATION', 'EDUCATION']])

_objNB = naive_bayes.CategoricalNB(alpha = 0.01)
thisModel = _objNB.fit(xTrain, yTrain)

y_predProb = thisModel.predict_proba(xTrain)
df_y_predProb = pandas.DataFrame(y_predProb, columns=thisModel.classes_)
X = df_y_predProb.iloc[:,1]

d = 0.05

def matrix_B (x, delta):

    x_min = numpy.min(x)
    x_max = numpy.max(x)
    x_mean = numpy.mean(x)

    # Loop through the bin width candidates
    x_middle = delta * numpy.round(x_mean / delta)
    n_bin_left = numpy.ceil((x_middle - x_min) / delta)
    n_bin_right = numpy.ceil((x_max - x_middle) / delta)
    x_low = x_middle - n_bin_left * delta

    # Assign observations to bins starting from 0
    list_boundary = []
    n_bin = n_bin_left + n_bin_right
    bin_index = 0
    bin_boundary = x_low
    for i in numpy.arange(n_bin):
        bin_boundary = bin_boundary + delta
        bin_index = numpy.where(x > bin_boundary, i+1, bin_index)
        list_boundary.append(bin_boundary)
        
    return list_boundary

matrix_boundary = matrix_B (X,d)

plt.figure(figsize = (10,6), dpi = 200)
plt.hist(X, bins = matrix_boundary, align = 'mid')
plt.title('Histogram')
plt.ylabel('Proportion of Observations')
plt.xlabel('Distribution of predicted probability of CAR_USE = private')
plt.grid(axis = 'y')
plt.show() 

