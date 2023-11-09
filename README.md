# Vehicle-Usage-Prediction-Using-Classification-Tree-and-Naive-Bayes-Models
CS 484: Introduction to Machine Learning Assignment

# CS 484: Introduction to Machine Learning
## Spring Semester 2023 - Assignment 3

In this assignment, provided with the `claim_history.xlsx`, you will train models to predict the usage of a vehicle using 10,302 observations. Your models will use the variables outlined below to predict vehicle usage (`CAR_USE`), which can be either Commercial or Private.

### Label Field

- **CAR_USE**: Vehicle Usage (Commercial, Private)

### Nominal Predictor

- **CAR_TYPE**: Vehicle Type (Minivan, Panel Truck, Pickup, SUV, Sports Car, Van)
- **OCCUPATION**: Occupation of Vehicle Owner (Blue Collar, Clerical, Doctor, Home Maker, Lawyer, Manager, Professional, Student, Unknown)

### Ordinal Predictor

- **EDUCATION**: Highest Education Level of Vehicle Owner (Below High School, High School, Bachelors, Masters, Doctors)

### Data Preparation

We will only use observations with no missing values in the aforementioned variables. All 100% complete observations will be used for training.

For each observation, calculate the predicted probabilities for `CAR_USE` categories. The observation will be classified in the category with the highest predicted probability. In case of ties, choose the Private category.

## Question 1: Classification Tree Model (50 points)

Train a classification tree model with:

- Maximum depth of two
- Entropy as the split criterion
- Assignment to the left or right child node based on the evaluation of the splitting criterion

### Part (a) - Leaf Nodes Description (20 points)

Describe the leaf nodes of the classification tree, including:
1. Splitting Criterion
2. Number of Observations
3. Predicted Probabilities of CAR_USE
4. Predicted CAR_USE category
5. Split Entropy Value

### Part (b) - Fictitious Person: Professional with Doctors Education and Minivan (10 points)

Calculate the Car Usage probabilities for a fictitious person with these attributes.

### Part (c) - Fictitious Person: Student with Below High School Education and Sports Car (10 points)

Calculate the Car Usage probabilities for another fictitious person with these attributes.

### Part (d) - Histogram of Predicted Probabilities (5 points)

Generate a histogram for the predicted probabilities of `CAR_USE = Private` with a bin width of 0.05. The vertical axis should represent the proportion of observations.

### Part (e) - Misclassification Rate (5 points)

Determine the misclassification rate of the Classification Tree model.

## Question 2: Naïve Bayes Model (50 points)

Train a Naïve Bayes model with a Laplace/Lidstone value of 0.01 for cell counts when computing row probabilities.

### Part (a) - Class Probabilities (10 points)

What are the Class Probabilities?

### Part (b) - Cross-Tabulation (10 points)

Cross-tabulate the label variable by each predictor. Include frequency counts and row probabilities in each label class.

### Part (c) - Fictitious Person: Professional with Doctors Education and Minivan (10 points)

Calculate the Car Usage probabilities for a fictitious person with these attributes.

### Part (d) - Fictitious Person: Student with Below High School Education and Sports Car (10 points)

Calculate the Car Usage probabilities for another fictitious person with these attributes.

### Part (e) - Histogram of Predicted Probabilities (5 points)

Generate a histogram for the predicted probabilities of `CAR_USE = Private` with a bin width of 0.05. The vertical axis should represent the proportion of observations.

### Part (f) - Misclassification Rate (5 points)

Determine the misclassification rate of the Naïve Bayes model.
