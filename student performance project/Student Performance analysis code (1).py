#made by Dharmendra Choudhary......VIT university,vellore,Tamil Nadu
#import tkinter as tk
#from tkinter import scrolledtext,ttk
#from PIL import Image,ImageTk
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error

#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
#from flask import Flask, render_template
#window=tk.TK()
#window.title("Student performance Analysis")
#Reading the dataset and assigns it to the DataFrame df 
df = pd.read_csv(r"C:\Users\hp\Downloads\StudentPerformanceAnalysisProject-main\Student dataset example1.csv")
column_data_types = df.dtypes


#Removing a column named "PlaceofBirth" from the DataFrame df using the drop() function. 
df = df.drop('Place of Birth',axis=1)
print("The column names after dropping place of birth column")
print(df.columns)
print("The statistical describe of dataset:")
#It prints the descriptive statistics of the DataFrame df using the describe() function.
print (df.describe())
#It creates scatter plots for each categorical variable (gender, Relation, Topic, etc.) against the target variable Class using seaborn's scatterplot() function. Each plot shows the relationship between a categorical variable and the target variable.
#ls = ['gender','Relation','Topic','SectionID','GradeID','NationalITy','Class','StageID','Semester','ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays']
ls=[ 'Section', 'Age', 'Parents Education', 'Parents Occupation',
       'InterestingTopic', 'Caste',  'Absent Days',
       'ParentschoolSatisfaction', 'Semester',
       'No. of times visited library in this sem', 'Single Parents(Y/N)',
       'No. of study materials refering from Online resources', 'Backlog',
       'Previous CGPA', 'Family support for education(Y/N)', 'District',
       'Hobby']
for i in ls:
    g= sns.catplot(x=i,data=df,kind='count')
print("The shapes of dataset:")
print (df.shape)

#preprocessing
#It prepares the data for modeling by separating the target variable (Class) from the input features (X) using the pop() function.
target = df.pop('class')
#It then applies one-hot encoding to convert categorical variables into numerical format using get_dummies() function.
X = pd.get_dummies(df)
#It encodes the target variable Class using LabelEncoder() to convert it into numerical format.
le = LabelEncoder()
y = le.fit_transform(target)
#It splits the dataset into training and testing sets using train_test_split() function with a test size of 30% and sets the random state to ensure reproducibility.
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)
#It standardizes the features using StandardScaler() to transform the data to have a mean of 0 and a standard deviation of 1.
ss = StandardScaler()
#print X_train
X_train_std = ss.fit_transform(X_train)
X_test_std = ss.fit_transform(X_test)
#It prints the feature importance scores sorted in descending order.

#dimensionality_reduction






feat_labels = X.columns[:70]
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("Top features:")
for f in range(X_train.shape[1]):
    if indices[f] < len(feat_labels) and indices[f] < len(importances):
        print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

# Plotting feature importances
plt.figure(figsize=(8, 8))
sns.barplot(y=importances[indices], x=feat_labels[indices])
plt.title('Feature Importances')
plt.xlabel('Features', fontsize=14, fontweight='bold', rotation=90)  # Replace ylabel with xlabel
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)  # Add grid lines
plt.tight_layout()  # Adjust layout to prevent overlapping
plt.show()




"""feat_labels = X.columns[:89]
forest = RandomForestClassifier(n_estimators=1000,random_state=0,n_jobs=-1)
forest.fit(X_train,y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print("the vale of X_train.shape[1]",X_train.shape[1])
"""#for f in range(X_train.shape[1]):
    #print("%2d) %-*s %f" % (f + 1, 30,feat_labels[indices[f]],importances[indices[f]]))"""
#for f in range((X_train.shape[1])):
   # if indices[f] < len(feat_labels) and indices[f] < len(importances):
""" print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

h = sns.barplot(importances[indices])#y=feat_labels[indices])
plt.show()"""
#removing dimensions

X_train_new = X_train
X_test_new = X_test
#print("the X_train_new columns:",X_train_new.columns)
#print("the X_test_new columns:",X_test_new.columns)
#c1=set(X_train_new.columns)
#c2=set(X_test_new.columns)
#print("the union of 
#Wprint("union",c1.intersection(c2))


ls = ['No. of study materials refering from Online resources','No. of times visited library in this sem','Backlog',
      'Absent Days','Previous cGPA']

for i in X_train.columns:
    if i in ls:
        pass
    else:
        X_train_new.drop(i , axis=1, inplace=True)

for i in X_test.columns:
    if i in ls:
        pass
    else:
        X_test_new.drop(i , axis=1, inplace=True)


#spot checking algorithms

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

#accuracy=[]
# evaluate each model in turn
results = []
names = []
print("The mean and standard deviation of Linear Regression algorithm")
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7,shuffle=True)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Calculate additional evaluation metrics for classification algorithms
"""for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='accuracy')
    accuracy = cv_results.mean()
    msg = "%s Accuracy: %f" % (name, accuracy)
    print(msg)"""

# Standardize the dataset
pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledLASSO', Pipeline([('Scaler', StandardScaler()),('LASSO',Lasso())])))
pipelines.append(('ScaledEN', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN',KNeighborsRegressor())])))
pipelines.append(('ScaledCART', Pipeline([('Scaler', StandardScaler()),('CART',DecisionTreeRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVR', SVR())])))
results = []
names = []
print("The mean and standard deviation of  scaled  Regression algorithm")
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=7,shuffle=True)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    # Calculate additional evaluation metrics for classification algorithms
"""for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='accuracy')
    accuracy = cv_results.mean()
    msg = "%s Accuracy: %f" % (name, accuracy)
    print(msg)
"""

fig = plt.figure()
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names)
plt.show()

#lasso algorithm tuning
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
k_values = np.array([.1,.11,.12,.13,.14,.15,.16,.09,.08,.07,.06,.05,.04])
param_grid = dict(alpha=k_values)
model = Lasso()
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)


print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
print(" The mean test score and std test score , params of tunned lasso algorithm:")
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#using ensembles
ensembles = []
ensembles.append(('ScaledAB', Pipeline([('Scaler', StandardScaler()),('AB',AdaBoostRegressor())])))
ensembles.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GBM',GradientBoostingRegressor())])))
ensembles.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF',RandomForestRegressor())])))
ensembles.append(('ScaledET', Pipeline([('Scaler', StandardScaler()),('ET',ExtraTreesRegressor())])))
results = []
names = []
print("The mean and standard deviation of Ensemble Methods:  ")
for name, model in ensembles:
    kfold = KFold(n_splits=10, random_state=7,shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    # Calculate additional evaluation metrics for regression algorithms
"""for name, model in models:
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    cv_results = cross_val_score(model, X_train_new, y_train, cv=kfold, scoring='neg_mean_squared_error')
    mse = -cv_results.mean()
    msg = "%s Mean Squared Error: %f" % (name, mse)
    print(msg)
"""
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Scaled Ensemble Algorithm Comparison')
#ax = fig.add_subplot(111)

plt.boxplot(results)
plt.xticks(range(1, len(names) + 1), names)
plt.show()

# Tune scaled AdaboostRegressor
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,150,200,250,300,350,400]))
model = AdaBoostRegressor(random_state=7)
kfold = KFold(n_splits=10, random_state=7,shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kfold)
grid_result = grid.fit(rescaledX, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
print("The mean and standard deviation ,params of  tunned AdaboostRegressor:")
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
# prepare the model
scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=7, n_estimators=400)
model.fit(rescaledX, y_train)

# transform the validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)
print("Ytest",y_test)
print("predictions",predictions)
print("MSE:")
print(mean_squared_error(y_test,predictions))



"""# Define threshold values for classification
thresholds = {
    'good': 80,
    'satisfactory': 60,
    'average': 40,
    'bad': 0
}"""

# Predict using the best-performing model (Gradient Boosting Regressor)
"""scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
model = GradientBoostingRegressor(random_state=7, n_estimators=400)
model.fit(rescaledX, y_train)

# Transform the validation dataset
rescaledValidationX = scaler.transform(X_test)
predictions = model.predict(rescaledValidationX)"""
#from sklearn.metrics import mean_absolute_error

# Train the best algorithm on the entire training dataset
#best_model = GradientBoostingRegressor(random_state=7, n_estimators=400)
#best_model.fit(X_train, y_train)

# Make predictions on the testing dataset
#predictions = model.predict(df.values)
#model.fit(df.values)
# Calculate Mean Absolute Error (MAE)
#mae = mean_absolute_error(X_test, predictions)

# Calculate accuracy rate for each student
"""accuracy_rates = []
for i, actual in enumerate(predictions):
    pred = y_test[i]
    accuracy = 1 - abs(actual - pred) / actual  # Accuracy rate calculation
    accuracy_rates.append(accuracy)

# Print accuracy rate for each student
for i, accuracy in enumerate(accuracy_rates):
    print("Student %d: Accuracy Rate = %.2f%%" % (i+1, accuracy * 100))"""
"""
accuracy_rates = []
for i, pred in enumerate(predictions):
    actual = df[i]
    accuracy = 1 - abs(actual - pred) / actual  # Accuracy rate calculation
    accuracy_rates.append(accuracy)

# Print accuracy rate for each student
for i, accuracy in enumerate(accuracy_rates):
    print("Student %d: Accuracy Rate = %.2f%%" % (i+1, accuracy * 100))
# Classify student performance based on predictions"""
"""performance_labels = []
for prediction in predictions:
    for label, threshold in thresholds.items():
        if prediction >= threshold:
            performance_labels.append(label)
            break  # Once the label is assigned, break out of the loop

# Print the performance labels for each student
for index, label in enumerate(performance_labels):
    print("Student", index + 1, "performance:", label)
import matplotlib.pyplot as plt

# Count occurrences of each performance category
performance_counts = {label: performance_labels.count(label) for label in thresholds.keys()}

# Plotting pie chart
plt.figure(figsize=(8, 8))
plt.pie(performance_counts.values(), labels=performance_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Student Performance Categories')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

"""

    # You can calculate other regression metrics like Mean Absolute Error (MAE) and R-squared score as well
