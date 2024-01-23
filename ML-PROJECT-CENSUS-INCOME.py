#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import seaborn as sns


# In[2]:


df = pd.read_csv('census-income.csv')
df.head()


# # Exploratory Data Analysis (EDA)

# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.duplicated().sum()


# In[7]:


df.drop_duplicates(inplace=True)


# In[8]:


df = df.rename(columns={'Unnamed: 14': 'Annual-income'})
df.head()


# In[9]:


df.isnull().sum()


# In[10]:


df.apply(lambda x: x.unique())


# # DATA MANIPULATION

# In[11]:


df[(df['workclass']=='Private') & (df['native-country']!='United-States')].shape


# In[12]:


# Extracting the “education” column and storing it in “census_ed
census_ed=df['education']
census_ed.head(5)


# In[13]:


# Extracting all the columns from “age” to “relationship” and storoing it in “census_seq”
census_seq=df.iloc[:,:8]
census_seq.head(5)


# In[14]:


# Extracting the column number “5”, “8”, “11” and storing it in “census_col”.
census_col=df.iloc[:,[5,8,11]]
census_col.head(5)


# In[15]:


df.head()


# In[16]:


# Extracting all the male employees who work in state-gov and storing it in “male_gov”.
male_gov=df[(df['sex']=='Male') & (df['workclass']=='State-gov')]
male_gov.head(5)


# In[17]:


# Extracting all the 39 year olds who have bachelor's degree or who are native of US and storing the result in “census_us”.
census_us=df[(df['age']==39)& ((df['education']=='Bachelors') |(df['native-country']=='United-States'))]
census_us.head(5)


# In[18]:


# Extracting 200 random rows from the “census” data frame and storing it in “census_200”.
census_200=df.sample(200)
census_200.head(5)


# In[19]:


# Get the count of different levels of the “workclass” column
df['workclass'].value_counts()


# In[20]:


# Calculate the mean of the “capital.gain” column grouped according to “workclass”.
df.groupby('workclass')['capital-gain'].mean()


# In[21]:


# Create a separate dataframe with the details of males and females from the census data that has income more than 50,000.
high_income_df = df[df['Annual-income'] == '>50K']  # Filtering data for income more than 50,000
males_df = high_income_df[high_income_df['sex'] == 'Male']  # Filtering data for males
females_df = high_income_df[high_income_df['sex'] == 'Female']  # Filtering data for females

# Print the separate dataframes
print("i) Separate DataFrame for Males with Income >50K:")
print(males_df.head(5))


# In[22]:


print("\nSeparate DataFrame for Females with Income >50K:")
print(females_df.head(5))


# In[23]:


# Calculate the percentage of people from the United States who are private employees and earn less than 50,000 annually.
us_priv_less50k = df[(df['native-country'] == 'United-States') & (df['workclass'] == 'Private') & (df['Annual-income'] == '<=50K')]
per_us_priv_less50k = (len(us_priv_less50k) / len(df[df['native-country'] == 'United-States'])) * 100

# Print the result
print("\nj) Percentage of people from the United States who are private employees and earn less than 50,000 annually:")
print(per_us_priv_less50k)


# In[24]:


# Calculate the percentage of married people in the census data.
married_percentage = (len(df[df['marital-status'].str.startswith('Married')]) / len(df)) * 100

# Print the result
print("\nk) Percentage of married people in the census data:")
print(married_percentage)


# In[25]:


# Calculate the percentage of high school graduates earning more than 50,000 annually.
hs_graduates_more50k = df[(df['education'] == 'HS-grad') & (df['Annual-income'] == '>50K')]
per_hs_graduates_more50k = (len(hs_graduates_more50k) / len(df[df['education'] == 'HS-grad'])) * 100

# Print the result
print("\nl) Percentage of high school graduates earning more than 50,000 annually:")
print(per_hs_graduates_more50k)


# # DATA VISUALIZATION

# In[26]:


# Set the style for seaborn
sns.set(style="whitegrid")

# Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(df['age'], bins=20, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()
print("The histogram shows the distribution of ages in the dataset, providing insights into the age demographics.")


# In[27]:


# Education Level Counts
plt.figure(figsize=(12, 6))
sns.countplot(x='education', data=df, palette='viridis')
plt.title('Education Level Counts')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
print("This countplot displays the distribution of education levels, helping to understand the education" 
      "background of individuals.")


# In[28]:


# Relationship Status Pie Chart
plt.figure(figsize=(8, 8))
df['relationship'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Distribution of Relationship Status')
plt.show()
print("The pie chart illustrates the distribution of relationship statuses," 
      "providing a visual representation of the proportions.")


# In[29]:


# Occupation Bar Chart
plt.figure(figsize=(12, 6))
sns.countplot(x='occupation', data=df, palette='muted')
plt.title('Occupation Distribution')
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()
print("This bar chart shows the distribution of different occupations, offering insights into the" 
      "diversity of occupations in the dataset.")


# In[30]:


# Income Distribution by Workclass
plt.figure(figsize=(14, 8))
sns.violinplot(x='workclass', y='hours-per-week', hue='Annual-income', data=df, split=True, inner='quart', palette='Set2')
plt.title('Income Distribution by Workclass and Weekly Hours')
plt.xlabel('Workclass')
plt.ylabel('Hours per Week')
plt.xticks(rotation=45)
plt.show()
print("The violin plot visualizes the distribution of weekly working hours for different workclasses," 
      "segmented by income levels. It helps identify patterns and variations in the data.")


# # DATA PREPROCESSING

# In[31]:


value_to_count = '?' 
df.isin([value_to_count]).any()


# In[32]:


count_of_question_marks = (df == value_to_count).sum().sum()
count_of_question_marks


# In[33]:


# Replace '?' with 'unknown'
df.replace(value_to_count, 'unknown', inplace=True)
df.apply(lambda x: x.unique())


# Encoding categorical data (independent variables)

# In[34]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
df['workclass'] = labelencoder.fit_transform(df['workclass'])
df['education'] = labelencoder.fit_transform(df['education'])
df['marital-status'] = labelencoder.fit_transform(df['marital-status'])
df['occupation'] = labelencoder.fit_transform(df['occupation'])
df['relationship'] = labelencoder.fit_transform(df['relationship'])
df['race'] = labelencoder.fit_transform(df['race'])
df['sex'] = labelencoder.fit_transform(df['sex'])
df['native-country'] = labelencoder.fit_transform(df['native-country'])


# Encoding categorical data (dependent variables)

# In[35]:


df['Annual-income'] = labelencoder.fit_transform(df['Annual-income'])


# # MACHINE LEARNING MODELS

# Linear Regression

# In[36]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Taking 'education.num' as the independent variable and 'hours-per-week' as the dependent variable
X = df[['education-num']]
y = df['hours-per-week']


# In[37]:


# Dividing the dataset into training and test sets in a 70:30 ratio.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Displaying the shapes of training and test sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# In[38]:


# Train a simple linear regression model
linear_reg = LinearRegression()
linear_reg.fit(X_train, y_train)


# In[39]:


# Predict values on the training set
y_train_pred = linear_reg.predict(X_train)

# Calculate the error in prediction
error_train = y_train - y_train_pred

# Display the first few rows of actual values, predicted values, and errors
prediction_results_train = pd.DataFrame({'Actual': y_train, 'Predicted': y_train_pred, 'Error': error_train})
print(prediction_results_train.head())


# In[40]:


# Predict values on the test set
y_test_pred = linear_reg.predict(X_test)

# Calculate the RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

# Display the RMSE
print("Root Mean Squared Error (RMSE):", rmse)


# # Logistic Regression

#  Simple Logistic Regression

# In[41]:


# Divide the dataset into training and test sets in 65:35 ratio
X_train, X_test, y_train, y_test = train_test_split(df['occupation'], df['Annual-income'], test_size=0.35, random_state=42)


# In[42]:


# Build a logistic regression model
from sklearn.linear_model import LogisticRegression
logis_simple = LogisticRegression()
logis_simple.fit(X_train.values.reshape(-1, 1), y_train)


# In[43]:


# Predict values on the test set
y_pred = logis_simple.predict(X_test.values.reshape(-1, 1))


# In[44]:


# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[45]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='navy')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='darkred')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# Multiple Logistic Regression with few independent variables

# In[46]:


# Divide the dataset into training and test sets in 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(
    df[['age', 'workclass', 'education']], df['Annual-income'], test_size=0.20, random_state=42)


# In[47]:


# Build a logistic regression model with multiple independent variables
logis_multi = LogisticRegression()
logis_multi.fit(X_train, y_train)

# Predict values on the test set
y_pred = logis_multi.predict(X_test)


# In[48]:


# Evaluate the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[49]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='navy')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='darkred')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# Multiple Logistic Regression with all the independent variables

# In[50]:


X=df.iloc[:,:-1]
X


# In[51]:


y=df.iloc[:,-1]
y


# In[52]:


# SPLITTING THE DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[53]:


# Standardisation
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[54]:


# Initialize the Logistic Regression model & Train the model
logis = LogisticRegression(random_state=42)
logis.fit(X_train_scaled, y_train)
# Make predictions on the testing set
y_pred = logis.predict(X_test_scaled)


# In[55]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[56]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='#2E4053')  # Dark blue
sns.kdeplot(y_pred, label='Predicted', fill=True, color='#D35400')  # Dark orange
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# Decision Tree Classifier

# In[57]:


# Initialize the Random Forest model & Train the model
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)


# In[58]:


# Make predictions on the testing set
y_pred = dt.predict(X_test)


# In[59]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[60]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='navy')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='darkred')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# Random Forest Classifier

# In[61]:


# Initialize the Random Forest model & Train the model
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=12,random_state=42)
rfc.fit(X_train, y_train)
# Make predictions on the testing set
y_pred = rfc.predict(X_test)


# In[62]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[63]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='navy')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='darkred')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# Feature Importances

# In[64]:


# Access feature importances
feature_importances = rfc.feature_importances_

# Display feature importances
for feature, importance in zip(X.columns, feature_importances):
    print(f"{feature}: {importance}")


# Random Forest Hyperparameter Tuning with GridSearchCV 

# In[65]:


from sklearn.model_selection import GridSearchCV
# Define the hyperparameters and their possible values to search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy')

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("Best hyperparameters:", grid_search.best_params_)

# Evaluate the model on the test set
accuracy = grid_search.score(X_test, y_test)
print("Accuracy on the test set:", accuracy)


# Random Forest Hyperparameter Tuning with RandomizedSearchCV

# In[66]:


from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from scipy.stats import randint

# Define the hyperparameters and their possible values to search
param_dist = {
    'classifier__n_estimators': randint(50, 200),
    'classifier__max_depth': [None] + list(range(10, 31)),
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Create a ColumnTransformer for preprocessing (replace this with your specific preprocessing steps)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']),
        ('cat', 'passthrough', ['workclass', 'marital-status', 'occupation'])
    ])

# Create a Pipeline with preprocessing and the classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)

# Fit the model to the training data
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
accuracy = random_search.score(X_test, y_test)
print("Accuracy on the test set:", accuracy)


# XGB Classifier

# In[67]:


from xgboost import XGBClassifier
# Create an XGBoost classifier
xgb_classifier = XGBClassifier(random_state=42, objective='binary:logistic')

# Train the XGBoost model
xgb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = xgb_classifier.predict(X_test)


# In[68]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[69]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='navy')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='darkred')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# XGB Claasifier Hyperparameter Tuning with RandomizedSearchCV

# In[70]:


# Define the hyperparameters and their possible values to search
param_dist = {
    'classifier__n_estimators': randint(100, 300),
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10, None],
    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
    'classifier__subsample': [0.8, 0.9, 1.0],
    'classifier__min_child_weight': [1, 2, 3, 4]
}


# In[ ]:





# In[71]:


# Create a Pipeline with preprocessing and the classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42, objective='binary:logistic'))
])

# Create a RandomizedSearchCV object
random_search = RandomizedSearchCV(
    pipeline, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)


# In[72]:


# Fit the model to the training data
random_search.fit(X_train, y_train)

# Print the best parameters found
print("Best hyperparameters:", random_search.best_params_)

# Evaluate the model on the test set
accuracy = random_search.score(X_test, y_test)
print("Accuracy on the test set:", accuracy)


# Support Vector Machine (SVM)

# In[73]:


from sklearn.svm import SVC
from scipy.stats import loguniform
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy.stats import randint

# Create an SVM classifier
svm_classifier = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)

# Create a full pipeline with preprocessing and the classifier
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', svm_classifier)
])

# Train the SVM model
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)


# In[74]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[75]:


# Create a figure and axis
plt.figure(figsize=(10, 6))

# Plot the kernel density estimate for actual and predicted values with dark colors
sns.kdeplot(y_test, label='Actual', fill=True, color='navy')
sns.kdeplot(y_pred, label='Predicted', fill=True, color='darkred')
plt.xlabel('Values')
plt.ylabel('Density')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.show()


# # SELECTING THE BEST MODEL

# In[76]:


# Initialize lists to store results
models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVM']
accuracies = []
conf_matrices = []
class_reports = []

# Function to evaluate model and store results
def evaluate_model(model_name, model, X_test, y_test):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    accuracies.append(accuracy)
    conf_matrices.append(conf_matrix)
    class_reports.append(class_report)

# Evaluate each model
evaluate_model('Logistic Regression', logis, X_test_scaled, y_test)
evaluate_model('Decision Tree', dt, X_test, y_test)
evaluate_model('Random Forest', rfc, X_test, y_test)
evaluate_model('XGBoost', xgb_classifier, X_test, y_test)
evaluate_model('SVM', pipeline, X_test, y_test)

# Create a DataFrame for results
results_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'Confusion Matrix': conf_matrices,
    'Classification Report': class_reports
})

# Print the DataFrame
print(results_df)


# In[77]:


# Visualize confusion matrices using heatmaps
plt.figure(figsize=(10, 5))
for i, model_name in enumerate(models, start=1):
    plt.subplot(2, 3, i)
    sns.heatmap(conf_matrices[i-1], annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')

plt.tight_layout()
plt.show()


# In[78]:


# Select the best model based on the highest accuracy
best_model_index = results_df['Accuracy'].idxmax()
best_model_name = results_df.loc[best_model_index, 'Model']
best_model_accuracy = results_df.loc[best_model_index, 'Accuracy']

print(f"The best model is {best_model_name} with an accuracy of {best_model_accuracy:.2f}")


# In[ ]:




