# Census_Income_Prediction

**Project Summary**

The Census Income Prediction Project is an in-depth exploration of machine learning applied to a dataset derived from the 1994 US census. With over 48,000 records, the project aims to predict whether an individual earns more or less than $50,000 annually based on various socio-economic attributes. It encompasses data loading, exploratory data analysis (EDA), data manipulation, visualization, preprocessing, and the implementation of multiple machine learning models for predictive classification.

**Dataset Overview**

The dataset provides a diverse range of features, including age, education level, workclass, marital status, occupation, race, sex, and more. The target variable is annual income, categorized as above or below $50,000. This project utilizes the dataset to gain insights into the socio-economic landscape and build predictive models for income classification.

**Exploratory Data Analysis (EDA)**

Initial steps involve loading the dataset and conducting EDA to understand its structure and characteristics. Techniques like checking shape, handling duplicates, and renaming columns provide a robust foundation. Exploration of unique values and manipulations, including filtering and extraction, aids in understanding the dataset.

**Data Visualization**

Visualizations play a crucial role in understanding distribution and relationships within the dataset. Using Seaborn and Matplotlib, the project generates insights into age distribution, education levels, relationship statuses, occupation distribution, and income distribution by workclass. These visualizations aid in identifying patterns and trends.

**Data Preprocessing**

Preprocessing ensures the data is ready for machine learning models. Handling missing values, replacing placeholders, and encoding categorical variables are vital steps. The project replaces '?' values with 'unknown' and uses Label Encoding to convert categorical variables into a numerical format.

**Machine Learning Models**

The predictive task involves implementing various machine learning algorithms to classify individuals into income groups. The project explores Linear Regression, Logistic Regression, Decision Tree Classifier, Random Forest Classifier, XGBoost Classifier, and Support Vector Machine (SVM).

**Linear Regression**

A simple linear regression model predicts hours worked per week based on education level. The model's performance is assessed through Root Mean Squared Error (RMSE), offering insights into the relationship between education and working hours.

**Logistic Regression**

Simple and multiple logistic regression models predict income levels. Models are evaluated in terms of accuracy, classification reports, and confusion matrices, providing a nuanced understanding of their predictive capabilities.

**Decision Tree Classifier**

A decision tree classifier predicts income groups. The model's performance is assessed using accuracy, classification reports, and confusion matrices.

**Random Forest Classifier**

The Random Forest Classifier, a more complex model, is implemented and evaluated. Feature importances are analyzed to understand variables influencing predictions.

**XGBoost Classifier**

The XGBoost Classifier, known for its performance, is explored. The model's predictive capabilities are evaluated using accuracy, classification reports, and confusion matrices.

**Support Vector Machine (SVM)**

An SVM classifier predicts income levels. The model's performance is assessed based on accuracy, classification reports, and confusion matrices.

**Model Evaluation and Selection**

All implemented models are rigorously evaluated, and results are compared. Confusion matrices visualize true positives, true negatives, false positives, and false negatives. The model with the highest accuracy is selected as the most promising for predicting income levels.

**Conclusion**

The Census Income Prediction Project serves as a comprehensive guide to exploring and predicting income levels based on socio-economic attributes. It covers every step of the data science pipeline, from data loading to model evaluation. Insights from exploratory data analysis illuminate the demographic landscape, while machine learning models offer predictive capabilities.

Recommendations for further improvements, such as feature engineering or additional data sources, are discussed. The project provides a valuable resource for data science enthusiasts, practitioners, and anyone interested in leveraging machine learning for socio-economic predictions.
