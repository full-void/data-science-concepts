# Data Science Concepts

This repository contains Jupyter notebooks detailing Data Science Concepts and my notes on them. 

It should be useful for both a quick review and a reference to fall back upon.

The notebooks contain both vital theory and practical examples with real-life datasets.

The details in the notebooks discussed are:

## Linear Regression Model Development
This notebook discusses on creation of Linear Regression Models as a way of prediction. Specifically, it discusses:
* Simple Linear Regression
    - Getting the equation
* Multiple Linear Regression
* Model evaluation with Visualization
    - Regression Plot
    - Residual Plot
    - Distribution Plot for MLR
* Polynomial Regression
* Pipelines
* Regularization - L2 or Ridge Regression
* Grid Search
* In-sample Evaluation metrics like R-squared, RMSE, MAE
* Determining a good model

## Exploratory Data Analysis

This notebook details some exploratory data analysis approaches. I also go through a realworld database on cars and conduct the analysis to find important variables to predict price. Specifically, it talks about:

* Loading data and finding important metadata about column types.
* Finding relationship between numerical data using Pearson correlation.
    - Visualising relationship using regression plots.
* Finding relationship between numerical and categorical data.
    - Using boxplots to visualize the data.
    - Conducting ANOVA analysis to check for statistically significant relationship.
* Descriptive Statistical Analysis
    - Using pandas' `describe()` method to get important statistics such as 5 number summary and other statistically important values.
    - Finding counts of unique values and making decisions on legibility of the data.
* Grouping of the data.
    - Finding important information such as means of the data divided by groups.
    - Visualizing using Heatmaps
* P-values and correlation

## Model Evaluation and Refinement

This notebook talks about methods to evaluate a model. It talks about:

* Training and Testing
* Evaluation with metrics (such as Jaccard Index, F1-score, RMSE or R-squared)
* K-Fold Cross-validation
    - Getting descriptive stats
    - Using various metrics
* Overfitting, Underfitting and Model Selection

## Logistic Regression

This notebook discusses Logistic Regression as a way to approach classification problems. It details upon:

* What is Logistic Regression
    - Difference From Linear Regression
    - Advantages
    - Disadvatages
* Creating the Model
    - About the dataset, Loading data, Pre-processing
    - Train/Test
    - Modeling with sklearn
    - Getting Probability for classes
    - Evaluation
        + Jaccard Index
        + Confusion Matrix
        + Classification Report

## Decision Trees

This notebook discussed Decision Trees as a way to approach classification problems. Specifically, it details on:

* What is  Decision Tree?
    - Advantages
    - Disadvantages
* Practical Problem
* The Data
    - Pre-processing
    - Label Encoding
    - Split into train/test
* Modeling
* Prediction and Evaluation
* Grid Search
* Evaluation of Best Decision Tree from Grid Search
* Conclusion

## K Nearest Neighbors

This notebook contains info about K Nearest Neighbors algorithm for classification problems. It goes in detail about:

* What is K-Nearest Neighors Algorithm
    - Advantages
    - Disadvantages
    - Choosing K
* Creating the Model
* Evaluation
* Tuning K

## Data Wrangling

This notebook contains concepts and notes on data wrangling: preparing data for further analysis and model development. We go through a life-like example and clean a data from the UCI Machine Learning Repository. Specifically, it discusses:

* Setting Headers
* Correcting the Data
    - Identifying the missing values
        + Converting other markers (such as ?) to NAN
        + Analysing for missing data
    - Dealing with missing values
        + Dropping data
        + Replacing data with mean and frequency
    - Correcting the data format
* Data Standardization
* Data Noramalization
* Binning
    - Visualizing Bins
* Encoding Categorical Variables
    - Label Encoding
    - One-hot Encoding

## SQL and Pandas

This notebook details some important and commonly used information on Pandas Python Library and SQL. Specifically, it talks about:

* Connecting to a database with Magic Commands in IPython SQL Interface.
* Reading the dataset.
* Storing the dataset in a database.
* Getting data from database and converting it to a pandas dataframe.
* Filtering data with multiple and single conditions (both Pandas and SQL)
* High-level description of data (Pandas and SQL)
* Sorting (Pandas and SQL)
* Useful Pandas functions