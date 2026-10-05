# Load necessary libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values

# TODO: Check how many values are missing (NaN)
print("How many values missing?")
print(data.isnull().sum())#this gives the number of missing values in each column with their sum

# Option 1: Drop the entire column with missing values
# TODO: Add code to drop the 'bmi' column and verify
data_option1=data.copy()#which like taking the backup of the original data to avoid losing it
data_option1.drop(['bmi'],axis=1, inplace=True)#axis=1 means we are dropping the 'column' and inplace=True means we are modifying the original dataframe which cannot be restored.
print("\nOption 1: Drop the 'bmi' column")
print(data_option1.isnull().sum())
# Option 2: Drop rows with missing values
# TODO: Add code to drop rows with missing values and verify
data_option2=data.copy()
data_option2.dropna(inplace=True)#dropna() drops all the rows with missing values.
print("\nOption 2: Drop rows with missing values")
print(data_option2.isnull().sum())

# Option 3: Fill missing values with mean (SimpleImputer)
# TODO: Add code to fill missing values in the 'bmi' column using SimpleImputer
data_option3=data.copy()
imputer = SimpleImputer(strategy="mean")#SimpleImputer is used to fill the missing values with mean, median, or most frequent value. Here we are using mean.In which through the mean we going to fill the missing values in the 'bmi' column..
data_option3["bmi"] = imputer.fit_transform(data_option3[["bmi"]])#fit_transform() is used to fit the imputer on the data and transform the data. Here we are fitting the imputer on the 'bmi' column and transforming the 'bmi' column.
print("\nOption3: Fill missing values with mean (SimpleImputer)")
print(data_option3.isnull().sum())

