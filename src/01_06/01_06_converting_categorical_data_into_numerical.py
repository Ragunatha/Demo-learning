import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values
# Option 3: Fill missing values with mean (SimpleImputer)
# Reload dataset to ensure fresh state
imputer = SimpleImputer(strategy="mean")
data["bmi"] = imputer.fit_transform(data[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data.isnull().sum())
#In this I am just converting categorical data into numerical data using Label Encoding and One Hot Encoding. Label Encoding is used to convert categorical data into numerical data by assigning a unique integer to each category. One Hot Encoding is used to convert categorical data into numerical data by creating a new binary column for each category.
#for predictive analysis we need to convert categorical data into numerical data because most of the machine learning algorithms can only work with numerical data. So, we need to convert categorical data into numerical data before feeding it to the machine learning algorithms.
# Label Encoding: Encode 'sex' and 'smoker' columns
# TODO: Create a label encoder instance and encode the 'sex' column
le_sexcolumn=LabelEncoder()#definig the label encoder instance for 'sex' column
data["sex"] = le_sexcolumn.fit_transform(data["sex"])
print("\nSKlearn Label Encoding for 'sex' column:")
print(dict(zip(le_sexcolumn.classes_, le_sexcolumn.transform(le_sexcolumn.classes_))))#mapping of each category to its corresponding integer value. Here we are using zip() function to create a dictionary of the mapping of each category to its corresponding integer value.
#The zip() function takes two or more iterables as input and returns an iterator that generates tuples containing elements from the input iterables. In this case, we are using zip() function to create a dictionary of the mapping of each category to its corresponding integer value.
print(data["sex"].head(10))
# TODO: Create a label encoder instance and encode the 'smoker' column
le_smokercolumn=LabelEncoder()#definig the label encoder instance for 'smoker' column
data["smoker"] = le_smokercolumn.fit_transform(data["smoker"])
print("\nSKlearn Label Encoding for 'smoker' column:")
print(dict(zip(le_smokercolumn.classes_, le_smokercolumn.transform(le_smokercolumn.classes_))))
print(data["smoker"].head(10))
# One Hot Encoding: Encode the 'region' column
#Createing a one hot encoder instance and encode the 'region' column
Ohe=OneHotEncoder(sparse_output=False,drop='first')#defining the one hot encoder instance for 'region' column. sparse=False means we are returning a dense array instead of a sparse matrix.
region_encoded = Ohe.fit_transform(data[["region"]])#fit_transform() is used to fit the one hot encoder on the data and transform the data. Here we are fitting the one hot encoder on the 'region' column and transforming the 'region' column.
region_columns = Ohe.get_feature_names_out(["region"])#get_feature_names_out() is used to get the feature names of the one hot encoded data. Here we are getting the feature names of the one hot encoded 'region' column.
# TODO: Convert the result into a DataFrame with appropriate column names
region_df = pd.DataFrame(region_encoded,columns=region_columns)#converting the one hot encoded data into a DataFrame with appropriate column names.
data.drop(columns=["region"],inplace=True)#dropping the original 'region' column from the dataset.inplace=True means pandas changes data directly. You don’t need to assign the result back.
data=pd.concat([data,region_df],axis=1)#concatenating the original dataset with the one hot encoded 'region' column. axis=1 means we are concatenating the dataframes column-wise.
print("\nSKlearn One Hot Encoding for 'region' column:")
print(region_df.head(10))

#Display the updated dataframe after one hot encoding
print("\nUpdated DataFrame after One Hot Encoding:")
print(data.head(15))

