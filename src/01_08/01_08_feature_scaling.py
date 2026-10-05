import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder,MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv("input/insurance.csv")

# Display the first 15 rows of the dataset
print("First 15 rows of the dataset:")
print(data.head(15))

# Handling Missing Values
# Option 3: Fill missing values with mean (SimpleImputer)
imputer = SimpleImputer(strategy="mean")
data["bmi"] = imputer.fit_transform(data[["bmi"]])
print("\nOption 3: Fill missing values with mean (SimpleImputer)")
print(data.isnull().sum())

# Encoding Categorical Variables
# Label encode 'sex' and 'smoker'
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['smoker'] = le.fit_transform(data['smoker'])

# One hot encode 'region'
ohe = OneHotEncoder(sparse_output=False, drop='first')
region_encoded = ohe.fit_transform(data[['region']])
region_columns = ohe.get_feature_names_out(['region'])
region_df = pd.DataFrame(region_encoded, columns=region_columns)

# Combine numerical and encoded columns
X_num = data[['age', 'bmi', 'children']].copy()
X_final = pd.concat([X_num, region_df, data['sex'], data['smoker']], axis=1)

# Assign response variable
y_final = data['charges']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=0)

# TODO: Normalize the training and test sets using MinMaxScaler
n_scaler = MinMaxScaler()
X_train_normalized = n_scaler.fit_transform(X_train)#fit_transform does two things: it learns the scaling limits from the data, then scales that data. For X_train, MinMaxScaler learns each feature’s minimum and maximum and maps them to the 0–1 range
X_test_normalized = n_scaler.transform(X_test)#transform() is used to scale the test data using the scaling limits learned from the training data. It applies the same scaling transformation to the test data based on the minimum and maximum values learned from the training data.
print("\nNormalized training data datasets:\t",X_train_normalized)
print("\nNormalized test data datasets:\t",X_test_normalized)
# TODO: Standardize the training and test sets using StandardScaler
s_scaler = StandardScaler()
X_train_standardized = s_scaler.fit_transform(X_train)#fit_transform does two things: it learns the scaling limits from the data, then scales that data. For X_train, StandardScaler learns each feature’s mean and standard deviation and maps them to the 0–1 range
X_test_standardized = s_scaler.transform(X_test)#transform() is used to scale the test data using the scaling limits learned from the training data. It applies the same scaling transformation to the test data based on the mean and standard deviation values learned from the training data.
print("\nStandardized training data datasets:\t",X_train_standardized)
print("\nStandardized test data datasets:\t",X_test_standardized)
