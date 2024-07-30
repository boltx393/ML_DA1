# This is the **Digital Assessment** - 1 for the Laboratory course for Machine Learning (BCSE209L) <br> 
# ***
# The task is to create a dataset and then perform: <br>
# 1 - Data Manipulation <br>
# 2 - Data Pre-Processing <br>
# 3 - Implement Find-S and Candidate Elimination Algorithm <br>

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv(r'C:\Users\iamaa\Downloads\database.csv')
df

# **1 - Data Manipulation Operations** <br>
# <br>
# A - Insert New Sample <br>
# B - Delete Particular Sample <br>
# C - Update Particular Sample <br>

# A - Insert New Sample 

# New sample
new_sample = pd.DataFrame({
    'Name': ['Raghav Mittal'],
    'Branch': ['Computer Science'],
    'Degree': ['B.Tech'],
    'Summer Internship': ['No'],
    'Credits Completed': [125],
    'Credits Required': [151],
    'Backlog/Arrear': ['Yes'],
    'Age': [22],
    'CGPA': [7.5],
    'Eligible for placements': ['No']
})

# Concatenate the new sample
df = pd.concat([df, new_sample], ignore_index=True)
df

# b - Delete Particular Sample 

# Deleting the record with name Swati Kumar 

df = df[df['Name'] != 'Swati Kumar']
df

# c - Update Particular Sample 

# Updating the CGPA of Jai Patel to 8.5

df.loc[df['Name'] == 'Jai Patel', 'CGPA'] = 8.5
df

# **2 - Data Pre-Processing** <br><br>
# 
# A - Find Number of Missing Value <br>
# B - Replace missing values by mean, median and mode operations <br> 
# C - Apply encoding techniques for: <br> 
#     i. independent variables <br> 
#     ii. Dependent variables

# A - Find Number of Missing Values 
missing_values = df.isnull().sum()

#B - Replace missing values by mean, median and mode operations

# Replace missing values with mean - CGPA
df.loc[:, 'CGPA'] = df['CGPA'].fillna(df['CGPA'].mean())

# Replace missing values with median - Age
df.loc[:, 'Age'] = df['Age'].fillna(df['Age'].median())

# Replace missing values with mode - Branch
df.loc[:, 'Branch'] = df['Branch'].fillna(df['Branch'].mode()[0])

df


# C - Apply Encoding Techniques: 

# 1 - For Independent Variables 

df_encoded = pd.get_dummies(df, columns=['Branch', 'Degree', 'Summer Internship', 'Backlog/Arrear'])
df_encoded

# C - Apply Encoding Techniques: 

# 1 - For Dependent Variables 

label_encoder = LabelEncoder()
df_encoded['Eligible for placements'] = label_encoder.fit_transform(df_encoded['Eligible for placements'])
df_encoded

# Extract features and target
X = df.drop('Eligible for placements', axis=1).values
y = df['Eligible for placements'].values

# Convert to list of examples
examples = np.hstack((X, y.reshape(-1, 1))).tolist()

# Find-S Algorithm
def find_s(examples, target_attribute):
    # Initialize the hypothesis with the most specific hypothesis
    hypothesis = ['?' for _ in range(len(examples[0]) - 1)]
    
    for example in examples:
        if example[-1] == target_attribute:
            for i in range(len(hypothesis)):
                if hypothesis[i] == '?' or hypothesis[i] == example[i]:
                    hypothesis[i] = example[i]
                else:
                    hypothesis[i] = '?'
                    
    return hypothesis

# Candidate Elimination Algorithm
def candidate_elimination(examples, target_attribute):
    specific_hypothesis = ['?' for _ in range(len(examples[0]) - 1)]
    general_hypotheses = [['?' for _ in range(len(examples[0]) - 1)]]
    
    for example in examples:
        if example[-1] == target_attribute:
            for i in range(len(specific_hypothesis)):
                if specific_hypothesis[i] == '?' or specific_hypothesis[i] == example[i]:
                    specific_hypothesis[i] = example[i]
                else:
                    specific_hypothesis[i] = '?'
            
            new_general_hypotheses = []
            for general in general_hypotheses:
                new_general = general[:]
                for i in range(len(new_general)):
                    if general[i] == '?' or general[i] == example[i]:
                        new_general[i] = general[i]
                    else:
                        new_general[i] = '?'
                new_general_hypotheses.append(new_general)
            
            general_hypotheses = new_general_hypotheses
        else:
            new_general_hypotheses = []
            for general in general_hypotheses:
                if not all([g == '?' or g == e for g, e in zip(general, example)]):
                    new_general = general[:]
                    for i in range(len(new_general)):
                        if new_general[i] == '?':
                            new_general[i] = example[i]
                        else:
                            new_general[i] = '?'
                    new_general_hypotheses.append(new_general)
            
            general_hypotheses = new_general_hypotheses

    return specific_hypothesis, general_hypotheses

target = 1 
hypothesis = find_s(examples, target)
print("Find-S Hypothesis:", hypothesis)

specific_hypothesis, general_hypotheses = candidate_elimination(examples, target)
print("Specific Hypothesis:", specific_hypothesis)
print("General Hypotheses:", general_hypotheses)


