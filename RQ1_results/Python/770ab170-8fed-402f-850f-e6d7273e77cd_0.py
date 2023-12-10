import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.cluster import KMeans
import zipfile
import os

# Specify the location of the zip file and the folder to unzip to
zip_path = "path_to_your_zip_file"
unzip_folder = "path_to_folder_to_unzip_to"

# Create a ZipFile Object and load zip file in it
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Extract all the contents of zip file in different directory
    zip_ref.extractall(unzip_folder)

# Specify the file paths
powers_file_path = os.path.join(unzip_folder, 'super_hero_powers.csv')
heroes_file_path = os.path.join(unzip_folder, 'heroes_information.csv')

# Load the data into pandas DataFrames
powers_df = pd.read_csv(powers_file_path)
heroes_df = pd.read_csv(heroes_file_path)

# Merge the dataframes
merged_df = pd.merge(heroes_df, powers_df, left_on='name', right_on='hero_names')

# Identify the numerical and categorical columns
num_cols = ['Height', 'Weight']
cat_cols = ['Eye color', 'Race', 'Hair color', 'Publisher', 'Skin color', 'Alignment']

# Handling missing data
for col in num_cols:
    merged_df[col].fillna(merged_df[col].median(), inplace=True)

for col in cat_cols:
    merged_df[col].fillna(merged_df[col].mode()[0], inplace=True)

# Encoding categorical variables
merged_df = pd.get_dummies(merged_df, columns=cat_cols)

# Normalizing numerical variables
for col in num_cols:
    merged_df[col] = (merged_df[col] - merged_df[col].min()) / (merged_df[col].max() - merged_df[col].min())

# We will use only the numerical columns for clustering
numerical_df = merged_df.select_dtypes(include=['float64', 'int64', 'uint8'])

# Perform the clustering
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(numerical_df)

# Add the cluster labels to the original dataframe
merged_df['Cluster'] = clusters

merged_df[['name', 'Gender', 'Height', 'Weight', 'Publisher_Marvel Comics', 'Publisher_DC Comics', 'Cluster']].head()
