import pandas as pd 
from sklearn.preprocessing import OrdinalEncoder

# Loading the dataset
df = pd.read_pickle('model_dev2/data/raw/Cancer_Rates.pkl')
print(df)

# column names
df.columns

# Cleaning column names
df.columns = df.columns.str.lower().str.replace(' ', '_')
df.columns

# Keeping columns
to_keep = [
    'colorectal', 
    'lung_bronc', 
    'breast_can',
    'prostate_c',
    'urinary_sy',
    'all_cancer'
]
df = df[to_keep]

#Drop missing values
df.dropna(inplace=True)
df

df.dtypes

# Perform encoding on colorectal 

enc = OrdinalEncoder()
enc.fit(df[['colorectal']])
df['colorectal'] = enc.transform(df[['colorectal']])
df['colorectal']

# Creating a dataframe with mapping for colorectal 
df_mapping_colorectal = pd.DataFrame(enc.categories_[0], columns=['colorectal'])
df_mapping_colorectal['colorectal_ordinal'] = df_mapping_colorectal.index
df_mapping_colorectal

# Saving the mapping as a CSV for colorectal 
df_mapping_colorectal.to_csv('model_dev2/data/processed/mapping_colorectal.csv', index=False)

# Perform encoding on breast_can

enc = OrdinalEncoder()
enc.fit(df[['breast_can']])
df['breast_can'] = enc.transform(df[['breast_can']])
df['breast_can']

# Creating a dataframe with mapping for breast_can
df_mapping_breast_can = pd.DataFrame(enc.categories_[0], columns=['breast_can'])
df_mapping_breast_can['breast_can_ordinal'] = df_mapping_breast_can.index
df_mapping_breast_can

# Saving the mapping as a CSV for breast_can
df_mapping_breast_can.to_csv('model_dev2/data/processed/mapping_breast_can.csv', index=False)

# Perform encoding on prostate_c

enc = OrdinalEncoder()
enc.fit(df[['prostate_c']])
df['prostate_c'] = enc.transform(df[['prostate_c']])
df['prostate_c']

# Creating a dataframe with mapping for prostate_c
df_mapping_prostate_c = pd.DataFrame(enc.categories_[0], columns=['prostate_c'])
df_mapping_prostate_c['prostate_c_ordinal'] = df_mapping_prostate_c.index
df_mapping_prostate_c

# Saving the mapping as a CSV for prostate_c
df_mapping_prostate_c.to_csv('model_dev2/data/processed/mapping_prostate_c.csv', index=False)

# Perform encoding on all_cancer
enc = OrdinalEncoder()
enc.fit(df[['all_cancer']])
df['all_cancer'] = enc.transform(df[['all_cancer']])
df['all_cancer']

# Creating a dataframe with mapping for all_cancer
df_mapping_all_cancer = pd.DataFrame(enc.categories_[0], columns=['all_cancer'])
df_mapping_all_cancer['all_cancer_ordinal'] = df_mapping_all_cancer.index
df_mapping_all_cancer

# Saving the mapping as a CSV for all_cancer
df_mapping_all_cancer.to_csv('model_dev2/data/processed/mapping_all_cancer.csv', index=False)

# Saving all mapping
df.to_csv('model_dev2/data/processed/cancer_data_processed.csv', index=False)