import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_pickle('model_dev1/data/raw/shooting_data.pk1')

df.columns

# Column names are already cleaned 

# Data types
df.dtypes

# Keeping columns
to_keep = [
    'boro',
    'perp_sex',
    'perp_race',
    'vic_sex',
    'vic_race',
]

df = df[to_keep]


# Perform encoding on boro

enc = OrdinalEncoder()
enc.fit(df[['boro']])
df['boro'] = enc.transform(df[['boro']])
df['boro']

# Creating a dataframe with mapping for boro
df_mapping_boro = pd.DataFrame(enc.categories_[0], columns=['boro'])
df_mapping_boro['boro_ordinal'] = df_mapping_boro.index
df_mapping_boro

# Saving the mapping as a CSV for boro
df_mapping_boro.to_csv('model_dev1/data/processed/mapping_boro.csv', index=False)



# Perform encoding on perp sex
enc = OrdinalEncoder()
enc.fit(df[['perp_sex']])
df['perp_sex'] = enc.transform(df[['perp_sex']])
df['perp_sex']

# Creating a dataframe with mapping for perp sex
df_mapping_perp_sex = pd.DataFrame(enc.categories_[0], columns=['perp_sex'])
df_mapping_perp_sex['perp_sex_ordinal'] = df_mapping_perp_sex.index
df_mapping_perp_sex

# Saving the mapping as a CSV for perp sex
df_mapping_perp_sex.to_csv('model_dev1/data/processed/mapping_perp_sex.csv', index=False)


# Perform encoding on perp race
enc = OrdinalEncoder()
enc.fit(df[['perp_race']])
df['perp_race'] = enc.transform(df[['perp_race']])
df['perp_race']

# Creating a dataframe with mapping for perp race
df_mapping_perp_race = pd.DataFrame(enc.categories_[0], columns=['perp_race'])
df_mapping_perp_race['perp_race_ordinal'] = df_mapping_perp_race.index
df_mapping_perp_race

# Saving the mapping as a CSV for perp race
df_mapping_perp_race.to_csv('model_dev1/data/processed/mapping_perp_race.csv', index=False)


# Saving all mapping
df.to_csv('model_dev1/data/processed/shooting_data_processed.csv', index=False)