import pandas as pd

# download data
datalink = 'https://data.cityofnewyork.us/resource/833y-fsy8.csv'

df = pd.read_csv(datalink)
df.size
print(df)

# Saving the CSV
df.to_csv('model_dev1/data/raw/shooting_data.csv', index=False)

# Save as pickle file
df.to_pickle('model_dev1/data/raw/shooting_data.pk1')

