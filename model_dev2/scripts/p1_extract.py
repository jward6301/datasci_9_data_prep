import pandas as pd

# Loading the dataset

datalink = 'https://data-lakecountyil.opendata.arcgis.com/datasets/lakecountyil::cancer-rates.csv?where=1=1&outSR=%7B%22latestWkid%22%3A3435%2C%22wkid%22%3A102671%7D'
df = pd.read_csv(datalink)
df.size

# Saving as a CSV file
df.to_csv('model_dev2/data/raw/Cancer_Rates.csv', index=False)

# Saving as a Pickle file
df.to_pickle('model_dev2/data/raw/Cancer_Rates.pkl')