import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler


# Indepdent Variables or predictors: All other columns except 'boro'
# Dependent Variables or target: 'boro'

# Loading the processed dataset
df = pd.read_csv('model_dev1/data/processed/mapping_perp_race.csv')
df = pd.read_csv('model_dev1/data/processed/mapping_perp_sex.csv')


df.dropna(inplace=True)
len(df)

print(df)

X = df.drop('perp_sex', axis=1)
y = df['perp_sex']  
print(X)
print(y) 

# StandardScaler
scaler = StandardScaler()
scaler.fit(X) 

X_scaled = scaler.transform(X)
X_scaled


# Splitting the scaled data into training, validation, and testing
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

(X_train.shape, X_val.shape, X_test.shape)

# Saving it
pickle.dump(X_train, open('model_dev1/model/X_train.sav', 'wb'))
pickle.dump(X.columns, open('model_dev1/model/X_columns.sav', 'wb'))
pickle.dump(scaler, open('model_dev1/model/scaler.sav', 'wb'))