import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Indepdent Variables or predictors: All variables except breast_can
# Dependent Variable or target: breast_can

# Loading the processed dataset
df = pd.read_csv('model_dev2/data/processed/cancer_data_processed.csv')
print(df)

df.dropna(inplace=True)
len(df)

print(df)

X = df.drop('breast_can', axis=1)
y = df['breast_can']  
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
pickle.dump(X_train, open('model_dev2/model/X_train.sav', 'wb'))
pickle.dump(X.columns, open('model_dev2/model/X_columns.sav', 'wb'))
pickle.dump(scaler, open('model_dev2/model/scaler.sav', 'wb'))