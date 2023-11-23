# datasci_9_data_prep
HHA 507 Week 9 Assignment

## 1. Datasets
1. This data set was taken from data.gov. It focuses on NYPD Shooting Incidents from 2006 - 2023. The columns represent pertitant information to each case, such as the perp's characterstics (age, race), the victim's characteristics, the location of the shooting, the borough, time of occurance, a description and more. 
* Intended machine learning task is classification. 
* This dataset will be the focus in `model_dev1`.
2. This data set was also taken from data.gov. It focuses on cancer rates in Lake County, Illinois. The columns represent different cancers focused on and their zipcodes. It was last updated in September 2023. 
* Intended machine learning task is regression.
* This dataset will be the focus in `model_dev2`.

## 2. Data Cleaning and Transformation Plan and 3. Data Cleaning Execution (Optional Challenge):
1. Create two folders to host the two different datasets. The two folders will be named `model_dev1` and `model_dev2`.
2. In each folder, create 3 new folders named: `data`, `model` and `scripts`. In the `data` folder, create two more folders named `processed` and `raw`. 
3. In both of the `scripts` folder, create 3 files named `p1_extract.py`, `p2_transform.py` and `p3_compute.py`. Each of these files will host the necessary sections. 
4. In the `p1_extract.py`, load in the dataset, then save it as a CSV and pickle file into the `raw` folder. 
5. In the `p2_transform.py`, load in the dataset using the pickle file from the `raw` folder. Steps 6-8 will occur in this same file. 
6. To clean the dataset, I will clean the column names by removing spaces, capitalized letters and special characters. I will also check for missing values and drop them as needed. I will then keep only the necessary columns and drop the remaining. I will also check the data types.
7. Once all of the cleaning is completed, I will move onto transforming the data. I will do this by using ordinal encoding and mapping. On each encoding and mapping for the necessary columns, I will save the file to the `processed` folder. 
8. Once that is complete, I will also save the whole encoded dataframe to the `processed` folder for possible future reference. 
* The code was run through the python terminal.
* To view my code click here: https://github.com/jward6301/datasci_9_data_prep/blob/main/model_dev1/scripts/p2_transform.py
 

## 4. Dataset Splitting:
1. For both `model_dev1` and `model_dev2` follow these steps.
2. In the `p3_compute.py` file, load in the processed data frames either individually or using the whole one. (For `model_dev1`, they were loaded individually, while for `model_dev2`, the whole processed dataset was loaded. ) 
3. List the independent variables or features, which will generally be all of the columns minus the dependent variable or target variable. Also list the dependent variable or target variable. I also dropped all missing values again to ensure that there were none missing. 
4. Utilize standard scaler, `X_scaled` and split the scaled data into training, validation, and testing. All of the code is available to view in my `p3_compute.py` files.
5. I then saved al of this to the model folder using the following commands and changing them as needed: `pickle.dump(X_train, open('model_dev2/model/X_train.sav', 'wb'))`
`pickle.dump(X.columns, open('model_dev2/model/X_columns.sav', 'wb'))`
`pickle.dump(scaler, open('model_dev2/model/scaler.sav', 'wb'))`.
* The code was run through the python terminal
* To view my code click here: https://github.com/jward6301/datasci_9_data_prep/blob/main/model_dev1/scripts/p3_compute.py
## 5. Errors and Issues
* I ran into one error during the dataset splitting section. For `model_dev1`, I kept receving error codes that it could not convert string to float. This was fixed by loading each individual processed dataset rather than the whole processed dataset. 
* The assignment utilzied code from Professor Williams. 