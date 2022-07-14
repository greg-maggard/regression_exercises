import pandas as pd
import numpy as np
# train test split from sklearn
from sklearn.model_selection import train_test_split
# imputer from sklearn
from sklearn.impute import SimpleImputer
from env import get_db_url
from sklearn.model_selection import train_test_split

import acquire
import wrangle

#Clean Titanic Data Files:

def clean_titanic_data(df):
    '''
    Takes in a titanic dataframe and returns a cleaned dataframe
    Arguments: df - a pandas dataframe with the expected feature names and columns
    Returns: clean_df - a dataframe with the cleaning operations performed on it
    '''
    #Drop Duplicates
    df.drop_duplicates(inplace = True)
    #Drop Columns
    columns_to_drop = ['embarked', 'class', 'passenger_id', 'deck']
    df = df.drop(columns = columns_to_drop)
    #Encoded Categorical Variables
    dummy_df = pd.get_dummies(df[['sex', 'embark_town']], dummy_na = False, drop_first = [True, True])
    df = pd.concat([df, dummy_df], axis = 1)
    return df.drop(columns =  ['sex', 'embark_town'])

def impute_age(train, validate, test):
    '''
    Imputes the mean age of train to all three datasets.
    '''
    imputer = SimpleImputer(strategy = 'mean', missing_values = np.nan)
    imputer = imputer.fit(train[['age']])
    train[['age']] = imputer.transform(train[['age']])
    validate[['age']] = imputer.transform(validate[['age']])
    test[['age']] = imputer.transform(test[['age']])
    return train, validate, test

def prep_titanic_data(df):
    df = clean_titanic_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.survived, random_state = 1234)
    train, validate = train_test_split(train, train_size = .7, stratify = train.survived, random_state = 1234)
    train, validate, test = impute_age(train, validate, test)
    return train, validate, test

#--------------------------------------------------------------------------------------------------------------------------------------------

#Cleaning Iris Data:
def clean_iris_data(df):
    #Dropping unneeded columns
    columns_to_drop = ['species_id', 'measurement_id']
    df = df.drop(columns = columns_to_drop)
    #renaming species_name column
    df = df.rename(columns = {'species_name' : 'species'})
    #creating dummy variables of species
    dummy_df = pd.get_dummies(df[['species']], drop_first = True)
    #concatenating dummy variables onto original dataframe
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def prep_iris_data(df):
    df = clean_iris_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.species, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.8, stratify = train.species, random_state = 1234)
    return train, validate, test

#--------------------------------------------------------------------------------------------------------------------------------------------

#Prepare Telco Data:
def clean_telco_data(df):
    #Replacing empty cells with nulls:
    df = df.replace(' ', np.nan)
    #Dropping rows with nulls:
    df = df.dropna()
    #Dropping unneeded columns:
    columns_to_drop = ['contract_type_id', 'payment_type_id', 'internet_service_type_id',]
    df = df.drop(columns = columns_to_drop)
    #creating dummy variables of species
    dummy_df = pd.get_dummies(df[['gender', 'partner', 'dependents', 'phone_service', 'multiple_lines', 'online_security', 'online_backup', 'device_protection', 'tech_support', 'streaming_tv', 'streaming_movies', 'paperless_billing', 'contract_type', 'payment_type', 'internet_service_type']], drop_first = True)
    #concatenating dummy variables onto original dataframe
    df = pd.concat([df, dummy_df], axis = 1)
    return df

def prep_telco_data(df):
    df = clean_telco_data(df)
    train, test = train_test_split(df, train_size = 0.8, stratify = df.churn, random_state = 1234)
    train, validate = train_test_split(train, train_size = 0.8, stratify = train.churn, random_state = 1234)
    return train, validate, test

#--------------------------------------------------------------------------------------------------------------------------------------------

def scale_zillow_data(train, validate, test):
    
    '''
    Takes in train, validate, and test sets, creates copies of those sets, and
    scales the copies. Returns train_scaled, validate_scaled, and test_scaled 
    DataFrames.
    '''
    
    #Defining the columns that need to be scaled:
    scaled_columns = ['bedroomcnt', 'bathroomcnt', 'calculatedfinishedsquarefeet', 'taxvaluedollarcnt']
    
    #Creating scalable copies of the train, validate, and test sets:
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    
    #Creating the scaler object:
    scaler = MinMaxScaler()
    scaler.fit(train[scaled_columns])
    
    #Applying the scaler to the scalable colums within the train, validate, test copies:
    train_scaled[scaled_columns] = scaler.transform(train[scaled_columns])
    validate_scaled[scaled_columns] = scaler.transform(validate[scaled_columns])
    test_scaled[scaled_columns] = scaler.transform(test[scaled_columns])

    #Returning scaled dataframes:
    return train_scaled, validate_scaled, test_scaled

