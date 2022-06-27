import numpy as np
import pandas as pd
import os

def get_zillow_data():
    filename = "zillow.csv"

    if os.path.isfile(filename):
        return pd.read_csv(filename, index_col = 0)
    else:
        # read the SQL query into a dataframe
        df = pd.read_sql('''SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips, propertylandusedesc 
        FROM properties_2017 
        JOIN propertylandusetype on properties_2017.propertylandusetypeid = propertylandusetype.propertylandusetypeid 
        WHERE propertylandusedesc = "Single Family Residential";''', get_db_url('zillow'))

        # Write that dataframe to disk for later. Called "caching" the data for later.
        df.to_csv(filename)

        # Return the dataframe to the calling code
        return df

def clearing_fips(df):
    '''This function takes in a DataFrame of unprepared Zillow information and generates a new
    'county' column, with the county name based on the FIPS code. Drops the 'fips' column and returns
    the new DataFrame.
    '''
    # create a list of our conditions
    fips = [
        (df['fips'] == 6037.0),
        (df['fips'] == 6059.0),
        (df['fips'] == 6111.0)
        ]
    # create a list of the values we want to assign for each condition
    counties = ['Los Angeles County', 'Orange County', 'Ventura County']
    # create a new column and use np.select to assign values to it using our lists as arguments
    df['county'] = np.select(fips, counties)
    df = df.drop(columns = 'fips')
    return df

def wrangle_zillow():
    '''Function to import zillow data from database and create a CSV cache of the file. 
    Function runs the clearing_fips function to generate counties and drop fips column, 
    then drops rows containing nulls, as well as rows with 0 bathrooms, 0 bedrooms, and 
    less than 12 sqft. Finally, converts 'bedroomcnt' and 'yearbuilt' columns to integers.
    Return wrangled DataFrame.
    '''
    #Acquire Data:
    df = get_zillow_data()
    #Run clearing_fips function:
    df = clearing_fips(df)
    #Drop Null Values:
    df = df.dropna()
    #Drop listings that have 0.0 bathrooms, 0.0 bedrooms, and are under the 120 sqft legal minimum as required by California to be considered a residence:
    df = df.drop(df[(df.bedroomcnt == 0.0) & (df.bathroomcnt == 0.0) & (df.calculatedfinishedsquarefeet < 120.0)].index)
    #Converting 'bedroomcnt' and 'yearbuilt' columns to 'int' type:
    df = df.astype({'bedroomcnt' : int, 'yearbuilt': int})
    return df
