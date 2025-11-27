"""Download utility for IBTrACS data."""

# Imports
import os
import pandas as pd
from urllib import request

def download_data(basin="NA",data_dir="./data/ibtracs",overwrite=True) -> str:

    """
    Download IBTrACS data from desired basin.

    Args:
        basin (str): the basin shortname used in IBTrACS datasets
        datadir (str): the directory to download the data
        overwrite (bool): option to overwrite the file if it already exists

    Retuns:
        str: the path to the downloaded data
    """

    # Set the URL
    url = 'https://www.ncei.noaa.gov/data/'+\
          'international-best-track-archive-for-climate-stewardship-ibtracs/'+\
          'v04r01/access/csv/ibtracs.'+basin+'.list.v04r01.csv'

    # Set the file path
    file_path = os.path.join(data_dir,f'ibtracs_{basin}.csv')

    # Download the file if it doesn't already exists
    if overwrite or not os.path.exists(file_path):
            request.urlretrieve(url,file_path)

    return file_path

def read_data(file_path,subset_season=False,year_start=2010,year_end=2025) -> pd.DataFrame:

    """
    Read IBTrACS data to a pandas data frame, subset seasons if needed.

    Args:
        file_path (str): the path of the IBTrACS file to read
        subset_season (bool): option to subset the data based on season
        year_start (int): the season to start the subset
        year_end (int): the season to end the subset

    Returns:
        pd.DataFrame: the dataframe containing IBTrACS data
    """

    # Read the data from the CSV
    df = pd.read_csv(file_path,low_memory=False,skiprows=range(1,2))

    # Only keep a handful of columns
    keep_columns = ['SID','SEASON','NUMBER','NAME','ISO_TIME',
                    'NATURE','LAT','LON','WMO_WIND','WMO_PRES','TRACK_TYPE',
                    'DIST2LAND','LANDFALL','IFLAG','STORM_SPEED','STORM_DIR']
    df = df[keep_columns]

    # Convert time strings to datetimes for better querying
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
    df['SEASON'] = pd.to_numeric(df['SEASON'])
    df['NUMBER'] = pd.to_numeric(df['NUMBER'])
    df['LAT'] = pd.to_numeric(df['LAT'])
    df['LON'] = pd.to_numeric(df['LON'])

    # Subset seasons
    if subset_season:
        df = df[(df['SEASON'] >= year_start) & (df['SEASON'] <= year_end)]

    return df
