"""Download utility for IBTrACS data."""

# Imports
import os
import pandas as pd
from urllib import request
import goes
import datetime
from pyproj import Proj
import numpy as np

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
                    'DIST2LAND','LANDFALL','IFLAG','STORM_SPEED','STORM_DIR',
                    'USA_SSHS']
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

def filter_to_goes_east_bounds(df):

    """
    Filter IBTrACS dataframe to observations visible on the GOES-East disk.

    Args:
        df (pd.DataFrame): the dataframe containing IBTrACS data

    Returns:
        pd.DataFrame: the dataframe containing IBTrACS data 
            filtered to GOES-East disk.
    """
    
    # Restrict the data to the appropriate timespan of the GOES data
    df = df.loc[df['ISO_TIME']>=datetime.datetime(2017,4,10)].copy(True)

     # Set the parameters to download data
    date = datetime.datetime.now()-datetime.timedelta(days=30)
    product = 'ABI-L1b-RadF'
    band = 13

    # Get the GOES data
    ds = goes.get_goes_file(position='goes-east', date=date, band=band, product=product)

    # Get dataset projection data
    sat_height = ds.goes_imager_projection.perspective_point_height
    sat_lon = ds.goes_imager_projection.longitude_of_projection_origin
    sat_sweep = ds.goes_imager_projection.sweep_angle_axis

    # The projection x and y coordinates equals the scanning angle (in radians) multiplied by the satellite height
    x = ds.variables['x'][:] * sat_height
    y = ds.variables['y'][:] * sat_height

    # Create X and Y meshgrids
    X, Y = np.meshgrid(x, y)

    # Create a pyproj geostationary map object
    p = Proj(proj='geos', h=sat_height, lon_0=sat_lon, sweep=sat_sweep)

    # Get latitudes and longitudes
    lons, lats = p(X, Y, inverse=True)

    # Get a simple bounding box based on min/max lat/lons
    lons = np.where(lons==1e+30,np.nan,lons)
    lats = np.where(lats==1e+30,np.nan,lats)
    min_lat = np.nanmin(lats[lats != -np.inf])
    max_lat = np.nanmax(lats[lats != np.inf])
    min_lon = np.nanmin(lons[lons != -np.inf])
    max_lon = np.nanmax(lons[lons != np.inf])

    # Query IBTraCS data based on bounding box
    df = df.loc[
        (df['LAT'] >= min_lat) & 
        (df['LAT'] <= max_lat) & 
        (df['LON'] >= min_lon) & 
        (df['LON'] <= max_lon)
    ]

    # Create empty list
    drop_inds = []

    # Reset indices of dataframe
    df = df.reset_index()

    for i, row in df[['LAT','LON']].iterrows():
        
        # Cast latitude and longitude to float
        track_lat = float(row["LAT"])
        track_lon = float(row["LON"])

        # Convert lon/lat to x/y
        track_x,track_y = p(track_lon,track_lat)
        if track_x==np.inf or track_y==np.inf or track_x==np.nan or track_y==np.nan:
            drop_inds.append(i)

        # Get the closest point to the IBTrACS data
        x_ind = np.nanargmin(abs(x-track_x))
        y_ind = np.nanargmin(abs(y-track_y))

        # Check that none of 50 points on any side of the storm are off of the disc
        check_size = 50
        off_disc = np.isnan(lons[y_ind-check_size:y_ind+check_size,x_ind-check_size:x_ind+check_size]).any()
        off_disc += np.isinf(lons[y_ind-check_size:y_ind+check_size,x_ind-check_size:x_ind+check_size]).any()

        # If the points are off the disc, append the dataframe index to drop after looping
        if off_disc:
            drop_inds.append(i)

    # Drop any indices that fell off the disc
    df = df.drop(df.index[drop_inds])

    return df

def interpolate_meteorological_data(df):

    """
    Interpolate many wind speed and pressure values where it is missing
    
    Args:
        df (pd.DataFrame): the dataframe containing IBTrACS data

    Returns:
        pd.DataFrame: the dataframe containing IBTrACS data 
            filtered to GOES-East disk.
    """

     # Format IBTrACS data to do the interpolation
    df.replace(' ',np.nan,inplace=True)
    df['WMO_PRES'] = df['WMO_PRES'].astype(float)
    df['WMO_WIND'] = df['WMO_WIND'].astype(float)

    # Interpolate values for odd-numbered hours that were originally missing
    groups = []
    for sid, group in df.groupby('SID'):
        group[['WMO_WIND','WMO_PRES']] = group[['WMO_WIND','WMO_PRES']].interpolate(method='linear',axis=0, limit_direction='both')
        groups.append(group)
    df = pd.concat(groups)

    # Replace NAs that weren't able to be interpolated back the ' '
    df = df.fillna(' ')

    return df

if __name__ == '__main__':
    
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='IBTrACS Downloading and Preparation')

    parser.add_argument(
        '--data_dir',
        default="/Users/dylanwhite/Projects/tropical-cv/data/ibtracs",
        type=str
    )
    parser.add_argument(
        '--basin',
        default='NA',
        type=str
    )
    parser.add_argument(
        '--overwrite',
        action='store_true'
    )
    args = parser.parse_args()

    # Download or set the file path for the IBTrACS data
    file_path = download_data(
        basin=args.basin,
        data_dir=args.data_dir,
        overwrite=args.overwrite
    )

    # Read the data from the CSV
    df = read_data(file_path)

    # Restrict the data to the appropriate timespan of the GOES data
    df = filter_to_goes_east_bounds(df)

    output_file = Path(file_path).parent.joinpath('ibtracs_goes_east.csv')
    df.to_csv(output_file)