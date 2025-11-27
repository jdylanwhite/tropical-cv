"""Download utility for NOAA GOES satellite imagery."""

# Imports
import xarray as xr
import s3fs
from datetime import datetime, timedelta

def day_of_year(date) -> datetime:
    '''
    Take a datetime date and get the number of days since Jan 1 of that same year.

    Args:
        date (datetime): the date to get the day of year

    Returns:
        (datetime): the day of the year for the provided datetime
    '''
    year = date.year
    firstDay = datetime(year, 1, 1)
    return (date - firstDay).days + 1

def get_satellite_for_date(position, date):
    """
    Get the correct GOES satellite number based on position and date.
    
    Args:
        position (str): the position, 'goes-east' or 'goes-west', to get the satellite name
        date (datetime): date to retrieve data for
    
    Returns:
        str: satellite name (e.g., 'goes16', 'goes19')
    """
    
    if position == 'goes-east':
        # GOES-East transitions
        if date >= datetime(2025, 4, 7):
            return 'goes19'
        else:
            return 'goes16'
    
    elif position == 'goes-west':
        # GOES-West transitions
        if date >= datetime(2023, 1, 10):
            return 'goes18'
        else:
            return 'goes17'
    
    else:
        raise ValueError(f"Position must be 'goes-east' or 'goes-west', got: {position}")

def get_goes_file(position='goes-east', date=None, hour=18, band=2, product='ABI-L1b-RadF'):
    """
    Get a GOES file from AWS S3 (public bucket, no credentials needed)
    
    Args:
        position (str): 'goes-east' or 'goes-west' (automatically selects correct satellite for date)
            OR specific satellite: 'goes16', 'goes17', 'goes18', 'goes19'
        date (datetime): date to retrieve data for
        hour (int): hour of day (UTC)
        band (int): ABI band number
        product (str): Product name (e.g., 'ABI-L1b-RadF')
    
    Returns:
        xarray.Dataset: the downloaded GOES NetCDF dataset
    """
    
    if date is None:
        date = datetime.now() - timedelta(days=30)
    
    # If position is 'goes-east' or 'goes-west', determine the satellite
    if position in ['goes-east', 'goes-west']:
        satellite = get_satellite_for_date(position, date)
        print(f"Using {satellite} for {position} on {date.strftime('%Y-%m-%d')}")
    else:
        # Assume it's a specific satellite name
        satellite = position
    
    # Determine bucket name
    bucket_map = {
        'goes16': 'noaa-goes16',
        'goes17': 'noaa-goes17',
        'goes18': 'noaa-goes18',
        'goes19': 'noaa-goes19'
    }
    
    if satellite not in bucket_map:
        raise ValueError(f"Unknown satellite: {satellite}")
    
    bucket = bucket_map[satellite]
    
    # Determine scan mode based on date
    if date < datetime(2019, 4, 2, 16):
        scan_mode = "M3"
    else:
        scan_mode = "M6"
    
    # Build the S3 path
    year = date.year
    day = day_of_year(date)
    
    prefix = f'{product}/{year}/{day:03.0f}/{hour:02.0f}/'
    pattern = f'OR_{product}-{scan_mode}C{band:02.0f}'
    
    # Initialize s3fs (anonymous access for public buckets)
    fs = s3fs.S3FileSystem(anon=True)
    
    # List files matching the pattern
    s3_path = f'{bucket}/{prefix}'
    try:
        files = fs.ls(s3_path)
        # Filter for the specific band and pattern
        matching_files = [f for f in files if pattern in f]
        
        if not matching_files:
            print(f"No files found matching pattern: {pattern}")
            print(f"Checked path: {s3_path}")
            return None
        
        # Get the first file
        file_path = matching_files[0]
        print(f"Found file: {file_path.split('/')[-1]}")
        
        # Open with xarray
        with fs.open(file_path, 'rb') as f:
            ds = xr.open_dataset(f, engine='h5netcdf')
            # Load into memory to avoid issues when file handle closes
            ds = ds.load()
            
        return ds
        
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return None
