"""Download utility for NOAA GOES satellite imagery."""

# Imports
import xarray as xr
import s3fs
import datetime
from pyproj import Proj
import numpy as np

def day_of_year(date) -> datetime.datetime:
    '''
    Take a datetime date and get the number of days since Jan 1 of that same year.

    Args:
        date (datetime): the date to get the day of year

    Returns:
        (datetime): the day of the year for the provided datetime
    '''
    year = date.year
    firstDay = datetime.datetime(year, 1, 1)
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
        if date >= datetime.datetime(2025, 4, 7):
            return 'goes19'
        else:
            return 'goes16'
    
    elif position == 'goes-west':
        # GOES-West transitions
        if date >= datetime.datetime(2023, 1, 10):
            return 'goes18'
        else:
            return 'goes17'
    
    else:
        raise ValueError(f"Position must be 'goes-east' or 'goes-west', got: {position}")

def get_goes_file(position='goes-east', date=None, hour=None, band=2, product='ABI-L1b-RadF'):
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
        date = datetime.datetime.now() - datetime.timedelta(days=30)
    
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
    if date < datetime.datetime(2019, 4, 2, 16):
        scan_mode = "M3"
    else:
        scan_mode = "M6"
    
    # Build the S3 path
    year = date.year
    day = day_of_year(date)
    if hour is None:
        hour = date.hour
    
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

def extract_tile_around_latlon(ds, lat, lon, tile_size=512):
    """Extract a tile of specified size centered on a lat/lon coordinate.

    Args:
        ds (xarray.Dataset): GOES dataset containing radiance data and 
            projection information.
        lat (float): Latitude of center point in degrees.
        lon (float): Longitude of center point in degrees.
        tile_size (int, optional): Size of tile in pixels. Defaults to 512.

    Returns:
        tuple: A tuple containing:
            - tile (numpy.ndarray): Extracted tile of shape (tile_size, tile_size).
            - tile_info (dict): Dictionary containing information about the tile:
                - center_lat (float): Latitude of tile center.
                - center_lon (float): Longitude of tile center.
                - center_pixel (tuple of int): (x_idx, y_idx) pixel coordinates 
                  of center.
                - pixel_bounds (tuple of int): (x_start, x_end, y_start, y_end) 
                  pixel boundaries.
                - tile_size (int): Requested tile size.
                - x_coords (numpy.ndarray): X coordinates in radians for the tile.
                - y_coords (numpy.ndarray): Y coordinates in radians for the tile.

    Raises:
        ValueError: If the point (lat, lon) is not visible from the satellite.

    Examples:
        >>> tile, info = extract_tile_around_latlon(ds, lat=26.0, lon=-94.0, 
        ...                                          tile_size=512)
        >>> print(f"Tile shape: {tile.shape}")
        Tile shape: (512, 512)
    """
    
    # Get projection parameters
    sat_height = ds.goes_imager_projection.perspective_point_height
    sat_lon = ds.goes_imager_projection.longitude_of_projection_origin
    sat_sweep = ds.goes_imager_projection.sweep_angle_axis
    
    # Get x and y coordinates in radians - convert to numpy arrays
    x_rad = ds.variables['x'][:].values
    y_rad = ds.variables['y'][:].values
    
    # Create pyproj projection
    p = Proj(proj='geos', h=sat_height, lon_0=sat_lon, sweep=sat_sweep)
    
    # Convert lat/lon to projection x/y coordinates (in meters)
    x_meters, y_meters = p(lon, lat)
    
    # Debug prints
    print(f"Input: lat={lat}, lon={lon}")
    print(f"Projected: x_meters={x_meters}, y_meters={y_meters}")
    
    # Check if projection was successful (returns inf if point not visible)
    if np.isinf(x_meters) or np.isinf(y_meters) or np.isnan(x_meters) or np.isnan(y_meters):
        raise ValueError(f"Point ({lat}, {lon}) is not visible from satellite at {sat_lon}°")
    
    # Convert meters back to radians
    x_center = x_meters / sat_height
    y_center = y_meters / sat_height
    
    print(f"Radians: x_center={x_center}, y_center={y_center}")
    print(f"x_rad range: [{x_rad.min()}, {x_rad.max()}]")
    print(f"y_rad range: [{y_rad.min()}, {y_rad.max()}]")
    
    # Find the closest pixel indices
    x_idx = int(np.argmin(np.abs(x_rad - x_center)))
    y_idx = int(np.argmin(np.abs(y_rad - y_center)))
    
    print(f"Center pixel: x_idx={x_idx}, y_idx={y_idx}")
    
    # Calculate tile boundaries
    half_tile = tile_size // 2
    
    x_start = max(0, x_idx - half_tile)
    x_end = min(len(x_rad), x_idx + half_tile)
    y_start = max(0, y_idx - half_tile)
    y_end = min(len(y_rad), y_idx + half_tile)
    
    # Extract the tile
    tile = ds.Rad[y_start:y_end, x_start:x_end].values
    
    # Get actual tile size (might be smaller at edges)
    actual_height, actual_width = tile.shape
    
    # Pad if necessary to ensure tile_size x tile_size
    if actual_height < tile_size or actual_width < tile_size:
        padded_tile = np.zeros((tile_size, tile_size))
        padded_tile[:actual_height, :actual_width] = tile
        tile = padded_tile
        print(f"Warning: Tile padded. Original size: {actual_height}x{actual_width}")
    
    # Store tile information
    tile_info = {
        'center_lat': lat,
        'center_lon': lon,
        'center_pixel': (x_idx, y_idx),
        'pixel_bounds': (x_start, x_end, y_start, y_end),
        'tile_size': tile_size,
        'x_coords': x_rad[x_start:x_end],
        'y_coords': y_rad[y_start:y_end]
    }
    
    return tile, tile_info
