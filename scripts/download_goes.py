'''Download GOES imagery to be used by the dataloader.'''
# TODO make sure files don't already exist before downloading data
# TODO save JSON of storm indices, lat/lons, TS cats, original image name, full tile coords

# Import modules
import datetime
import numpy as np
from pyproj import Proj
import os
import pandas as pd
from tqdm import tqdm
import argparse
import json
from pathlib import Path

# Import functions I've written
import goes
import ibtracs

def get_dataset_slice(ds_var,x_ind,y_ind,buffer_size):

    # Calculate the slice bounds
    y_start = max(0, y_ind - buffer_size)
    y_end = min(ds_var.shape[0], y_ind + buffer_size)
    x_start = max(0, x_ind - buffer_size)
    x_end = min(ds_var.shape[1], x_ind + buffer_size)

    # Extract the available data
    data_slice = ds_var[y_start:y_end, x_start:x_end]

    # Calculate padding needed on each side
    pad_top = max(0, buffer_size - y_ind)
    pad_bottom = max(0, (y_ind + buffer_size) - ds_var.shape[0])
    pad_left = max(0, buffer_size - x_ind)
    pad_right = max(0, (x_ind + buffer_size) - ds_var.shape[1])

    padded_data = data_slice.pad(
        y=(pad_top, pad_bottom),
        x=(pad_left, pad_right),
        mode='constant',
        constant_values=np.nan
    )

    if 0 in padded_data.shape:
        raise IndexError('0 in data slice shape!')

    return padded_data

def remove_free_pixels(free,x_ind,y_ind,buffer_size):
    
    # Calculate the slice bounds
    y_start = max(0, y_ind - buffer_size)
    y_end = min(free.shape[0], y_ind + buffer_size)
    x_start = max(0, x_ind - buffer_size)
    x_end = min(free.shape[1], x_ind + buffer_size)

    # Extract the available data
    free[y_start:y_end, x_start:x_end] = 0

    return free

def download_data(
    ibtracs_path,
    training_data_dir,
    band=13,
    start_date=None,
    end_date=None,
    sample_size=1200,
    limit=None
):

    # Set up a dictionary for tracking downloaded tiles
    # We may have already ran this script, 
    # so check if it exists before starting from scratch
    output_path = os.path.join(args.training_data_dir,'image_data.json')
    if os.path.exists(output_path):
        with open(output_path,'r') as f:
            image_data = json.load(f)
        if len(image_data['images'])==0:
            image_id = 0
        else:
            image_id = max([int(x['id']) for x in image_data['images']])+1
    else:
        image_data = {
            'images': [],
        }
        image_id = 0

    # Read the IBTrACS data that we've already filtered 
    # for intersection on the GOES disk 
    df = ibtracs.read_data(ibtracs_path)
    df = df.loc[df['NATURE']=='TS'].reset_index()
    
    # Filter the dates of the IBTrACS for what we wish to download
    if start_date is not None:
        start_date = datetime.datetime.fromisoformat(start_date)
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'],utc=True)
        df = df.loc[(df['ISO_TIME'] >= start_date)].copy(deep=True)
    if end_date is not None:
        end_date = datetime.datetime.fromisoformat(end_date)
        df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'],utc=True)
        df = df.loc[(df['ISO_TIME'] <= end_date)].copy(deep=True)

    # Limit the data to a random sample, if applicable
    if limit is not None and len(df)>limit:
        df = df.sample(limit)
    
    # Set the buffer (in pixels) size
    buffer_size = sample_size // 2
    
    # Loop through all of the dates in the IBTrACS dataframe
    dates = df['ISO_TIME'].unique()
    for date in tqdm(dates,total=len(dates),desc="Downloading GOES Imagery"):

        # Convert the numpy.datetime64 object into a datetime object
        dt = datetime.datetime.fromisoformat(str(date)[:-3])

        # Check if we've already downloaded this data
        prefix = dt.strftime("%Y%m%d_%HZ")
        check_file_list = list(Path(training_data_dir).joinpath('positive').glob(f'{prefix}*'))
        if len(check_file_list) > 0:
            print(f'You have already downloaded data from this disk, skipping {prefix}')
            continue

        try:
        
            # Get the GOES data
            ds = goes.get_goes_file(position='goes-east', date=dt, band=band, product='ABI-L1b-RadF')
         
            # Get dataset projection data
            sat_height = ds.goes_imager_projection.perspective_point_height
            sat_lon = ds.goes_imager_projection.longitude_of_projection_origin
            sat_sweep = ds.goes_imager_projection.sweep_angle_axis
         
            # The projection x and y coordinates equals the scanning 
            # angle (in radians) multiplied by the satellite height
            x = ds.variables['x'][:] * sat_height
            y = ds.variables['y'][:] * sat_height
         
            # Create an array of ones for the negative training samples
            nx, ny = ds.Rad.shape
            free = np.ones((ny,nx))
         
            # Create a pyproj geostationary map object
            p = Proj(proj='geos', h=sat_height, lon_0=sat_lon, sweep=sat_sweep)
         
            # Get the subset of the dataframe that matches this date
            df_subset = df.loc[df['ISO_TIME'] == date]
         
            # Loop through all of the subset dataframe rows:
            for i, row in df_subset.iterrows():
         
                # Get the latitudes/longitudes of the track data
                track_lat = float(row['LAT'])
                track_lon = float(row['LON'])
         
                # Convert lon/lat to x/y
                track_x,track_y = p(track_lon,track_lat)
         
                # Get the closest point to the IBTrACS data
                x_ind = np.nanargmin(abs(x-track_x))
                y_ind = np.nanargmin(abs(y-track_y))
         
                # Set the file name
                file_name = dt.strftime("%Y%m%d_%HZ")+f'_x{x_ind:05.0f}_y{y_ind:05.0f}_hw{sample_size}.nc'
                file_path = os.path.join(training_data_dir,'positive',file_name)
         
                # Save as netCDF
                rad_tile = get_dataset_slice(ds.Rad,x_ind,y_ind,buffer_size)
                #rad_tile = ds.Rad[y_ind-buffer_size:y_ind+buffer_size,x_ind-buffer_size:x_ind+buffer_size]
                rad_tile.to_netcdf(file_path, encoding={'y': {'dtype': 'float32'}, 'x': {'dtype': 'float32'}})

                # Append to COCO-style data dictionary
                image_data['images'].append({
                    'category':'positive',
                    'file_name':file_path,
                    'id':image_id,
                    'width':int(rad_tile.shape[1]),
                    'height':int(rad_tile.shape[0]),
                    'band':int(band),
                    'original_file':str(ds.dataset_name),
                    'original_ul':[int(y_ind-buffer_size),int(x_ind-buffer_size)],
                    'track_coordinates':[float(track_lon),float(track_lat)],
                    'date':str(date),
                    'df_index':i
                })
                image_id += 1
         
                # Mark these pixels as containing a positive sample
                free = remove_free_pixels(free,x_ind,y_ind,buffer_size)
         
            # Now go find some random negative samples that don't overlap with locations where free == 0
            negative_count = 0
            max_attempt_count = 5*len(df_subset)
            attempt_count = 0
            while negative_count < len(df_subset):
         
                # Get a random location on the image
                x_ind = np.random.randint(buffer_size,nx-buffer_size,size=1)[0]
                y_ind = np.random.randint(buffer_size,ny-buffer_size,size=1)[0]
         
                # Check that it doesn't overlap with previous images
                if (free[y_ind-buffer_size:y_ind+buffer_size,x_ind-buffer_size:x_ind+buffer_size] == 1).all():
         
                    # Set the file name
                    file_name = dt.strftime("%Y%m%d_%HZ")+f'_x{x_ind:05.0f}_y{y_ind:05.0f}_hw{sample_size}.nc'
                    file_path = os.path.join(training_data_dir,'negative',file_name)
         
                    # Save as netCDF
                    rad_tile = get_dataset_slice(ds.Rad,x_ind,y_ind,buffer_size)
                    #rad_tile = ds.Rad[y_ind-buffer_size:y_ind+buffer_size,x_ind-buffer_size:x_ind+buffer_size]
                    rad_tile.to_netcdf(file_path, encoding={'y': {'dtype': 'float32'}, 'x': {'dtype': 'float32'}})

                    # Append to COCO-style data dictionary
                    image_data['images'].append({
                        'category':'negative',
                        'file_name':file_path,
                        'id':image_id,
                        'width':int(rad_tile.shape[1]),
                        'height':int(rad_tile.shape[0]),
                        'band':int(band),
                        'original_file':str(ds.dataset_name),
                        'original_ul':[int(y_ind-buffer_size),int(x_ind-buffer_size)],
                        'track_coordinates':[-9999,-9999],
                        'date':str(date),
                        'df_index':-9999
                    })
                    image_id += 1

                    # Also mark these pixels, since we have a sample already taken from them
                    #free[y_ind-buffer_size:y_ind+buffer_size,x_ind-buffer_size:x_ind+buffer_size] = 0
                    free = remove_free_pixels(free,x_ind,y_ind,buffer_size)

                    # Advance the count of negative training sample images obtained
                    negative_count = negative_count + 1
         
                # Advance the count of attempts and break if it has been going for too long
                attempt_count = attempt_count + 1
                if attempt_count > max_attempt_count:
                    print('Breaking')
                    break
         
            # Delete the dataset to save memory
            del(ds)
    
        except Exception as e:
           print(f'Error processing training data from {dt}: {e}')

        # Write output file on each iteration in case something fails
        with open(output_path,'w') as f:
            json.dump(image_data,f)

    return image_data
           
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='GOES Tile Downloader')
    parser.add_argument(
        '--ibtracs_path',
        default="/Users/dylanwhite/Documents/Projects/tropical-cv/data/ibtracs/ibtracs_goes_east.csv",
        type=str
    )
    parser.add_argument(
        '--start_date',
        default=None,
        type=str
    )
    parser.add_argument(
        '--end_date',
        default=None,
        type=str
    )
    parser.add_argument(
        '--limit',
        default=None,
        type=int
    )
    parser.add_argument(
        '--sample_size',
        default=1200,
        type=int
    )
    parser.add_argument(
        '--band',
        default=13,
        type=int
    )
    parser.add_argument(
        '--training_data_dir',
        default="/Users/dylanwhite/Documents/Projects/tropical-cv/data/training",
        type=str
    )
    args = parser.parse_args()
    
    image_data = download_data(
        args.ibtracs_path,
        args.training_data_dir,
        band=args.band,
        start_date=args.start_date,
        end_date=args.end_date,
        sample_size=args.sample_size,
        limit=args.limit
    )