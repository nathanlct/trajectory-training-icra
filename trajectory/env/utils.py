import boto3
import itertools
import json
from math import atan2, cos, radians, sin, sqrt
import numpy as np


def lat_long_distance(pos1, pos2):
    """Returns distance in meters between two (latitude, longitude) points (Haversine formula)"""
    (lat1, long1), (lat2, long2) = pos1, pos2
    R = 6371e3  # radius of the Earth in meters
    dlat = radians(lat2 - lat1)
    dlong = radians(long2 - long1)
    a = sin(dlat / 2) * sin(dlat / 2) \
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlong / 2) * sin(dlong / 2)
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def get_bearing(lat1,lon1,lat2,lon2):
    dLon = lon2 - lon1;
    y = np.sin(dLon) * np.cos(lat2);
    x = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dLon);
    brng = np.rad2deg(np.arctan2(y, x));
    if brng < 0: brng+= 360
    return brng

def get_driving_direction(bearing):
    if(bearing> 340 and bearing <360):return 'West'
    elif(bearing> 150 and bearing <190):return 'East'
    else:return None
    
def get_valid_lat_long(lat,long):
    is_valid = False
    if long < -86.58 and long > -86.685 and lat > 35.98 and lat < 36.07: is_valid = True
    
    return is_valid

def pairwise(iterable):
    """Return successive overlapping pairs taken from the input iterable."""
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def partition(iterable, pred):
    "Use a predicate to partition entries into false entries and true entries"
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)

def moving_sum(array, chunk_size):
    """Moving sum. Returns an array of size len(array) - chunk_size + 1."""
    return np.convolve(array, np.ones(chunk_size), 'valid')

def moving_average(array, chunk_size):
    """Moving average. Returns an array of size len(array) - chunk_size + 1."""
    return np.convolve(array, np.ones(chunk_size), 'valid') / chunk_size

def counter(limit=None):
    """Equivalent to range(limit), with range(None) counting up to infinity."""
    i = 0
    while True:
        yield i
        i += 1
        if limit is not None and i >= limit:
            break

def duration_to_str(delta_t):
    """Convert a duration (in seconds) into a human-readable string."""
    delta_t = int(delta_t)
    s_out = ''
    for time_s, unit in [(86400, 'd'), (3600, 'h'), (60, 'm'), (1, 's')]:
        count = delta_t // time_s
        delta_t %= time_s
        if count > 0 or unit == 's':
            s_out += f'{count}{unit}'
    return s_out

def dict_to_json(data, path):
    """Save a dictionary into a .json file at path."""
    class Encoder(json.JSONEncoder):
        def default(self, obj):
            try:
                return json.JSONEncoder.default(self, obj)
            except TypeError:
                return str(obj)

    with open(path, 'w') as fp:
        json.dump(data, fp, indent=4, cls=Encoder)

def get_last_or(arr, default=None):
    """Return the last element of {arr}, or {default} if {arr} is empty."""
    if len(arr) == 0:
        return default
    return arr[-1]

def get_first_element(arr):
    """Returns arr[0]...[0]."""
    val = arr
    try:
        while True:
            val = val[0]
    except:
        return val

def upload_to_s3(bucket_name, bucket_key, file_path, metadata={}, log=False):
    s3 = boto3.resource("s3")
    s3.Bucket(bucket_name).upload_file(str(file_path), str(bucket_key),
                                       ExtraArgs={"Metadata": metadata})
    if log:
        print(f'Uploaded {file_path} to s3://{bucket_name}/{bucket_key}')
