import numpy as np
import pandas as pd
import geopandas as gpd

import pyproj

import shapely
import shapely.wkt
from shapely.geometry import Point, Polygon, MultiLineString
from shapely.ops import transform

wgs84 = pyproj.Proj("+init=EPSG:4326")
etrs89_utm32 = pyproj.Proj("+init=EPSG:25832")
project = lambda x, y: pyproj.transform(wgs84, etrs89_utm32, x, y)
inverse = lambda x, y: pyproj.transform(etrs89_utm32, wgs84, x, y)

def to_utm32(geom_wgs84):
    return transform(project, geom_wgs84)

def to_wgs84(geom_utm32):
    return transform(inverse, geom_utm32)

def offset_line(geom, scale):
    if geom.type == 'MultiLineString':
        return MultiLineString([
            transform(inverse, transform(project, g).parallel_offset(distance = scale, side = 'right'))
        for g in geom.geoms])
    elif geom.type == 'LineString':
        return transform(inverse, transform(project, geom).parallel_offset(distance = scale, side = 'right'))

def distance(geom_a, geom_b):
    """ Calculates distance between to WGS84 geometries in meters. """
    geom_a_ = transform(project, geom_a)
    geom_b_ = transform(project, geom_b)
    return geom_a_.distance(geom_b_)

def distances(geodataframe, ref_geom):
    """ Calculates distance between all items in geodataframe and ref_geom (both in WGS84) in meters. """
    res = pd.DataFrame(index = geodataframe.index, columns = ['Distance'])
    ref_geom_ = transform(project, ref_geom)
    
    for i1, r1 in geodataframe.iterrows():
        geom_ = transform(project, r1.geometry)
        res.loc[i1, 'Distance'] = ref_geom_.distance(geom_)
            
    return res

def distance_matrix(geodataframe):
    """ 
        WARNING: Not optimized 
        alculates distance between all pairs in geodataframe (in WGS84) in meters.
    """
    res = pd.DataFrame(index = geodataframe.index, columns = geodataframe.index)
    
    for i1, r1 in geodataframe.iterrows():
        for i2, r2 in geodataframe.iterrows():
            res.loc[i1, i2] = distance(r1.geometry, r2.geometry)
            
    return res

def length(geom):
    geom_ = transform(project, geom)
    return geom_.length

def delta_line(geom, point, offset = 0, delta = .5):
    """ Calculate a small delta line on geom around point """
    p = geom.project(point) + offset
    l = geom.length
    # Correct out of range offset
    if p < 0: 
        p = 0
    elif p > l:
        p = geom.length
        
    dp1 = p - delta if p - delta > 0 else 0
    dp2 = p + delta if p + delta < l else l
    
    while dp2 <= dp1:
        dp2 = dp2 + delta
        
    return np.array([
        np.array(geom.interpolate(dp1).coords[0]),
        np.array(geom.interpolate(dp2).coords[0])
    ])

def line_angle(coords_a, coords_b):
    coords_a_ = np.asarray(coords_a)
    coords_b_ = np.asarray(coords_b)
    assert coords_a_.shape == (2, 2)
    assert coords_b_.shape == (2, 2)
    a = np.diff(coords_a_, axis = 0).flatten()
    b = np.diff(coords_b_, axis = 0).flatten()
    angle_ab = (np.arctan2(a[0], a[1]) - np.arctan2(b[0], b[1])) * 180 / np.pi
    while angle_ab < -180:
        angle_ab = angle_ab + 360
    while angle_ab > 180:
        angle_ab = angle_ab - 360
    return angle_ab
