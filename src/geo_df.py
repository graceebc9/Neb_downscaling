import pandas as pd 
from src.utils import load_pc_shp

def load_geodf(nrows):
    df = pd.read_csv('/Users/gracecolverd/NebulaDataset/final_dataset/NEBULA_englandwales_domestic_filtered.csv', nrows=nrows)
    pcs_load = df.postcode.str[0:2].unique().tolist()
    pcs_load = df.postcode.str.extract('([A-Z]+)').iloc[:,0].unique().tolist()
    print(pcs_load)
    pc_shp = load_pc_shp(pcs_load)

    geo_df = pc_shp.merge(df, left_on='POSTCODE', right_on='postcode', how='inner')
    geo_df = geo_df.to_crs('EPSG:4326')
    geo_df['latitude'] = geo_df.geometry.centroid.y 
    geo_df['longitude'] = geo_df.geometry.centroid.x
    return geo_df

def create_geo_df(df):
    pcs_load = df.postcode.str[0:2].unique().tolist()
    pcs_load = df.postcode.str.extract('([A-Z]+)').iloc[:,0].unique().tolist()
    print('Postcodes to load ' , pcs_load)
    pc_shp = load_pc_shp(pcs_load)
    
    geo_df = pc_shp.merge(df, left_on='POSTCODE', right_on='postcode', how='inner')
    geo_df['latitude'] = geo_df.geometry.centroid.y 
    geo_df['longitude'] = geo_df.geometry.centroid.x
    geo_df = geo_df.to_crs('EPSG:4326')


    return geo_df