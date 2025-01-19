import os 
import pandas as pd 
import geopandas as gpd 

def load_pc_shp(pcs_to_load, pc_path='/Volumes/T9/2024_Data_downloads/codepoint_polygons_edina/Download_all_postcodes_2378998'):
    ll = []
    for pc in pcs_to_load:
        if len(pc)==1:
            path = os.path.join(pc_path, f'codepoint-poly_5267291/one_letter_pc_code/{pc}/{pc}.shp')
        else:
            path = os.path.join(pc_path,  f'codepoint-poly_5267291/two_letter_pc_code/{pc}.shp' ) 
        sd = gpd.read_file(path)    
        ll.append(sd) 
    pc_shp = pd.concat(ll)
    return pc_shp 


