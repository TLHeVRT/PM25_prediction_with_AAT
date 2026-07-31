import argparse

import ee
import pandas as pd


parser = argparse.ArgumentParser(
    description="Extract station-level geographic features with Earth Engine."
)
parser.add_argument(
    "--project",
    required=True,
    help="Google Cloud project ID with Earth Engine access."
)
args = parser.parse_args()
ee.Initialize(project=args.project)


# Input: map the new CSV schema to the legacy names used by downstream output.
input_columns = {
    'station_id': '站点编号',
    'station_name': '站点名称',
    'city': '城市',
    'longitude': '经度',
    'latitude': '纬度'
}
df = pd.read_csv(
    'stations.csv',
    dtype={'station_id': 'string'},
    encoding='utf-8-sig'
)
missing_columns = set(input_columns) - set(df.columns)
if missing_columns:
    raise ValueError(
        f"stations.csv is missing required columns: {sorted(missing_columns)}"
    )
df = df.rename(columns=input_columns)


# Convert station coordinates to an Earth Engine FeatureCollection.
features = []
for _, row in df.iterrows():
    geometry = ee.Geometry.Point([row['经度'], row['纬度']])
    feature = ee.Feature(
        geometry,
        {'station_id': str(row['站点编号'])}
    )
    features.append(feature)
station_fc = ee.FeatureCollection(features)


# Source datasets and derived raster layers.
dem = ee.Image('USGS/SRTMGL1_003')
landcover = ee.ImageCollection('ESA/WorldCover/v200').first()
urban_mask = landcover.eq(50)
pop = (
    ee.ImageCollection('WorldPop/GP/100m/pop')
    .filter(ee.Filter.eq('country', 'CHN'))
    .filter(ee.Filter.date('2020-01-01', '2020-12-31'))
    .mosaic()
)

# Convert people per approximately 100 m x 100 m cell to people per km^2.
pop_density = pop.multiply(100)

ndvi = (
    ee.ImageCollection('MODIS/061/MOD13A1')
    .filterDate('2021-01-01', '2021-12-31')
    .select('NDVI')
    .mean()
    .multiply(0.0001)
)


# Extract point elevation and buffer-based geographic features.
def extract_features(feature):
    elevation = dem.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=feature.geometry(),
        scale=30
    ).get('elevation')

    buffer_5km = feature.geometry().buffer(5000)
    buffer_3km = feature.geometry().buffer(3000)

    # Terrain roughness is the standard deviation of 90 m sampled elevation.
    roughness_5km = dem.reduceRegion(
        reducer=ee.Reducer.stdDev(),
        geometry=buffer_5km,
        scale=90
    ).get('elevation')

    # The mean of the binary built-up mask gives the 3 km urban ratio.
    urban_ratio_3km = urban_mask.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer_3km,
        scale=30
    ).get('Map')

    ndvi_mean_5km = ndvi.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer_5km,
        scale=500
    ).get('NDVI')

    pop_density_5km = pop_density.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=buffer_5km,
        scale=100
    ).get('population')

    return feature.set({
        'elevation': elevation,
        'terrain_roughness_5km': roughness_5km,
        'urban_ratio_3km': urban_ratio_3km,
        'ndvi_mean_5km': ndvi_mean_5km,
        'population_density_5km': pop_density_5km
    })


results_fc = station_fc.map(extract_features)
features_list = results_fc.getInfo()['features']
extracted_data = [item['properties'] for item in features_list]
df_gee = pd.DataFrame(extracted_data)

# Output: preserve the legacy intermediate file schema.
df_final = pd.merge(
    df,
    df_gee,
    left_on='站点编号',
    right_on='station_id',
    how='left'
)
df_final = df_final.drop(columns=['station_id'])
df_final.to_csv('station_features_gee.csv', index=False)
print("Earth Engine feature extraction completed: station_features_gee.csv")
