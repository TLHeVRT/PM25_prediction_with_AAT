import geopandas as gpd
import pandas as pd
from pyproj import CRS
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Input: geographic features produced by run_ee.py.
df = pd.read_csv('station_features_gee.csv')
gdf_stations = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df['经度'], df['纬度']),
    crs='EPSG:4326'
)
coastline = gpd.read_file(
    'ne_10m_coastline.shp/ne_10m_coastline.shp'
)


# Use an Asia-wide equidistant projection for candidate search, then calculate
# the final distance in a station-centered azimuthal equidistant projection.
search_crs = 'ESRI:102026'
gdf_stations_search = gdf_stations.to_crs(search_crs)
coastline_search = coastline.to_crs(search_crs)
coastline_sindex = coastline_search.sindex


def distance_to_coast_km(index):
    station_search = gdf_stations_search.geometry.iloc[index]
    _, nearest_distances = coastline_sindex.nearest(
        station_search,
        return_all=True,
        return_distance=True
    )

    # The safety margin prevents candidate loss from search-projection error.
    search_radius = float(nearest_distances.min()) * 2.0 + 100_000.0
    candidate_indices = coastline_sindex.query(
        station_search,
        predicate='dwithin',
        distance=search_radius
    )

    station = gdf_stations.geometry.iloc[index]
    local_crs = CRS.from_proj4(
        f'+proj=aeqd +lat_0={station.y} +lon_0={station.x} '
        '+datum=WGS84 +units=m +no_defs'
    )
    station_local = gpd.GeoSeries(
        [station],
        crs=gdf_stations.crs
    ).to_crs(local_crs).iloc[0]
    coastline_local = coastline.iloc[candidate_indices].to_crs(local_crs)
    return coastline_local.distance(station_local).min() / 1000.0


df['distance_to_coast_km'] = [
    distance_to_coast_km(index) for index in range(len(df))
]


# Standardize latitude, longitude, and elevation before K-Means clustering.
cluster_features = df[['纬度', '经度', 'elevation']]
cluster_features = cluster_features.fillna(cluster_features.median())
cluster_features_scaled = StandardScaler().fit_transform(cluster_features)
kmeans = KMeans(n_clusters=6, random_state=42)
df['climate_zone'] = kmeans.fit_predict(cluster_features_scaled)
df = pd.get_dummies(df, columns=['climate_zone'], prefix='zone')


# Median-impute and standardize the continuous output features.
numeric_features = [
    'elevation',
    'terrain_roughness_5km',
    'urban_ratio_3km',
    'ndvi_mean_5km',
    'population_density_5km',
    'distance_to_coast_km'
]
df[numeric_features] = df[numeric_features].fillna(
    df[numeric_features].median()
)
scaler = StandardScaler()
df[numeric_features] = scaler.fit_transform(df[numeric_features])


# Output: preserve the legacy final file schema and column order.
final_columns = (
    ['站点编号', '纬度', '经度']
    + numeric_features
    + [column for column in df.columns if column.startswith('zone_')]
)
df_final = df[final_columns]
df_final.to_csv('station_features.csv', index=False)
print("Local feature processing completed: station_features.csv")
