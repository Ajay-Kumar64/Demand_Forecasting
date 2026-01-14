from pathlib import Path
import geopandas as gpd
import h3


H3_RESOLUTION = 8


def build_zone_h3_lookup(shapefile_path: Path) -> dict:
    # Read shapefile
    gdf = gpd.read_file(shapefile_path)

    # Step 1: Project to NYC local CRS (meters)
    gdf = gdf.to_crs(epsg=2263)

    # Step 2: Compute centroid in projected CRS
    gdf["centroid"] = gdf.geometry.centroid

    # Step 3: Reproject centroids back to WGS84
    centroids = gpd.GeoSeries(gdf["centroid"], crs=2263).to_crs(epsg=4326)

    gdf["lat"] = centroids.y
    gdf["lon"] = centroids.x

    # Step 4: Convert centroid → H3
    gdf["h3_cell"] = gdf.apply(
        lambda r: h3.latlng_to_cell(
            r["lat"],
            r["lon"],
            H3_RESOLUTION
        ),
        axis=1
    )

    # Final mapping
    return dict(zip(gdf["LocationID"], gdf["h3_cell"]))
