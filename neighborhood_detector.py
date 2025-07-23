#!/usr/bin/env python3
"""
Montreal Neighborhood Detector
Uses OpenStreetMap Overpass API to determine neighborhoods based on GPS coordinates.
Outputs enhanced CSV files with accurate neighborhood assignments.

Requirements:
pip install pandas requests shapely

Usage:
python neighborhood_detector.py

Input files: businesses_with_gps.csv, violations_with_gps.csv
Output files: businesses_with_neighborhoods.csv, violations_with_neighborhoods.csv
"""

import pandas as pd
import requests
import json
import os
import time
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import warnings

warnings.filterwarnings('ignore')

# Configuration
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
NEIGHBORHOOD_CACHE_FILE = "montreal_neighborhoods_osm.json"
REQUEST_TIMEOUT = 120
RATE_LIMIT_DELAY = 1  # seconds between API calls

def fetch_montreal_neighborhoods_from_osm():
    """
    Fetch Montreal neighborhood boundaries from OpenStreetMap using Overpass API.
    Returns a dictionary with neighborhood names and their polygon boundaries.
    """
    print("\n=== FETCHING MONTREAL NEIGHBORHOODS FROM OPENSTREETMAP ===")
    
    # Check if we have cached data
    if os.path.exists(NEIGHBORHOOD_CACHE_FILE):
        try:
            with open(NEIGHBORHOOD_CACHE_FILE, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            print(f"✓ Loaded {len(cached_data)} neighborhoods from cache: {NEIGHBORHOOD_CACHE_FILE}")
            return cached_data
        except Exception as e:
            print(f"⚠️ Error loading cache: {e}, fetching fresh data...")
    
    # Comprehensive Overpass query for Montreal neighborhoods
    overpass_query = """
    [out:json][timeout:120];
    (
      // Get Montreal metropolitan area
      relation["name"~"Montréal|Montreal"]["admin_level"~"^(6|8)$"];
      map_to_area -> .montreal_area;
      
      // Get all administrative boundaries within Montreal area
      (
        // Official boroughs and neighborhoods
        relation["admin_level"~"^(9|10)$"]["name"](area.montreal_area);
        
        // Named places and quarters
        way["place"~"^(neighbourhood|suburb|quarter|hamlet)$"]["name"](area.montreal_area);
        relation["place"~"^(neighbourhood|suburb|quarter|hamlet)$"]["name"](area.montreal_area);
        
        // Specific well-known Montreal neighborhoods
        way["name"~"Quartier chinois|Little Italy|Villeray|Plateau|Mile End|Rosemont|Outremont|Verdun|Westmount"]["place"](area.montreal_area);
        relation["name"~"Quartier chinois|Little Italy|Villeray|Plateau|Mile End|Rosemont|Outremont|Verdun|Westmount"]["place"](area.montreal_area);
        
        // Areas with specific landuse that are neighborhoods
        way["landuse"="residential"]["name"]["addr:city"~"Montreal|Montréal"](area.montreal_area);
        
        // Historic or cultural districts
        way["historic"="district"]["name"](area.montreal_area);
        relation["historic"="district"]["name"](area.montreal_area);
      );
    );
    out geom;
    """
    
    try:
        print("Querying OpenStreetMap Overpass API for Montreal neighborhoods...")
        print("This may take 1-2 minutes...")
        
        response = requests.post(
            OVERPASS_API_URL, 
            data=overpass_query, 
            timeout=REQUEST_TIMEOUT,
            headers={'User-Agent': 'Montreal-Neighborhood-Detector/1.0'}
        )
        response.raise_for_status()
        
        data = response.json()
        elements = data.get('elements', [])
        print(f"✓ Received {len(elements)} elements from OpenStreetMap")
        
        neighborhoods = {}
        processed_count = 0
        
        for element in elements:
            if 'tags' not in element or 'name' not in element['tags']:
                continue
                
            name = element['tags']['name'].strip()
            
            # Skip if no name or generic names
            if not name or name.lower() in ['montreal', 'montréal', 'unnamed', 'unknown']:
                continue
            
            try:
                polygons = []
                
                if element['type'] == 'way' and 'geometry' in element:
                    # Create polygon from way geometry
                    coords = [(node['lon'], node['lat']) for node in element['geometry']]
                    if len(coords) >= 3:
                        # Close the polygon if not already closed
                        if coords[0] != coords[-1]:
                            coords.append(coords[0])
                        
                        polygon = Polygon(coords)
                        if polygon.is_valid and polygon.area > 0:
                            polygons.append(polygon)
                
                elif element['type'] == 'relation' and 'members' in element:
                    # Handle relation - try to build multipolygon
                    outer_ways = []
                    inner_ways = []
                    
                    # Collect outer and inner ways
                    for member in element['members']:
                        if member['type'] == 'way' and 'geometry' in member:
                            coords = [(node['lon'], node['lat']) for node in member['geometry']]
                            if len(coords) >= 3:
                                if coords[0] != coords[-1]:
                                    coords.append(coords[0])
                                
                                if member.get('role') == 'outer':
                                    outer_ways.append(coords)
                                elif member.get('role') == 'inner':
                                    inner_ways.append(coords)
                                else:
                                    # Default to outer if role is unclear
                                    outer_ways.append(coords)
                    
                    # Create polygons from outer ways
                    for way_coords in outer_ways:
                        try:
                            polygon = Polygon(way_coords)
                            if polygon.is_valid and polygon.area > 0:
                                polygons.append(polygon)
                        except Exception:
                            continue
                
                # Process the polygons we found
                if polygons:
                    if len(polygons) == 1:
                        final_geometry = polygons[0]
                    else:
                        # Merge multiple polygons
                        try:
                            final_geometry = unary_union(polygons)
                        except Exception:
                            final_geometry = polygons[0]  # Use first polygon as fallback
                    
                    # Convert to serializable format
                    if isinstance(final_geometry, Polygon):
                        coords_list = list(final_geometry.exterior.coords)
                    elif isinstance(final_geometry, MultiPolygon):
                        # For multipolygon, use the largest polygon
                        largest_polygon = max(final_geometry.geoms, key=lambda p: p.area)
                        coords_list = list(largest_polygon.exterior.coords)
                    else:
                        continue
                    
                    # Store neighborhood data
                    neighborhoods[name] = {
                        'type': 'Polygon',
                        'coordinates': coords_list,
                        'admin_level': element['tags'].get('admin_level', 'unknown'),
                        'place_type': element['tags'].get('place', 'unknown'),
                        'area': final_geometry.area,
                        'source': element['type']
                    }
                    
                    processed_count += 1
                    if processed_count % 10 == 0:
                        print(f"  Processed {processed_count} neighborhoods...")
                    
            except Exception as e:
                print(f"  ⚠️ Error processing {name}: {e}")
                continue
        
        print(f"✓ Successfully processed {len(neighborhoods)} Montreal neighborhoods")
        
        # Show some examples of what we found
        print(f"\nSample neighborhoods found:")
        for i, (name, data) in enumerate(list(neighborhoods.items())[:10]):
            area_km2 = data['area'] * 111320 * 111320 / 1000000  # Rough conversion to km²
            print(f"  • {name} (admin_level: {data['admin_level']}, ~{area_km2:.2f} km²)")
        
        # Cache the results
        try:
            with open(NEIGHBORHOOD_CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(neighborhoods, f, ensure_ascii=False, indent=2)
            print(f"✓ Cached neighborhoods to {NEIGHBORHOOD_CACHE_FILE}")
        except Exception as e:
            print(f"⚠️ Error caching data: {e}")
        
        return neighborhoods
        
    except requests.RequestException as e:
        print(f"✗ Error fetching from Overpass API: {e}")
        print("This could be due to:")
        print("  - Network connectivity issues")
        print("  - Overpass API server overload")
        print("  - Query timeout (try again later)")
        return {}
    except Exception as e:
        print(f"✗ Error processing OpenStreetMap data: {e}")
        return {}

def get_neighborhood_from_coordinates(lat, lon, osm_neighborhoods):
    """
    Determine neighborhood based on GPS coordinates using OSM boundaries.
    Returns neighborhood name or None if not found.
    """
    if pd.isna(lat) or pd.isna(lon) or not osm_neighborhoods:
        return None
    
    # Basic bounds check for Montreal area
    if not (45.3 <= lat <= 45.8 and -74.1 <= lon <= -73.3):
        return None
    
    point = Point(lon, lat)  # Note: Shapely Point takes (longitude, latitude)
    
    # Sort neighborhoods by area (smallest first) for more precise matching
    sorted_neighborhoods = sorted(
        osm_neighborhoods.items(),
        key=lambda x: x[1].get('area', float('inf'))
    )
    
    for neighborhood_name, neighborhood_data in sorted_neighborhoods:
        try:
            coords = neighborhood_data['coordinates']
            polygon = Polygon(coords)
            
            if polygon.is_valid and polygon.contains(point):
                return neighborhood_name
        except Exception:
            continue
    
    return None

def parse_gps_column(gps_string):
    """
    Parse GPS string to extract latitude and longitude.
    Handles various formats like "lat,lon" or "POINT(lon lat)".
    """
    if pd.isna(gps_string) or not gps_string:
        return None, None
    
    try:
        gps_str = str(gps_string).strip()
        
        # Handle POINT(lon lat) format
        if gps_str.startswith('POINT(') and gps_str.endswith(')'):
            coords_part = gps_str[6:-1]  # Remove 'POINT(' and ')'
            coords = coords_part.split()
            if len(coords) >= 2:
                lon = float(coords[0])
                lat = float(coords[1])
                return lat, lon
        
        # Handle "lat,lon" or "lon,lat" format
        elif ',' in gps_str:
            coords = gps_str.split(',')
            if len(coords) >= 2:
                coord1 = float(coords[0].strip())
                coord2 = float(coords[1].strip())
                
                # Determine which is lat and which is lon based on Montreal bounds
                # Montreal: lat ~45.4-45.8, lon ~-74.1 to -73.3
                if 45.0 <= coord1 <= 46.0 and -75.0 <= coord2 <= -73.0:
                    return coord1, coord2  # lat, lon
                elif 45.0 <= coord2 <= 46.0 and -75.0 <= coord1 <= -73.0:
                    return coord2, coord1  # lat, lon (swapped)
                else:
                    # Default assumption: first is lat, second is lon
                    return coord1, coord2
        
        # Handle space-separated coordinates
        elif ' ' in gps_str:
            coords = gps_str.split()
            if len(coords) >= 2:
                coord1 = float(coords[0])
                coord2 = float(coords[1])
                
                # Same logic as above
                if 45.0 <= coord1 <= 46.0 and -75.0 <= coord2 <= -73.0:
                    return coord1, coord2
                elif 45.0 <= coord2 <= 46.0 and -75.0 <= coord1 <= -73.0:
                    return coord2, coord1
                else:
                    return coord1, coord2
    
    except (ValueError, IndexError):
        pass
    
    return None, None

def process_csv_with_neighborhoods(input_file, output_file, osm_neighborhoods, lat_col=None, lon_col=None, gps_col=None):
    """
    Process a CSV file and add neighborhood information based on GPS coordinates.
    Can handle either separate lat/lon columns or a combined GPS column.
    """
    print(f"\n=== PROCESSING {input_file} ===")
    
    try:
        # Load the CSV file
        df = pd.read_csv(input_file)
        print(f"✓ Loaded {len(df)} records from {input_file}")
        print(f"Columns available: {list(df.columns)}")
        
        # Determine how to extract coordinates
        if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
            # Use separate lat/lon columns
            print(f"✓ Using separate columns: {lat_col}, {lon_col}")
            df['parsed_lat'] = df[lat_col]
            df['parsed_lon'] = df[lon_col]
            valid_coords = df[['parsed_lat', 'parsed_lon']].notna().all(axis=1)
            
        elif gps_col and gps_col in df.columns:
            # Parse GPS column
            print(f"✓ Parsing GPS data from column: {gps_col}")
            print(f"Sample GPS values: {df[gps_col].dropna().head(3).tolist()}")
            
            # Parse GPS coordinates
            gps_parsed = df[gps_col].apply(parse_gps_column)
            df['parsed_lat'] = gps_parsed.apply(lambda x: x[0] if x else None)
            df['parsed_lon'] = gps_parsed.apply(lambda x: x[1] if x else None)
            valid_coords = df[['parsed_lat', 'parsed_lon']].notna().all(axis=1)
            
        else:
            print(f"✗ No suitable GPS columns found in {input_file}")
            print(f"Expected: ({lat_col}, {lon_col}) or {gps_col}")
            return False
        
        print(f"✓ Found {valid_coords.sum()} records with valid GPS coordinates")
        
        if valid_coords.sum() == 0:
            print(f"⚠️ No valid GPS coordinates found in {input_file}")
            return False
        
        # Apply neighborhood detection
        print("Determining neighborhoods... (this may take a few minutes)")
        
        neighborhoods = []
        processed = 0
        found_neighborhoods = 0
        
        for idx, row in df.iterrows():
            if valid_coords.iloc[idx]:
                lat = row['parsed_lat']
                lon = row['parsed_lon']
                neighborhood = get_neighborhood_from_coordinates(lat, lon, osm_neighborhoods)
                neighborhoods.append(neighborhood)
                
                if neighborhood:
                    found_neighborhoods += 1
            else:
                neighborhoods.append(None)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed}/{len(df)} records... ({found_neighborhoods} neighborhoods found)")
        
        # Add neighborhood column
        df['neighborhood'] = neighborhoods
        
        # Drop temporary parsing columns if they were created
        if 'parsed_lat' in df.columns and lat_col != 'parsed_lat':
            df = df.drop(['parsed_lat', 'parsed_lon'], axis=1)
        
        # Show statistics
        neighborhood_counts = df['neighborhood'].value_counts()
        print(f"\n✓ Neighborhood assignment complete:")
        print(f"  Records with neighborhoods: {found_neighborhoods}/{len(df)} ({found_neighborhoods/len(df)*100:.1f}%)")
        print(f"  Unique neighborhoods found: {len(neighborhood_counts)}")
        
        print(f"\nTop neighborhoods in {input_file}:")
        for neighborhood, count in neighborhood_counts.head(10).items():
            if neighborhood:
                print(f"  {neighborhood}: {count} records")
        
        # Save the enhanced CSV
        df.to_csv(output_file, index=False)
        print(f"✓ Saved enhanced data to {output_file}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_file}: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to process both CSV files and add neighborhood information.
    """
    print("🗺️ MONTREAL NEIGHBORHOOD DETECTOR")
    print("Using OpenStreetMap data via Overpass API")
    print("=" * 50)
    
    # Check if input files exist
    businesses_file = 'businesses_with_gps.csv'
    violations_file = 'violations_with_gps.csv'
    
    if not os.path.exists(businesses_file):
        print(f"✗ Input file not found: {businesses_file}")
        return
    
    if not os.path.exists(violations_file):
        print(f"✗ Input file not found: {violations_file}")
        return
    
    # Fetch neighborhood boundaries from OpenStreetMap
    osm_neighborhoods = fetch_montreal_neighborhoods_from_osm()
    
    if not osm_neighborhoods:
        print("✗ Failed to obtain neighborhood boundaries from OpenStreetMap")
        print("Cannot proceed without boundary data.")
        return
    
    print(f"\n✅ Successfully loaded {len(osm_neighborhoods)} neighborhood boundaries")
    
    # Process businesses file
    business_success = process_csv_with_neighborhoods(
        businesses_file,
        'businesses_with_neighborhoods.csv',
        osm_neighborhoods,
        lat_col='latitude',
        lon_col='longitude'
    )
    
    # Process violations file (uses GPS column instead of separate lat/lon)
    violations_success = process_csv_with_neighborhoods(
        violations_file,
        'violations_with_neighborhoods.csv', 
        osm_neighborhoods,
        gps_col='gps'
    )
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*50}")
    
    if business_success:
        print("✅ businesses_with_neighborhoods.csv - Successfully created")
    else:
        print("❌ businesses_with_neighborhoods.csv - Failed to create")
    
    if violations_success:
        print("✅ violations_with_neighborhoods.csv - Successfully created")
    else:
        print("❌ violations_with_neighborhoods.csv - Failed to create")
    
    if business_success and violations_success:
        print("\n🎉 All files processed successfully!")
        print("You can now use the enhanced CSV files for neighborhood-based analysis.")
    else:
        print("\n⚠️ Some files failed to process. Check the error messages above.")
    
    print(f"\nCache file created: {NEIGHBORHOOD_CACHE_FILE}")
    print("(Delete this file to force fresh data download from OpenStreetMap)")

if __name__ == "__main__":
    main()