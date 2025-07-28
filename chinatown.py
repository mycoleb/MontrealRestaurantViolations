#!/usr/bin/env python3
"""
Montreal Business Violations Neighborhood Analysis

This script analyzes business violation data for Montreal, assigns businesses to
neighborhoods, and creates an interactive map showing violation ratios by neighborhood.

Requirements:
    pip install pandas geopandas folium shapely requests

Usage:
    python montreal_analysis.py

Files needed in the same directory:
    - businesses_with_gps_corrected.csv
    - violations_with_gps_corrected.csv
"""

import pandas as pd
import requests
import folium
from shapely.geometry import Point, Polygon
import geopandas as gpd
from collections import defaultdict
import json
import time
from folium import plugins
import os
import sys
import ssl
from geopy.geocoders import Nominatim
from sklearn.cluster import DBSCAN
import numpy as np

# Disable SSL certificate verification for problematic sources
ssl._create_default_https_context = ssl._create_unverified_context

def load_data():
    """Load the business and violations data from CSV files."""
    print("Loading data...")
    
    # Check if files exist
    business_file = 'businesses_with_gps_corrected.csv'
    violations_file = 'violations_with_gps_corrected.csv'
    
    if not os.path.exists(business_file):
        print(f"Error: {business_file} not found in current directory")
        return None, None
    
    if not os.path.exists(violations_file):
        print(f"Error: {violations_file} not found in current directory")
        return None, None
    
    try:
        # Load businesses data (using corrected GPS as primary, original as fallback)
        businesses = pd.read_csv(business_file)
        violations = pd.read_csv(violations_file)
        
        # Clean and prepare GPS coordinates
        businesses = businesses.dropna(subset=['latitude', 'longitude'])
        businesses = businesses[(businesses['latitude'] != 0) & (businesses['longitude'] != 0)]
        
        print(f"Loaded {len(businesses)} businesses with valid GPS coordinates")
        print(f"Loaded {len(violations)} violations")
        
        return businesses, violations
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def get_montreal_neighborhoods():
    """Get Montreal neighborhood boundaries from Open Data Montreal API."""
    print("Fetching Montreal neighborhood boundaries...")
    
    # Try multiple Montreal Open Data API endpoints
    urls = [
        # Reference residential neighborhoods (quartiers de référence)
        "https://donnees.montreal.ca/dataset/f38c91a1-e33f-4475-a112-3b84b1c60c1e/resource/a80e611f-5336-4306-ba2a-fd657f0f00fa/download/quartierreferencehabitation.geojson",
        # Administrative boundaries (boroughs)
        "https://donnees.montreal.ca/dataset/9797a946-9da8-41ec-8815-f6b276dec7e9/resource/e18bfd07-edc8-4ce8-8a5a-3b617662a794/download/limites-administratives-agglomeration.geojson",
        # GitHub backup source
        "https://raw.githubusercontent.com/blackmad/neighborhoods/master/montreal.geojson"
    ]
    
    for i, url in enumerate(urls):
        try:
            print(f"Trying source {i+1}/3: {url.split('/')[-1]}")
            response = requests.get(url, timeout=30, verify=False)
            if response.status_code == 200:
                neighborhoods = gpd.read_file(url)
                print(f"Successfully loaded {len(neighborhoods)} neighborhoods from source {i+1}")
                print(f"Available columns: {list(neighborhoods.columns)}")
                
                # Standardize column names - look for common neighborhood name fields
                possible_name_fields = ['NOM', 'NAME', 'nom', 'name', 'QUARTIER', 'quartier', 'ARROND', 'arrond']
                name_field = None
                
                for field in possible_name_fields:
                    if field in neighborhoods.columns:
                        name_field = field
                        print(f"Found name field: {field}")
                        break
                
                if name_field and name_field != 'NOM':
                    neighborhoods = neighborhoods.rename(columns={name_field: 'NOM'})
                elif 'NOM' not in neighborhoods.columns:
                    # Create generic names if no name field found
                    neighborhoods['NOM'] = [f'Area_{i}' for i in range(len(neighborhoods))]
                    print("Warning: No neighborhood name field found, using generic names")
                
                # Print sample of neighborhood names for debugging
                print(f"Sample neighborhood names: {list(neighborhoods['NOM'].head(10))}")
                
                return neighborhoods
            else:
                print(f"Failed to fetch from source {i+1} (status: {response.status_code})")
                
        except Exception as e:
            print(f"Error with source {i+1}: {e}")
            continue
    
    print("All neighborhood data sources failed")
    return None

def get_neighborhoods_alternative_sources():
    """Try alternative sources for neighborhood data."""
    print("Trying alternative neighborhood data sources...")
    
    alternative_urls = [
        "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/montreal.geojson",
        "https://raw.githubusercontent.com/codeforgermany/click_that_hood/master/public/data/montreal.geojson"
    ]
    
    for i, url in enumerate(alternative_urls):
        try:
            print(f"Trying alternative source {i+1}: {url.split('/')[-1]}")
            response = requests.get(url, timeout=30, verify=False)
            if response.status_code == 200:
                neighborhoods = gpd.read_file(url)
                print(f"Successfully loaded {len(neighborhoods)} neighborhoods from alternative source {i+1}")
                
                # Standardize column names
                if 'name' in neighborhoods.columns:
                    neighborhoods = neighborhoods.rename(columns={'name': 'NOM'})
                elif 'NOM' not in neighborhoods.columns:
                    neighborhoods['NOM'] = [f'Area_{i}' for i in range(len(neighborhoods))]
                
                return neighborhoods
            else:
                print(f"Failed to fetch from alternative source {i+1}")
                
        except Exception as e:
            print(f"Error with alternative source {i+1}: {e}")
            continue
    
    return None

def get_neighborhoods_overpass_api():
    """Try to get neighborhood data from Overpass API."""
    print("Trying Overpass API for neighborhood data...")
    
    try:
        # Overpass query for Montreal administrative boundaries
        overpass_url = "http://overpass-api.de/api/interpreter"
        overpass_query = """
        [out:json][timeout:25];
        (
          relation["admin_level"~"^(8|9|10)$"]["place"~"suburb|neighbourhood"]["name"]["boundary"="administrative"](45.4,-73.8,45.7,-73.4);
        );
        out geom;
        """
        
        response = requests.post(overpass_url, data={'data': overpass_query}, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if data['elements']:
                print(f"Found {len(data['elements'])} administrative boundaries from Overpass API")
                # Convert to GeoDataFrame (simplified - would need more complex processing)
                return None  # For now, return None as this requires complex OSM data processing
        
    except Exception as e:
        print(f"Error with Overpass API: {e}")
    
    return None

def get_neighborhoods_nominatim(businesses, max_businesses=1500):
    """Use Nominatim to identify neighborhoods for business locations."""
    print(f"Using Nominatim to identify neighborhoods for up to {max_businesses} businesses...")
    
    try:
        geolocator = Nominatim(user_agent="montreal_violations_analysis")
        neighborhood_assignments = {}
        
        # Sample businesses if we have too many
        business_sample = businesses.sample(min(max_businesses, len(businesses))) if len(businesses) > max_businesses else businesses
        
        for idx, (_, business) in enumerate(business_sample.iterrows()):
            if idx % 100 == 0:
                print(f"Processed {idx}/{len(business_sample)} businesses...")
            
            try:
                location = geolocator.reverse(f"{business['latitude']}, {business['longitude']}", timeout=10)
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    neighborhood = (address.get('suburb') or 
                                  address.get('neighbourhood') or 
                                  address.get('quarter') or 
                                  address.get('city_district') or 
                                  'Unknown')
                    neighborhood_assignments[business['business_id']] = neighborhood
                
                # Add delay to respect Nominatim usage policy
                time.sleep(0.1)
                
            except Exception as e:
                if idx < 10:  # Only print first few errors
                    print(f"Error geocoding business {business['business_id']}: {e}")
                continue
        
        print(f"Successfully identified neighborhoods for {len(neighborhood_assignments)} businesses")
        return neighborhood_assignments
        
    except Exception as e:
        print(f"Error in Nominatim geocoding: {e}")
        return {}

def create_neighborhood_polygons_from_businesses(businesses, neighborhood_assignments):
    """Create approximate neighborhood polygons from business clusters."""
    print("Creating neighborhood polygons from business clusters...")
    
    try:
        # Group businesses by neighborhood
        neighborhood_businesses = defaultdict(list)
        for business_id, neighborhood in neighborhood_assignments.items():
            business = businesses[businesses['business_id'] == business_id]
            if not business.empty:
                neighborhood_businesses[neighborhood].append([
                    business.iloc[0]['latitude'], 
                    business.iloc[0]['longitude']
                ])
        
        neighborhoods_list = []
        
        for neighborhood, coords in neighborhood_businesses.items():
            if len(coords) < 3:  # Need at least 3 points for a polygon
                continue
            
            try:
                # Use DBSCAN to cluster points and create convex hull
                coords_array = np.array(coords)
                
                # Create a simple convex hull around the points
                from scipy.spatial import ConvexHull
                if len(coords) >= 3:
                    hull = ConvexHull(coords_array)
                    hull_coords = [(coords_array[vertex][1], coords_array[vertex][0]) for vertex in hull.vertices]
                    polygon = Polygon(hull_coords)
                    
                    neighborhoods_list.append({
                        'NOM': neighborhood,
                        'geometry': polygon
                    })
            except Exception as e:
                print(f"Error creating polygon for {neighborhood}: {e}")
                continue
        
        if neighborhoods_list:
            neighborhoods_gdf = gpd.GeoDataFrame(neighborhoods_list, crs='EPSG:4326')
            print(f"Created {len(neighborhoods_gdf)} neighborhood polygons")
            return neighborhoods_gdf
        
    except Exception as e:
        print(f"Error creating neighborhood polygons: {e}")
    
    return None

def improve_unknown_assignments(business_neighborhoods, violations, max_unknown_to_process=500):
    """Try to improve neighborhood assignments for businesses marked as 'Unknown'."""
    print("Improving neighborhood assignments for unknown businesses...")
    
    unknown_businesses = business_neighborhoods[business_neighborhoods['neighborhood'] == 'Unknown']
    
    if len(unknown_businesses) == 0:
        return business_neighborhoods
    
    print(f"Found {len(unknown_businesses)} businesses with unknown neighborhoods")
    
    # Sample if too many unknowns
    if len(unknown_businesses) > max_unknown_to_process:
        unknown_sample = unknown_businesses.sample(max_unknown_to_process)
        print(f"Processing sample of {max_unknown_to_process} unknown businesses")
    else:
        unknown_sample = unknown_businesses
    
    try:
        geolocator = Nominatim(user_agent="montreal_violations_analysis_improve")
        
        for idx, (business_idx, business) in enumerate(unknown_sample.iterrows()):
            if idx % 50 == 0:
                print(f"Improved {idx}/{len(unknown_sample)} unknown businesses...")
            
            try:
                location = geolocator.reverse(f"{business['latitude']}, {business['longitude']}", timeout=10)
                if location and location.raw.get('address'):
                    address = location.raw['address']
                    neighborhood = (address.get('suburb') or 
                                  address.get('neighbourhood') or 
                                  address.get('quarter') or 
                                  address.get('city_district') or 
                                  'Unknown')
                    
                    if neighborhood != 'Unknown':
                        business_neighborhoods.loc[business_idx, 'neighborhood'] = neighborhood
                
                time.sleep(0.1)  # Respect rate limits
                
            except Exception as e:
                continue
    
    except Exception as e:
        print(f"Error improving unknown assignments: {e}")
    
    return business_neighborhoods

def create_chinatown_polygon():
    """Create a special polygon for Quartier Chinois."""
    chinatown_coords = [
        [45.509282, -73.561113],
        [45.507696, -73.562595], 
        [45.507041, -73.561215],
        [45.505557, -73.562643],
        [45.504930, -73.561120],
        [45.508093, -73.558214]
    ]
    
    # Convert to Polygon (note: shapely expects (lon, lat) format)
    polygon_coords = [(coord[1], coord[0]) for coord in chinatown_coords]
    return Polygon(polygon_coords)

def assign_neighborhoods(businesses, neighborhoods_gdf):
    """Assign each business to a neighborhood using spatial joins."""
    print("Assigning businesses to neighborhoods...")
    
    # Create GeoDataFrame from businesses
    business_points = gpd.GeoDataFrame(
        businesses,
        geometry=gpd.points_from_xy(businesses.longitude, businesses.latitude),
        crs='EPSG:4326'
    )
    
    # Initialize neighborhood column
    business_points['neighborhood'] = None
    
    # Ensure neighborhoods are in the same CRS
    if neighborhoods_gdf is not None:
        if neighborhoods_gdf.crs != 'EPSG:4326':
            neighborhoods_gdf = neighborhoods_gdf.to_crs('EPSG:4326')
    
    # Create Chinatown as a special neighborhood
    chinatown_poly = create_chinatown_polygon()
    chinatown_gdf = gpd.GeoDataFrame(
        {'NOM': ['Quartier Chinois'], 'geometry': [chinatown_poly]},
        crs='EPSG:4326'
    )
    
    # First, assign businesses to Chinatown
    try:
        chinatown_businesses = gpd.sjoin(business_points, chinatown_gdf, how='inner', predicate='within')
        business_points.loc[chinatown_businesses.index, 'neighborhood'] = 'Quartier Chinois'
        print(f"Found {len(chinatown_businesses)} businesses in Quartier Chinois")
    except Exception as e:
        print(f"Error assigning Chinatown businesses: {e}")
    
    # Assign remaining businesses to other neighborhoods if available
    if neighborhoods_gdf is not None:
        try:
            # Remove Chinatown businesses from consideration for other neighborhoods
            remaining_businesses = business_points[business_points['neighborhood'].isna()]
            
            # Assign remaining businesses to other neighborhoods
            business_neighborhoods = gpd.sjoin(remaining_businesses, neighborhoods_gdf, how='left', predicate='within')
            
            # Update the main dataframe with neighborhood assignments
            business_points.loc[business_neighborhoods.index, 'neighborhood'] = business_neighborhoods['NOM']
            
        except Exception as e:
            print(f"Error in spatial join: {e}")
    
    # Handle businesses not assigned to any neighborhood
    unassigned = business_points['neighborhood'].isna().sum()
    if unassigned > 0:
        print(f"Warning: {unassigned} businesses could not be assigned to neighborhoods")
        business_points['neighborhood'] = business_points['neighborhood'].fillna('Unknown')
    
    return business_points

def calculate_violation_ratios(business_neighborhoods, violations):
    """Calculate violation ratios for each neighborhood."""
    print("Calculating violation ratios...")
    
    # Count businesses per neighborhood
    business_counts = business_neighborhoods['neighborhood'].value_counts()
    
    # Merge violations with business neighborhoods
    violations_with_neighborhoods = violations.merge(
        business_neighborhoods[['business_id', 'neighborhood']], 
        on='business_id', 
        how='left'
    )
    
    # Count violations per neighborhood
    violation_counts = violations_with_neighborhoods['neighborhood'].value_counts()
    
    # Calculate ratios
    ratios = {}
    for neighborhood in business_counts.index:
        business_count = business_counts[neighborhood]
        violation_count = violation_counts.get(neighborhood, 0)
        ratio = violation_count / business_count if business_count > 0 else 0
        ratios[neighborhood] = {
            'businesses': business_count,
            'violations': violation_count,
            'ratio': ratio
        }
    
    return ratios

def create_interactive_map(neighborhoods_gdf, ratios):
    """Create an interactive Folium map with neighborhood polygons and violation ratios."""
    print("Creating interactive map...")
    
    # Center the map on Montreal
    montreal_center = [45.5017, -73.5673]
    m = folium.Map(location=montreal_center, zoom_start=11, tiles='OpenStreetMap')
    
    # Color scale for violation ratios
    max_ratio = max([data['ratio'] for data in ratios.values()]) if ratios else 1
    
    def get_color(ratio):
        """Get color based on violation ratio."""
        if ratio == 0:
            return 'green'
        elif ratio < max_ratio * 0.3:
            return 'yellow'
        elif ratio < max_ratio * 0.6:
            return 'orange'
        else:
            return 'red'
    
    # Add Chinatown polygon with enhanced tooltip
    chinatown_poly = create_chinatown_polygon()
    chinatown_coords = [[coord[1], coord[0]] for coord in chinatown_poly.exterior.coords]
    
    chinatown_ratio = ratios.get('Quartier Chinois', {'businesses': 0, 'violations': 0, 'ratio': 0})
    chinatown_color = get_color(chinatown_ratio['ratio'])
    
    # Enhanced tooltip with better styling
    chinatown_tooltip = f"""
    <div style='font-family: Arial; font-size: 14px; font-weight: bold; color: #333;'>
        <strong>🏮 Quartier Chinois</strong><br>
        <span style='font-size: 12px; font-weight: normal;'>
            Violation Ratio: {chinatown_ratio['ratio']:.3f}<br>
            Businesses: {chinatown_ratio['businesses']}<br>
            Violations: {chinatown_ratio['violations']}
        </span>
    </div>
    """
    
    folium.Polygon(
        locations=chinatown_coords,
        color=chinatown_color,
        weight=3,
        fillColor=chinatown_color,
        fillOpacity=0.6,
        popup=folium.Popup(
            f"""<div style='font-family: Arial; padding: 10px;'>
            <h4 style='margin: 0 0 10px 0; color: #d63031;'>🏮 Quartier Chinois</h4>
            <table style='font-size: 12px; border-collapse: collapse;'>
                <tr><td style='padding: 2px 10px 2px 0; font-weight: bold;'>Businesses:</td><td>{chinatown_ratio['businesses']}</td></tr>
                <tr><td style='padding: 2px 10px 2px 0; font-weight: bold;'>Violations:</td><td>{chinatown_ratio['violations']}</td></tr>
                <tr><td style='padding: 2px 10px 2px 0; font-weight: bold;'>Violation Ratio:</td><td>{chinatown_ratio['ratio']:.3f}</td></tr>
            </table>
            </div>""",
            max_width=300
        ),
        tooltip=folium.Tooltip(
            chinatown_tooltip,
            permanent=False,
            sticky=True,
            style="background-color: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 5px; padding: 8px;"
        )
    ).add_to(m)
    
    # Add other neighborhood polygons with enhanced tooltips
    if neighborhoods_gdf is not None:
        for idx, row in neighborhoods_gdf.iterrows():
            neighborhood_name = row.get('NOM', f'Neighborhood_{idx}')
            
            # Try to get a better name from multiple possible fields
            if neighborhood_name.startswith('Area_') or neighborhood_name.startswith('Neighborhood_'):
                # Look for other name fields
                for possible_field in ['name', 'NAME', 'nom', 'NOM', 'QUARTIER', 'quartier', 'ARROND', 'arrond']:
                    if possible_field in row and pd.notna(row[possible_field]) and str(row[possible_field]).strip():
                        neighborhood_name = str(row[possible_field]).strip()
                        break
            
            # Skip if this is Chinatown (already added)
            if neighborhood_name == 'Quartier Chinois':
                continue
                
            neighborhood_data = ratios.get(neighborhood_name, {'businesses': 0, 'violations': 0, 'ratio': 0})
            
            # Skip neighborhoods with no businesses
            if neighborhood_data['businesses'] == 0:
                continue
            
            try:
                # Convert geometry to GeoJSON-like format for Folium
                if row.geometry.geom_type == 'Polygon':
                    coords = [[point[1], point[0]] for point in row.geometry.exterior.coords]
                elif row.geometry.geom_type == 'MultiPolygon':
                    coords = []
                    for polygon in row.geometry.geoms:
                        coords.append([[point[1], point[0]] for point in polygon.exterior.coords])
                else:
                    continue
                
                color = get_color(neighborhood_data['ratio'])
                
                # Enhanced tooltip for regular neighborhoods
                enhanced_tooltip = f"""
                <div style='font-family: Arial; font-size: 14px; font-weight: bold; color: #333;'>
                    <strong>📍 {neighborhood_name}</strong><br>
                    <span style='font-size: 12px; font-weight: normal;'>
                        Violation Ratio: {neighborhood_data['ratio']:.3f}<br>
                        Businesses: {neighborhood_data['businesses']}<br>
                        Violations: {neighborhood_data['violations']}
                    </span>
                </div>
                """
                
                enhanced_popup = f"""
                <div style='font-family: Arial; padding: 10px;'>
                    <h4 style='margin: 0 0 10px 0; color: #0984e3;'>📍 {neighborhood_name}</h4>
                    <table style='font-size: 12px; border-collapse: collapse;'>
                        <tr><td style='padding: 2px 10px 2px 0; font-weight: bold;'>Businesses:</td><td>{neighborhood_data['businesses']}</td></tr>
                        <tr><td style='padding: 2px 10px 2px 0; font-weight: bold;'>Violations:</td><td>{neighborhood_data['violations']}</td></tr>
                        <tr><td style='padding: 2px 10px 2px 0; font-weight: bold;'>Violation Ratio:</td><td>{neighborhood_data['ratio']:.3f}</td></tr>
                    </table>
                </div>
                """
                
                if row.geometry.geom_type == 'Polygon':
                    folium.Polygon(
                        locations=coords,
                        color=color,
                        weight=2,
                        fillColor=color,
                        fillOpacity=0.3,
                        popup=folium.Popup(enhanced_popup, max_width=300),
                        tooltip=folium.Tooltip(
                            enhanced_tooltip,
                            permanent=False,
                            sticky=True,
                            style="background-color: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 5px; padding: 8px;"
                        )
                    ).add_to(m)
                else:  # MultiPolygon
                    for coord_set in coords:
                        folium.Polygon(
                            locations=coord_set,
                            color=color,
                            weight=2,
                            fillColor=color,
                            fillOpacity=0.3,
                            popup=folium.Popup(enhanced_popup, max_width=300),
                            tooltip=folium.Tooltip(
                                enhanced_tooltip,
                                permanent=False,
                                sticky=True,
                                style="background-color: rgba(255,255,255,0.95); border: 2px solid #333; border-radius: 5px; padding: 8px;"
                            )
                        ).add_to(m)
                        
            except Exception as e:
                print(f"Error adding polygon for {neighborhood_name}: {e}")
                continue
    
    # Add an enhanced legend with better styling
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 250px; height: 140px; 
                background-color: rgba(255,255,255,0.95); 
                border: 2px solid #333; 
                border-radius: 10px;
                z-index:9999; 
                font-size:14px; 
                padding: 15px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif;">
    <h4 style="margin: 0 0 10px 0; color: #333; text-align: center;">Violation Ratio Legend</h4>
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <div style="width: 20px; height: 15px; background-color: green; margin-right: 10px; border: 1px solid #333;"></div>
        <span>No violations (0)</span>
    </div>
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <div style="width: 20px; height: 15px; background-color: yellow; margin-right: 10px; border: 1px solid #333;"></div>
        <span>Low (0-30% of max)</span>
    </div>
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <div style="width: 20px; height: 15px; background-color: orange; margin-right: 10px; border: 1px solid #333;"></div>
        <span>Medium (30-60% of max)</span>
    </div>
    <div style="display: flex; align-items: center; margin: 5px 0;">
        <div style="width: 20px; height: 15px; background-color: red; margin-right: 10px; border: 1px solid #333;"></div>
        <span>High (60%+ of max)</span>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add a title to the map
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50%; transform: translateX(-50%);
                background-color: rgba(255,255,255,0.95); 
                border: 2px solid #333; 
                border-radius: 10px;
                z-index:9999; 
                font-size:18px; 
                padding: 10px 20px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.3);
                font-family: Arial, sans-serif;
                font-weight: bold;
                color: #333;">
        Montreal Business Violations by Neighborhood
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m

def main():
    """Main function to run the complete analysis."""
    print("Montreal Business Violations Neighborhood Analysis")
    print("=" * 55)
    
    try:
        # Load data
        businesses, violations = load_data()
        
        if businesses is None or violations is None:
            print("Failed to load data. Exiting.")
            return None, None
        
        # Get neighborhood boundaries with multiple fallbacks
        neighborhoods_gdf = get_montreal_neighborhoods()
        
        if neighborhoods_gdf is None:
            print("Official Montreal data failed. Trying alternative sources...")
            neighborhoods_gdf = get_neighborhoods_alternative_sources()
        
        if neighborhoods_gdf is None:
            print("Alternative sources failed. Trying Overpass API...")
            neighborhoods_gdf = get_neighborhoods_overpass_api()
        
        if neighborhoods_gdf is None:
            print("All official sources failed. Using Nominatim to identify neighborhoods...")
            
            # Use Nominatim to identify neighborhoods from business locations
            neighborhood_assignments = get_neighborhoods_nominatim(businesses, max_businesses=1500)
            
            if neighborhood_assignments:
                # Create polygons from business clusters
                neighborhoods_gdf = create_neighborhood_polygons_from_businesses(businesses, neighborhood_assignments)
                
                if neighborhoods_gdf is not None:
                    print("Successfully created neighborhoods using Nominatim")
                else:
                    print("Could not create neighborhood polygons from Nominatim data")
                    neighborhoods_gdf = None
            else:
                print("Nominatim neighborhood detection failed")
                neighborhoods_gdf = None
        
        if neighborhoods_gdf is None:
            # Create minimal analysis with just Chinatown
            print("Creating minimal analysis with Chinatown only...")
            chinatown_poly = create_chinatown_polygon()
            
            # Check which businesses are in Chinatown
            chinatown_businesses = []
            for _, business in businesses.iterrows():
                point = Point(business['longitude'], business['latitude'])
                if chinatown_poly.contains(point):
                    chinatown_businesses.append(business['business_id'])
            
            # Calculate Chinatown ratios
            chinatown_violations = violations[violations['business_id'].isin(chinatown_businesses)]
            ratios = {
                'Quartier Chinois': {
                    'businesses': len(chinatown_businesses),
                    'violations': len(chinatown_violations),
                    'ratio': len(chinatown_violations) / len(chinatown_businesses) if chinatown_businesses else 0
                }
            }
            
            # Create simple map
            m = create_interactive_map(None, ratios)
        else:
            # Full analysis with all neighborhoods
            business_neighborhoods = assign_neighborhoods(businesses, neighborhoods_gdf)
            
            # Try to improve assignments for unknown businesses
            business_neighborhoods = improve_unknown_assignments(business_neighborhoods, violations, max_unknown_to_process=500)
            
            ratios = calculate_violation_ratios(business_neighborhoods, violations)
            m = create_interactive_map(neighborhoods_gdf, ratios)
        
        # Save the map
        output_file = 'montreal_business_violations_map.html'
        m.save(output_file)
        print(f"\nMap saved as '{output_file}'")
        print(f"Open this file in your web browser to view the interactive map.")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print("-" * 70)
        print(f"{'Neighborhood':<30} | {'Businesses':>10} | {'Violations':>10} | {'Ratio':>8}")
        print("-" * 70)
        
        for neighborhood, data in sorted(ratios.items(), key=lambda x: x[1]['ratio'], reverse=True):
            print(f"{neighborhood:<30} | {data['businesses']:>10} | {data['violations']:>10} | {data['ratio']:>8.3f}")
        
        return m, ratios
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None, None
# Load GeoJSON with real neighborhood names
url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/montreal.geojson"
neighborhoods = gpd.read_file(url)

# Ensure 'name' is renamed to 'NOM'
neighborhoods = neighborhoods.rename(columns={'name': 'NOM'})

# Calculate centroids
neighborhoods['centroid'] = neighborhoods.geometry.centroid

# Now you can get latitude and longitude
neighborhoods['latitude'] = neighborhoods.centroid.y
neighborhoods['longitude'] = neighborhoods.centroid.x

# Example: print name and center coordinates
print(neighborhoods[['NOM', 'latitude', 'longitude']].head())
if __name__ == "__main__":
    # Check if required packages are installed
    required_packages = {
        'geopandas': 'geopandas',
        'folium': 'folium', 
        'shapely': 'shapely',
        'geopy': 'geopy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    for package, install_name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(install_name)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install required packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        sys.exit(1)

    # Run the analysis
    map_obj, violation_ratios = main()
    
    if map_obj and violation_ratios:
        print(f"\nAnalysis complete! Found {len(violation_ratios)} neighborhoods.")
        print("Open 'montreal_business_violations_map.html' in your browser to view the interactive map.")
    else:
        print("\nAnalysis failed. Please check the error messages above.")