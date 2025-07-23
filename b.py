import pandas as pd
import folium
from folium import plugins
import numpy as np

def create_business_violations_map():
    """
    Create an interactive map showing businesses and violations with different colored pins.
    Blue pins for businesses, red pins for violations.
    """
    
    # Read the CSV files
    print("Loading data...")
    businesses_df = pd.read_csv('businesses_with_gps.csv')
    violations_df = pd.read_csv('violations_with_gps.csv')
    
    # Clean and prepare the data
    print("Preparing data...")
    
    # Parse GPS coordinates for businesses
    businesses_coords = []
    for idx, row in businesses_df.iterrows():
        gps = str(row['gps'])
        if gps and gps != 'nan' and gps != '':
            try:
                # Assuming GPS format is "lat,lon" or similar
                if ',' in gps:
                    lat, lon = gps.split(',')
                    lat, lon = float(lat.strip()), float(lon.strip())
                    # Validate coordinates are reasonable
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        businesses_coords.append({
                            'lat': lat,
                            'lon': lon,
                            'name': row['name'],
                            'address': row['address'],
                            'city': row['city'],
                            'type': row['type'],
                            'status': row['statut']
                        })
            except (ValueError, AttributeError):
                continue
    
    # Parse GPS coordinates for violations
    violations_coords = []
    for idx, row in violations_df.iterrows():
        gps = str(row['gps'])
        if gps and gps != 'nan' and gps != '':
            try:
                if ',' in gps:
                    lat, lon = gps.split(',')
                    lat, lon = float(lat.strip()), float(lon.strip())
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        violations_coords.append({
                            'lat': lat,
                            'lon': lon,
                            'description': row['description'],
                            'address': row['adresse'],
                            'city': row['ville'],
                            'amount': row['montant'],
                            'owner': row['proprietaire'],
                            'category': row['categorie']
                        })
            except (ValueError, AttributeError):
                continue
    
    print(f"Found {len(businesses_coords)} valid business coordinates")
    print(f"Found {len(violations_coords)} valid violation coordinates")
    
    if not businesses_coords and not violations_coords:
        print("No valid coordinates found. Please check your GPS data format.")
        return
    
    # Calculate center point for the map
    all_lats = ([coord['lat'] for coord in businesses_coords] + 
                [coord['lat'] for coord in violations_coords])
    all_lons = ([coord['lon'] for coord in businesses_coords] + 
                [coord['lon'] for coord in violations_coords])
    
    center_lat = np.mean(all_lats)
    center_lon = np.mean(all_lons)
    
    # Create the map
    print("Creating map...")
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add business markers (blue pins)
    business_group = folium.FeatureGroup(name='Businesses')
    for business in businesses_coords:
        popup_text = f"""
        <b>{business['name']}</b><br>
        Address: {business['address']}<br>
        City: {business['city']}<br>
        Type: {business['type']}<br>
        Status: {business['status']}
        """
        
        folium.Marker(
            location=[business['lat'], business['lon']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=business['name'],
            icon=folium.Icon(color='blue', icon='building', prefix='fa')
        ).add_to(business_group)
    
    # Add violation markers (red pins)
    violations_group = folium.FeatureGroup(name='Violations')
    for violation in violations_coords:
        popup_text = f"""
        <b>Violation</b><br>
        Description: {violation['description']}<br>
        Address: {violation['address']}<br>
        City: {violation['city']}<br>
        Amount: ${violation['amount']:,.2f}<br>
        Owner: {violation['owner']}<br>
        Category: {violation['category']}
        """
        
        folium.Marker(
            location=[violation['lat'], violation['lon']],
            popup=folium.Popup(popup_text, max_width=300),
            tooltip=f"Violation: {violation['description'][:50]}...",
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        ).add_to(violations_group)
    
    # Add feature groups to map
    business_group.add_to(m)
    violations_group.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add a marker cluster plugin for better performance with many markers
    if len(businesses_coords) + len(violations_coords) > 1000:
        print("Large dataset detected. Consider using marker clustering for better performance.")
    
    # Add full screen button
    plugins.Fullscreen().add_to(m)
    
    # Add measure control
    plugins.MeasureControl().add_to(m)
    
    # Save the map
    output_file = 'business_violations_map.html'
    m.save(output_file)
    print(f"Map saved as {output_file}")
    print(f"Open {output_file} in your web browser to view the interactive map.")
    
    return m

def create_heatmap_version():
    """
    Alternative version: Create heatmaps for density visualization
    """
    print("\nCreating heatmap version...")
    
    # Read the CSV files
    businesses_df = pd.read_csv('businesses_with_gps.csv')
    violations_df = pd.read_csv('violations_with_gps.csv')
    
    # Parse coordinates
    business_coords = []
    violation_coords = []
    
    # Parse business coordinates
    for idx, row in businesses_df.iterrows():
        gps = str(row['gps'])
        if gps and gps != 'nan' and gps != '':
            try:
                if ',' in gps:
                    lat, lon = gps.split(',')
                    lat, lon = float(lat.strip()), float(lon.strip())
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        business_coords.append([lat, lon])
            except (ValueError, AttributeError):
                continue
    
    # Parse violation coordinates
    for idx, row in violations_df.iterrows():
        gps = str(row['gps'])
        if gps and gps != 'nan' and gps != '':
            try:
                if ',' in gps:
                    lat, lon = gps.split(',')
                    lat, lon = float(lat.strip()), float(lon.strip())
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        violation_coords.append([lat, lon])
            except (ValueError, AttributeError):
                continue
    
    if not business_coords and not violation_coords:
        print("No valid coordinates for heatmap.")
        return
    
    # Calculate center
    all_points = business_coords + violation_coords
    center_lat = np.mean([point[0] for point in all_points])
    center_lon = np.mean([point[1] for point in all_points])
    
    # Create heatmap
    m_heat = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles='OpenStreetMap'
    )
    
    # Add heatmaps
    if business_coords:
        plugins.HeatMap(
            business_coords, 
            name='Business Density',
            radius=15,
            blur=10,
            gradient={0.2: 'blue', 0.4: 'lightblue', 0.6: 'cyan', 1: 'white'}
        ).add_to(m_heat)
    
    if violation_coords:
        plugins.HeatMap(
            violation_coords, 
            name='Violation Density',
            radius=15,
            blur=10,
            gradient={0.2: 'red', 0.4: 'orange', 0.6: 'yellow', 1: 'white'}
        ).add_to(m_heat)
    
    folium.LayerControl().add_to(m_heat)
    
    # Save heatmap
    heatmap_file = 'business_violations_heatmap.html'
    m_heat.save(heatmap_file)
    print(f"Heatmap saved as {heatmap_file}")
    
    return m_heat

if __name__ == "__main__":
    # Install required packages if not already installed
    print("Make sure you have the required packages installed:")
    print("pip install pandas folium numpy")
    print()
    
    try:
        # Create the main map with pins
        map_obj = create_business_violations_map()
        
        # Create heatmap version
        heatmap_obj = create_heatmap_version()
        
        print("\nMap creation completed!")
        print("Files created:")
        print("- business_violations_map.html (pin map)")
        print("- business_violations_heatmap.html (heatmap)")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find CSV file - {e}")
        print("Make sure 'businesses_with_gps.csv' and 'violations_with_gps.csv' are in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your data format and try again.")