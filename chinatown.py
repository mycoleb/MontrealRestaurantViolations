import pandas as pd
import folium
import numpy as np
import requests
import json

# Constants
CHINATOWN_POLYGON = [
    [45.509282, -73.561113],
    [45.507696, -73.562595], 
    [45.507041, -73.561215],
    [45.505557, -73.562643],
    [45.504930, -73.561120],
    [45.508093, -73.558214]
]

def point_in_polygon_simple(point, polygon_coords):
    """Simple point-in-polygon check using ray casting"""
    x, y = point[1], point[0]  # lng, lat
    n = len(polygon_coords)
    inside = False
    
    p1x, p1y = polygon_coords[0][1], polygon_coords[0][0]  # lng, lat
    for i in range(1, n + 1):
        p2x, p2y = polygon_coords[i % n][1], polygon_coords[i % n][0]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def point_in_bounds(point, bounds):
    """Check if a point is within rectangular bounds"""
    lat, lng = point
    return (bounds[0][0] <= lat <= bounds[1][0] and 
            bounds[0][1] <= lng <= bounds[1][1])

def get_color(ratio):
    """Get color based on violation ratio"""
    if ratio >= 0.8:
        return '#d73027'
    elif ratio >= 0.6:
        return '#fc8d59'
    elif ratio >= 0.4:
        return '#ffffbf'
    elif ratio >= 0.2:
        return '#91bfdb'
    else:
        return '#1a9850'

def safe_ratio(numerator, denominator):
    """Calculate ratio safely, avoiding division by zero"""
    return numerator / denominator if denominator > 0 else 0

def generate_popup_html(name, businesses, violations, ratio):
    """Generate HTML for neighborhood popup"""
    emoji = "🏮" if name == "Chinatown" else "📍"
    return f"""
    <div style="font-family: Arial, sans-serif; width: 300px; padding: 10px;">
        <h2 style="margin: 0 0 15px 0; color: #2c3e50; font-size: 20px; 
                   border-bottom: 3px solid #3498db; padding-bottom: 8px;">
            {emoji} {name}
        </h2>
        <div style="font-size: 15px; line-height: 1.6;">
            <p style="margin: 10px 0;"><strong>🏪 Food Establishments:</strong> {businesses}</p>
            <p style="margin: 10px 0;"><strong>⚠️ Health Violations:</strong> {violations}</p>
            <p style="margin: 10px 0;"><strong>📊 Violation Ratio:</strong> 
               <span style="font-weight: bold; color: {get_color(ratio)}; font-size: 18px;">
               {ratio:.3f}</span>
            </p>
        </div>
    </div>
    """

def generate_tooltip_html(name, businesses, violations, ratio):
    """Generate HTML for neighborhood tooltip"""
    emoji = "🏮" if name == "Chinatown" else "📍"
    return f"""
    <div style="font-family: Arial, sans-serif; padding: 8px; background: white; 
                border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.2);">
        <div style="font-size: 17px; font-weight: bold; color: #2c3e50; margin-bottom: 5px;">
            {emoji} {name}
        </div>
        <div style="font-size: 14px; color: #34495e;">
            <strong>Violation Ratio:</strong> {ratio:.3f}<br>
            <strong>Details:</strong> {violations} violations / {businesses} establishments
        </div>
    </div>
    """

def create_expanded_fallback_neighborhoods():
    """Create expanded fallback neighborhoods with better coverage"""
    return {
        "Ville-Marie": {
            "bounds": [[45.485, -73.580], [45.520, -73.540]]
        },
        "Chinatown": {
            "polygon": CHINATOWN_POLYGON
        },
        "Le Plateau-Mont-Royal": {
            "bounds": [[45.510, -73.595], [45.535, -73.560]]
        },
        "Outremont": {
            "bounds": [[45.515, -73.620], [45.535, -73.590]]
        },
        "Rosemont–La Petite-Patrie": {
            "bounds": [[45.530, -73.595], [45.560, -73.565]]
        },
        "Villeray–Saint-Michel–Parc-Extension": {
            "bounds": [[45.545, -73.640], [45.575, -73.600]]
        },
        "Ahuntsic-Cartierville": {
            "bounds": [[45.540, -73.700], [45.580, -73.640]]
        },
        "Côte-des-Neiges–Notre-Dame-de-Grâce": {
            "bounds": [[45.470, -73.650], [45.510, -73.590]]
        },
        "Le Sud-Ouest": {
            "bounds": [[45.460, -73.580], [45.485, -73.540]]
        },
        "Verdun": {
            "bounds": [[45.445, -73.580], [45.470, -73.540]]
        },
        "Mercier–Hochelaga-Maisonneuve": {
            "bounds": [[45.535, -73.560], [45.570, -73.500]]
        },
        "Saint-Léonard": {
            "bounds": [[45.580, -73.620], [45.610, -73.580]]
        },
        "Anjou": {
            "bounds": [[45.595, -73.595], [45.625, -73.555]]
        },
        "Rivière-des-Prairies–Pointe-aux-Trembles": {
            "bounds": [[45.635, -73.560], [45.675, -73.480]]
        },
        "Montréal-Nord": {
            "bounds": [[45.605, -73.650], [45.635, -73.610]]
        },
        "Saint-Laurent": {
            "bounds": [[45.505, -73.750], [45.545, -73.700]]
        },
        "Pierrefonds-Roxboro": {
            "bounds": [[45.475, -73.900], [45.515, -73.820]]
        },
        "L'Île-Bizard–Sainte-Geneviève": {
            "bounds": [[45.440, -73.950], [45.480, -73.880]]
        },
        "Lachine": {
            "bounds": [[45.415, -73.700], [45.445, -73.650]]
        },
        "LaSalle": {
            "bounds": [[45.395, -73.650], [45.425, -73.600]]
        },
        # Additional expanded coverage areas
        "Greater Downtown": {
            "bounds": [[45.480, -73.590], [45.515, -73.530]]
        },
        "East Montreal": {
            "bounds": [[45.500, -73.530], [45.570, -73.480]]
        },
        "West Montreal": {
            "bounds": [[45.450, -73.750], [45.520, -73.650]]
        },
        "North Montreal": {
            "bounds": [[45.570, -73.700], [45.620, -73.600]]
        },
        "South Montreal": {
            "bounds": [[45.400, -73.620], [45.460, -73.530]]
        }
    }

def assign_neighborhood_improved(lat, lng, neighborhoods):
    """Improved neighborhood assignment with better fallback"""
    # Check Chinatown first (highest priority)
    if point_in_polygon_simple([lat, lng], CHINATOWN_POLYGON):
        return 'Chinatown'
    
    # Check other neighborhoods with bounds
    for neighborhood, data in neighborhoods.items():
        if neighborhood == 'Chinatown':
            continue
        if 'bounds' in data and point_in_bounds([lat, lng], data['bounds']):
            return neighborhood
    
    return 'Other'

def identify_food_establishments_flexible(businesses_df):
    """More flexible food establishment identification"""
    
    # Expanded food keywords including French terms
    food_keywords = [
        # English terms
        'restaurant', 'food', 'café', 'bar', 'bakery', 'grocery', 'market', 
        'bistro', 'pizza', 'convenience', 'catering', 'deli', 'butcher',
        'seafood', 'chicken', 'burger', 'sandwich', 'coffee', 'tea',
        # French terms
        'alimentaire', 'boulangerie', 'épicerie', 'marché', 'pizzeria', 
        'dépanneur', 'traiteur', 'boucherie', 'poissonnerie', 'pâtisserie',
        'fromagerie', 'charcuterie', 'brasserie', 'crêperie', 'sandwicherie'
    ]
    
    # Try different approaches to identify food establishments
    food_establishments = pd.DataFrame()
    
    # Method 1: Direct keyword matching in 'type' column
    if 'type' in businesses_df.columns:
        food_mask = businesses_df['type'].str.lower().str.contains(
            '|'.join(food_keywords), na=False, regex=True
        )
        food_establishments = businesses_df[food_mask].copy()
        print(f"✅ Method 1 (type column): Found {len(food_establishments)} food establishments")
    
    # Method 2: If that fails, try other common column names
    if len(food_establishments) == 0:
        alternative_columns = ['category', 'business_type', 'sector', 'industry', 'description']
        for col in alternative_columns:
            if col in businesses_df.columns:
                food_mask = businesses_df[col].str.lower().str.contains(
                    '|'.join(food_keywords), na=False, regex=True
                )
                food_establishments = businesses_df[food_mask].copy()
                print(f"✅ Method 2 ({col} column): Found {len(food_establishments)} food establishments")
                if len(food_establishments) > 0:
                    break
    
    # Method 3: If still no results, be more permissive and include all businesses
    if len(food_establishments) == 0:
        print("⚠️  No food establishments found with keyword matching. Using all businesses.")
        food_establishments = businesses_df.copy()
        print(f"✅ Method 3 (all businesses): Using {len(food_establishments)} establishments")
    
    return food_establishments

def create_montreal_violations_map_fixed():
    """Fixed version of Montreal violations map creator"""
    
    print("🗺️  Montreal Health Violations Map Generator (FIXED VERSION)")
    print("=" * 60)
    
    # Load CSV files with better error handling
    try:
        businesses_df = pd.read_csv('businesses_with_gps.csv')
        violations_df = pd.read_csv('violations_with_gps.csv')
        print(f"✅ Loaded {len(businesses_df)} businesses and {len(violations_df)} violations")
        
        # Print column information
        print(f"📊 Business columns: {list(businesses_df.columns)}")
        print(f"📊 Violations columns: {list(violations_df.columns)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find CSV files: {e}")
        print("💡 Make sure you have 'businesses_with_gps.csv' and 'violations_with_gps.csv' in the current directory")
        return None
    
    # Clean data more thoroughly
    print(f"\n🧹 Cleaning data...")
    original_business_count = len(businesses_df)
    original_violation_count = len(violations_df)
    
    # Clean businesses data
    businesses_df = businesses_df.dropna(subset=['latitude', 'longitude'])
    print(f"   - Removed {original_business_count - len(businesses_df)} businesses without coordinates")
    
    # Clean violations data  
    violations_df = violations_df.dropna(subset=['business_id'])
    print(f"   - Removed {original_violation_count - len(violations_df)} violations without business_id")
    
    # Check coordinate validity (Montreal is around 45.5°N, 73.6°W)
    valid_coords = (
        (businesses_df['latitude'] >= 45.3) & (businesses_df['latitude'] <= 45.8) &
        (businesses_df['longitude'] >= -74.0) & (businesses_df['longitude'] <= -73.3)
    )
    businesses_df = businesses_df[valid_coords]
    print(f"   - Kept {len(businesses_df)} businesses with valid Montreal coordinates")
    
    if len(businesses_df) == 0:
        print("❌ No businesses with valid coordinates found!")
        return None
    
    # Use improved food establishment identification
    food_establishments = identify_food_establishments_flexible(businesses_df)
    
    if len(food_establishments) == 0:
        print("❌ No food establishments found!")
        return None
    
    # Use expanded fallback neighborhoods for better coverage
    neighborhoods = create_expanded_fallback_neighborhoods()
    print(f"✅ Using {len(neighborhoods)} neighborhood boundaries")
    
    # Assign neighborhoods with improved method
    print(f"\n🏘️  Assigning neighborhoods...")
    food_establishments['neighborhood'] = food_establishments.apply(
        lambda row: assign_neighborhood_improved(row['latitude'], row['longitude'], neighborhoods),
        axis=1
    )
    
    # Map violations to neighborhoods
    print(f"⚠️  Mapping violations to neighborhoods...")
    
    # Create business lookup for faster access
    business_lookup = food_establishments.set_index('business_id')[['latitude', 'longitude', 'neighborhood']].to_dict('index')
    
    violations_with_neighborhood = []
    mapped_count = 0
    
    for _, violation in violations_df.iterrows():
        business_id = violation['business_id']
        if business_id in business_lookup:
            violations_with_neighborhood.append({
                'business_id': business_id,
                'neighborhood': business_lookup[business_id]['neighborhood']
            })
            mapped_count += 1
    
    print(f"   - Successfully mapped {mapped_count} violations to neighborhoods")
    
    if mapped_count == 0:
        print("❌ No violations could be mapped to businesses!")
        print("💡 Check that business_id values match between the two CSV files")
        return None
    
    violations_neighborhood_df = pd.DataFrame(violations_with_neighborhood)
    
    # Calculate statistics
    business_counts = food_establishments.groupby('neighborhood').size().to_dict()
    violation_counts = violations_neighborhood_df.groupby('neighborhood').size().to_dict()
    
    print(f"\n📊 Statistics:")
    print(f"   - Found businesses in {len(business_counts)} neighborhoods")
    print(f"   - Found violations in {len(violation_counts)} neighborhoods")
    
    # Check if we have meaningful data
    total_businesses_assigned = sum(business_counts.values())
    total_violations_assigned = sum(violation_counts.values())
    
    print(f"   - Total businesses assigned: {total_businesses_assigned}")
    print(f"   - Total violations assigned: {total_violations_assigned}")
    
    # Special check for Chinatown
    chinatown_businesses = business_counts.get('Chinatown', 0)
    chinatown_violations = violation_counts.get('Chinatown', 0)
    print(f"   - Chinatown: {chinatown_businesses} businesses, {chinatown_violations} violations")
    
    # Create map
    print(f"\n🗺️  Creating interactive map...")
    m = folium.Map(location=[45.5017, -73.5673], zoom_start=11, tiles='OpenStreetMap')
    
    # Add title
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50%; width: 600px; height: 90px; 
                margin-left: -300px; background-color: white; border:2px solid grey; z-index:9999; 
                font-size:18px; text-align: center; padding: 10px; border-radius: 5px;">
    <p style="margin: 5px 0; font-weight: bold;">Montreal Health Violations by Neighborhood</p>
    <p style="font-size:14px; margin: 5px 0;">Hover over neighborhoods to see names and ratios</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Add neighborhoods to map
    neighborhoods_added = 0
    
    for neighborhood_name, data in neighborhoods.items():
        business_count = business_counts.get(neighborhood_name, 0)
        violation_count = violation_counts.get(neighborhood_name, 0)
        
        # Only add neighborhoods that have businesses
        if business_count > 0:
            ratio = safe_ratio(violation_count, business_count)
            
            popup_content = generate_popup_html(neighborhood_name, business_count, violation_count, ratio)
            tooltip_content = generate_tooltip_html(neighborhood_name, business_count, violation_count, ratio)
            
            style = {
                'fillColor': get_color(ratio),
                'color': '#333333', 
                'weight': 3 if neighborhood_name == 'Chinatown' else 2,
                'fillOpacity': 0.8 if neighborhood_name == 'Chinatown' else 0.7,
                'opacity': 0.8
            }
            
            if neighborhood_name == 'Chinatown':
                # Special handling for Chinatown polygon
                folium.Polygon(
                    locations=CHINATOWN_POLYGON,
                    **style,
                    popup=folium.Popup(popup_content, max_width=350),
                    tooltip=folium.Tooltip(tooltip_content, sticky=True)
                ).add_to(m)
                neighborhoods_added += 1
                
            elif 'bounds' in data:
                # Handle rectangular bounds
                bounds = data['bounds']
                rectangle_coords = [
                    [bounds[0][0], bounds[0][1]], [bounds[0][0], bounds[1][1]],
                    [bounds[1][0], bounds[1][1]], [bounds[1][0], bounds[0][1]]
                ]
                folium.Polygon(
                    locations=rectangle_coords,
                    **style,
                    popup=folium.Popup(popup_content, max_width=350),
                    tooltip=folium.Tooltip(tooltip_content, sticky=True)
                ).add_to(m)
                neighborhoods_added += 1
    
    print(f"   - Added {neighborhoods_added} neighborhoods to map")
    
    # Add sample business markers for debugging (optional)
    if len(food_establishments) > 0:
        print(f"🎯 Adding sample business markers for verification...")
        sample_businesses = food_establishments.sample(n=min(20, len(food_establishments)), random_state=42)
        
        for idx, business in sample_businesses.iterrows():
            folium.CircleMarker(
                location=[business['latitude'], business['longitude']],
                radius=3,
                popup=f"Business: {business.get('type', 'Unknown')}<br>Neighborhood: {business['neighborhood']}",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 220px; height: 180px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 15px; border-radius: 5px;">
    <h4 style="margin: 0 0 15px 0;">Violation Ratio Legend</h4>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#1a9850"></i> Low (0.0 - 0.2)</p>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#91bfdb"></i> Medium-Low (0.2 - 0.4)</p>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#ffffbf"></i> Medium (0.4 - 0.6)</p>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#fc8d59"></i> Medium-High (0.6 - 0.8)</p>
    <p style="margin: 5px 0;"><i class="fa fa-square" style="color:#d73027"></i> High (0.8+)</p>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Save map
    output_file = 'montreal_violations_map_fixed.html'
    m.save(output_file)
    
    print(f"\n✅ Map saved as '{output_file}'")
    
    # Print detailed neighborhood statistics
    print(f"\n📊 DETAILED NEIGHBORHOOD STATISTICS")
    print("=" * 80)
    
    all_neighborhoods = set(business_counts.keys()) | set(violation_counts.keys())
    neighborhood_stats = []
    
    for neighborhood in all_neighborhoods:
        business_count = business_counts.get(neighborhood, 0)
        violation_count = violation_counts.get(neighborhood, 0)
        ratio = safe_ratio(violation_count, business_count)
        
        if business_count > 0:  # Only show neighborhoods with food establishments
            neighborhood_stats.append((neighborhood, business_count, violation_count, ratio))
    
    # Sort by violation ratio (highest first)
    neighborhood_stats.sort(key=lambda x: x[3], reverse=True)
    
    print(f"{'NEIGHBORHOOD NAME':<40} {'RATIO':<8} {'VIOLATIONS':<12} {'ESTABLISHMENTS':<15}")
    print("-" * 80)
    
    for name, businesses, violations, ratio in neighborhood_stats:
        # Highlight Chinatown with special formatting
        if name == 'Chinatown':
            print(f"🏮 {name:<38} {ratio:>6.3f} {violations:>10d} {businesses:>13d}")
        else:
            print(f"{name:<40} {ratio:>6.3f} {violations:>10d} {businesses:>13d}")
    
    print(f"\n✅ Successfully processed {len(neighborhood_stats)} neighborhoods with food establishments")
    print(f"✅ Open '{output_file}' in your browser to see the interactive map!")
    
    if neighborhoods_added == 0:
        print(f"\n⚠️  WARNING: No neighborhoods were added to the map!")
        print(f"💡 This suggests that businesses aren't being properly assigned to neighborhoods.")
        print(f"💡 Run the debug version first to identify the issue.")
    
    return m, neighborhood_stats

# Main execution
if __name__ == "__main__":
    print("🚀 Running Fixed Montreal Violations Map Generator")
    print("💡 If you see issues, run the debug version first!")
    print("=" * 60)
    
    result = create_montreal_violations_map_fixed()
    
    if result:
        print("\n🎉 SUCCESS! Map generation completed!")
        print("💡 TIP: Red dots show sample business locations for verification")
        print("💡 TIP: Hover over neighborhoods to see their names and statistics!")
        print("💡 TIP: Click on neighborhoods for detailed information!")
        print("🏮 TIP: Look for Chinatown as a special highlighted area!")
    else:
        print("\n❌ Map generation failed.")
        print("💡 Try running the debug version to identify the problem:")
        print("   python debug_version.py")