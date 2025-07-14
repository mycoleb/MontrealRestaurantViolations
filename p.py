import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings
import base64
from io import BytesIO
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Read the data
print("Loading data...")
businesses = pd.read_csv('businesses.csv')
violations = pd.read_csv('violations.csv')

print(f"Loaded {len(businesses)} businesses and {len(violations)} violations")

# Data preprocessing
# Convert date columns to datetime
violations['date'] = pd.to_datetime(violations['date'], unit='s', errors='coerce')
violations['date_jugement'] = pd.to_datetime(violations['date_jugement'], unit='s', errors='coerce')
violations['date_statut'] = pd.to_datetime(violations['date_statut'], unit='s', errors='coerce')

businesses['date_statut'] = pd.to_datetime(businesses['date_statut'], unit='s', errors='coerce')

# Extract year and month for temporal analysis
violations['year'] = violations['date'].dt.year
violations['month'] = violations['date'].dt.month
violations['year_month'] = violations['date'].dt.to_period('M')

# Merge violations with business information
merged_data = violations.merge(businesses[['business_id', 'name', 'type', 'latitude', 'longitude']], 
                               on='business_id', how='left')

# Function to convert plot to base64 string
def plot_to_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    buffer.close()
    plt.close()
    return f"data:image/png;base64,{image_base64}"

# Start building HTML
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Montreal Restaurant Health Violations Analysis</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #555;
            border-bottom: 2px solid #ddd;
            padding-bottom: 10px;
            margin-top: 40px;
        }
        .plot-container {
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }
        .summary-stats {
            background-color: #e8f4f8;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .stat-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        .stat-box {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c5aa0;
        }
        .stat-label {
            font-size: 14px;
            color: #666;
            margin-top: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #2c5aa0;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <h1>Montreal Restaurant Health Violations Analysis</h1>
"""

# Calculate summary statistics
total_violations = len(violations)
unique_businesses = violations['business_id'].nunique()
avg_fine = violations['montant'].mean()
total_fines = violations['montant'].sum()
date_range = f"{violations['date'].min().strftime('%Y-%m-%d')} to {violations['date'].max().strftime('%Y-%m-%d')}"

html_content += f"""
    <div class="summary-stats">
        <h2>Summary Statistics</h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-value">{total_violations:,}</div>
                <div class="stat-label">Total Violations</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{unique_businesses:,}</div>
                <div class="stat-label">Businesses with Violations</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${avg_fine:,.2f}</div>
                <div class="stat-label">Average Fine</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">${total_fines:,.2f}</div>
                <div class="stat-label">Total Fines Collected</div>
            </div>
        </div>
        <p style="margin-top: 20px;"><strong>Date Range:</strong> {date_range}</p>
    </div>
"""

# 1. Violations over time
plt.figure(figsize=(10, 6))
violations_by_month = violations.groupby('year_month').size()
violations_by_month.plot(kind='line', linewidth=2, color='#2c5aa0')
plt.title('Health Violations Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Number of Violations')
plt.grid(True, alpha=0.3)
plt.tight_layout()

html_content += """
    <div class="plot-container">
        <h2>Temporal Analysis</h2>
        <img src="{}" alt="Violations Over Time">
    </div>
""".format(plot_to_base64())

# 2. Top 10 establishments with most violations
plt.figure(figsize=(10, 6))
top_violators = violations['etablissement'].value_counts().head(10)
top_violators.plot(kind='barh', color='coral')
plt.title('Top 10 Establishments by Violation Count', fontsize=16, fontweight='bold')
plt.xlabel('Number of Violations')
plt.tight_layout()

html_content += """
    <div class="plot-container">
        <h2>Top Violators</h2>
        <img src="{}" alt="Top 10 Violators">
    </div>
""".format(plot_to_base64())

# 3. Violations by category
plt.figure(figsize=(10, 8))
category_counts = violations['categorie'].value_counts()
colors = plt.cm.Set3(np.linspace(0, 1, len(category_counts)))
plt.pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', colors=colors)
plt.title('Distribution of Violations by Category', fontsize=16, fontweight='bold')
plt.axis('equal')

html_content += """
    <div class="plot-container">
        <h2>Violation Categories</h2>
        <img src="{}" alt="Violations by Category">
    </div>
""".format(plot_to_base64())

# 4. Average fine amount by category
plt.figure(figsize=(10, 6))
avg_fine_by_category = violations.groupby('categorie')['montant'].mean().sort_values()
avg_fine_by_category.plot(kind='bar', color='lightgreen')
plt.title('Average Fine Amount by Category', fontsize=16, fontweight='bold')
plt.ylabel('Average Fine ($)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

html_content += """
    <div class="plot-container">
        <h2>Financial Impact by Category</h2>
        <img src="{}" alt="Average Fines by Category">
    </div>
""".format(plot_to_base64())

# 5. Yearly trend
plt.figure(figsize=(10, 6))
yearly_violations = violations.groupby('year').size()
yearly_violations.plot(kind='line', marker='o', linewidth=2, markersize=8, color='#2c5aa0')
if len(yearly_violations) >= 3:
    ma3 = yearly_violations.rolling(window=3, center=True).mean()
    ma3.plot(linewidth=2, linestyle='--', color='red', label='3-Year Moving Avg')
plt.title('Yearly Violation Trends', fontsize=16, fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Number of Violations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

html_content += """
    <div class="plot-container">
        <h2>Yearly Trends</h2>
        <img src="{}" alt="Yearly Trends">
    </div>
""".format(plot_to_base64())

# 6. Fine amount distribution
plt.figure(figsize=(10, 6))
plt.hist(violations['montant'], bins=50, color='gold', edgecolor='black', alpha=0.7)
plt.axvline(violations['montant'].median(), color='red', linestyle='--', linewidth=2,
            label=f'Median: ${violations["montant"].median():.0f}')
plt.title('Distribution of Fine Amounts', fontsize=16, fontweight='bold')
plt.xlabel('Fine Amount ($)')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()

html_content += """
    <div class="plot-container">
        <h2>Fine Distribution</h2>
        <img src="{}" alt="Fine Distribution">
    </div>
""".format(plot_to_base64())

# Add top violations table
top_categories = violations['categorie'].value_counts().head(10)
html_content += """
    <div class="plot-container">
        <h2>Top Violation Types</h2>
        <table>
            <tr>
                <th>Violation Category</th>
                <th>Count</th>
                <th>Percentage</th>
            </tr>
"""

for category, count in top_categories.items():
    percentage = (count / total_violations) * 100
    html_content += f"""
            <tr>
                <td>{category}</td>
                <td>{count:,}</td>
                <td>{percentage:.1f}%</td>
            </tr>
"""

html_content += """
        </table>
    </div>
"""

# Add repeat offenders analysis
repeat_offenders = violations.groupby(['business_id', 'etablissement']).size().sort_values(ascending=False).head(10)
html_content += """
    <div class="plot-container">
        <h2>Repeat Offenders</h2>
        <table>
            <tr>
                <th>Establishment</th>
                <th>Number of Violations</th>
            </tr>
"""

for (business_id, name), count in repeat_offenders.items():
    html_content += f"""
            <tr>
                <td>{name}</td>
                <td>{count}</td>
            </tr>
"""

html_content += """
        </table>
    </div>
</body>
</html>
"""

# Save the HTML file
with open('montreal_violations_report.html', 'w', encoding='utf-8') as f:
    f.write(html_content)

print("\nAnalysis complete! Open 'montreal_violations_report.html' in your web browser to view the results.")

# Also save summary data to CSV
summary_data = {
    'Total Violations': total_violations,
    'Unique Businesses': unique_businesses,
    'Average Fine': f"${avg_fine:.2f}",
    'Total Fines': f"${total_fines:,.2f}",
    'Date Range': date_range,
    'Most Common Violation': violations['categorie'].mode()[0],
    'Establishment with Most Violations': violations['etablissement'].mode()[0]
}

pd.DataFrame([summary_data]).to_csv('violation_summary.csv', index=False)
print("Summary data saved to 'violation_summary.csv'")

# Create interactive map
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    import json
    
    print("\nCreating interactive map...")
    
    # Create a base map centered on Montreal
    montreal_map = folium.Map(
        location=[45.5017, -73.5673], 
        zoom_start=11,
        tiles='OpenStreetMap'
    )
    
    # Prepare data with valid coordinates
    map_data = merged_data[['latitude', 'longitude', 'etablissement', 'categorie', 'montant', 'date']].dropna()
    
    # Create a feature group for violations
    violations_layer = folium.FeatureGroup(name='Individual Violations')
    
    # Add marker cluster for better performance with many points
    marker_cluster = MarkerCluster().add_to(violations_layer)
    
    # Add individual markers with popup info
    for idx, row in map_data.iterrows():
        popup_html = f"""
        <div style="font-family: Arial; width: 200px;">
            <b>{row['etablissement']}</b><br>
            <hr style="margin: 5px 0;">
            <b>Category:</b> {row['categorie']}<br>
            <b>Fine:</b> ${row['montant']}<br>
            <b>Date:</b> {row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'N/A'}
        </div>
        """
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color='red', icon='exclamation-triangle', prefix='fa')
        ).add_to(marker_cluster)
    
    violations_layer.add_to(montreal_map)
    
    # Create density analysis by area
    # Group violations by rounded coordinates to create grid cells
    map_data['lat_round'] = map_data['latitude'].round(2)
    map_data['lon_round'] = map_data['longitude'].round(2)
    
    # Count violations per grid cell
    area_counts = map_data.groupby(['lat_round', 'lon_round']).size().reset_index(name='count')
    
    # Create choropleth-style circles for density
    density_layer = folium.FeatureGroup(name='Violation Density')
    
    # Calculate percentiles for color scaling
    percentiles = area_counts['count'].quantile([0.25, 0.5, 0.75, 0.9])
    
    for idx, row in area_counts.iterrows():
        # Determine color based on count
        if row['count'] <= percentiles[0.25]:
            color = '#fee8c8'  # Light orange
            radius = 200
        elif row['count'] <= percentiles[0.5]:
            color = '#fdbb84'  # Medium orange
            radius = 300
        elif row['count'] <= percentiles[0.75]:
            color = '#e34a33'  # Dark orange
            radius = 400
        else:
            color = '#b30000'  # Dark red
            radius = 500
        
        folium.Circle(
            location=[row['lat_round'], row['lon_round']],
            radius=radius,
            popup=f"Violations in area: {row['count']}",
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.6,
            weight=2
        ).add_to(density_layer)
    
    density_layer.add_to(montreal_map)
    
    # Create heatmap layer with better parameters
    heat_layer = folium.FeatureGroup(name='Heat Map')
    heat_data = [[row['latitude'], row['longitude']] for idx, row in map_data.iterrows()]
    
    HeatMap(
        heat_data,
        min_opacity=0.2,
        max_zoom=18,
        radius=25,
        blur=20,
        gradient={
            0.0: 'blue',
            0.25: 'cyan',
            0.5: 'lime',
            0.75: 'yellow',
            1.0: 'red'
        }
    ).add_to(heat_layer)
    
    heat_layer.add_to(montreal_map)
    
    # Add layer control
    folium.LayerControl().add_to(montreal_map)
    
    # Add a title to the map
    title_html = '''
    <div style="position: fixed; 
                top: 10px; 
                left: 50%; 
                transform: translateX(-50%);
                z-index: 1000;
                background-color: white;
                padding: 10px;
                border: 2px solid grey;
                border-radius: 5px;
                font-family: Arial;
                font-size: 16px;
                font-weight: bold;">
        Montreal Restaurant Health Violations Map
    </div>
    '''
    montreal_map.get_root().html.add_child(folium.Element(title_html))
    
    # Add statistics to map
    stats_html = f'''
    <div style="position: fixed; 
                bottom: 30px; 
                left: 10px; 
                z-index: 1000;
                background-color: white;
                padding: 15px;
                border: 2px solid grey;
                border-radius: 5px;
                font-family: Arial;
                font-size: 14px;">
        <b>Map Statistics:</b><br>
        Total Violations Shown: {len(map_data)}<br>
        Unique Locations: {len(area_counts)}<br>
        <br>
        <b>Density Legend:</b><br>
        <span style="color: #fee8c8;">●</span> Low (≤{int(percentiles[0.25])} violations)<br>
        <span style="color: #fdbb84;">●</span> Medium (≤{int(percentiles[0.5])} violations)<br>
        <span style="color: #e34a33;">●</span> High (≤{int(percentiles[0.75])} violations)<br>
        <span style="color: #b30000;">●</span> Very High (>{int(percentiles[0.75])} violations)
    </div>
    '''
    montreal_map.get_root().html.add_child(folium.Element(stats_html))
    
    # Save the map
    montreal_map.save('montreal_violations_map.html')
    print("Interactive map saved as 'montreal_violations_map.html'")
    
    # Also create a per-capita analysis if we have business density data
    # Calculate violations per business in each area
    business_density = businesses.groupby(['city']).size()
    violation_density = violations.groupby(['ville']).size()
    
    # Create per capita dataframe
    per_capita = pd.DataFrame({
        'businesses': business_density,
        'violations': violation_density
    }).fillna(0)
    per_capita['violations_per_business'] = per_capita['violations'] / per_capita['businesses']
    per_capita = per_capita[per_capita['businesses'] > 0].sort_values('violations_per_business', ascending=False)
    
    # Add per capita analysis to the main HTML report
    per_capita_html = """
        <div class="plot-container">
            <h2>Violations Per Business by City</h2>
            <table>
                <tr>
                    <th>City</th>
                    <th>Total Businesses</th>
                    <th>Total Violations</th>
                    <th>Violations per Business</th>
                </tr>
    """
    
    for city, row in per_capita.head(15).iterrows():
        per_capita_html += f"""
                <tr>
                    <td>{city}</td>
                    <td>{int(row['businesses']):,}</td>
                    <td>{int(row['violations']):,}</td>
                    <td>{row['violations_per_business']:.2f}</td>
                </tr>
        """
    
    per_capita_html += """
            </table>
        </div>
    """
    
    # Insert this into the main HTML before the closing tags
    html_content = html_content.replace('</body>', per_capita_html + '</body>')
    
    # Re-save the updated HTML
    with open('montreal_violations_report.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
except ImportError:
    print("\nTo create an interactive map, install folium: pip install folium")