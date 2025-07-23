import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import subprocess
from datetime import datetime
import warnings
import shutil
import numpy as np

warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --- Configuration ---
OUTPUT_BASE_FOLDER = 'violation_animations'
VIDEO_1_FILENAME = 'violations_overview_analysis.mp4'  # Parts 1 & 2
VIDEO_2_FILENAME = 'violations_per_capita_analysis.mp4'  # Part 3
FPS = 5 # Frames per second for the video (adjust for speed)
DPI = 100 # Dots per inch for image quality
PAUSE_FRAMES = FPS * 2 # 2-second pause at the end of each section

# FIXED: Set figure size to ensure even width/height for FFmpeg
FIGURE_WIDTH = 14  # This will result in 1400 pixels at 100 DPI (even number)
FIGURE_HEIGHT = 8   # This will result in 800 pixels at 100 DPI (even number)

# Creator credit text
CREATOR_TEXT_EN = "Created by Mycole Brown"
CREATOR_TEXT_FR = "Créé par Mycole Brown"

# --- Load Data ---
print("Loading data...")
try:
    # Use the with_neighborhoods CSV files as specified
    violations = pd.read_csv('violations_with_neighborhoods.csv')
    print(f"✓ Loaded {len(violations)} violations with neighborhood data")
    
    businesses = pd.read_csv('businesses_with_neighborhoods.csv')
    print(f"✓ Loaded {len(businesses)} businesses with neighborhood data")
except FileNotFoundError as e:
    print(f"✗ Error: {e}")
    print("Please ensure the 'violations_with_neighborhoods.csv' and 'businesses_with_neighborhoods.csv' files are in the same directory.")
    exit()

# --- Data Inspection ---
print("\n=== DATA INSPECTION ===")
print(f"Columns in violations CSV: {list(violations.columns)}")
print(f"Date column sample values: {violations['date'].head(10).tolist()}")
print(f"Date column dtype: {violations['date'].dtype}")

# Check for GPS data
if 'gps' in violations.columns:
    gps_violations = violations['gps'].notna().sum()
    print(f"✓ GPS data available in violations: {gps_violations} records with GPS")
    if gps_violations > 0:
        print(f"Sample GPS data: {violations['gps'].dropna().head(3).tolist()}")
else:
    print("⚠️  No GPS data in violations CSV")

if 'gps' in businesses.columns:
    gps_businesses = businesses['gps'].notna().sum()
    print(f"✓ GPS data available in businesses: {gps_businesses} records with GPS")
else:
    print("⚠️  No GPS data in businesses CSV")

# Examine neighborhood data
violations_with_neighborhoods = violations['neighborhood'].notna().sum()
businesses_with_neighborhoods = businesses['neighborhood'].notna().sum()

print(f"\nViolations with neighborhoods: {violations_with_neighborhoods}/{len(violations)} ({violations_with_neighborhoods/len(violations)*100:.1f}%)")
print(f"Businesses with neighborhoods: {businesses_with_neighborhoods}/{len(businesses)} ({businesses_with_neighborhoods/len(businesses)*100:.1f}%)")

if violations_with_neighborhoods > 0:
    print(f"Unique neighborhoods in violations: {len(violations['neighborhood'].dropna().unique())}")
    print(f"Sample violation neighborhoods: {sorted(violations['neighborhood'].dropna().unique())[:10]}")

if businesses_with_neighborhoods > 0:
    print(f"Unique neighborhoods in businesses: {len(businesses['neighborhood'].dropna().unique())}")
    print(f"Sample business neighborhoods: {sorted(businesses['neighborhood'].dropna().unique())[:10]}")

# --- Data Preprocessing ---
print("\n=== DATA PREPROCESSING ===")

# Convert dates (YYYYMMDD format)
try:
    violations['date'] = pd.to_datetime(violations['date'], format='%Y%m%d', errors='coerce')
    print(f"✓ Successfully converted dates from YYYYMMDD format")
    print(f"Date range: {violations['date'].min()} to {violations['date'].max()}")
except Exception as e:
    print(f"✗ Error converting dates: {e}")
    exit()

# Check for null dates after conversion
null_dates = violations['date'].isnull().sum()
print(f"Null dates after conversion: {null_dates}")

if null_dates > 0:
    print(f"Dropping {null_dates} rows with invalid dates")
    violations.dropna(subset=['date'], inplace=True)

if len(violations) == 0:
    print("✗ No valid dates found! Exiting.")
    exit()

print(f"✓ Final dataset: {len(violations)} violations with valid dates")

# Extract year_month for grouping
violations['year_month'] = violations['date'].dt.to_period('M')

# Sort data by date to ensure correct temporal progression
violations.sort_values('date', inplace=True)

# Handle missing columns
if 'categorie' not in violations.columns:
    print("Warning: 'categorie' column not found. Using default category.")
    violations['categorie'] = 'Unknown'
else:
    print(f"✓ Found {violations['categorie'].nunique()} unique categories")

# Clean neighborhood data - use both ville and neighborhood columns
print("Processing neighborhood data...")

# For violations, use neighborhood if available, otherwise fall back to ville
violations['final_neighborhood'] = violations['neighborhood'].fillna(violations.get('ville', 'Unknown')).fillna('Unknown')

# For businesses, use neighborhood if available, otherwise fall back to city
businesses['final_neighborhood'] = businesses['neighborhood'].fillna(businesses.get('city', 'Unknown')).fillna('Unknown')

print(f"Final violations neighborhoods: {violations['final_neighborhood'].nunique()} unique")
print(f"Final business neighborhoods: {businesses['final_neighborhood'].nunique()} unique")

# Process GPS data if available
if 'gps' in violations.columns and 'gps' in businesses.columns:
    print("Processing GPS coordinates...")
    
    def parse_gps(gps_string):
        if pd.isna(gps_string):
            return None, None
        try:
            coords = str(gps_string).split(',')
            if len(coords) >= 2:
                lat = float(coords[0].strip())
                lon = float(coords[1].strip())
                return lat, lon
        except:
            pass
        return None, None
    
    violations[['parsed_lat', 'parsed_lon']] = violations['gps'].apply(
        lambda x: pd.Series(parse_gps(x))
    )
    businesses[['parsed_lat', 'parsed_lon']] = businesses['gps'].apply(
        lambda x: pd.Series(parse_gps(x))
    )
    
    valid_violation_coords = violations[['parsed_lat', 'parsed_lon']].notna().all(axis=1).sum()
    valid_business_coords = businesses[['parsed_lat', 'parsed_lon']].notna().all(axis=1).sum()
    
    print(f"✓ Parsed coordinates - Violations: {valid_violation_coords}, Businesses: {valid_business_coords}")

# Get the top neighborhoods by actual violation count for better per capita analysis
print(f"\nTop neighborhoods by violation count:")
top_violation_neighborhoods = violations['final_neighborhood'].value_counts().head(15)
print(top_violation_neighborhoods)

print(f"\nTop neighborhoods by business count:")
top_business_neighborhoods = businesses['final_neighborhood'].value_counts().head(15)
print(top_business_neighborhoods)

# Find neighborhoods that have both significant violations AND businesses for per capita analysis
violation_neighborhoods = set(violations['final_neighborhood'].value_counts().head(20).index)
business_neighborhoods = set(businesses['final_neighborhood'].value_counts().head(20).index)
common_neighborhoods = violation_neighborhoods.intersection(business_neighborhoods)

print(f"\nNeighborhoods with both violations and businesses (top candidates for per capita): {len(common_neighborhoods)}")
print(f"Common neighborhoods: {sorted(common_neighborhoods)}")

# UPDATED: Specify neighborhoods of interest and combine with top performers
specified_neighborhoods = ['Quartier chinois', 'Little Italy', 'Villeray']
print(f"\nSpecified neighborhoods to include: {specified_neighborhoods}")

# Check which specified neighborhoods exist in the data (case-insensitive)
existing_specified = []
for neighborhood in specified_neighborhoods:
    # Check violations
    violations_matches = violations[violations['final_neighborhood'].str.contains(neighborhood, case=False, na=False)]
    businesses_matches = businesses[businesses['final_neighborhood'].str.contains(neighborhood, case=False, na=False)]
    
    print(f"\nChecking for '{neighborhood}':")
    if len(violations_matches) > 0:
        matched_neighborhoods = violations_matches['final_neighborhood'].value_counts()
        print(f"  Violations matches: {matched_neighborhoods.to_dict()}")
        # Use the most common match
        exact_name = matched_neighborhoods.index[0]
        existing_specified.append(exact_name)
    else:
        print(f"  No violations found for '{neighborhood}'")
    
    if len(businesses_matches) > 0:
        matched_businesses = businesses_matches['final_neighborhood'].value_counts()
        print(f"  Business matches: {matched_businesses.to_dict()}")
    else:
        print(f"  No businesses found for '{neighborhood}'")

# Combine specified neighborhoods with top common neighborhoods
target_neighborhoods_for_per_capita = list(set(existing_specified + list(common_neighborhoods)[:10]))

# Ensure we have at least some neighborhoods to analyze
if len(target_neighborhoods_for_per_capita) == 0:
    # Fallback to top violation neighborhoods
    target_neighborhoods_for_per_capita = list(violation_neighborhoods)[:10]
    print(f"⚠️ Fallback: Using top violation neighborhoods")

print(f"\nFinal neighborhoods for per capita analysis ({len(target_neighborhoods_for_per_capita)}): {target_neighborhoods_for_per_capita}")

# Count businesses per neighborhood
businesses_per_neighborhood = businesses.groupby('final_neighborhood').size()
print(f"✓ Business counts calculated for {len(businesses_per_neighborhood)} neighborhoods")

# Show business counts for our target neighborhoods
print(f"\nBusiness counts for target neighborhoods:")
for neighborhood in target_neighborhoods_for_per_capita:
    business_count = businesses_per_neighborhood.get(neighborhood, 0)
    violation_count = violations[violations['final_neighborhood'] == neighborhood].shape[0]
    print(f"  {neighborhood}: {business_count} businesses, {violation_count} violations")

# --- Create Base Output Directory ---
if os.path.exists(OUTPUT_BASE_FOLDER):
    shutil.rmtree(OUTPUT_BASE_FOLDER)
    print(f"✓ Cleared existing base output folder: {OUTPUT_BASE_FOLDER}")
os.makedirs(OUTPUT_BASE_FOLDER)
print(f"✓ Created base output folder: {OUTPUT_BASE_FOLDER}")

# --- Enhanced Frame Generation Functions ---
def add_creator_credit(ax):
    """Add creator credit to the plot"""
    # Add English credit
    ax.text(0.02, 0.98, CREATOR_TEXT_EN, transform=ax.transAxes, 
            fontsize=10, fontweight='bold', color='#333333',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Add French credit
    ax.text(0.02, 0.93, CREATOR_TEXT_FR, transform=ax.transAxes, 
            fontsize=10, fontweight='bold', color='#333333',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def generate_single_line_frames(data_series, title_prefix, output_subfolder, ylabel='Number of Violations'):
    """Generate frames for single line plot (violations per month)"""
    print(f"\n--- Generating frames for {output_subfolder} ---")
    
    folder_path = os.path.join(OUTPUT_BASE_FOLDER, output_subfolder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✓ Created output folder: {folder_path}")

    frame_count = 0
    months = data_series.index.unique().sort_values()
    y_max = data_series.max() * 1.1 if not data_series.empty else 10
    
    print(f"✓ Single index data: {len(months)} months, max value: {data_series.max()}")
    
    if len(months) == 0:
        print(f"✗ No data to plot for {output_subfolder}")
        return 0

    for i, current_month in enumerate(months):
        try:
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

            # Data for current month and all previous months
            data_to_plot = data_series.loc[data_series.index <= current_month]
            if not data_to_plot.empty:
                ax.plot(range(len(data_to_plot)), data_to_plot.values,
                        marker='o', linestyle='-', color='#2c5aa0', linewidth=2, markersize=6)
                # Use numeric x-axis with month labels
                tick_positions = range(0, len(data_to_plot), max(1, len(data_to_plot)//10))
                tick_labels = [str(data_to_plot.index[pos]) for pos in tick_positions if pos < len(data_to_plot)]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_ylim(0, y_max)
            ax.set_title(f'{title_prefix}\n{current_month.strftime("%B %Y")}',
                        fontsize=18, fontweight='bold')
            ax.set_xlabel('Timeline', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add creator credit
            add_creator_credit(ax)
            
            plt.tight_layout()

            # Save the frame
            frame_filename = os.path.join(folder_path, f'frame_{i:04d}.png')
            plt.savefig(frame_filename, dpi=DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
            frame_count += 1
            
            # Print progress every 20 frames
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{len(months)} frames...")

        except Exception as e:
            print(f"✗ Error generating frame {i} for {output_subfolder}: {e}")
            plt.close()
            continue
    
    # Add pause frames at the end
    if frame_count > 0:
        last_frame_file = os.path.join(folder_path, f'frame_{frame_count-1:04d}.png')
        for j in range(PAUSE_FRAMES):
            pause_frame_file = os.path.join(folder_path, f'frame_{frame_count:04d}.png')
            try:
                shutil.copy2(last_frame_file, pause_frame_file)
                frame_count += 1
            except Exception as e:
                print(f"✗ Error creating pause frame: {e}")
                break

    print(f"✓ Generated {frame_count} total frames for {output_subfolder}")
    return frame_count

def generate_multiline_frames(data_series, title_prefix, output_subfolder, ylabel='Number of Violations', neighborhoods_to_show=None, top_n=10):
    """Generate frames for multiple line plots (categories/neighborhoods over time)"""
    print(f"\n--- Generating frames for {output_subfolder} ---")
    
    folder_path = os.path.join(OUTPUT_BASE_FOLDER, output_subfolder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✓ Created output folder: {folder_path}")

    frame_count = 0
    months = data_series.index.get_level_values(0).unique().sort_values()
    
    # Get items to track
    if neighborhoods_to_show is not None:
        # Use specified neighborhoods
        available_items = []
        for item in neighborhoods_to_show:
            if item in data_series.index.get_level_values(1):
                available_items.append(item)
        top_items = pd.Index(available_items)
        print(f"✓ Using specified neighborhoods: {list(top_items)}")
    else:
        # Get top categories/neighborhoods based on total violations
        top_items = data_series.groupby(level=1).sum().nlargest(top_n).index
        print(f"✓ Using top {top_n} items by total: {list(top_items)}")
    
    if len(top_items) == 0:
        print(f"✗ No items to track for {output_subfolder}")
        return 0
    
    # Create a color palette for the lines
    colors = plt.cm.Set3(np.linspace(0, 1, len(top_items)))
    
    y_max = data_series.max() * 1.1 if not data_series.empty else 10
    
    print(f"✓ MultiIndex data: {len(months)} months, tracking {len(top_items)} items")
    print(f"Month range: {months[0]} to {months[-1]}")

    for i, current_month in enumerate(months):
        try:
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

            # Plot lines for each top category/neighborhood up to current month
            legend_labels = []
            for j, item in enumerate(top_items):
                # Get data for this item up to current month
                item_data = []
                item_months = []
                
                for month in months[:i+1]:
                    if (month, item) in data_series.index:
                        value = data_series.loc[(month, item)]
                    else:
                        value = 0
                    item_data.append(value)
                    item_months.append(month)
                
                if item_data:
                    ax.plot(range(len(item_data)), item_data,
                            marker='o', linestyle='-', color=colors[j], 
                            linewidth=2, markersize=4, alpha=0.8)
                    legend_labels.append(str(item)[:25] + ('...' if len(str(item)) > 25 else ''))
            
            # Set up the plot
            ax.set_ylabel(ylabel, fontsize=14)
            ax.set_ylim(0, y_max)
            ax.set_title(f'{title_prefix}\n{current_month.strftime("%B %Y")}',
                        fontsize=18, fontweight='bold')
            ax.set_xlabel('Timeline', fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add legend
            if legend_labels:
                ax.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # Set x-axis labels
            if i > 0:
                tick_positions = range(0, i+1, max(1, (i+1)//10))
                tick_labels = [str(months[pos]) for pos in tick_positions if pos <= i]
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels, rotation=45, ha='right')
            
            # Add creator credit
            add_creator_credit(ax)
            
            plt.tight_layout()

            # Save the frame
            frame_filename = os.path.join(folder_path, f'frame_{i:04d}.png')
            plt.savefig(frame_filename, dpi=DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()
            frame_count += 1
            
            # Print progress every 20 frames
            if (i + 1) % 20 == 0:
                print(f"  Generated {i + 1}/{len(months)} frames...")

        except Exception as e:
            print(f"✗ Error generating frame {i} for {output_subfolder}: {e}")
            plt.close()
            continue
    
    # Add pause frames at the end
    if frame_count > 0:
        last_frame_file = os.path.join(folder_path, f'frame_{frame_count-1:04d}.png')
        for j in range(PAUSE_FRAMES):
            pause_frame_file = os.path.join(folder_path, f'frame_{frame_count:04d}.png')
            try:
                shutil.copy2(last_frame_file, pause_frame_file)
                frame_count += 1
            except Exception as e:
                print(f"✗ Error creating pause frame: {e}")
                break

    print(f"✓ Generated {frame_count} total frames for {output_subfolder}")
    return frame_count

def create_video_from_folders(folder_list, output_filename):
    """Create a video from a list of frame folders"""
    print(f"\n=== CREATING VIDEO '{output_filename}' ===")
    
    # Get list of all image files in order
    image_files = []
    
    for subfolder in folder_list:
        folder_path = os.path.join(OUTPUT_BASE_FOLDER, subfolder)
        if os.path.exists(folder_path):
            files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
            print(f"✓ Found {len(files)} PNG files in {subfolder}")
            
            # Add full paths
            for f in files:
                full_path = os.path.join(folder_path, f)
                image_files.append(full_path)
        else:
            print(f"✗ Subfolder {subfolder} not found!")

    print(f"✓ Total image files for video: {len(image_files)}")

    if len(image_files) == 0:
        print("✗ No image files found! Cannot create video.")
        return False

    # Create a single directory with all images sequentially numbered
    combined_folder = os.path.join(OUTPUT_BASE_FOLDER, f'combined_frames_{output_filename.replace(".mp4", "")}')
    if os.path.exists(combined_folder):
        shutil.rmtree(combined_folder)
    os.makedirs(combined_folder)
    
    # Copy all images to combined folder with sequential naming
    frame_num = 0
    for img_file in image_files:
        if os.path.exists(img_file):
            new_name = f'frame_{frame_num:06d}.png'
            shutil.copy2(img_file, os.path.join(combined_folder, new_name))
            frame_num += 1
    
    print(f"✓ Created combined frames folder with {frame_num} images")
    
    # Enhanced FFmpeg command with scaling filter to ensure even dimensions
    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output files
        '-framerate', str(FPS),
        '-i', os.path.join(combined_folder, 'frame_%06d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Force even width/height
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-preset', 'medium',
        '-movflags', '+faststart',  # Optimize for web playback
        output_filename
    ]
    
    try:
        print(f"Running FFmpeg command: {' '.join(ffmpeg_command)}")
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"✓ Video '{output_filename}' created successfully!")
        
        # Check video file size
        if os.path.exists(output_filename):
            video_size = os.path.getsize(output_filename)
            print(f"✓ Video file size: {video_size:,} bytes ({video_size/1024/1024:.1f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating video {output_filename}:")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Try alternative approach with different scaling
        print("Trying alternative FFmpeg approach...")
        alt_command = [
            'ffmpeg',
            '-y',
            '-framerate', str(FPS),
            '-i', os.path.join(combined_folder, 'frame_%06d.png'),
            '-vf', 'scale=1400:800',  # Force specific even dimensions
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '23',
            '-preset', 'fast',
            output_filename
        ]
        
        try:
            alt_result = subprocess.run(alt_command, check=True, capture_output=True, text=True)
            print(f"✓ Video '{output_filename}' created successfully with alternative method!")
            return True
        except subprocess.CalledProcessError as e2:
            print(f"✗ Alternative method also failed: {e2.stderr}")
            return False
        
    except FileNotFoundError:
        print("✗ Error: FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
        print("Download from: https://ffmpeg.org/download.html")
        return False

# --- Generate Visualizations ---

# Visualization 1: Violations per Month
print("\n=== VISUALIZATION 1: VIOLATIONS PER MONTH ===")
violations_per_month = violations.groupby('year_month').size().sort_index()

# Fill missing months to ensure smooth animation
min_month = violations['year_month'].min()
max_month = violations['year_month'].max()
full_month_range = pd.period_range(start=min_month, end=max_month, freq='M')
violations_per_month_full = violations_per_month.reindex(full_month_range, fill_value=0)

frames_1 = generate_single_line_frames(violations_per_month_full, 'Health Violations in Montreal Per Month', 
                                      'violations_per_month', 'Number of Violations')

# Visualization 2: Violations by Category per Month
print("\n=== VISUALIZATION 2: VIOLATIONS BY CATEGORY ===")
violations_by_category_month = violations.groupby(['year_month', 'categorie']).size()

frames_2 = generate_multiline_frames(violations_by_category_month, 'Health Violations in Montreal by Category', 
                                    'violations_by_category_per_month', 'Number of Violations', top_n=10)

# Visualization 3: Violations per Capita by Neighborhood per Month (ENHANCED WITH GPS)
print("\n=== VISUALIZATION 3: VIOLATIONS PER CAPITA BY NEIGHBORHOOD (WITH GPS ENHANCEMENT) ===")

# Calculate violations per capita more systematically
violations_per_capita_data = []

# Group violations by month and neighborhood (using final_neighborhood)
violations_by_neighborhood_month = violations.groupby(['year_month', 'final_neighborhood']).size()

# Get all unique months
all_months = violations['year_month'].unique()

# Use the neighborhoods we identified (including specified ones)
neighborhoods_to_analyze = target_neighborhoods_for_per_capita

print(f"Analyzing per capita for neighborhoods: {neighborhoods_to_analyze}")

# If GPS data is available, print some geographic insights
if 'parsed_lat' in violations.columns and 'parsed_lon' in violations.columns:
    print("\nGPS-based geographic insights:")
    for neighborhood in neighborhoods_to_analyze[:5]:  # Show top 5 neighborhoods
        neighborhood_violations = violations[violations['final_neighborhood'] == neighborhood]
        valid_coords = neighborhood_violations[['parsed_lat', 'parsed_lon']].notna().all(axis=1)
        
        if valid_coords.sum() > 0:
            lat_range = neighborhood_violations.loc[valid_coords, 'parsed_lat'].agg(['min', 'max'])
            lon_range = neighborhood_violations.loc[valid_coords, 'parsed_lon'].agg(['min', 'max'])
            print(f"  {neighborhood}: {valid_coords.sum()} violations with GPS coords")
            print(f"    Lat range: {lat_range['min']:.4f} to {lat_range['max']:.4f}")
            print(f"    Lon range: {lon_range['min']:.4f} to {lon_range['max']:.4f}")

for month in all_months:
    for neighborhood in neighborhoods_to_analyze:
        # Get violations for this month and neighborhood
        if (month, neighborhood) in violations_by_neighborhood_month.index:
            violation_count = violations_by_neighborhood_month.loc[(month, neighborhood)]
        else:
            violation_count = 0
        
        # Get business count for this neighborhood (using final_neighborhood)
        if neighborhood in businesses_per_neighborhood.index:
            business_count = businesses_per_neighborhood.loc[neighborhood]
        else:
            business_count = 1  # Avoid division by zero
        
        # Calculate per capita (violations per business)
        per_capita = violation_count / business_count if business_count > 0 else 0
        
        violations_per_capita_data.append({
            'year_month': month,
            'neighborhood': neighborhood,
            'violations_per_capita': per_capita
        })

# Convert to DataFrame and then to Series with MultiIndex
per_capita_df = pd.DataFrame(violations_per_capita_data)
violations_per_capita_series = per_capita_df.set_index(['year_month', 'neighborhood'])['violations_per_capita']

print(f"Per capita data shape: {violations_per_capita_series.shape}")
print(f"Max per capita value: {violations_per_capita_series.max():.4f}")

# Print summary of what we're tracking
neighborhoods_in_per_capita = violations_per_capita_series.index.get_level_values(1).unique()
print(f"Neighborhoods in per capita analysis: {list(neighborhoods_in_per_capita)}")

# Print some sample data for each neighborhood
for neighborhood in neighborhoods_in_per_capita:
    neighborhood_data = violations_per_capita_series.xs(neighborhood, level=1)
    total_violations = violations[violations['final_neighborhood'] == neighborhood].shape[0]
    business_count = businesses_per_neighborhood.get(neighborhood, 0)
    print(f"{neighborhood}: {total_violations} total violations, {business_count} businesses, max per capita: {neighborhood_data.max():.4f}")

frames_3 = generate_multiline_frames(violations_per_capita_series, 'Health Violations per Business by Montreal Neighborhood', 
                                    'violations_per_capita_per_month', 'Violations per Business', 
                                    neighborhoods_to_show=neighborhoods_to_analyze)

total_frames = frames_1 + frames_2 + frames_3
print(f"\n=== TOTAL FRAMES GENERATED: {total_frames} ===")

# --- Create Two Separate Videos ---

# Video 1: Parts 1 & 2 (Overview)
success_1 = create_video_from_folders(['violations_per_month', 'violations_by_category_per_month'], VIDEO_1_FILENAME)

# Video 2: Part 3 (Per Capita Analysis)
success_2 = create_video_from_folders(['violations_per_capita_per_month'], VIDEO_2_FILENAME)

# --- Final Cleanup ---
print("\n=== CLEANUP ===")
keep_frames = input("Keep frame files for inspection? (y/N): ").lower().startswith('y')

if not keep_frames and os.path.exists(OUTPUT_BASE_FOLDER):
    shutil.rmtree(OUTPUT_BASE_FOLDER)
    print(f"✓ Removed temporary frames folder: {OUTPUT_BASE_FOLDER}")
else:
    print(f"✓ Keeping frames folder: {OUTPUT_BASE_FOLDER}")

print(f"\n=== ANALYSIS COMPLETE ===")
if success_1:
    print(f"✓ Created overview video: {VIDEO_1_FILENAME}")
if success_2:
    print(f"✓ Created per capita video: {VIDEO_2_FILENAME}")

# Final summary
print(f"\n=== SUMMARY ===")
print(f"Total frames generated: {total_frames}")
print(f"Creator credit added: {CREATOR_TEXT_EN} / {CREATOR_TEXT_FR}")

# GPS data summary
if 'gps' in violations.columns:
    gps_violations = violations['gps'].notna().sum()
    print(f"GPS-enabled violations: {gps_violations}/{len(violations)} ({gps_violations/len(violations)*100:.1f}%)")
if 'gps' in businesses.columns:
    gps_businesses = businesses['gps'].notna().sum()
    print(f"GPS-enabled businesses: {gps_businesses}/{len(businesses)} ({gps_businesses/len(businesses)*100:.1f}%)")

# Summary of target neighborhoods
print(f"\nTarget neighborhoods included in per capita analysis:")
for i, neighborhood in enumerate(target_neighborhoods_for_per_capita, 1):
    violation_count = violations[violations['final_neighborhood'] == neighborhood].shape[0]
    business_count = businesses_per_neighborhood.get(neighborhood, 0)
    specified_status = "✓ SPECIFIED" if neighborhood in existing_specified else ""
    print(f"  {i:2d}. {neighborhood}: {violation_count} violations, {business_count} businesses {specified_status}")

if success_1 and success_2:
    print("✅ Both videos created successfully with GPS data and creator credits!")
    print("✅ Specified neighborhoods (Quartier chinois, Little Italy, Villeray) included where data available!")
elif success_1 or success_2:
    print("⚠️  One video created successfully, check the other")
else:
    print("❌ Video creation failed - check FFmpeg installation")