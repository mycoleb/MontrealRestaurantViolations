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

# --- Load Data ---
print("Loading data...")
try:
    violations = pd.read_csv('violations.csv')
    print(f"✓ Loaded {len(violations)} violations")
    
    businesses = pd.read_csv('businesses.csv')
    print(f"✓ Loaded {len(businesses)} businesses")
except FileNotFoundError as e:
    print(f"✗ Error: {e}. Please ensure both violations.csv and businesses.csv are in the same directory.")
    exit()

# --- Data Inspection ---
print("\n=== DATA INSPECTION ===")
print(f"Columns in violations.csv: {list(violations.columns)}")
print(f"Date column sample values: {violations['date'].head(10).tolist()}")
print(f"Date column dtype: {violations['date'].dtype}")

# Examine neighborhood data
print(f"\nUnique neighborhoods in violations (ville): {len(violations['ville'].dropna().unique())}")
print(f"Unique neighborhoods in businesses (city): {len(businesses['city'].dropna().unique())}")

# Show actual neighborhood names
print(f"\nSample violation neighborhoods: {sorted(violations['ville'].dropna().unique())[:10]}")
print(f"Sample business neighborhoods: {sorted(businesses['city'].dropna().unique())[:10]}")

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

# Clean neighborhood data
print("Processing neighborhood data...")
violations['neighborhood'] = violations['ville'].fillna('Unknown')
businesses['neighborhood'] = businesses['city'].fillna('Unknown')

# Get the top neighborhoods by actual violation count for better per capita analysis
print(f"\nTop neighborhoods by violation count:")
top_violation_neighborhoods = violations['neighborhood'].value_counts().head(15)
print(top_violation_neighborhoods)

print(f"\nTop neighborhoods by business count:")
top_business_neighborhoods = businesses['neighborhood'].value_counts().head(15)
print(top_business_neighborhoods)

# Find neighborhoods that have both significant violations AND businesses for per capita analysis
violation_neighborhoods = set(violations['neighborhood'].value_counts().head(20).index)
business_neighborhoods = set(businesses['neighborhood'].value_counts().head(20).index)
common_neighborhoods = violation_neighborhoods.intersection(business_neighborhoods)

print(f"\nNeighborhoods with both violations and businesses (top candidates for per capita): {len(common_neighborhoods)}")
print(f"Common neighborhoods: {sorted(common_neighborhoods)}")

# Use the top 10 neighborhoods that have both violations and businesses
target_neighborhoods_for_per_capita = list(common_neighborhoods)[:10]
print(f"\nUsing these neighborhoods for per capita analysis: {target_neighborhoods_for_per_capita}")

# Count businesses per neighborhood
businesses_per_neighborhood = businesses.groupby('neighborhood').size()
print(f"✓ Business counts calculated for {len(businesses_per_neighborhood)} neighborhoods")

# --- Create Base Output Directory ---
if os.path.exists(OUTPUT_BASE_FOLDER):
    shutil.rmtree(OUTPUT_BASE_FOLDER)
    print(f"✓ Cleared existing base output folder: {OUTPUT_BASE_FOLDER}")
os.makedirs(OUTPUT_BASE_FOLDER)
print(f"✓ Created base output folder: {OUTPUT_BASE_FOLDER}")

# --- Enhanced Frame Generation Functions ---
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
            plt.figure(figsize=(14, 8))

            # Data for current month and all previous months
            data_to_plot = data_series.loc[data_series.index <= current_month]
            if not data_to_plot.empty:
                plt.plot(range(len(data_to_plot)), data_to_plot.values,
                         marker='o', linestyle='-', color='#2c5aa0', linewidth=2, markersize=6)
                # Use numeric x-axis with month labels
                tick_positions = range(0, len(data_to_plot), max(1, len(data_to_plot)//10))
                tick_labels = [str(data_to_plot.index[pos]) for pos in tick_positions if pos < len(data_to_plot)]
                plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
            
            plt.ylabel(ylabel, fontsize=14)
            plt.ylim(0, y_max)
            plt.title(f'{title_prefix}\n{current_month.strftime("%B %Y")}',
                      fontsize=18, fontweight='bold')
            plt.xlabel('Timeline', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()

            # Save the frame
            frame_filename = os.path.join(folder_path, f'frame_{i:04d}.png')
            plt.savefig(frame_filename, dpi=DPI, bbox_inches='tight')
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
            plt.figure(figsize=(14, 8))

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
                    plt.plot(range(len(item_data)), item_data,
                             marker='o', linestyle='-', color=colors[j], 
                             linewidth=2, markersize=4, alpha=0.8)
                    legend_labels.append(str(item)[:25] + ('...' if len(str(item)) > 25 else ''))
            
            # Set up the plot
            plt.ylabel(ylabel, fontsize=14)
            plt.ylim(0, y_max)
            plt.title(f'{title_prefix}\n{current_month.strftime("%B %Y")}',
                      fontsize=18, fontweight='bold')
            plt.xlabel('Timeline', fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add legend
            if legend_labels:
                plt.legend(legend_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            
            # Set x-axis labels
            if i > 0:
                tick_positions = range(0, i+1, max(1, (i+1)//10))
                tick_labels = [str(months[pos]) for pos in tick_positions if pos <= i]
                plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
            
            plt.tight_layout()

            # Save the frame
            frame_filename = os.path.join(folder_path, f'frame_{i:04d}.png')
            plt.savefig(frame_filename, dpi=DPI, bbox_inches='tight')
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
    
    # FFmpeg command using image sequence
    ffmpeg_command = [
        'ffmpeg',
        '-y',
        '-framerate', str(FPS),
        '-i', os.path.join(combined_folder, 'frame_%06d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        '-preset', 'medium',
        output_filename
    ]
    
    try:
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"✓ Video '{output_filename}' created successfully!")
        
        # Check video file size
        if os.path.exists(output_filename):
            video_size = os.path.getsize(output_filename)
            print(f"✓ Video file size: {video_size:,} bytes ({video_size/1024/1024:.1f} MB)")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error creating video {output_filename}:")
        print(f"STDERR: {e.stderr}")
        return False
        
    except FileNotFoundError:
        print("✗ Error: FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
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

# Visualization 3: Violations per Capita by Neighborhood per Month (FIXED)
print("\n=== VISUALIZATION 3: VIOLATIONS PER CAPITA BY NEIGHBORHOOD (FIXED) ===")

# Calculate violations per capita more systematically
violations_per_capita_data = []

# Group violations by month and neighborhood
violations_by_neighborhood_month = violations.groupby(['year_month', 'neighborhood']).size()

# Get all unique months
all_months = violations['year_month'].unique()

# Use the neighborhoods we identified as having both violations and businesses
neighborhoods_to_analyze = target_neighborhoods_for_per_capita

print(f"Analyzing per capita for neighborhoods: {neighborhoods_to_analyze}")

for month in all_months:
    for neighborhood in neighborhoods_to_analyze:
        # Get violations for this month and neighborhood
        if (month, neighborhood) in violations_by_neighborhood_month.index:
            violation_count = violations_by_neighborhood_month.loc[(month, neighborhood)]
        else:
            violation_count = 0
        
        # Get business count for this neighborhood
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
    total_violations = violations[violations['neighborhood'] == neighborhood].shape[0]
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