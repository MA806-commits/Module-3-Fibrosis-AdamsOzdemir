from termcolor import colored
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from scipy.integrate import trapezoid
from pathlib import Path 

#   SETUP PATHS
base_path = Path(__file__).parent
project_root = base_path.parent 
images_folder = project_root / "images" 
csv_path = project_root / "Filenames and Depths for Students.csv"

#   LOAD METADATA
master_df = pd.read_csv(csv_path)
# Standardize column names to remove hidden spaces
master_df.columns = master_df.columns.str.strip()

# Standardize "Filenames" to match your CSV and handle case sensitivity
master_df["Filenames"] = master_df["Filenames"].str.strip().str.lower()

#   CHOOSE IMAGES
target_filenames = [
    "images/MASK_SK658 Slobe ch010159.jpg",
    "images/MASK_SK658 Slobe ch010158.jpg",
    "images/MASK_SK658 Slobe ch010157.jpg",
    "images/MASK_SK658 Slobe ch010156.jpg",
    "images/MASK_SK658 Slobe ch010149.jpg",
    "images/MASK_SK658 Slobe ch010147.jpg",
]

# Standardize search list to lowercase
search_list = [name.strip().lower() for name in target_filenames]

DEPTH_COL = "Depth from lung surface (in micrometers) where image was acquired"

# Filter the CSV for chosen images
selected_metadata = master_df[master_df["Filenames"].isin(search_list)].copy()
results = [] 

print(colored(f"Processing {len(selected_metadata)} selected images...", "yellow"))

#   IMAGE PROCESSING LOOP
for index, row in selected_metadata.iterrows():
    fname_lower = row['Filenames']
    depth = row[DEPTH_COL]

    # Use project_root because the filename already contains "images/"
    full_path = project_root / fname_lower
    img = cv2.imread(str(full_path), 0)

    if img is None: 
        print(colored(f"Error: Could not load {fname_lower} at {full_path}", "red"))
        continue 

    #   PIXEL COUNTING LOGIC 
    # Using 127 binary threshold for sparse black and white lung images
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    white_px = np.sum(binary == 255)
    total_px = binary.size
    percent = (100 * (white_px / total_px)) if total_px > 0 else 0 

    # Adding Filename here so it appears in the final CSV output
    results.append({
        "Filename": fname_lower,
        "Depth": depth, 
        "White Percent": percent
    })

    print(f'Successfully processed {fname_lower} ({percent:.2f}%)')

#   DATA HANDLING & MATH
if not results:
    print(colored("No data found! Check filenames in your target_filenames list.", "red"))
    exit()

# Create DataFrame and sort by depth for correct math/plotting
df_results = pd.DataFrame(results).sort_values('Depth')

# EXPORT TO CSV (Final Project Requirement)
df_results.to_csv("Fibrosis_Analysis_Summary.csv", index=False)
print(colored("\nResults successfully saved to 'Fibrosis_Analysis_Summary.csv'", "green"))

x = df_results["Depth"].values
y = df_results["White Percent"].values 

# Calculate Total Fibrosis Load (Area Under Curve) - Above and Beyond
total_load = trapezoid(y, x)
print(colored(f"Total Fibrosis Load (Area Under the Curve): {total_load:.2f}", 'cyan'))

# Formatting the depths list for the terminal
depth_string = ", ".join(map(str, x)) 
print(colored(f"Depths Analyzed: {depth_string} microns", 'green'))

#   INTERPOLATION
interpolate_depth = float(input(colored(f"\nEnter Depth to Interpolate (microns): ", "yellow")))
i_func = interp1d(x, y, kind="linear", fill_value="extrapolate")
interp_point = i_func(interpolate_depth)
print(colored(f"Interpolated Fibrosis at {interpolate_depth} microns: {interp_point:.2f}%", "green"))

#    VISUALIZATION
plt.figure(figsize=(10, 5))
plt.plot(x, y, 'o-', label='Calculated Data', linewidth=2)
plt.plot(interpolate_depth, interp_point, 'ro', markersize=12, label='Interpolated Point')
plt.fill_between(x, y, color='blue', alpha=0.1, label='Total Fibrosis Volume (AUC)')

plt.title("Pulmonary Fibrosis Analysis: Depth vs. White Pixel %")
plt.xlabel("Depth (microns)")
plt.ylabel('% White Pixels (Fibrosis Indicator)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
