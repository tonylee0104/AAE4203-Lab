import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Load data from CSV files (pseudoranges, satellite positions, errors)
# These represent measured pseudoranges and corrections for each satellite/epoch
satellite_positions = np.loadtxt("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/satellite_positions.csv", delimiter=',')  # (max_num_sats, num_epochs*3)
pseudoranges_meas = np.loadtxt("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/pseudoranges_meas.csv", delimiter=',')  # (max_num_sats, num_epochs)
satellite_clock_bias = np.loadtxt("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/satellite_clock_bias.csv", delimiter=',')  # (max_num_sats, num_epochs)
ionospheric_delay = np.loadtxt("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/ionospheric_delay.csv", delimiter=',')  # (max_num_sats, num_epochs)
tropospheric_delay = np.loadtxt("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/tropospheric_delay.csv", delimiter=',')  # (max_num_sats, num_epochs)

# Function to convert ECEF to ENU (local coordinates relative to reference)
def ecef_to_enu(lat0, lon0, h0, x, y, z):
    """
    Convert ECEF coordinates to ENU coordinates relative to a reference point
    """
    # WGS84 parameters
    a = 6378137.0  # semi-major axis
    e = 0.0818191908426  # eccentricity
    
    # Convert reference point from geodetic to ECEF
    N0 = a / np.sqrt(1 - e**2 * np.sin(np.radians(lat0))**2)
    x0 = (N0 + h0) * np.cos(np.radians(lat0)) * np.cos(np.radians(lon0))
    y0 = (N0 + h0) * np.cos(np.radians(lat0)) * np.sin(np.radians(lon0))
    z0 = (N0 * (1 - e**2) + h0) * np.sin(np.radians(lat0))
    
    # Compute differences
    dx = x - x0
    dy = y - y0
    dz = z - z0
    
    # Rotation matrix from ECEF to ENU
    sin_lat = np.sin(np.radians(lat0))
    cos_lat = np.cos(np.radians(lat0))
    sin_lon = np.sin(np.radians(lon0))
    cos_lon = np.cos(np.radians(lon0))
    
    # ENU coordinates
    east = -sin_lon * dx + cos_lon * dy
    north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    up = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    
    return east, north, up

# Estimate initial reference position by averaging ground truth ECEF positions
def estimate_reference_position(df):
    """
    Estimate reference position (lat, lon, height) from ECEF coordinates
    by averaging the positions
    """
    # Convert ECEF to approximate geodetic (simple method for reference)
    x_mean = df['ecefX'].mean()
    y_mean = df['ecefY'].mean()
    z_mean = df['ecefZ'].mean()
    
    # Simple conversion to geodetic (approximate)
    p = np.sqrt(x_mean**2 + y_mean**2)
    theta = np.arctan2(z_mean * 6378137.0, p * 6356752.314245)
    
    lat = np.arctan2(z_mean + 0.006739496742276 * 6356752.314245 * np.sin(theta)**3,
                     p - 0.006739496742276 * 6378137.0 * np.cos(theta)**3)
    lon = np.arctan2(y_mean, x_mean)
    
    N = 6378137.0 / np.sqrt(1 - 0.00669437999014 * np.sin(lat)**2)
    h = p / np.cos(lat) - N
    
    return np.degrees(lat), np.degrees(lon), h

# Load ground truth data (high-precision measurements for error comparison)
Ground_truth = r"C:\Users\abc66\Dropbox\Year 4\SEM 1\Guidance and navigation\LAB\lab\Result\UBX\NAV-HPPOSECEF.csv"
# Check if file exists
if not os.path.exists(Ground_truth):
    print(f"Error: File not found at {Ground_truth}")
    print("Please check the file path and try again.")
else:
    try:
        # Read the data
        df = pd.read_csv(Ground_truth)
        
        # Convert from cm to meters (assuming the data is in cm based on typical GPS formats)
        df['ecefX'] = df['ecefX'] / 100
        df['ecefY'] = df['ecefY'] / 100
        df['ecefZ'] = df['ecefZ'] / 100
        
        # Estimate reference position
        lat0, lon0, h0 = estimate_reference_position(df)
        
        # We will compute ENU later with shared reference
        
        # Print detailed statistics
        print(f"\n=== Trajectory Statistics ===")
        print(f"Total points: {len(df)}")
        print(f"Position accuracy range: {df['pAcc'].min():.1f} - {df['pAcc'].max():.1f} cm")
        
        # Calculate speeds (approximate)
        
        dx = np.diff(df['ecefX'])
        dy = np.diff(df['ecefY'])
        dz = np.diff(df['ecefZ'])
        
        
    except Exception as e:
        print(f"Error processing the file: {e}")
        print("Please check if the file format is correct.")

# Get the number of epochs
num_epochs = pseudoranges_meas.shape[1]
max_num_sats = pseudoranges_meas.shape[0]

# Initialize variables to store estimated positions
estimated_positions = []  # List to store estimated receiver positions
estimated_clock_biases = []  # List to store estimated receiver clock biases

# Set initial receiver position (you can set this to a known approximate position)
receiver_position = np.array([0.0, 0.0, 0.0])  # Units: meters
c = 299792458.0  # Speed of light, meters per second

# Core SPP function: Least squares solver for position and clock bias
def least_squares_solution(satellite_positions, receiver_position, pseudoranges_meas, satellite_clock_bias,
                           ionospheric_delay, tropospheric_delay):
    receiver_clock_bias = 0.0  # Receiver clock bias, in meters
    for j in range(10):  # Maximum iterations (iterative refinement)
        # Compute geometric distances (rho_i)
        estimated_distances = np.linalg.norm(satellite_positions - receiver_position, axis=1)
        # Correct pseudoranges for known errors (P_corrected = P + dt_s - I - T)
        corrected_pseudoranges = pseudoranges_meas + satellite_clock_bias - ionospheric_delay - tropospheric_delay
        # Compute residuals (delta P = P_corrected - (rho + dt_r))
        pseudoranges_diff = corrected_pseudoranges - (estimated_distances + receiver_clock_bias)
        # Build design matrix G (partial derivatives: -unit_vector for position, 1 for clock)
        G = np.zeros((len(satellite_positions), 4))
        for i in range(len(satellite_positions)):
            p_i = satellite_positions[i] - receiver_position
            r_i = estimated_distances[i]
            G[i, :3] = -p_i / r_i
            G[i, 3] = 1.0
        # Solve least squares: delta_x = (G^T G)^-1 G^T delta_P (using lstsq for stability)
        delta_p, residuals, rank, s = np.linalg.lstsq(G, pseudoranges_diff, rcond=None)
        receiver_position += delta_p[:3]
        receiver_clock_bias += delta_p[3]
        # Check convergence
        if np.linalg.norm(delta_p[:3]) < 1e-4:
            break
    return receiver_position, receiver_clock_bias

# ECEF to LLA conversion
def ecef_to_lla(x, y, z):
    # WGS84 ellipsoid constants
    a = 6378137.0  # Semi-major axis, meters
    e2 = 6.69437999014e-3  # Square of eccentricity
    # Longitude calculation
    lon = np.arctan2(y, x)
    # Latitude and altitude initial estimation
    p = np.sqrt(x ** 2 + y ** 2)
    lat = np.arctan2(z, p * (1 - e2))  # Initial latitude
    lat_prev = 0
    N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - N
    # Iterative computation
    while np.abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = a / np.sqrt(1 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e2 * N / (N + h)))
    # Convert from radians to degrees
    lat_deg = np.degrees(lat)
    lon_deg = np.degrees(lon)
    return lat_deg, lon_deg, h

# ECEF to ENU conversion (alternative implementation)
def ecef_to_enu(x, y, z, x_ref, y_ref, z_ref):
    # Convert reference point to lat, lon, h
    lat_ref, lon_ref, h_ref = ecef_to_lla(x_ref, y_ref, z_ref)
    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)
    # Compute difference vector
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref
    # Transformation matrix
    t = np.array([
        [-np.sin(lon_ref), np.cos(lon_ref), 0],
        [-np.sin(lat_ref)*np.cos(lon_ref), -np.sin(lat_ref)*np.sin(lon_ref), np.cos(lat_ref)],
        [np.cos(lat_ref)*np.cos(lon_ref), np.cos(lat_ref)*np.sin(lon_ref), np.sin(lat_ref)]
    ])
    # Apply transformation
    enu = t @ np.array([dx, dy, dz])
    return enu

# Main SPP loop: Process each epoch
for epoch in range(num_epochs):
    # Extract data for current epoch
    p_l1_epoch = pseudoranges_meas[:, epoch]
    sat_clock_err_epoch = satellite_clock_bias[:, epoch]
    ion_error_l1_epoch = ionospheric_delay[:, epoch]
    tropo_error_epoch = tropospheric_delay[:, epoch]
    sat_pos_epoch = satellite_positions[:, epoch*3:(epoch+1)*3]  # Columns for current epoch
    # Exclude invalid (NaN) data
    valid_idx = ~np.isnan(p_l1_epoch) & \
                ~np.isnan(sat_clock_err_epoch) & \
                ~np.isnan(ion_error_l1_epoch) & \
                ~np.isnan(tropo_error_epoch) & \
                ~np.isnan(sat_pos_epoch[:, 0]) & \
                ~np.isnan(sat_pos_epoch[:, 1]) & \
                ~np.isnan(sat_pos_epoch[:, 2])
    # Check if enough satellites (>=4)
    if np.sum(valid_idx) < 4:
        print(f"Epoch {epoch+1}: Not enough satellites, skipping this epoch.")
        if epoch > 0:
            # Use previous position
            estimated_positions.append(estimated_positions[-1])
            estimated_clock_biases.append(estimated_clock_biases[-1])
        else:
            # Use initial position
            estimated_positions.append(receiver_position.copy())
            estimated_clock_biases.append(0.0)
        continue
    # Extract valid data
    p_l1_valid = p_l1_epoch[valid_idx]
    sat_clock_err_valid = sat_clock_err_epoch[valid_idx]
    ion_error_l1_valid = ion_error_l1_epoch[valid_idx]
    tropo_error_valid = tropo_error_epoch[valid_idx]
    sat_pos_valid = sat_pos_epoch[valid_idx, :]
    # Use previous epoch's position as initial estimate
    if epoch > 0:
        receiver_position = estimated_positions[-1].copy()
    else:
        receiver_position = np.array([0.0, 0.0, 0.0])  # Initial position
    # Run least squares
    estimated_position, estimated_receiver_clock_bias = least_squares_solution(
        sat_pos_valid, receiver_position, p_l1_valid, sat_clock_err_valid, ion_error_l1_valid, tropo_error_valid
    )
    # Store results
    estimated_positions.append(estimated_position.copy())
    estimated_clock_biases.append(estimated_receiver_clock_bias)

# Convert estimated ECEF to LLA
lat_list = []
lon_list = []
alt_list = []
for pos in estimated_positions:
    lat, lon, alt = ecef_to_lla(pos[0], pos[1], pos[2])
    lat_list.append(lat)
    lon_list.append(lon)
    alt_list.append(alt)

# Compute ENU relative to first estimated position
enu_list = []
x_ref, y_ref, z_ref = estimated_positions[0]  # Reference point
for pos in estimated_positions:
    enu = ecef_to_enu(pos[0], pos[1], pos[2], x_ref, y_ref, z_ref)
    enu_list.append(enu)

# Extract ENU components
east_list = [enu[0] for enu in enu_list]
north_list = [enu[1] for enu in enu_list]
up_list = [enu[2] for enu in enu_list]

# Print last epoch's LLA
print(f"Estimated Position in Latitude, Longitude, Altitude for last epoch:")
print(f"Latitude: {lat_list[-1]} degrees")
print(f"Longitude: {lon_list[-1]} degrees")
print(f"Altitude: {alt_list[-1]} meters")

# Compute LLA and ENU for ground truth data
measured_lat_list = []
measured_lon_list = []
measured_alt_list = []
measured_enu_list = []
for i in range(len(df)):
    x = df['ecefX'].iloc[i]
    y = df['ecefY'].iloc[i]
    z = df['ecefZ'].iloc[i]
    lat, lon, alt = ecef_to_lla(x, y, z)
    measured_lat_list.append(lat)
    measured_lon_list.append(lon)
    measured_alt_list.append(alt)
    enu = ecef_to_enu(x, y, z, x_ref, y_ref, z_ref)
    measured_enu_list.append(enu)

# Extract ground truth ENU
measured_east_list = [enu[0] for enu in measured_enu_list]
measured_north_list = [enu[1] for enu in measured_enu_list]
measured_up_list = [enu[2] for enu in measured_enu_list]

# Compute errors
east_errors = np.array(east_list) - np.array(measured_east_list)
north_errors = np.array(north_list) - np.array(measured_north_list)
up_errors = np.array(up_list) - np.array(measured_up_list)
horizontal_errors = np.sqrt(east_errors**2 + north_errors**2)
three_d_errors = np.sqrt(east_errors**2 + north_errors**2 + up_errors**2)

# Compute RMS errors
east_rms = np.sqrt(np.mean(east_errors**2))
north_rms = np.sqrt(np.mean(north_errors**2))
up_rms = np.sqrt(np.mean(up_errors**2))
horizontal_rms = np.sqrt(np.mean(horizontal_errors**2))
three_d_rms = np.sqrt(np.mean(three_d_errors**2))

# Print error stats
print("\n=== Positioning Error Statistics ===")
print("East Errors (m):")
print(f"Mean: {np.mean(east_errors):.3f}, RMS: {east_rms:.3f}, Std: {np.std(east_errors):.3f}, Min: {np.min(east_errors):.3f}, Max: {np.max(east_errors):.3f}")
print("North Errors (m):")
print(f"Mean: {np.mean(north_errors):.3f}, RMS: {north_rms:.3f}, Std: {np.std(north_errors):.3f}, Min: {np.min(north_errors):.3f}, Max: {np.max(north_errors):.3f}")
print("Up Errors (m):")
print(f"Mean: {np.mean(up_errors):.3f}, RMS: {up_rms:.3f}, Std: {np.std(up_errors):.3f}, Min: {np.min(up_errors):.3f}, Max: {np.max(up_errors):.3f}")
print("2D Horizontal Errors (m):")
print(f"Mean: {np.mean(horizontal_errors):.3f}, RMS: {horizontal_rms:.3f}, Std: {np.std(horizontal_errors):.3f}, Min: {np.min(horizontal_errors):.3f}, Max: {np.max(horizontal_errors):.3f}")
print("3D Errors (m):")
print(f"Mean: {np.mean(three_d_errors):.3f}, RMS: {three_d_rms:.3f}, Std: {np.std(three_d_errors):.3f}, Min: {np.min(three_d_errors):.3f}, Max: {np.max(three_d_errors):.3f}")

# Visualization: 2D and 3D trajectories
plt.figure(figsize=(16, 6))
# Subplot 1: Latitude vs Longitude (2D)
plt.subplot(1, 2, 1)
plt.plot(measured_lon_list, measured_lat_list, 'b.-', label='Measured')
plt.plot(lon_list, lat_list, 'r.-', label='Estimated')
plt.scatter(measured_lon_list[0], measured_lat_list[0], color='green', s=100, label='Start (Measured)', zorder=5, marker='o')
plt.scatter(lon_list[0], lat_list[0], color='lime', s=50, label='Start (Estimated)', zorder=5, marker='o')
plt.scatter(measured_lon_list[-1], measured_lat_list[-1], color='darkblue', s=100, label='End (Measured)', zorder=5, marker='s')
plt.scatter(lon_list[-1], lat_list[-1], color='darkred', s=50, label='End (Estimated)', zorder=5, marker='s')
plt.title('GNSS Trajectory')
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.legend()
plt.grid(True)
# Subplot 2: 3D ENU Trajectory
ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot(measured_east_list, measured_north_list, measured_up_list, 'b.-', label='Measured')
ax.plot(east_list, north_list, up_list, 'r.-', label='Estimated')
ax.scatter(measured_east_list[0], measured_north_list[0], measured_up_list[0], color='green', s=100, label='Start (Measured)', zorder=5, marker='o')
ax.scatter(east_list[0], north_list[0], up_list[0], color='lime', s=50, label='Start (Estimated)', zorder=5, marker='o')
ax.scatter(measured_east_list[-1], measured_north_list[-1], measured_up_list[-1], color='darkblue', s=100, label='End (Measured)', zorder=5, marker='s')
ax.scatter(east_list[-1], north_list[-1], up_list[-1], color='darkred', s=50, label='End (Estimated)', zorder=5, marker='s')
ax.set_title('3D ENU Trajectory')
ax.set_xlabel('East (m)')
ax.set_ylabel('North (m)')
ax.set_zlabel('Up (m)')
ax.legend()
ax.grid(True)
plt.tight_layout()

# Error plots over time
epochs = np.arange(len(east_errors))
plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(epochs, east_errors, label='East Error')
plt.plot(epochs, north_errors, label='North Error')
plt.plot(epochs, up_errors, label='Up Error')
plt.title('Directional Errors over Time')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(epochs, horizontal_errors, label='2D Horizontal Error')
plt.title('2D Horizontal Error over Time')
plt.ylabel('Error (m)')
plt.legend()
plt.grid(True)
plt.subplot(3,1,3)
plt.plot(epochs, three_d_errors, label='3D Error')
plt.title('3D Error over Time')
plt.ylabel('Error (m)')
plt.xlabel('Epoch')
plt.legend()
plt.grid(True)
plt.tight_layout()

# East vs North Error Distribution
plt.figure(figsize=(8, 6))
plt.scatter(north_errors, east_errors, alpha=0.5)
plt.title('East vs North Error Distribution')
plt.xlabel('North Error (m)')
plt.ylabel('East Error (m)')
plt.grid(True)
plt.axhline(0, color='black', linestyle='--', linewidth=0.5)
plt.axvline(0, color='black', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Statistical Error Distribution (Histograms)
plt.figure(figsize=(10, 5))
plt.suptitle('Statistical Error Distribution Analysis (2D and 3D Error Histograms)')
plt.subplot(1,2,1)
plt.hist(horizontal_errors, bins=20)
plt.axvline(np.mean(horizontal_errors), color='r', linestyle='--', label='Mean')
plt.axvline(horizontal_rms, color='g', linestyle='-', label='RMS')
ax = plt.gca()
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
offset = (xmax - xmin) * 0.02
mean_val = np.mean(horizontal_errors)
rms_val = horizontal_rms
plt.text(mean_val + offset, ymax * 0.9, f'Mean: {mean_val:.3f} m', color='r', ha='left')
plt.text(rms_val + offset, ymax * 0.8, f'RMS: {rms_val:.3f} m', color='g', ha='left')
plt.title('2D Horizontal Error Histogram')
plt.legend()
plt.subplot(1,2,2)
plt.hist(three_d_errors, bins=20)
plt.axvline(np.mean(three_d_errors), color='r', linestyle='--', label='Mean')
plt.axvline(three_d_rms, color='g', linestyle='-', label='RMS')
ax = plt.gca()
ymin, ymax = ax.get_ylim()
xmin, xmax = ax.get_xlim()
offset = (xmax - xmin) * 0.02
mean_val = np.mean(three_d_errors)
rms_val = three_d_rms
plt.text(mean_val + offset, ymax * 0.9, f'Mean: {mean_val:.3f} m', color='r', ha='left')
plt.text(rms_val + offset, ymax * 0.8, f'RMS: {rms_val:.3f} m', color='g', ha='left')
plt.title('3D Error Histogram')
plt.legend()
plt.tight_layout()

# Error Evolution (with smoothing)
epochs = np.arange(len(east_errors))
smooth_2d = pd.Series(horizontal_errors).rolling(window=5).mean()
smooth_3d = pd.Series(three_d_errors).rolling(window=5).mean()
plt.figure(figsize=(12, 6))
plt.plot(epochs, horizontal_errors, label='2D Horizontal Error', alpha=0.5)
plt.plot(epochs, three_d_errors, label='3D Error', alpha=0.5)
plt.plot(epochs, smooth_2d, label='2D Smooth', color='blue')
plt.plot(epochs, smooth_3d, label='3D Smooth', color='orange')
plt.title('Error Evolution Analysis - 2D and 3D Positional Errors vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Positional Error (m)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Save results to CSV
try:
    estimated_positions_df = pd.DataFrame(estimated_positions, columns=['X', 'Y', 'Z'])
    estimated_positions_df['Latitude'] = lat_list
    estimated_positions_df['Longitude'] = lon_list
    estimated_positions_df['Altitude'] = alt_list
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/estimated_positions.csv"), exist_ok=True)
    
    estimated_positions_df.to_csv("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/estimated_positions.csv", index=False)
    print("Estimated positions saved successfully!")
    
except Exception as e:
    print(f"Error saving estimated positions: {e}")

# Save errors to CSV
try:
    errors_df = pd.DataFrame({
        'East Error': east_errors,
        'North Error': north_errors,
        'Up Error': up_errors,
        'Horizontal Error': horizontal_errors,
        '3D Error': three_d_errors
    })
    
    errors_df.to_csv("C:/Users/abc66/Dropbox/Year 4/SEM 1/Guidance and navigation/LAB/lab/Result/positioning_errors.csv", index=False)
    print("Positioning errors saved successfully!")
    
except Exception as e:
    print(f"Error saving positioning errors: {e}")