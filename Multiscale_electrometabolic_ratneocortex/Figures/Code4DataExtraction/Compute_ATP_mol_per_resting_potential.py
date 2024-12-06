import h5py    
import numpy as np  
from multiprocessing import Pool
# Importing Pandas to create DataFrame
import pandas as pd
import time

def find_NO_AP_window(voltage, threshold, window_start, window_end, time_step, step_size):
    """
    Automatically find a window that does not contain an action potential (AP), starting from a 
    given window and moving backward if an AP is detected.
    
    Parameters:
    - voltage (numpy array): Voltage array in mV.
    - threshold (float): Voltage threshold for detecting an action potential.
    - window_start (int): Start time of the window in ms.
    - window_end (int): End time of the window in ms.
    - time_step (float): Time step in ms (e.g., 0.025 ms).
    - step_size (int): How much to shift the window back in ms each time.
    
    Returns:
    - (start_idx, end_idx): Indices of the no-action-potential window in the voltage array.
    """
    # Convert window times to sample indices
    window_start_idx = int(window_start / time_step)
    window_end_idx = int(window_end / time_step)
    step_samples = int(step_size / time_step)  # Step size in number of samples
    
    # Search backward through time windows to detect a window without an action potential
    for t_end_idx in range(window_end_idx, step_samples - 1, -step_samples):
        t_start_idx = max(0, t_end_idx - step_samples)
        
        # Check if voltage stays below the threshold within this window
        below_threshold = np.all(voltage[t_start_idx:t_end_idx] <= threshold)
        if below_threshold:
            return t_start_idx, t_end_idx
    
    # Return None if no suitable window is found
    return None

def compute_molecules_per_AP(atp_in_AP, time_step):
    """
    Compute the number of molecules involved per action potential (AP).
    
    Parameters:
    - atp_in_AP (numpy array): Array of ATP concentration values during the AP window.
    - time_step (float): Time step in ms.
    
    Returns:
    - energy_molecules (float): Number of energy molecules consumed during the AP.
    """
    molecules = 1e-3 * 6.0e+23 * 2e-12  # Avogadro's number and conversion factors
    dtao = len(atp_in_AP)  # Total time duration in ms
    energy_molecules = abs(atp_in_AP[0] - atp_in_AP[-1]) / (dtao / 1000) * molecules  # per second
    return energy_molecules

def read_voltage_window(result_path, row_idx, start_time, end_time):
    """
    Reads the voltage data for a specific row (row_idx) within a given time window (start_time, end_time).
    
    Parameters:
    - result_path (str): Path to the HDF5 file containing voltage data.
    - row_idx (int): The row index of the voltage data to read.
    - start_time (float): The start of the time window (in ms).
    - end_time (float): The end of the time window (in ms).
    - time_array (array-like): Array of time points corresponding to the voltage data.
    
    Returns:
    - voltage_window (array-like): The voltage values within the specified time window.
    """
    with h5py.File(result_path+'ndam_v.h5', 'r') as f:
        voltage_window = f['report']['All']['data'][int(start_time/0.025):int(end_time/0.025),row_idx]
    return voltage_window

def read_ATP_window(result_path, row_idx, start_time, end_time):
    """
    Reads the voltage data for a specific row (row_idx) within a given time window (start_time, end_time).
    
    Parameters:
    - result_path (str): Path to the HDF5 file containing voltage data.
    - row_idx (int): The row index of the voltage data to read.
    - start_time (float): The start of the time window (in ms).
    - end_time (float): The end of the time window (in ms).
    - time_array (array-like): Array of time points corresponding to the voltage data.
    
    Returns:
    - voltage_window (array-like): The voltage values within the specified time window.
    """
    with h5py.File(result_path+'ndam_atpi.h5', 'r') as f:
        atp_window = f['report']['All']['data'][int(start_time):int(end_time),row_idx]
    return atp_window

def Find_mol_per_AP(row_idx):
    """
    Parallelizes the process of finding action potential amplitudes across neurons.
    
    Parameters:
    - row_idx: The index of the neuron in the mapping.
    
    Returns:
    - molecules_consumed (list or np.ndarray): An array of molecules consumed during the action potential.
    """
    read_result = h5py.File(result_path + 'ndam_atpi.h5')
    
    mapping = read_result['report']['All']['mapping']['node_ids'][:]
    idx = np.where(mapping == row_idx)[0]

    voltage_row = read_voltage_window(result_path, idx, 0, 3000)
    ATP_row = read_ATP_window(result_path, idx, 0, 3000)

    no_ap_window = find_NO_AP_window(voltage_row, threshold, window_start, window_end, time_step, step_size)
    
    if no_ap_window is not None:
        start_idx, end_idx = no_ap_window
        
        # Define new window around the detected action potential for ATP analysis
        buffer_before = 0 # ms
        buffer_after = 0 # ms
        t1 = max(0, int(start_idx - buffer_before))
        t2 = min(len(voltage_row), int(end_idx + buffer_after))
        
        t1met = int(t1 * time_step)
        t2met = int(t2 * time_step)
    
        # Analyze ATP changes in the window around the action potential
        atp_window = ATP_row[t1met:t2met]  # Extract the ATP data in this window
        min_atp_idx = np.argmin(atp_window)  # Find minimum ATP value in this range
        
        # Compute the number of molecules consumed during the action potential
        molecules_consumed = compute_molecules_per_AP(atp_window[:min_atp_idx + 1], time_step)

        # Normalize output if it's a list or array
        if isinstance(molecules_consumed, list) or isinstance(molecules_consumed, np.ndarray):
            return np.array(molecules_consumed)  # Ensure it's a numpy array for consistency
        else:
            return np.array([molecules_consumed])  # Return as an array of one element if scalar

    return np.nan  # or some fixed value if no action potential is found
    

def parallel_find_AP(nodes_indices):
    
    pool = Pool(28)
    resting_potentials = pool.map(Find_mol_per_AP, nodes_indices)
    
    pool.close()
    pool.join()
    
    
    return np.array(resting_potentials)

threshold = -30  # Action potential voltage threshold in mV
time_step = 0.025  # Time step in ms
step_size = 100  # Step size for backward search in ms

# Define the initial window to search for action potentials
window_start = 2700  # ms
window_end = 2800  # ms

# DATA can be downloaded from 10.5281/zenodo.14187063
nodes_ids = np.load('Data/spiking_nodes_met.npy')

# RESULTS can be downloaded from 10.5281/zenodo.14187063
result_path ='/gpfs/bbp.cscs.ch/data/project/proj137/farinaNGV/Paper_results/my_simulation/reporting_metabolism_young/'

# Creating Empty DataFrame and Storing it in variable df
df = pd.DataFrame()
df['ids'] = nodes_ids
print('Start')
tic = time.perf_counter()
molecules = parallel_find_AP(nodes_ids)
toc = time.perf_counter()
print(f"Computed in {toc - tic:0.4f} seconds")
print(molecules)

#np.save('Data/Molecules_ATPi_resting_state_met.npy', molecules)

df['ATPmolXresting'] = molecules

df.to_csv('Data/Molecules_ATPi_resting_state_met.csv', index=False)