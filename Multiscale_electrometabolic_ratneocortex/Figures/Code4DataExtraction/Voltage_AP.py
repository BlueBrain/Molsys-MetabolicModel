import h5py    
import numpy as np  
from multiprocessing import Pool
# Importing Pandas to create DataFrame
import pandas as pd
import time

def read_result(result_path):
    read_result = h5py.File(result_path)
    result = read_result['report']['All']['data']

    return(result)

def mapping_extract(result_path):
    read_result = h5py.File(result_path)
    result = read_result['report']['All']['data']
        
    extract_mapping = read_result['report']['All']['mapping']['node_ids'][:]
    
    t0 = read_result['report']['All']['mapping']['time'][0]
    T =read_result['report']['All']['mapping']['time'][1]
    dt = read_result['report']['All']['mapping']['time'][2]

    
    N= (T-t0)/dt
    time_line = np.linspace(t0,T, int(N))
    N2 = (T-t0)/0.025
    voltage_time_line = np.linspace(t0,T,int(N2))
    return(extract_mapping, time_line, voltage_time_line)


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
    with h5py.File(result_path, 'r') as f:
        voltage_window = f['report']['All']['data'][int(start_time/0.025):int(end_time/0.025),row_idx]
    return voltage_window

def read_voltage_matrix_window(result_path, start_time, end_time):
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
    with h5py.File(result_path, 'r') as f:
        voltage_window = f['report']['All']['data'][int(start_time/0.025):int(end_time/0.025), :]
    return voltage_window


def find_action_potential_amplitude(args):
    voltage_trace = args
    threshold=-30
    window_size=50
    
    """
    Extracts the action potential amplitude from a voltage trace.
    
    Parameters:
    - voltage_trace (array-like): 1D array of voltage values (in mV).
    - time_trace (array-like): 1D array of time values corresponding to the voltage trace.
    - threshold (float): Voltage threshold to detect an action potential (default -30 mV).
    - window_size (int): Number of time points before and after the peak to include in the analysis (default 5).
    
    Returns:
    - float: The action potential amplitude (peak voltage - resting potential).
    - tuple: Indices of the start and end of the window (time window) around the peak.
    """
    
    # Step 1: Check if there is any action potential above the threshold
    if np.all(voltage_trace <= threshold):
        return np.nan  # No action potential, return NaN
    
    else:
        # Step 2: Identify the peak of the action potential
        peak_idx = np.argmax(voltage_trace)  # Index of the peak voltage
        peak_voltage = voltage_trace[peak_idx]
        
        # Step 3: Define a time window around the peak
        start_idx = max(0, peak_idx - window_size)  # Ensure we don't go out of bounds
        end_idx = min(len(voltage_trace), peak_idx + window_size)
        # Step 4: Compute resting potential as the minimum voltage in the window
        voltage_window = voltage_trace[start_idx:end_idx]
        resting_potential = np.min(voltage_window)
        # Step 5: Compute the amplitude as the difference between peak voltage and resting potential
        amplitude = peak_voltage - resting_potential
        return amplitude#, (start_idx, end_idx)

def parallel_find_AP(voltage_matrix):
    
    # Prepare the arguments for parallel execution
    args = [(voltage_matrix[:, i]) for i in range(np.shape(voltage_matrix)[1])]
    
    pool = Pool(28)
    resting_potentials = pool.map(find_action_potential_amplitude, args)
    
    pool.close()
    pool.join()
    
    
    return np.array(resting_potentials)

# result_path ='/gpfs/bbp.cscs.ch/data/scratch/proj137/farina/Met_Aug/v03/sim14/'

# sim_results_path = result_path + 'reporting_3000_node_sets_v03_single_node_removed/'
# voltage_met = read_result(sim_results_path + 'ndam_v.h5')
# mapping_met,time_line_met, volt_time_met = mapping_extract(sim_results_path+ 'ndam_nai.h5')

# # Creating Empty DataFrame and Storing it in variable df
# df = pd.DataFrame()

# df['ids'] = mapping_met
# number_cells= len(df['ids'])
# print('Start')

# indicate the path where voltage is stored:  can be downloaded from 10.5281/zenodo.14187063
result_path ='/gpfs/bbp.cscs.ch/data/project/proj137/farinaNGV/Paper_results/my_simulation/reporting_neurodamus/ndam_v.h5'


# Creating Empty DataFrame and Storing it in variable df
df = pd.DataFrame()

read_result = h5py.File(result_path)
df['ids'] = read_result['report']['All']['mapping']['node_ids'][:]
number_cells= len(df['ids'])
print('Start')

matrix_read = read_voltage_matrix_window(result_path, 0, 3000)
tic = time.perf_counter()

# amplitude = find_action_potential_amplitude((voltage_met[:,110],volt_time_met))
# print(f"Action Potential Amplitude: {amplitude} mV")

amplitudes_AP = parallel_find_AP(matrix_read)
toc = time.perf_counter()
print(f"Computed in {toc - tic:0.4f} seconds")
df['amplitudes_AP'] = amplitudes_AP

df.to_csv('Data/voltage_amplitude_AP_ndam.csv', index=False)