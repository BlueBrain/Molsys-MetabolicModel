
import h5py    
import numpy as np  
from multiprocessing import Pool
# Importing Pandas to create DataFrame
import pandas as pd
import time


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



def find_resting_potential_for_row(args):
    """
    Iteratively checks time windows for action potentials and calculates the resting potential if no action potential is found.
    
    Parameters:
    - result_path (str): Path to the HDF5 file containing voltage data.
    - row_idx (int): Row index of the voltage data.
    - time_array (array-like): Time values (in ms) corresponding to the voltage data.
    - t0, t1 (float): Start and end time for the initial window (in ms).
    - max_time (float): Maximum time to scan (in ms).
    - threshold (float): Voltage threshold to detect action potentials (default -30 mV).
    - shift (float): Time shift (default 100 ms).
    
    Returns:
    - resting_potential (float): Calculated resting potential or NaN if no valid window is found.
    """
    voltage_row, row_idx, t0, t1= args

    threshold=-30
    shift=100
    max_time = 3000

    while t1 <= max_time:
        # Check if there is any action potential in this window
        if np.any(voltage_row> threshold):
            t0 += shift
            t1 += shift
            voltage_row = read_voltage_window(result_path, row_idx, t0, t1)
        else:
             # Compute the mean voltage for this window (resting potential)
             resting_potential = np.mean(voltage_row)
             return resting_potential
    return np.nan
    

def parallel_find_resting_potential(matrix_voltage, t0,t1):
    """
    Computes the resting membrane potentials for each row of a voltage matrix using multiprocessing.
    
    Parameters:
    - result_path (str): Path to the HDF5 file containing voltage data.
    - time_array (array-like): Time values (in ms) corresponding to the voltage data.
    - n_rows (int): Number of rows to process (each row corresponds to a cell).
    - t0, t1 (float): Start and end time for the initial window (in ms).
    - n_jobs (int): Number of CPU cores to use (-1 uses all available cores).
    
    Returns:
    - np.array: Resting potentials for each row.
    """

    # Prepare arguments for parallel processing
    args = [(matrix_voltage[:,i], i, t0, t1) for i in range(np.shape(matrix_voltage)[1])]
    
    pool = Pool(28)
    resting_potentials = pool.map(find_resting_potential_for_row, args)

    pool.close()    # No more tasks will be submitted to the pool
    pool.join()     # Wait for the worker processes to complete

    return np.array(resting_potentials)

result_path ='/gpfs/bbp.cscs.ch/data/scratch/proj137/farina/Met_Aug/v03/sim14/reporting_3000_node_sets_v03_single_node_removed_aged_metabolism/ndam_v.h5'

# Creating Empty DataFrame and Storing it in variable df
df = pd.DataFrame()

read_result = h5py.File(result_path)
df['ids'] = read_result['report']['All']['mapping']['node_ids'][:]
number_cells= len(df['ids'])
print('Start')


matrix_read = read_voltage_matrix_window(result_path, 500, 600)

print(np.shape(matrix_read))

tic = time.perf_counter()
#idx = 996
#resting_potentials = find_resting_potential_for_row([matrix_read[:,idx ], idx , 500, 600])
resting_potentials = parallel_find_resting_potential(matrix_read, 500, 600)
#print(resting_potentials)
toc = time.perf_counter()
print(f"Computed in {toc - tic:0.4f} seconds")

df['resting_potential'] = resting_potentials

df.to_csv('/gpfs/bbp.cscs.ch/data/project/proj137/farinaNGV/Notebooks/Paper/Data/voltage_rest_aged.csv', index=False)

print('Saved')