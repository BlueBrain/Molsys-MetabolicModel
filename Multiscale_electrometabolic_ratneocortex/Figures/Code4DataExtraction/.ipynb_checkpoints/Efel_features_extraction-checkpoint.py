import h5py    
import numpy as np  
# Importing Pandas to create DataFrame
import pandas as pd
import time
import efel
import numpy as np
import multiprocessing


def main():
    """Main"""
    
    # Load data
    time = np.load('Data_Efel/Time_voltage.npy')  # Time vector
    voltage_selected = np.load('Data_Efel/ndam/Voltage_spiking_ndam.npy')  # Voltage data for each node
    nodes_ids = np.load('Data_Efel/ndam/spiking_nodes_ndam.npy')  # Node IDs associated with each trace
    
    traces = []
    num_traces = voltage_selected.shape[1]  # Number of voltage traces
    
    for i in range(num_traces):
        # Extract time and voltage data for each trace
        trace_time = time
        trace_voltage = voltage_selected[:, i]
        
        # Create the trace dictionary
        trace = {
            'T': trace_time,
            'V': trace_voltage,
            'stim_start': [250],
            'stim_end': [3000]
        }
        
        traces.append(trace)

    # Use multiprocessing to calculate feature values in parallel
    pool = multiprocessing.Pool()
    traces_results = efel.get_feature_values(
        #traces, ['spike_count'], parallel_map=pool.map
        traces, ['AP_amplitude', 'peak_voltage','peak_time','steady_state_voltage_stimend', 'spike_count', 'maximum_voltage'], parallel_map=pool.map
    )
    
    # Initialize an empty list to collect data for DataFrame
    data_for_df = []
    
    # Iterate through each trace result and prepare data for DataFrame
    for trace_number, trace_results in enumerate(traces_results):
        row = {'Node_ID': nodes_ids[trace_number]}
        for feature_name, feature_values in trace_results.items():
            # If feature has multiple values, save as a list, else directly as a scalar
            row[feature_name] = list(feature_values) if len(feature_values) > 1 else feature_values[0]

        data_for_df.append(row)
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data_for_df)
    
    # Print or save the DataFrame to a CSV file
    print(df)
    df.to_csv('Data_Efel/ndam/Features_extraction_ndam.csv', index=False)

if __name__ == '__main__':
    main()