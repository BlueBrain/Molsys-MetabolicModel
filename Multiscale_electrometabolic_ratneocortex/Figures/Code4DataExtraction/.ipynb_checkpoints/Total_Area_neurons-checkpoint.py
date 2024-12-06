import h5py    
import numpy as np  
import bluepysnap
import matplotlib.pyplot as plt
import bluepysnap
from bluepysnap import Simulation as snap_sim
from pathlib import Path
from neuron import h
from bluecellulab import CircuitSimulation
from bluecellulab.graph import build_graph, plot_graph
from bluecellulab import Cell, Simulation
from bluecellulab.circuit.circuit_access import EmodelProperties
from multiprocessing import Pool

def Compute_area(cell_id):
    cell1 = node_population.get(cell_id)

    hoc_name = cell1['model_template'].replace("hoc:", "")
    hoc_file = "/gpfs/bbp.cscs.ch/project/proj137/home/mandge/optimisation/release/v7/"+hoc_name +".hoc"
    morph_file ="/gpfs/bbp.cscs.ch/project/proj83/entities/fixed-ais-L23PC-2020-12-10/ascii/" + str(cell1['morphology']) + '.asc'

    v_init = -70
    holding_current = f['nodes']['All']['0']['dynamics_params']['holding_current'][cell_id]
    threshold_current = f['nodes']['All']['0']['dynamics_params']['threshold_current'][cell_id]
    emodel_properties = EmodelProperties(threshold_current=threshold_current,
                                    holding_current=holding_current)
    cell = Cell(hoc_file, morph_file, template_format="v6", emodel_properties=emodel_properties)
    cell.cell
    total_area = 0
    for sec in h.allsec():
        # iterate over all the segments in the section
        for seg in sec:
             if "myelin" in sec.name():
                 continue
             else:
                 total_area += seg.area()
    return(total_area)

def parallel_find_Area(nodes_indices):
    
    # Prepare the arguments for parallel execution
    #args = [(voltage_matrix[:, i]) for i in range(np.shape(voltage_matrix)[1])]
    
    pool = Pool(28)
    areas = pool.map(Compute_area, nodes_indices)
    
    pool.close()
    pool.join()
    
    
    return np.array(areas)

# read Circuit
circuit = bluepysnap.Circuit('/gpfs/bbp.cscs.ch/data/scratch/proj137/farina/Met_Aug/v03/v03_config/circuit_config.json') 
node_population = circuit.nodes["All"]
pandas_nodes = node_population.get()

import pandas as pd
df =  pd.read_csv('mapping.csv')

f =h5py.File("/gpfs/bbp.cscs.ch/project/proj137/home/mandge/mm/compute_currents_v3/nodes_new.h5",'r')

ids = list(df.ids)
extract_areas = parallel_find_Area(ids[:])

#np.save('ComputeArea/areas.npy', extract_areas)

df['Area'] = extract_areas
df.to_csv('area.csv', index=False)