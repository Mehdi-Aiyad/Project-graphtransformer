import os
import json
import itertools
import time
{
    "size_min" : 5,
    "size_max" : 35,
    "p" : 0.55,
    "q" : 0.25,
}
# C

t = time.time()

grid_min = [5,15,25]
grid_max = [30,40,50,60]
grid_p = [0.55, 0.65, 0.75, 0.85]
grid_q = [0.10, 0.20, 0.30, 0.40]

for i, params in enumerate(itertools.product(grid_min, grid_max, grid_p, grid_q)):

    config = {
    "nb_clusters" : 6,
    "size_min" : params[0],
    "size_max" : params[1],
    "p" : params[2],
    "q" : params[3],
    "nb_graphs_train" : 1,
    "nb_graphs_test" : 1000,
    "nb_graphs_val" : 1
    }
    
    with open('configs/data_config.json' , "w") as f:
        json.dump(config,f)
        
    os.system("python3 data/SBMs/LOAD_CLUSTER_DATA.py")
    os.system("python3 main_eval.py")


    
print("Total time")
print(time.time()-t)