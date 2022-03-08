How to reproduce our results : 

  ------------------- TRAINING THE MODEL ------------------------------------
  1) Follow the reproducibility intructions of the main branch 
  2) run the model training with the command :  python main_SBMs_node_classification.py --config 'configs/SBMs_GraphTransformer_LapPE_CLUSTER_500k_sparse_graph_BN.json'
      (train the best model for the Clustering problem according to the paper : Position encoding with Laplacian eigenvectors, Batch Normalization and Sparse Graphs.)
  3) Reduce batch size to 16 in the config file if the training is too slow or you get an "out of memory" error. Stop the model after approximately 120 epochs.
  4) Find the trained model in .pkl format out/SBMs_sparse_LapPE_BN/checkpoints.
  
  ------------------- GENERATE NEW SBM GRAPHS --------------------------------
  5) Set the parameters you want in configs/data_config.json
  6) run the data generation with the command : python data/SBMs/LOAD_CLUSTER_DATA.py (run the LOAD_CLUSTER_DATA.py file)
  7) find the generated data in data/SBMs/SBM_CLUSTER.pkl (ignore the _train,_test,_val as they are only intermediary files)
  
  ------------------- TEST THE TRAINED MODEL ON THE GENERATED DATA -------------
  8) go to main_eval.py in line 143 and choose the model .pkl file from out/SBMs_sparse_LapPE_BN/checkpoints that you want to evaluate
  9) run model_eval.py 
  
  ------------------- TEST WITH WITH PARAMETER VARIATION (GRID) ------------------
  10) set the grids for parameter variation in main.py and run main.py
  11) find the results in json format (the parameters and the associated accuracy) in out/SBMs_sparse_LapPE_BN/results/comparison.json
  12) use the gnn_result_plots.ipynb jupyter notebook to visualize and calculate the SNR
  
  -------------------- RETRAIN THE MODEL ON THE PARAMETER CONFIGURATION SBMS IT PERFORMED POORLY AGAINST -----------------------------
  13) modify the parameters in configs/data_config.json and generate new training data 
  14) go to main_eval.py in line 143 and choose the model .pkl file from out/SBMs_sparse_LapPE_BN/checkpoints that you want to retrain
  15) follow the "TEST WITH PARAMETER VARIATION (GRID) procedure to test the new model
  
  -------------------- TRAINING A NEW MODEL WITH FEWER LAYERS AND WITH DROPOUT ----------------------------------------------------
  16) modify the SBMs_GraphTransformer_LapPE_CLUSTER_500k_sparse_graph_BN.json file in configs to add dropout in line 33 (it is set to 0)
  17) modify the same file to reduce the number of layers (the parameter L) from 10 to 3 in line 26.
  18) launch the training following the "TRAINING THE MODEL" procedure above
  
  Remarks : 
    - The code for the baseline spectral clustering and comparison with our model is in the root directory : baselineComp.ipynb
    - The code tweak we mentioned in the presentation to relieve the bottleneck in data loading is in data/SBMs.py line 47 (you can compare with data/SBMs_old.py)
    - A json file with all our test results (374 tests) is located in out/SBMs_sparse_LapPE_BN/results : comparison_v0.json
