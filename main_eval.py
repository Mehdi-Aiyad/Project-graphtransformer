#!/usr/bin/env python
# coding: utf-8


import dgl
import numpy as np
import os
import socket
import time
import random
import glob
import argparse, json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from tqdm import tqdm
from nets.SBMs_node_classification.graph_transformer_net import GraphTransformerNet
from train.train_SBMs_node_classification import evaluate_network
import json
from data.data import LoadData 
from nets.SBMs_node_classification.load_net import gnn_model 
from tensorboardX import SummaryWriter



config_file = 'configs/SBMs_GraphTransformer_LapPE_CLUSTER_500k_sparse_graph_BN.json'
with open(config_file, 'r') as f:
    config = json.load(f)
    
data_config_file = 'configs/data_config.json'
with open(data_config_file, 'r') as f:
    data_config = json.load(f)



net_params = config['net_params']
net_params['in_dim']= 7
net_params['n_classes'] = 6



def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:',torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


device = gpu_setup(config['gpu']['use'], config['gpu']['id'])
net_params['device'] = device


def view_model_param(MODEL_NAME, net_params):
    model = gnn_model(MODEL_NAME, net_params)
    total_param = 0
    print("MODEL DETAILS:\n")
    #print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', MODEL_NAME, total_param)
    return total_param



def eval_pipeline(MODEL_NAME, dataset, params, net_params, dirs):
    
    start0 = time.time()
    per_epoch_time = []
    
    DATASET_NAME = dataset.name
    
    if net_params['lap_pos_enc']:
        st = time.time()
        print("[!] Adding Laplacian positional encoding.")
        dataset._add_laplacian_positional_encodings(net_params['pos_enc_dim'])
        print('Time LapPE:',time.time()-st)
        
    if net_params['wl_pos_enc']:
        st = time.time()
        print("[!] Adding WL positional encoding.")
        dataset._add_wl_positional_encodings()
        print('Time WL PE:',time.time()-st)
    
    if net_params['full_graph']:
        st = time.time()
        print("[!] Converting the given graphs to full graphs..")
        dataset._make_full_graph()
        print('Time taken to convert to full graphs:',time.time()-st)
        
    trainset, valset, testset = dataset.train, dataset.val, dataset.test
        
    root_log_dir, root_ckpt_dir, write_file_name, write_config_file = dirs
    device = net_params['device']
    
    # Write network and optimization hyper-parameters in folder config/
    
    with open(write_config_file + '.txt', 'w') as f:
        f.write("""Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n\nTotal Parameters: {}\n\n"""                .format(DATASET_NAME, MODEL_NAME, params, net_params, net_params['total_param']))
        
    log_dir = os.path.join(root_log_dir, "RUN_" + str(0))
    writer = SummaryWriter(log_dir=log_dir)

    # setting seeds
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])
    if device.type == 'cuda':
        torch.cuda.manual_seed(params['seed'])
    
    print("Test Graphs: ", len(testset))
    print("Number of Classes: ", net_params['n_classes'])

    model = gnn_model(MODEL_NAME, net_params)
    model = model.to(device)

    model.load_state_dict(torch.load('out/ModelsParams/epoch_113.pkl'))
    model.eval()

    
    test_loader = DataLoader(testset, batch_size=params['batch_size'], shuffle=False, collate_fn=dataset.collate)   
    _, test_acc = evaluate_network(model, device, test_loader, epoch=1)
    print("Test Accuracy: {:.4f}".format(test_acc))
    print("TOTAL TIME TAKEN: {:.4f}s".format(time.time()-start0))

    writer.close()

    """
        Write the results in out_dir/results folder
    """
    js_file = open(write_file_name+'.json', "r")
    res = json.load(js_file)
    
    res.append({"Data_Parameters" : data_config, "Accuracy" : test_acc }) 
    js_file.close()
    
    
    with open(write_file_name + '.json', 'w') as f:
        json.dump(res,f)



class DotDict(dict):
    def __init__(self, **kwds):
        self.update(kwds)
        self.__dict__ = self

if not os.path.isfile(write_file_name+'.json'):
    with open(write_file_name+'.json', 'w+') as f:
        json.dump([], f)
        
params = config['params']
net_params["batch_size"] = params['batch_size']
DATASET_NAME = config['dataset']
dataset = LoadData(DATASET_NAME)



out_dir = config['out_dir']
MODEL_NAME =  config['model']
root_log_dir = out_dir + 'logs/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
root_ckpt_dir = out_dir + 'checkpoints/' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
write_file_name = out_dir + 'results/comparison'
write_config_file = out_dir + 'configs/config_' + MODEL_NAME + "_" + DATASET_NAME + "_GPU" + str(config['gpu']['id']) + "_" + time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y')
if not os.path.exists(out_dir + 'results'):
    os.makedirs(out_dir + 'results')
if not os.path.exists(out_dir + 'results'):
    os.makedirs(out_dir + 'logs')
if not os.path.exists(out_dir + 'checkpoints'):
    os.makedirs(out_dir + 'checkpoints')
if not os.path.exists(out_dir + 'configs'):
    os.makedirs(out_dir + 'configs')
dirs = root_log_dir, root_ckpt_dir, write_file_name, write_config_file


net_params['total_param'] = view_model_param(MODEL_NAME, net_params)


eval_pipeline(MODEL_NAME, dataset, params, net_params, dirs)





