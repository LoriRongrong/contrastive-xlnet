import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, random_split, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import roc_auc_score
import argparse
import csv
from transformers import XLNetForSequenceClassification

usemoco=True
if usemoco:
    model = XLNetForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="xlnet-base-cased",
            num_labels=2) # num_labels used to be 2 but now I have to do something to match
    checkpoint = torch.load('./moco_model/moco.tar')
    print(checkpoint.keys())
    print(checkpoint['arch'])
    state_dict = checkpoint['state_dict']
    for key in list(state_dict.keys()):
        
        if 'encoder_q' in key:
            new_key = key[10:]
            state_dict[new_key] = state_dict[key]   
        del state_dict[key]
    for key in list(state_dict.keys()):
        if key == 'logits_proj.2.weight':
            # print(state_dict['logits_proj.0.weight'].shape)
            # print(state_dict['logits_proj.2.weight'].shape)
            new_key = 'logits_proj.weight'
            state_dict[new_key] = state_dict[key]

            del state_dict[key]
        if key == 'logits_proj.2.bias':
            new_key = 'logits_proj.bias'
            state_dict[new_key] = state_dict[key]
            
            del state_dict[key]
        if key == 'logits_proj.0.weight' or key == 'logits_proj.0.bias':
            del state_dict[key]
    
    # state_dict['logits_proj.weight'] = state_dict['logits_proj.weight'][:1000, :]
    # state_dict['logits_proj.bias'] = state_dict['logits_proj.bias'][:1000]
    model.load_state_dict(state_dict)						
    fc_features = model.logits_proj.in_features
    model.logits_proj = nn.Linear(fc_features, 2)
    model.save_pretrained('./moco_model/') # add a new line to save additional information
    torch.save(model.state_dict(), "./moco_model/moco.p")
    print('finished')
