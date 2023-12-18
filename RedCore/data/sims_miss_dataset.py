import os
import json

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from torchmetrics import Accuracy, ConfusionMatrix

import numpy as np
import random
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from data.temperature_scaling import ModelWithTemperature

from data.base_dataset import BaseDataset





labels_en = ['negative', 'weakly negative', 'neural', 'weakly positive', 'positive']



def temperature_scale(temperature, logits):
    # 进行calibration
    temperature = temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
    return logits / temperature

def train_temperature(temperature, val_logits, val_labels):
    nll_criterion = nn.CrossEntropyLoss().to(cuda2)

    logits = val_logits.to(cuda2)
    labels = val_labels.to(cuda2)

    # calibration之前的效果
    before_temperature_nll = nll_criterion(logits, labels).item()
    print('Before calibration - NLL: %.3f' % (before_temperature_nll))

    # 训练temperature超参；full-batch
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def eval():
        optimizer.zero_grad()
        loss = nll_criterion(temperature_scale(temperature, logits), labels)
        loss.backward()
        return loss
    optimizer.step(eval)

    # calibration之后的效果
    after_temperature_nll = nll_criterion(temperature_scale(temperature, logits), labels).item()
    print('Optimal temperature: %.3f' % temperature.item())
    print('After calibration - NLL: %.3f' % (after_temperature_nll))



class SimsMissDataset(BaseDataset, Dataset):
    @staticmethod
    def modify_commandline_options(parser, isTrain=None):
        parser.add_argument('--cvNo', type=int, help='which cross validation set')
        parser.add_argument('--A_type', type=str, help='which audio feat to use')
        parser.add_argument('--V_type', type=str, help='which visual feat to use')
        parser.add_argument('--L_type', type=str, help='which lexical feat to use')
        parser.add_argument('--output_dim', type=int, help='how many label types in this dataset')
        parser.add_argument('--norm_method', type=str, choices=['utt', 'trn'], help='how to normalize input comparE feature')
        parser.add_argument('--corpus_name', type=str, default='SIMS', help='which dataset to use')
        return parser

    def __init__(self, opt, set_name):
        
        self.stage = set_name
        self.set_name = set_name
        #self.dataset_path = dataset_path + self.stage + '.json'
        #self.dataset_path = '/data/sunjun/datasets/' + self.stage + '1121.json'

        self.dataset_path = '/data3/sunjun/work/sunjun/work/data/SIMS/SIMS_20230524/' + self.stage + '.json'
        self.filename_label_list = []

        self.length_label = 0
        with open(self.dataset_path) as f:
            for example in json.load(f):
                self.length_label += 1
                tv = example['video_file']
                a = example['video_file'].split('_')
                a[-1] = a[-1].zfill(4)
                aa = ''
                for i, item in enumerate(a):
                    aa = aa + item + '_'
                aa = aa[0:-1]
                
                self.filename_label_list.append((tv, aa, example['txt_label'], example['audio_label'], example['visual_label'], example['video_label']))
                #self.filename_label_list.append((example['file_name'], example['video_label']))

        if self.stage != 'train':           # val && tst
            self.missing_index = torch.tensor([
                [1,0,0], # AZZ
                [0,1,0], # ZVZ
                [0,0,1], # ZZL
                [1,1,0], # AVZ
                [1,0,1], # AZL
                [0,1,1], # ZVL
            ] * self.length_label).long()
            self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl'] * self.length_label
        else:                           # trn
            self.missing_index = [
                [1,0,0], # AZZ
                [0,1,0], # ZVZ
                [0,0,1], # ZZL
                [1,1,0], # AVZ
                [1,0,1], # AZL
                [0,1,1], # ZVL
            ]
            self.miss_type = ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl']


        # set collate function
        self.manual_collate_fn = True


    def __len__(self):
        return len(self.filename_label_list)

    def __getitem__(self, idx):
        if self.stage != 'train':
            feat_idx = idx // 6         # totally 6 missing types
            missing_index = self.missing_index[idx]
            miss_type = self.miss_type[idx]
        else:
            feat_idx = idx
            _miss_i = random.choice(list(range(len(self.missing_index))))
            missing_index = torch.tensor(self.missing_index[_miss_i]).long()
            miss_type = self.miss_type[_miss_i]

        current_filename, current_filename_a, label_t, label_a, label_v, label_m = self.filename_label_list[feat_idx]
        #text_vector = np.load('/data/sunjun/datasets/SIMS_20230524/SIMS_text/layeroutput/' + self.stage + '/' + current_filename + '.npy')
        text_vector = np.load('/data3/sunjun/work/sunjun/work/data/SIMS/SIMS_20230524/SIMS_text/layeroutput/' + self.stage + '/' + current_filename + '.npy')
        text_vector = torch.from_numpy(text_vector)

        #video_vector = np.load('/data/sunjun/datasets/SIMS/video/feat/' + self.stage + '/' + current_filename + '.npy')
        video_vector = np.load('/data3/sunjun/work/sunjun/work/data/SIMS/SIMS_20230524/video/feat/' + self.stage + '/' + current_filename + '.npy')
        video_vector = torch.from_numpy(video_vector)

        #audio_vector = np.load('/data/sunjun/datasets/SIMS/audio/frame_vector/' + self.stage + '/' + current_filename_a + '.npy')
        audio_vector = np.load('/data3/sunjun/work/sunjun/work/data/SIMS/SIMS_20230524/audio/frame_vector/' + self.stage + '/' + current_filename_a + '.npy')
        audio_vector = torch.from_numpy(audio_vector)


        #return audio_calibrated, audio_vector, text_calibrated, text_vector, labels_ch.index(current_label)
        # return  text_vector, audio_vector, video_vector, labels_en.index(label_t), labels_en.index(label_a), labels_en.index(label_v), labels_en.index(label_m)

        #print('TTTTTTTTTTTTTTTT:', missing_index[0])
        return {
            'A_feat': audio_vector,
            'V_feat': video_vector,
            'L_feat': text_vector,
            'label_a': labels_en.index(label_a),
            'label_l': labels_en.index(label_t),
            'label_v': labels_en.index(label_v),
            'label': labels_en.index(label_m),
            'current_filename': current_filename,
            'current_filename_a': current_filename_a,
            'missing_index': missing_index,
            'miss_type': miss_type
        } if self.set_name == 'train' else{
            'A_feat': audio_vector, #* missing_index[0],
            'V_feat': video_vector, #* missing_index[1],
            'L_feat': text_vector, #* missing_index[2],
            'label_a': labels_en.index(label_a),
            'label_l': labels_en.index(label_t),
            'label_v': labels_en.index(label_v),
            'label': labels_en.index(label_m),
            'current_filename': current_filename,
            'current_filename_a': current_filename_a,
            'missing_index': missing_index,
            'miss_type': miss_type
        }

    def collate_fn(self, batch):
        A = [sample['A_feat'] for sample in batch]
        V = [sample['V_feat'] for sample in batch]
        L = [sample['L_feat'] for sample in batch]
        lengths = torch.tensor([len(sample) for sample in A]).long()
        A = pad_sequence(A, batch_first=True, padding_value=0)
        V = pad_sequence(V, batch_first=True, padding_value=0)
        L = pad_sequence(L, batch_first=True, padding_value=0)
        label = torch.tensor([sample['label'] for sample in batch])
        label_l = torch.tensor([sample['label_l'] for sample in batch])
        label_a = torch.tensor([sample['label_a'] for sample in batch])
        label_v = torch.tensor([sample['label_v'] for sample in batch])
        #int2name = [sample['int2name'] for sample in batch]
        missing_index = torch.cat([sample['missing_index'].unsqueeze(0) for sample in batch], axis=0)
        miss_type = [sample['miss_type'] for sample in batch]
        return {
            'A_feat': A, 
            'V_feat': V,
            'L_feat': L,
            'label': label,
            'label_a': label_a,
            'label_l': label_l,
            'label_v': label_v,
            'lengths': lengths,
            'missing_index': missing_index,
            'miss_type': miss_type
        }
        
   