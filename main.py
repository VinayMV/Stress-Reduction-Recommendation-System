#!/usr/bin/env python37
# -*- coding: utf-8 -*-

import os
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join
import csv
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from utils import collate_fn
from model import GraphRec
from dataloader import GRDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', default='datasets/CalmDown/', help='dataset directory path: datasets/CalmDown')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.01, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=100, help='the number of steps after which the learning rate decay')
parser.add_argument('--test', action='store_true', help='test')
args = parser.parse_args()
print(args)
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
flag=0

def main():
    print('Loading data...')
    with open(args.dataset_path + 'dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(args.dataset_path + 'list.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)
    
    train_data = GRDataset(train_set, u_items_list, i_users_list) 
    valid_data = GRDataset(valid_set, u_items_list, i_users_list) 
    test_data = GRDataset(test_set, u_items_list, i_users_list) 
    train_loader = DataLoader(train_data, batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    model = GraphRec(user_count+1, item_count+1, rate_count+1, args.embed_dim).to(device)
    epoch_list=[]
    mae_list=[]
    rmse_list=[]
    acc_list=[]
    best_acc_list=[]
    best_mae_list=[]
    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load('best_checkpoint.pth.tar')
        model.load_state_dict(ckpt['state_dict'])
        mae, rmse, acc, r_ij, uids, iids  = validate(test_loader, model)
        list_reccomendation = ['buffer','Listen to music' ,'Lie down / take a nap',  'Talk about the issue with someone', 'Spent quality time with family', 'Watch videos /movies', 'Meditation / exercise', 'Take short trips / walks', 'Engage with pets', 'Follow a healthy diet (cut down on processed food and added sugar)', 'Reading books of your interest', 'Avoid putting off tasks until last minute', 'Reduce caffeine intake', 'Avail medical help / counselling']  
        list_iid = []
        list_rating = []
        for i in range(uids.size(dim=0)):
            if uids[i] == uids[uids.size(dim=0)-1]  and (iids[i] !=14 and iids[i] != 15) :
                list_iid.append(int(iids[i]))
                list_rating.append(float(r_ij[i]))
        dictionary = dict(zip(list_iid,list_rating))
        s_dictionary = sorted(dictionary.items(), key = lambda x:x[1], reverse = True)
        count_recc = 0
        for i in dict(s_dictionary).keys():
             if count_recc >2:
                 break
             else:
                 print("Recommendation ",count_recc+1,": ",list_reccomendation[i])
                 count_recc = count_recc + 1
        print("Test: \n MAE: {:.4f},\n RMSE: {:.4f},\n ACCURACY: {:.4f}".format(mae, rmse, acc))
        with open('test_accuracy.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['MAE', 'RMSE', 'ACCURACY'])
            row = [mae, rmse, acc]
            writer.writerow(row)
        return

    optimizer = optim.RMSprop(model.parameters(), args.lr)
    criterion = nn.MSELoss()
    for epoch in tqdm(range(args.epoch)):
        trainForEpoch(train_loader, model, optimizer, epoch, args.epoch, criterion, log_aggr = 100)

        mae, rmse,acc = validate(valid_loader, model)
        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        torch.save(ckpt_dict, 'latest_checkpoint.pth.tar')

        if epoch == 0:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            torch.save(ckpt_dict, 'best_checkpoint.pth.tar')
            
        if epoch == 0:
            best_acc = acc
        elif acc > best_acc:
            best_acc = acc
            
        epoch_list.append(epoch)
        mae_list.append(mae)
        rmse_list.append(rmse)
        acc_list.append(acc)
        best_acc_list.append(best_acc)
        best_mae_list.append(best_mae)
        print(' Epoch {} \n Validation: \n MAE: {:.4f}, \n RMSE: {:.4f},\n ACCURACY: {:.4f},\n BEST ACCURACY: {:.4f},\n BEST MAE: {:.4f}'.format(epoch, mae, rmse,acc, best_acc,best_mae))
    with open('train_accuracy.csv','w') as f:
        writer = csv.writer(f)
        writer.writerow(['EPOCH', 'MAE', 'RMSE', 'ACCURACY', 'BEST ACCURACY', 'BEST MAE'])
        for i in range(len(epoch_list)):
            row = [epoch_list[i], mae_list[i], rmse_list[i], acc_list[i], best_acc_list[i], best_mae_list[i]]
            writer.writerow(row)
     

def trainForEpoch(train_loader, model, optimizer, epoch, num_epochs, criterion, log_aggr=1):
    model.train()

    sum_epoch_loss = 0

    start = time.time()
    for i, (uids, iids, labels, u_items, i_users) in tqdm(enumerate(train_loader), total=len(train_loader)):
        uids = uids.to(device)
        iids = iids.to(device)
        labels = labels.to(device)
        u_items = u_items.to(device)
        i_users = i_users.to(device)
        
        optimizer.zero_grad()
        outputs = model(uids, iids, u_items, i_users)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step() 
        
        scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)
        loss_val = loss.item()
        scheduler.step(epoch = epoch)
        sum_epoch_loss += loss_val

        iter_num = epoch * len(train_loader) + i + 1

        if i % log_aggr == 0:
            print('[TRAIN] epoch %d/%d batch loss: %.4f (avg %.4f) (%.2f im/s)'
                % (epoch + 1, num_epochs, loss_val, sum_epoch_loss / (i + 1),
                  len(uids) / (time.time() - start)))

        start = time.time()
def pixelAccuracy(pred_val, true_val):
    #print(pred_val)
    pixel_labeled = np.sum(true_val > -1)
    if pred_val > 5:
        pred_val = 5
    #pixel_correct = np.sum(pred_val >= true_val-0.8 and pred_val <= true_val + 0.8)*(true_val > 0)
    pixel_correct = np.sum(((np.round(pred_val) == true_val) + (np.round(pred_val) == (true_val-1)) + (np.round(pred_val) == (true_val+1))) * (true_val > 0))
    pixel_accuracy = pixel_correct / pixel_labeled
    return pixel_accuracy, pixel_correct, pixel_labeled

def validate(valid_loader, model):
    model.eval()
    errors = []
    with torch.no_grad():
        for uids, iids, labels, u_items, i_users in tqdm(valid_loader):
            uids = uids.to(device)
            iids = iids.to(device)
            labels = labels.to(device)
            u_items = u_items.to(device)
            i_users = i_users.to(device)
            preds = model(uids, iids, u_items, i_users)
            error = torch.abs(preds.squeeze(1) - labels)
            errors.extend(error.data.cpu().numpy().tolist())
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean(np.power(errors, 2)))
    pred_val = np.asarray(torch.squeeze(preds))
    true_val = np.asarray(torch.squeeze(labels))
    accuracy = np.empty(true_val.shape[0])
    correct = np.empty(true_val.shape[0])
    labeled = np.empty(true_val.shape[0])
    for i in range(true_val.shape[0]):
         accuracy[i], correct[i], labeled[i] = pixelAccuracy(pred_val[i],true_val[i])
    acc = 100.0 * np.sum(correct) / np.sum(labeled)

    if args.test :
        return mae, rmse, acc, preds, uids, iids
    else:
        return mae, rmse, acc



if __name__ == '__main__':
    main()
