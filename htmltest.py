import os
import time
import argparse
import pickle
import numpy as np
import random
from tqdm import tqdm
from os.path import join
import random
import pandas as pd
from scipy.io import loadmat
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
parser.add_argument('--dataset', default='CalmDown', help='dataset name: CalmDown')
parser.add_argument('--dataset_path', default='datasets/CalmDown/', help='dataset directory path: datasets/CalmDown')
parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=64, help='the dimension of embedding')
args = parser.parse_args()
print(args)
here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
workdir = 'datasets/'
def main(score, level):
    user_count = 0
    item_count = 0
    rate_count = 0

    click_f = np.loadtxt('datasets/CalmDown/ratings_data.txt', dtype = np.int32)
    click_list = []
    u_items_list = []
    u_users_list = []
    u_users_items_list = []
    i_users_list = []
    for s in click_f:
         uid = s[0]
         iid = s[1]
         label = s[2]
         if uid > user_count:
                  user_count = uid
         if iid > item_count:
                  item_count = iid
         if label > rate_count:
                  rate_count = label
         click_list.append([uid, iid, label])

    pos_list = []
    for i in range(len(click_list)):
             pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
    length=len(click_list)
    pos_list = list(set(pos_list))
    random.shuffle(pos_list)
    user_count=user_count+1
    uid=user_count
    iid14=score
    iid15=level
    for i in range(1,14):
             click_list.append([uid, i, 0])
    click_list.append([uid, 14, iid14])
    click_list.append([uid, 15, iid15])
    for i in range(length,length+15):
             pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))
    num_train = int(len(pos_list) * 0.8)
    num_test = int(len(pos_list) * 0.1)
    train_set = pos_list[:num_train]
    valid_set = pos_list[num_train:num_train + num_test]
    test_set = pos_list[num_train + num_test:]
    with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
        pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


    train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
    valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
    test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

    click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
    train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')
    for u in tqdm(range(user_count + 1)):
        hist = train_df[train_df['uid'] == u]
        u_items = hist['iid'].tolist()
        u_ratings = hist['label'].tolist()
        if u_items == []:
            u_items_list.append([(0, 0)])
        else:
            u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

    train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')
    for i in tqdm(range(item_count + 1)):
        hist = train_df[train_df['iid'] == i]
        i_users = hist['uid'].tolist()
        i_ratings = hist['label'].tolist()
        if i_users == []:
            i_users_list.append([(0, 0)])
        else:
            i_users_list.append([(uid, rating) for uid, rating in zip(i_users, i_ratings)])
	
    with open(workdir + args.dataset + '/list.pkl', 'wb') as f:
        pickle.dump(u_items_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump(i_users_list, f, pickle.HIGHEST_PROTOCOL)
        pickle.dump((user_count, item_count, rate_count), f, pickle.HIGHEST_PROTOCOL)  
    with open(args.dataset_path + 'dataset.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)
    with open(args.dataset_path + 'list.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        (user_count, item_count, rate_count) = pickle.load(f)    
    valid_data = GRDataset(valid_set, u_items_list, i_users_list) 
    test_data = GRDataset(test_set, u_items_list, i_users_list) 
    valid_loader = DataLoader(valid_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    test_loader = DataLoader(test_data, batch_size = args.batch_size, shuffle = False, collate_fn = collate_fn)
    model = GraphRec(user_count+1, item_count+1, rate_count+1, args.embed_dim).to(device)
    ckpt = torch.load('best_checkpoint.pth.tar')
    model.load_state_dict(ckpt['state_dict'])
    print("chkpt loaded")
    r_ij, uids, iids = validate(test_loader, model)
    list_reccomendation = ['buffer','Listen to music' ,'Lie down / take a nap',  'Talk about the issue with someone', 'Spent quality time with family', 'Watch videos /movies', 'Meditation / exercise', 'Take short trips / walks', 'Engage with pets', 'Follow a healthy diet (cut down on processed food and added sugar)', 'Reading books of your interest', 'Avoid putting off tasks until last minute', 'Reduce caffeine intake', 'Avail medical help / counselling']  
    list_iid = []
    list_rating = []
    recc = []
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
            recc.append(list_reccomendation[i])
            count_recc = count_recc + 1
    return recc
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
        return preds, uids, iids
if __name__ == '__main__':
    main(score, level)
