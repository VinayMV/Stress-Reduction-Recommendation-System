# -*- coding: utf-8 -*-

import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat

random.seed(1234)

workdir = 'datasets/'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CalmDown', help='dataset name: CalmDown')
parser.add_argument('--test_prop', default=0.1, help='the proportion of data used for test')
parser.add_argument('--testing', action='store_true', help='test data')
args = parser.parse_args()

# load data
click_f = np.loadtxt(workdir+'CalmDown/ratings_data.txt', dtype = np.int32)

click_list = []

u_items_list = []
u_users_list = []
u_users_items_list = []
i_users_list = []

user_count = 0
item_count = 0
rate_count = 0

for s in click_f:
	uid = s[0]
	iid = s[1]
	if args.dataset == 'CalmDown':
		label = s[2]

	if uid > user_count:
		user_count = uid
	if iid > item_count:
		item_count = iid
	if label > rate_count:
		rate_count = label
	click_list.append([uid, iid, label])
length=len(click_list)
pos_list = []
for i in range(len(click_list)):
	pos_list.append((click_list[i][0], click_list[i][1], click_list[i][2]))

# remove duplicate items in pos_list because there are some cases where a user may have different rate scores on the same item.
pos_list = list(set(pos_list))

# train, valid and test data split
random.shuffle(pos_list)
if args.testing:
        user_count=user_count+1
        uid=user_count
        iid14=int(input("Enter PSS Score:"))
        iid15=int(input("Enter PSS Level:"))
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
print('Train samples: {}, Valid samples: {}, Test samples: {}'.format(len(train_set), len(valid_set), len(test_set)))

with open(workdir + args.dataset + '/dataset.pkl', 'wb') as f:
	pickle.dump(train_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(valid_set, f, pickle.HIGHEST_PROTOCOL)
	pickle.dump(test_set, f, pickle.HIGHEST_PROTOCOL)


train_df = pd.DataFrame(train_set, columns = ['uid', 'iid', 'label'])
valid_df = pd.DataFrame(valid_set, columns = ['uid', 'iid', 'label'])
test_df = pd.DataFrame(test_set, columns = ['uid', 'iid', 'label'])

click_df = pd.DataFrame(click_list, columns = ['uid', 'iid', 'label'])
train_df = train_df.sort_values(axis = 0, ascending = True, by = 'uid')

"""
u_items_list: 存储每个用户交互过的物品iid和对应的评分，没有则为[(0, 0)]
"""
for u in tqdm(range(user_count + 1)):
	hist = train_df[train_df['uid'] == u]
	u_items = hist['iid'].tolist()
	u_ratings = hist['label'].tolist()
	if u_items == []:
		u_items_list.append([(0, 0)])
	else:
		u_items_list.append([(iid, rating) for iid, rating in zip(u_items, u_ratings)])

train_df = train_df.sort_values(axis = 0, ascending = True, by = 'iid')

"""
i_users_list: 存储与每个物品相关联的用户及其评分，没有则为[(0, 0)]
"""
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


