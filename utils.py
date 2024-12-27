import torch
import random

truncate_len = 30

"""
Ciao dataset info:
Avg number of items rated per user: 38.3
Avg number of users interacted per user: 2.7
Avg number of users connected per item: 16.4
"""

def collate_fn(batch_data):
    """This function will be used to pad the graph to max length in the batch
       It will be used in the Dataloader
    """
    uids, iids, labels = [], [], []
    u_items, i_users = [], []
    u_items_len, i_users_len = [], []

    for data, u_items_u, i_users_i in batch_data:

        (uid, iid, label) = data
        uids.append(uid)
        iids.append(iid)
        labels.append(label)

        # user-items    
        if len(u_items_u) <= truncate_len:
            u_items.append(u_items_u)
        else:
            u_items.append(random.sample(u_items_u, truncate_len))
        u_items_len.append(min(len(u_items_u), truncate_len))
        
        # item-users
        if len(i_users_i) <= truncate_len:
            i_users.append(i_users_i)
        else:
            i_users.append(random.sample(i_users_i, truncate_len))
        i_users_len.append(min(len(i_users_i), truncate_len))

    batch_size = len(batch_data)

    # padding
    u_items_maxlen = max(u_items_len)
    i_users_maxlen = max(i_users_len)
    
    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 2], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)
    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 2], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(labels), \
            u_item_pad, i_user_pad
