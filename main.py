import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from parser import parse_args
from data_loader import loader
from model import Recommender
from evaluate import test

import random
from time import time
from collections import defaultdict
import numpy as np
import torch
import dgl
import scipy
import os

n_users = 0
n_items = 0

def sampling(train_pairs, batch_start, batch_end, train_user_set, n_items):
    def negative_sample(user_item, train_user_set):
        neg_items = []
        for user, _ in user_item.cpu().numpy():
            user = int(user)
            while True:
                neg_item = np.random.randint(low=0, high=n_items, size=1)[0]
                if neg_item not in train_user_set[user]:
                    break
            neg_items.append(neg_item)
        return neg_items
    
    sample_dict = {}
    interaction_pairs = train_pairs[batch_start:batch_end].to(device)
    sample_dict['batch_users'] = interaction_pairs[:,0]
    sample_dict['batch_pos'] = interaction_pairs[:,1]
    sample_dict['batch_neg'] = torch.LongTensor(negative_sample(interaction_pairs, train_user_set)).to(device)

    return sample_dict



def main(args, device):
    #Load Data
    data_dir = os.getcwd() + '/data/' + args.dataset
    train_interaction = loader(data_dir + '/train.txt')
    test_interaction = loader(data_dir + '/test.txt')
    train_user_set = defaultdict(list)
    test_user_set = defaultdict(list)
    n_users = max(max(train_interaction[:, 0]), max(test_interaction[:, 0])) + 1
    n_items = max(max(train_interaction[:, 1]), max(test_interaction[:, 1])) + 1
    print(f'#Users : {n_users} | #Items : {n_items}')
    for u_id, i_id in train_interaction:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_interaction:
        test_user_set[int(u_id)].append(int(i_id))
    train_interaction = torch.LongTensor(np.array([[cf[0], cf[1]] for cf in train_interaction], np.int32))

    #Build Graph
    cf_graph = dgl.heterograph({('user', 'purchased', 'item') : (train_interaction[:,0],train_interaction[:,1])})

    model = Recommender(cf_graph, args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Start Training...")
    for epoch in range(args.epoch):
        index = np.arange(len(train_interaction))
        np.random.shuffle(index)
        train_pairs = train_interaction[index]

        loss = 0
        batch = 0
        train_start_time = time()
        while batch + args.batch_size <= len(train_pairs):
            input_batch = sampling(train_pairs, batch, batch + args.batch_size, train_user_set, n_items)
            
            batch_loss, u_embedding, i_embedding = model(input_batch)
            batch_loss = batch_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            loss += batch_loss
            batch += args.batch_size
        train_end_time = time()

        if epoch % 5 == 0:
            test_start_time = time()
            _, u_embedding, i_embedding = model(test=True)
            result = test(model, train_user_set, test_user_set, n_users, n_items, u_embedding, i_embedding)
            test_end_time = time()

            print(f"\nepoch : {epoch}\tTrain Time : {train_end_time-train_start_time}\tTest Time : {test_end_time-test_start_time}\tLoss : {loss.item()}")
            print(f"recall@20 : {result['recall'][0]}\tnDCG@20 : {result['ndcg'][0]}\tprecision@20 : {result['precision'][0]}\thit@20 : {result['hit_ratio'][0]}\n")
        else:
            print(f"Train Time : {train_end_time-train_start_time}\tLoss : {loss.item()}")

if __name__ == '__main__':
    seed = 2022
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    global args, device

    args = parse_args()
    device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")
    main(args, device)