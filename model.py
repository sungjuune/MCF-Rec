import struct
import dgl
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA

import numpy as np
import scipy.sparse as ss
import warnings
import pickle
from tqdm import tqdm
warnings.filterwarnings('ignore')


class Recommender(nn.Module):
    def __init__(self, G, args):
        super(Recommender, self).__init__()

        # initialize parser arguments
        self.dim = args.dim
        self.layers = args.n_layers
        self.decay = args.decay
        self.drop_rate = args.drop_out
        self.mess_drop_out = args.mess_drop_out
        self.device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

        # initialize Graph
        self.graph = G
        self.n_nodes = self.graph.num_nodes()
        self.n_users = self.graph.num_nodes(ntype='user')
        self.n_items = self.graph.num_nodes(ntype='item')
        
        # normalize interaction matrix (D^-1A)
        self.inter_mat = self.graph.adj(scipy_fmt='coo',)
        self.norm_inter_mat = self.norm_dense(self.inter_mat)
        self.norm_inter_mat = self.sp_mat_to_sp_tensor(self.norm_inter_mat).to(self.device)

        self.embedding_dict, self.weight_dict, self.DA = self.initialize_features()

        

    def sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def norm_dense(self, A):
        rowsum = np.array(A.sum(1))
        d_inv = np.power(np.float32(rowsum), -1)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = ss.diags(d_inv.squeeze())
        norm_inter_mat = d_mat_inv.dot(A)
        return norm_inter_mat

    
    def initialize_features(self):
        init = nn.init.xavier_uniform_

        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(init(torch.empty(self.n_users,self.dim))).to(self.device),
            'item_emb': nn.Parameter(init(torch.empty(self.n_items,self.dim))).to(self.device)
        })
        weight_dict = nn.ParameterDict()
        for l in range(self.layers):
            weight_dict.update({'W_%d'%l: nn.Parameter(init(torch.empty(self.dim, self.dim))).to(self.device)})
            weight_dict.update({'b_%d'%l: nn.Parameter(init(torch.empty(1, self.dim))).to(self.device)})

        DA = self.sparse_dropout(self.norm_inter_mat, self.drop_rate)

        return embedding_dict, weight_dict, DA


    def sparse_dropout(self, x, drop_rate):
        noise_shape = x._nnz()
        random_tensor = 1 - drop_rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)
        return out * (1. / (1 - drop_rate))

    def structure_propagation(self, id_u, id_i, l):
        
        user_emb = torch.matmul(torch.sparse.mm(self.DA, id_i), self.weight_dict['W_%d'%l]) + self.weight_dict['b_%d'%l]
        item_emb = torch.matmul(torch.sparse.mm(self.DA.t(), id_u), self.weight_dict['W_%d'%l]) + self.weight_dict['b_%d'%l]

        return torch.cat([user_emb, item_emb], 0)

    def forward(self, batch=None, test=False):
        #user_emb = torch.zeros(self.n_users, self.dim).to(self.device)
        #item_emb = torch.zeros(self.n_items, self.dim).to(self.device)

        id_user = self.embedding_dict['user_emb']
        id_item = self.embedding_dict['item_emb']
        total_embeddings = [torch.cat([id_user,id_item],0)]

        for l in range(self.layers):
            struct_emb = self.structure_propagation(id_user, id_item, l)
            struct_emb = nn.LeakyReLU(negative_slope=0.2)(struct_emb)
            struct_emb = nn.Dropout(self.mess_drop_out)(struct_emb)
            struct_emb = F.normalize(struct_emb, p=2, dim=1)
            total_embeddings += [struct_emb]

        total_embeddings = torch.cat(total_embeddings, 1)
        user_emb = total_embeddings[:self.n_users,:]
        item_emb = total_embeddings[self.n_users:,:]

        bpr_loss = 0
        if test==False:
            user = batch['batch_users']
            pos = batch['batch_pos']
            neg = batch['batch_neg']
            batch_user_emb = user_emb[user]
            batch_pos_emb, batch_neg_emb = item_emb[pos], item_emb[neg]
            bpr_loss = self.bpr_loss(batch_user_emb, batch_pos_emb, batch_neg_emb)

        return bpr_loss, user_emb, item_emb

    def rating(self, u_embeddings, i_embeddings):
        return torch.matmul(u_embeddings, (i_embeddings).t())
        

    def bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = (self.decay * regularizer) / batch_size

        return mf_loss + emb_loss 




