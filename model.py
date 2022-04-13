import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore')

class Recommender(nn.Module):
    def __init__(self, G, args):
        super(Recommender, self).__init__()

        self.dim = args.dim
        self.layers = args.n_layers
        self.decay = args.decay
        self.drop_rate = args.drop_out
        self.device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

        self.graph = G
        self.n_nodes = self.graph.num_nodes()
        self.n_users = self.graph.num_nodes(ntype='user')
        self.n_items = self.graph.num_nodes(ntype='item')
        
        self.inter_mat = self.graph.adj(scipy_fmt='coo',)
        self.norm_inter_mat = self.norm_dense(self.inter_mat)
        self.norm_inter_mat = self.sp_mat_to_sp_tensor(self.norm_inter_mat).to(self.device)

        self.initialize_features()

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
        self.node_emb = init(torch.empty(self.n_nodes, self.dim).to(self.device))
        self.node_emb = nn.Parameter(self.node_emb)

        self.train_weight = init(torch.empty(self.dim, self.dim)).to(self.device)
        self.train_weight = nn.Parameter(self.train_weight)

        self.bias = init(torch.empty(self.n_nodes, self.dim)).to(self.device)
        self.bias = nn.Parameter(self.bias)

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

    def graph_convolution(self, DA):
        
        user_emb = torch.mm(torch.sparse.mm(DA, self.node_emb[self.n_users: , :]), self.train_weight) + self.bias[:self.n_users]
        item_emb = torch.mm(torch.sparse.mm(DA.t(), self.node_emb[:self.n_users , :]), self.train_weight) + self.bias[self.n_users:]

        user_emb = F.normalize(nn.LeakyReLU(negative_slope=0.2)(user_emb), p=2, dim=1)
        item_emb = F.normalize(nn.LeakyReLU(negative_slope=0.2)(item_emb), p=2, dim=1)

        return torch.concat([user_emb, item_emb],0)

    def forward(self, batch=None, test=False):
        if test==False:    
            user = batch['batch_users']
            pos = batch['batch_pos']
            neg = batch['batch_neg']
        
        #Interaction Dropout
        DA = self.sparse_dropout(self.norm_inter_mat, self.drop_rate)
        
        for layer in range(self.layers):
            all_embedding = self.graph_convolution(DA)
            
        user_emb = all_embedding[:self.n_users, :]
        item_emb = all_embedding[self.n_users:, :]
        loss = 0

        if test==False:
            batch_user_emb = user_emb[user]
            batch_pos_emb, batch_neg_emb = item_emb[pos], item_emb[neg]
            loss = self.bpr_loss(batch_user_emb, batch_pos_emb, batch_neg_emb)

        return loss, user_emb, item_emb

    def rating(self, u_embeddings, i_embeddings):
        return torch.matmul(u_embeddings, i_embeddings.t())
        

    def bpr_loss(self, users, pos_items, neg_items):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        regularizer = (torch.norm(users) ** 2 + torch.norm(pos_items) ** 2 + torch.norm(neg_items) ** 2) / 2
        emb_loss = (self.decay * regularizer) / batch_size

        return mf_loss + emb_loss 
