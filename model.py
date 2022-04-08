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
        self.device = torch.device("cuda:"+str(args.gpu_id)) if args.cuda else torch.device("cpu")

        self.graph = G
        self.n_nodes = self.graph.num_nodes()
        self.n_users = self.graph.num_nodes(ntype='user')
        self.n_items = self.graph.num_nodes(ntype='item')
        
        self.inter_adj = self.graph.adjacency_matrix(scipy_fmt='coo',).todense()
        self.inter_adj = torch.FloatTensor(self.inter_adj)

        self.square_adjaceny = torch.zeros(self.n_nodes, self.n_nodes)
        self.square_adjaceny[:self.n_users, self.n_users:] = self.inter_adj
        self.square_adjaceny[self.n_users:, :self.n_users] = self.inter_adj.T
        self.square_adjaceny += torch.eye(self.n_nodes)
        self.square_adjaceny = self.square_adjaceny.to(self.device)

        self.density_mat = torch.FloatTensor(self.norm_dense(self.square_adjaceny.cpu())).to(self.device)

        self.DA = torch.matmul(self.density_mat, self.square_adjaceny)

        self.initialize_features()


    def norm_dense(self, A):
        rowsum = np.array(A.sum(1))
        d_inv = np.power(rowsum, -1)
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = ss.diags(d_inv).todense()
        return d_mat_inv

    
    def initialize_features(self):
        init = nn.init.xavier_uniform_
        self.graph.nodes['user'].data['cf_feature'] = init(torch.empty(self.graph.num_nodes(ntype='user'), self.dim))
        self.graph.nodes['item'].data['cf_feature'] = init(torch.empty(self.graph.num_nodes(ntype='item'), self.dim))
        user_feat = self.graph.nodes['user'].data['cf_feature']
        item_feat = self.graph.nodes['item'].data['cf_feature']
        
        self.node_feat = torch.cat([user_feat, item_feat]).to(self.device)
        self.node_feat = nn.Parameter(self.node_feat)

        self.train_weight = init(torch.empty(self.dim, self.dim)).to(self.device)
        self.train_weight = nn.Parameter(self.train_weight)

        self.bias = init(torch.empty(self.n_nodes, self.dim)).to(self.device)
        self.bias = nn.Parameter(self.bias)

    def graph_convolution(self, H):
        result = torch.mm(torch.sparse.mm(self.DA, H), self.train_weight)
        result = nn.LeakyReLU(negative_slope=0.2)(result + self.bias)

        return result

    def forward(self, batch=None, test=False):
        if test==False:    
            user = batch['batch_users']
            pos = batch['batch_pos']
            neg = batch['batch_neg']

        H = self.node_feat
        for layer in range(self.layers):
            H = self.graph_convolution(H)
            H = nn.Dropout(0.1)(H)
            H = F.normalize(H, p=2, dim=1)

        user_emb = H[:self.n_users]
        item_emb = H[self.n_users:]
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