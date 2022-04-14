import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as ss
import warnings
warnings.filterwarnings('ignore')

class Recommender(nn.Module):
    def __init__(self, G, KG_item, KG_user, args):
        super(Recommender, self).__init__()

        # initialize parser arguments
        self.dim = args.dim
        self.layers = args.n_layers
        self.decay = args.decay
        self.drop_rate = args.drop_out
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

        self.kg_item = KG_item
        self.kg_user = KG_user

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
        self.struct_node_emb = init(torch.empty(self.n_nodes, self.dim).to(self.device))

        self.train_weight = init(torch.empty(self.dim, self.dim)).to(self.device)
        self.train_weight = nn.Parameter(self.train_weight)

        self.train_weight_2 = init(torch.empty(self.dim, self.dim)).to(self.device)
        self.train_weight_2 = nn.Parameter(self.train_weight_2)

        self.bias = init(torch.empty(self.n_nodes, self.dim)).to(self.device)
        self.bias = nn.Parameter(self.bias)

        self.bias_2 = init(torch.empty(self.n_nodes, self.dim)).to(self.device)
        self.bias_2 = nn.Parameter(self.bias_2)

        ##################################################
        self.mask = torch.zeros(len(self.kg_item.etypes), (self.kg_item.num_nodes(ntype='tail')-self.kg_item.num_nodes(ntype='head'))).to(self.device)
        for rel in range(len(self.kg_item.etypes)):
            idx = torch.unique(self.kg_item.adj(etype=f'{rel}')._indices()[1] - self.kg_item.num_nodes(ntype='head'))
            self.mask[rel,idx] = 1

        self.re = init(torch.empty(self.mask.shape[0], self.mask.shape[1])).to(self.device)
        self.re = nn.Parameter(self.re)

        self.entity_emb = init(torch.empty(self.mask.shape[1], self.dim)).to(self.device)
        self.entity_emb = nn.Parameter(self.entity_emb)

        self.ir = init(torch.empty(self.n_items, self.mask.shape[0])).to(self.device)
        self.ir = nn.Parameter(self.ir)
        ##################################################
        self.mask_2 = torch.zeros(len(self.kg_user.etypes), (self.kg_user.num_nodes(ntype='tail')-self.kg_user.num_nodes(ntype='head'))).to(self.device)
        for rel in range(len(self.kg_user.etypes)):
            idx = torch.unique(self.kg_user.adj(etype=f'{rel}')._indices()[1] - self.kg_user.num_nodes(ntype='head'))
            self.mask_2[rel,idx] = 1

        self.re_2 = init(torch.empty(self.mask_2.shape[0], self.mask_2.shape[1])).to(self.device)
        self.re_2 = nn.Parameter(self.re_2)

        self.entity_emb_2 = init(torch.empty(self.mask_2.shape[1], self.dim)).to(self.device)
        self.entity_emb_2 = nn.Parameter(self.entity_emb_2)

        self.ur = init(torch.empty(self.n_users, self.mask_2.shape[0])).to(self.device)
        self.ur = nn.Parameter(self.ur)
        ##################################################
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

    def forward_propagation(self, DA):
        
        self.user_emb = torch.mm(torch.sparse.mm(DA, self.struct_node_emb[self.n_users: , :]), self.train_weight)
        self.item_emb = torch.mm(torch.sparse.mm(DA.t(), self.struct_node_emb[:self.n_users , :]), self.train_weight)

        masked_item = torch.mul(nn.Softmax(dim=1)(self.re), self.mask)
        rel_feat = torch.mm(masked_item, self.entity_emb)
        item_kg_emb = torch.mm(self.ir, rel_feat)

        masked_user = torch.mul(nn.Softmax(dim=1)(self.re_2), self.mask_2)
        rel_feat_2 = torch.mm(masked_user, self.entity_emb_2)
        user_kg_emb = torch.mm(self.ur, rel_feat_2)
        ##################################################
        self.user_emb_f = torch.mm(torch.sparse.mm(DA, item_kg_emb), self.train_weight_2)
        self.item_emb_f = torch.mm(torch.sparse.mm(DA.t(), user_kg_emb), self.train_weight_2)

        H_u = self.user_emb + self.user_emb_f + self.bias[:self.n_users]
        H_i = self.item_emb + self.item_emb_f + self.bias[self.n_users:]

        self.user_emb_final = F.normalize(nn.LeakyReLU(negative_slope=0.2)(H_u), p=2, dim=1)
        self.item_emb_final = F.normalize(nn.LeakyReLU(negative_slope=0.2)(H_i), p=2, dim=1)
        return self.user_emb_final, self.item_emb_final
        

    def forward(self, batch=None, test=False):
        if test==False:    
            user = batch['batch_users']
            pos = batch['batch_pos']
            neg = batch['batch_neg']
        
        #Interaction Dropout
        DA = self.sparse_dropout(self.norm_inter_mat, self.drop_rate)
        #struct_emb = self.struct_node_emb

        u = torch.zeros(self.n_users, self.dim).to(self.device)
        i = torch.zeros(self.n_items, self.dim).to(self.device)

        for layer in range(len(self.layers)):
            uu,ii = self.forward_propagation(DA)
            u += uu
            i += ii
        
        user_emb = u
        item_emb = i    
        #user_emb = all_emb[:self.n_users, :]
        #item_emb = all_emb[self.n_users:, :]
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
