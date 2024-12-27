import torch as torch
import dgl
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import dgl.nn as dglnn

from dgl.nn.pytorch import GATConv, SAGEConv, SGConv, DotGatConv, GraphConv, EdgeConv


device = torch.device("cuda")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class HeteroLinear(nn.Module):

    def __init__(self, in_feats:dict, out_dim) -> None:
        super().__init__()

        self.linear_dict = {}
        for k,v in in_feats.items():
            self.linear_dict[k] = nn.Linear(v, out_dim).to(device)
        

    def forward(self, inputs:dict):
        h = {}
        for k, v in inputs.items():

            h[k] = self.linear_dict[k](v.to(device))
        return h
#
#
class GraphTGI_hetero(nn.Module):

    def __init__(self, tf_in_features, tg_in_features, mirna_in_features, hidden_features, slope):
        super().__init__()

        self.sage = RGSAGE(tf_in_features, tg_in_features, mirna_in_features, hidden_features, slope,).to(device)

    def forward(self, g, g_x):

        graph = self.sage(g, g_x)
        return graph


class RGSAGE(nn.Module):

    def __init__(self, tf_in_feats, tg_in_feats, mirna_in_feats, hid_feats, slope):
        super().__init__()

        self.act1 = nn.Sequential(nn.LeakyReLU(slope), nn.Dropout(0.2)).to(device)

        size1 = 1024
        size2 = 512
        size3 = 256

        self.Heterolinear = HeteroLinear({'TF': tf_in_feats, 'tg': tg_in_feats, 'miRna': mirna_in_feats}, size1).to(device)

        self.conv1 = dglnn.HeteroGraphConv({
            'regulate': SAGEConv(size1, size2,  'pool', activation=self.act1),
            'regulate_1': SAGEConv(size1, size2,  'pool', activation=self.act1),
            'regulate_2': SAGEConv(size1, size2,  'pool', activation=self.act1),
            'regulate_3': SAGEConv(size1, size2,  'pool', activation=self.act1),
            'regulate_4': SAGEConv(size1, size2,  'pool', activation=self.act1),
            'TF_self_loop': SAGEConv(size1, size2,  'pool', activation=self.act1)
        }, aggregate='sum').to(device)

        self.conv2 = dglnn.HeteroGraphConv({
            'regulate': SAGEConv(size2, size3,  'pool', activation=self.act1),
            'regulate_1': SAGEConv(size2, size3,  'pool', activation=self.act1),
            'regulate_2': SAGEConv(size2, size3,  'pool', activation=self.act1),
            'regulate_3': SAGEConv(size2, size3,  'pool', activation=self.act1),
            'regulate_4': SAGEConv(size2, size3,  'pool', activation=self.act1),
            'TF_self_loop': SAGEConv(size2, size3,  'pool', activation=self.act1)
        }, aggregate='sum').to(device)

    def forward(self, graph, inputs):

        # inputs = node features
        # transfer to same size
        h = self.Heterolinear(inputs)

        h1 = self.conv1(graph, h)

        h_mid = {k: self.act1(v) for k, v in h1.items()}     # 还可以加norm

        h2 = self.conv2(graph, h_mid)

        node_types = graph.ntypes
        for i in range(len(node_types)):
            graph.apply_nodes(lambda nodes: {'h': h2[node_types[i]]}, ntype=node_types[i])
        return graph



# dot product predictor for link prediction
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        with graph.local_scope():

            graph.ndata['link_prediction_h'] = h
            graph.apply_edges(dgl.function.u_dot_v('link_prediction_h', 'link_prediction_h', 'link_prediction_score'), etype=etype)
            #return graph.edges[etype].data['link_prediction_score']

            return torch.sigmoid(graph.edges[etype].data['link_prediction_score'])





