from math import floor
from random import random
import random
import numpy as np
import pandas as pd
import torch as torch
import dgl
import torch.nn.functional as F



def load_feature_data():

    TF_seq_feature = pd.read_csv('./data/tf_embeddings.csv',  header=None)

    tg_seq_feature = pd.read_csv('./data/tg_embeddings.csv', header=None)

    mirna_seq_feature = pd.read_csv('./data/miRNA_embeddings.csv', header=None)

    return TF_seq_feature, tg_seq_feature, mirna_seq_feature


def sample():

    TF_associate_tg = pd.read_csv('./data/TF-target.csv', header=0)
    TF_associate_TF = pd.read_csv('./data/TF-TF.csv', header=0)
    TF_associate_TF1 = pd.read_csv('./data/TF-TF_1.csv', header=0)
    TF_associate_miRna = pd.read_csv('./data/TF-miRna.csv', header=0)
    miRna_associate_TF = pd.read_csv('./data/miRna-TF.csv', header=0)
    miRna_associate_tg = pd.read_csv('./data/miRna-target.csv', header=0)

    return TF_associate_tg, TF_associate_TF, TF_associate_TF1, TF_associate_miRna, miRna_associate_TF, miRna_associate_tg


def build_hetero_graph(TF_associate_tg_idx, TF_associate_TF_idx, TF_associate_TF_idx1, TF_associate_miRna_idx, miRna_associate_TF_idx, miRna_associate_tg_idx, random_seed, device):


    TF_seq_feature_origin, tg_seq_feature_origin, mirna_seq_feature_origin = load_feature_data()

    TF_associate_tg, TF_associate_TF, TF_associate_TF1, TF_associate_miRna, miRna_associate_TF, miRna_associate_tg = sample()


    TF_associate_tg_src_node = torch.tensor(data = TF_associate_tg['TF'][TF_associate_tg_idx].values, dtype=torch.int32, device = device)
    TF_associate_tg_dst_node = torch.tensor(data = TF_associate_tg['target'][TF_associate_tg_idx].values, dtype=torch.int32, device = device)



    TF_associate_tg_dst_node = TF_associate_tg_dst_node - 1647


    TF_associate_TF_src_node = torch.tensor(data = TF_associate_TF['TF1'][TF_associate_TF_idx].values, dtype=torch.int32, device = device)
    TF_associate_TF_dst_node = torch.tensor(data = TF_associate_TF['TF2'][TF_associate_TF_idx].values, dtype=torch.int32, device = device)


    TF_associate_TF_src_node1 = torch.tensor(data = TF_associate_TF1['TF'][TF_associate_TF_idx1].values, dtype=torch.int32, device = device)
    TF_associate_TF_dst_node1 = torch.tensor(data = TF_associate_TF1['TF'][TF_associate_TF_idx1].values, dtype=torch.int32, device = device)


    TF_associate_miRna_scr_node = torch.tensor(data = TF_associate_miRna['TF'][TF_associate_miRna_idx].values, dtype=torch.int32, device = device)
    TF_associate_miRna_dst_node = torch.tensor(data = TF_associate_miRna['miRna'][TF_associate_miRna_idx].values, dtype=torch.int32, device = device)


    TF_associate_miRna_dst_node = TF_associate_miRna_dst_node - 20040



    miRna_associate_TF_src_node = torch.tensor(data = miRna_associate_TF['miRna'][miRna_associate_TF_idx].values, dtype=torch.int32, device = device)
    miRna_associate_TF_dst_node = torch.tensor(data = miRna_associate_TF['TF'][miRna_associate_TF_idx].values, dtype=torch.int32, device = device)

    miRna_associate_TF_src_node = miRna_associate_TF_src_node - 20040


    miRna_associate_tg_src_node = torch.tensor(data = miRna_associate_tg['miRna'][miRna_associate_tg_idx].values, dtype=torch.int32, device = device)
    miRna_associate_tg_dst_node = torch.tensor(data = miRna_associate_tg['target'][miRna_associate_tg_idx].values, dtype=torch.int32, device = device)

    miRna_associate_tg_src_node = miRna_associate_tg_src_node - 20040
    miRna_associate_tg_dst_node = miRna_associate_tg_dst_node - 1647



    TF_associate_tg_lp_label = torch.tensor(data = TF_associate_tg['lp_label'][TF_associate_tg_idx].values, dtype=torch.int32, device=device)
    TF_associate_TF_lp_label = torch.tensor(data = TF_associate_TF['lp_label'][TF_associate_TF_idx].values, dtype=torch.int32, device=device)
    TF_associate_miRna_lp_label = torch.tensor(data = TF_associate_miRna['lp_label'][TF_associate_miRna_idx].values, dtype=torch.int32, device=device)
    miRna_associate_TF_lp_label = torch.tensor(data = miRna_associate_TF['lp_label'][miRna_associate_TF_idx].values, dtype=torch.int32, device=device)
    miRna_associate_tg_lp_label = torch.tensor(data = miRna_associate_tg['lp_label'][miRna_associate_tg_idx].values, dtype=torch.int32, device=device)

    TF_self_loop_node = torch.tensor(data = range(1647), device= device)

    hetero_graph = dgl.heterograph({
        ('TF','regulate','tg'): (TF_associate_tg_src_node, TF_associate_tg_dst_node),
        ('TF','regulate_1','TF'): (TF_associate_TF_src_node, TF_associate_TF_dst_node),
        ('TF','regulate_2','miRna'): (TF_associate_miRna_scr_node, TF_associate_miRna_dst_node),
        ('miRna', 'regulate_3','TF'): (miRna_associate_TF_src_node, miRna_associate_TF_dst_node),
        ('miRna','regulate_4','tg'): (miRna_associate_tg_src_node, miRna_associate_tg_dst_node),
        ('TF', 'TF_self_loop', 'TF'): (TF_self_loop_node, TF_self_loop_node),
    },
        idtype = torch.int32, device = device)

    TF_ids = hetero_graph.nodes('TF').tolist()
    tg_ids = hetero_graph.nodes('tg').tolist()
    miRna_ids = hetero_graph.nodes('miRna').tolist()


    TF_seq_feature = TF_seq_feature_origin.iloc[TF_ids]#.iloc[TF_ids]：根据TF_ids列表中的ID，按行提取对应的化学特征数据。
    tg_seq_feature = tg_seq_feature_origin.iloc[tg_ids]
    miRna_seq_feature = mirna_seq_feature_origin.iloc[miRna_ids]


    hetero_graph.nodes['TF'].data['feature'] = torch.as_tensor(TF_seq_feature.values, dtype = torch.float32, device = device)
    hetero_graph.nodes['tg'].data['feature'] = torch.as_tensor(tg_seq_feature.values, dtype = torch.float32, device = device)
    hetero_graph.nodes['miRna'].data['feature'] = torch.as_tensor(miRna_seq_feature.values, dtype = torch.float32, device = device)
    
    # add edges label
    hetero_graph.edges['TF','regulate','tg'].data['lp_label'] = TF_associate_tg_lp_label
    hetero_graph.edges['TF','regulate_1','TF'].data['lp_label'] = TF_associate_TF_lp_label
    hetero_graph.edges['regulate_2'].data['lp_label'] = TF_associate_miRna_lp_label
    hetero_graph.edges['regulate_3'].data['lp_label'] = miRna_associate_TF_lp_label
    hetero_graph.edges['regulate_4'].data['lp_label'] = miRna_associate_tg_lp_label

    hetero_graph.edges['TF_self_loop'].data['lp_label'] = torch.ones(len(TF_self_loop_node), device=device)

    return hetero_graph, TF_seq_feature.shape[1], tg_seq_feature.shape[1], miRna_seq_feature.shape[1]


def construct_negative_graph(graph, k, etypes, device):

    utype, _, vtype = etypes
    src, dst = graph.edges(etype=etypes)

    neg_src = src.repeat_interleave(1).to(torch.int32).to(device)

    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * 1,)).to(torch.int32).to(device)

    return dgl.heterograph(
        {etypes: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

def set_random_seed(random_seed):
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if random_seed == 0:
        torch.backends.cudann.deterministic = True
        torch.backends.cudnn.benchmark = False









