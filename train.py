import time
import numpy as np
import torch as torch
import torch.nn.functional as F
import dgl
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import metrics
from scipy import interp
from utils import build_hetero_graph, sample, construct_negative_graph, set_random_seed
from model import GraphTGI_hetero, HeteroDotProductPredictor


torch.set_printoptions(profile='default')


def Train_loop(epochs, hidden_size, dropout, slope, lr, wd, random_seed, device):
    # dgl.load_backend('pytorch')
    # dgl.backend.load_backend('pytorch')
    # set_random_seed(random_seed)

    TF_associate_tg_samples, TF_associate_TF_samples, TF_associate_TF_samples1, TF_associate_miRna_samples, miRna_associate_TF_samples, miRna_associate_tg_samples = sample()

    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    TF_associate_tg_train_index = []
    TF_associate_tg_test_index = []
    TF_associate_TF_train_index = []
    TF_associate_TF_test_index = []

    TF_associate_TF_train_index1 = []
    TF_associate_TF_test_index1 = []

    TF_associate_miRna_train_index = []
    TF_associate_miRna_test_index = []

    miRna_associate_TF_train_index = []
    miRna_associate_TF_test_index = []
    miRna_associate_tg_train_index = []
    miRna_associate_tg_test_index = []


    for train_idx, test_idx in kf.split(TF_associate_tg_samples):
        TF_associate_tg_train_index.append(train_idx)
        TF_associate_tg_test_index.append(test_idx)

    for train_idx, test_idx in kf.split(TF_associate_TF_samples):
        TF_associate_TF_train_index.append(train_idx)
        TF_associate_TF_test_index.append(test_idx)

    for train_idx, test_idx in kf.split(TF_associate_TF_samples1):
        TF_associate_TF_train_index1.append(train_idx)
        TF_associate_TF_test_index1.append(test_idx)

    for train_idx, test_idx in kf.split(TF_associate_miRna_samples):
        TF_associate_miRna_train_index.append(train_idx)
        TF_associate_miRna_test_index.append(test_idx)

    for train_idx, test_idx in kf.split(miRna_associate_TF_samples):
        miRna_associate_TF_train_index.append(train_idx)
        miRna_associate_TF_test_index.append(test_idx)

    for train_idx, test_idx in kf.split(miRna_associate_tg_samples):
        miRna_associate_tg_train_index.append(train_idx)
        miRna_associate_tg_test_index.append(test_idx)

    lp_auc_result = []
    lp_acc_result = []
    lp_pre_result = []
    lp_recall_result = []
    lp_f1_result = []
    lp_auprc_result = []

    lp_fprs = []
    lp_tprs = []

    my_threshold = 0.5

    # generate a new graph for training in each fold
    # generate the negative graph in each epoch
    for i_fold in range(len(TF_associate_tg_train_index)):
        print('=====================Training for fold', i_fold + 1, '==============================')

        train_graph, tf_io_feats, tg_io_feats, mirna_io_feats = build_hetero_graph(
            TF_associate_tg_idx=TF_associate_tg_train_index[i_fold],
            TF_associate_TF_idx=TF_associate_TF_train_index[i_fold],
            TF_associate_TF_idx1=TF_associate_TF_train_index1[i_fold],
            TF_associate_miRna_idx=TF_associate_miRna_train_index[i_fold],
            miRna_associate_TF_idx=miRna_associate_TF_train_index[i_fold],
            miRna_associate_tg_idx=miRna_associate_tg_train_index[i_fold],
            random_seed=random_seed,
            device=device)
        test_graph, test_tf_io_feats, test_tg_io_feats, test_mirna_io_feats = build_hetero_graph(
            TF_associate_tg_idx=TF_associate_tg_test_index[i_fold],
            TF_associate_TF_idx=TF_associate_TF_test_index[i_fold],
            TF_associate_TF_idx1=TF_associate_TF_test_index1[i_fold],
            TF_associate_miRna_idx=TF_associate_miRna_test_index[i_fold],
            miRna_associate_TF_idx=miRna_associate_TF_test_index[i_fold],
            miRna_associate_tg_idx=miRna_associate_tg_test_index[i_fold],
            random_seed=random_seed,
            device=device)

        # encoder
        model_encoder = GraphTGI_hetero(tf_io_feats, tg_io_feats, mirna_io_feats, hidden_size, slope).to(device)
        model_encoder = model_encoder.to(device)

        # init parameters
        model_parameters = list(model_encoder.parameters())

        for j in range(1, len(model_parameters)):
            if (j % 2 == 2):
                parameters = model_parameters[j]
                torch.nn.init.xavier_normal_(parameters, gain=torch.nn.init.calculate_gain('relu'))


        BCEloss = torch.nn.BCELoss()

        linkPredictor = HeteroDotProductPredictor()
        linkPredictor = linkPredictor.to(device)


        opt = torch.optim.Adam(model_encoder.parameters(), lr=lr)

        train_TF_feats = train_graph.nodes['TF'].data['feature']
        train_tg_feats = train_graph.nodes['tg'].data['feature']
        train_miRna_feats = train_graph.nodes['miRna'].data['feature']
        train_node_features = {'TF': train_TF_feats, 'tg': train_tg_feats, 'miRna': train_miRna_feats}

        test_TF_feats = test_graph.nodes['TF'].data['feature']
        test_tg_feats = test_graph.nodes['tg'].data['feature']
        test_miRna_feats = test_graph.nodes['miRna'].data['feature']
        test_node_features = {'TF': test_TF_feats, 'tg': test_tg_feats, 'miRna': test_miRna_feats}

        for epoch in range(epochs):
            start = time.time()

            # train

            for _ in range(10):
                # def train(train_graph, node_features, model_encoder, linkPredictor, BCEloss, opt, device):
                train_lp_loss, embedding = train(
                    train_graph,
                    train_node_features,
                    model_encoder,
                    linkPredictor,
                    BCEloss,
                    opt,
                    device)

            test_lp_loss, test_lp_scores, test_lp_labels = test(
                test_graph,
                test_node_features,
                model_encoder,
                linkPredictor,
                BCEloss,
                device)

            # metrics for link prediction
            # only measure the regulate type
            test_lp_label = test_lp_labels['regulate']
            test_lp_score = test_lp_scores['regulate']


            lp_val_label = test_lp_label.cpu().numpy()
            lp_val_score = test_lp_score.cpu().numpy()


            lp_val_auc = metrics.roc_auc_score(lp_val_label, lp_val_score)
            #Precision-Recall Curve AUC (AUPRC)
            precision, recall, _ = metrics.precision_recall_curve(lp_val_label, lp_val_score)
            lp_val_auprc = metrics.auc(recall, precision)

            lp_val_score = np.where(lp_val_score > my_threshold, 1, 0)
            lp_accuracy_val = metrics.accuracy_score(lp_val_label, lp_val_score)
            lp_precision_val = metrics.precision_score(lp_val_label, lp_val_score)
            lp_recall_val = metrics.recall_score(lp_val_label, lp_val_score)
            lp_f1_val = metrics.f1_score(lp_val_label, lp_val_score)

            end = time.time()
            print('Epoch:', epoch + 1,
                  'Train lp Loss: %.4f' % train_lp_loss,
                  'test lp Loss: %.4f' % test_lp_loss, )

            print('LP Acc: %.4f' % lp_accuracy_val, 'LP Pre: %.4f' % lp_precision_val,
                  'LP Recall: %.4f' % lp_recall_val, 'LP F1: %.4f' % lp_f1_val, 'LP AUC: %.4f' % lp_val_auc,
                  'LP AUPRC: %.4f' % lp_val_auprc,
                  'Time: %.2f' % (end - start))

        lp_loss, lp_scores, lp_labels = test(
            test_graph,
            test_node_features,
            model_encoder,
            linkPredictor,
            BCEloss,
            device)

        # link prediction

        lp_label = lp_labels['regulate']
        lp_score = lp_scores['regulate']

        lp_label = lp_label.cpu().numpy()
        lp_score = lp_score.cpu().numpy()


        lp_fpr, lp_tpr, thresholds = metrics.roc_curve(lp_label, lp_score)
        lp_auc = metrics.auc(lp_fpr, lp_tpr)

        # Precision-Recall Curve AUC
        precision, recall, _ = metrics.precision_recall_curve(lp_label, lp_score)
        lp_auprc = metrics.auc(recall, precision)

        lp_score = np.where(lp_score > my_threshold, 1, 0)

        lp_accuracy = metrics.accuracy_score(lp_label, lp_score)
        lp_precision = metrics.precision_score(lp_label, lp_score)
        lp_recall = metrics.recall_score(lp_label, lp_score)
        lp_f1 = metrics.f1_score(lp_label, lp_score)

        lp_auc_result.append(lp_auc)
        lp_acc_result.append(lp_accuracy)
        lp_pre_result.append(lp_precision)
        lp_recall_result.append(lp_recall)
        lp_f1_result.append(lp_f1)
        lp_auprc_result.append(lp_auprc)

        lp_fprs.append(lp_fpr)
        lp_tprs.append(lp_tpr)


        print('Fold:', i_fold + 1, 'Test LP Acc: %.4f' % lp_accuracy, 'Test LP Pre: %.4f' % lp_precision,
              'Test LP Recall: %.4f' % lp_recall, 'Test LP F1: %.4f' % lp_f1, 'Test LP AUC: %.4f' % lp_auc,'Test LP AUPRC: %.4f' % lp_auprc )


    print('training finished')

    return lp_auc_result, lp_acc_result, lp_pre_result, lp_recall_result, lp_f1_result, lp_auprc_result, lp_fprs, lp_tprs


def train(train_graph, node_features, model_encoder, linkPredictor, BCEloss, opt, device):
    model_encoder.train()


    graph = model_encoder(train_graph, node_features)
    embedding = graph.ndata['h']

    etypes = train_graph.canonical_etypes

    link_prediction_scores = {}
    link_prediction_labels = {}

    lp_score = []

    # link prediction
    for etype in etypes:

        u, e, d = etype
        sub_train_graph = train_graph.edge_type_subgraph([etype])

        # build negative graph
        train_negative_graph = construct_negative_graph(sub_train_graph, 1, etype, device)
        label = sub_train_graph.edges[etype].data['lp_label'].to(torch.int64)

        node_types = sub_train_graph.ntypes
        dec_h = {node_types[j]: graph.nodes[node_types[j]].data['h'] for j in range(len(node_types))}

        if u == d:
            dec_h = dec_h[u]

        # link prediction using dot product
        pos_link_prediction_score = linkPredictor(sub_train_graph, dec_h, etype)
        neg_link_prediction_score = linkPredictor(train_negative_graph, dec_h, etype)

        # generate label
        pos_link_prediction_label = torch.ones([len(label), 1], dtype=torch.float32).to(device)
        neg_link_prediction_label = torch.zeros([len(label), 1], dtype=torch.float32).to(device)

        link_prediction_score = torch.cat((pos_link_prediction_score, neg_link_prediction_score), 0)
        link_prediction_label = torch.cat((pos_link_prediction_label, neg_link_prediction_label), 0)

        link_prediction_scores[e] = link_prediction_score
        link_prediction_labels[e] = link_prediction_label

    # compute loss for link-prediction task
    # the goal of model is to predict links of 'regulate' etype precisely
    # so that the others link are weighted to 0.1
    lp_loss = 0
    for etype in etypes:
        u, e, d = etype
        if e != 'regulate':
            lp_loss += 0.5 * BCEloss(link_prediction_scores[e], link_prediction_labels[e])
        else:
            lp_loss += BCEloss(link_prediction_scores[e], link_prediction_labels[e])

    opt.zero_grad()
    lp_loss.backward()

    # Updating the model weights
    opt.step()

    return lp_loss, embedding


def test(test_graph, test_node_features, model_encoder, linkPredictor, BCEloss, device):
    model_encoder.eval()

    with torch.no_grad():

        # encode to generate embeddings

        graph = model_encoder(test_graph, test_node_features)
        embedding = graph.ndata['h']


        etypes = test_graph.canonical_etypes

        test_lp_labels = {}
        test_lp_scores = {}


        for etype in etypes:
            u, e, d = etype

            sub_test_graph = test_graph.edge_type_subgraph([etype])
            test_negative_graph = construct_negative_graph(sub_test_graph, 1, etype, device)

            label = sub_test_graph.edges[etype].data['lp_label'].to(torch.int64)

            # extract feature of dec_graph

            node_types = sub_test_graph.ntypes
            dec_h = {node_types[j]: graph.nodes[node_types[j]].data['h'] for j in range(len(node_types))}

            # # if the sub graph only have one type of node, pass tensor directly

            if u == d:
                dec_h = dec_h[u]

            # link prediction using dot product

            pos_link_prediction_score = linkPredictor(sub_test_graph, dec_h, etype)
            neg_link_prediction_score = linkPredictor(test_negative_graph, dec_h, etype)

            # label

            pos_link_prediction_label = torch.ones([len(label), 1], dtype=torch.float32).to(device)
            neg_link_prediction_label = torch.zeros([len(label), 1], dtype=torch.float32).to(device)

            link_prediction_score = torch.cat((pos_link_prediction_score, neg_link_prediction_score), 0)
            link_prediction_label = torch.cat((pos_link_prediction_label, neg_link_prediction_label), 0)

            # note that which label/score been added
            test_lp_scores[e] = link_prediction_score
            test_lp_labels[e] = link_prediction_label

        # compute loss for link-prediction task
        test_lp_loss = 0

        for etype in etypes:
            u, e, d = etype
            if e != 'regulate':
                test_lp_loss += 0.5 * BCEloss(test_lp_scores[e], test_lp_labels[e])
            else:
                test_lp_loss += BCEloss(test_lp_scores[e], test_lp_labels[e])

    return test_lp_loss, test_lp_scores, test_lp_labels


