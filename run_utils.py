import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn.metrics import roc_auc_score, average_precision_score
from constants import PARAMS
from src.utils import create_rel2idxs
from src.metrics import apk


def calc_all_scores(triplets_loader: data.DataLoader,
                    model: nn.Module, device: torch.device):
    all_scores = []
    for triplets in triplets_loader:
        all_scores.append(model(triplets.to(device).long()))
    return torch.cat(all_scores, dim=0)


def calc_classic_metrics(pos_scores: torch.Tensor, pos_triplets: torch.Tensor,
                         neg_scores: torch.Tensor, neg_triplets: torch.Tensor,
                         metrics_separately: bool = True):
    pos_triplets = pos_triplets.cpu().numpy().astype(int)
    neg_triplets = neg_triplets.cpu().numpy().astype(int)
    pos_scores = pos_scores.cpu().numpy().astype(int)
    neg_scores = neg_scores.cpu().numpy().astype(int)
    if metrics_separately:
        rel2idxs_pos = create_rel2idxs(pos_triplets)
        rel2idxs_neg = create_rel2idxs(neg_triplets)
    else:
        rel2idxs_pos = {-1: np.arange(pos_triplets.shape[0])}
        rel2idxs_neg = {-1: np.arange(neg_triplets.shape[0])}
    rels = set(rel2idxs_pos.keys()).union(set(rel2idxs_neg.keys()))
    rel_auc_roc_list = []
    rel_auc_pr_list = []
    rel_apk_list = []
    for rel in rels:
        rel_pos_scores = pos_scores[rel2idxs_pos[rel]]
        rel_neg_scores = neg_scores[rel2idxs_neg[rel]]
        rel_scores = np.concatenate([rel_pos_scores, rel_neg_scores])
        rel_labels = np.concatenate([np.ones_like(rel_pos_scores),
                                     np.zeros_like(rel_neg_scores)])

        rel_auc_pr_list.append(average_precision_score(rel_labels, rel_scores))
        rel_auc_roc_list.append(roc_auc_score(rel_labels, rel_scores))

        actual_idx = range(len(rel_pos_scores))
        # All local indexes with probability (sorted)
        predicted_idx_all = sorted(range(len(rel_scores)), reverse=True,
                                   key=lambda i: rel_scores[i])
        rel_apk_list.append(apk(actual_idx, predicted_idx_all, k=50))

    return {'auc_roc': np.mean(rel_auc_roc_list),
            'auprc': np.mean(rel_auc_pr_list),
            'ap50': np.mean(rel_apk_list)}


def evaluation(pos_loader, neg_loader, model, device, loss_func,
               metrics_separately: bool = True):
    pos_scores = calc_all_scores(pos_loader, model, device)
    neg_scores = calc_all_scores(neg_loader, model, device)
    metrics = calc_classic_metrics(pos_scores, pos_loader.dataset,
                                   neg_scores, neg_loader.dataset.neg_triplets,
                                   metrics_separately)
    loss = loss_func(pos_scores, neg_scores)
    return loss.item(), metrics


# code from https://github.com/samehkamaleldin/libkge/blob/master/libkge/model_selection/eval.py
def evaluation_ranking(test_triples: np.array, nb_ents: int,
                       model: nn.Module, device: torch.device,
                       known_triples=None, verbose=0):
    """
    Evaluate a knowledge graph embedding model using the standard link prediction evaluation protocol.
    Parameters
    ----------
    model : KnowledgeGraphEmbeddingModel
        Model object.
    test_triples : np.ndarray
        evaluation test triples.
    known_triples : np.ndarray or None
        array with all the true known triples. If None, no filtered metrics are computed, only raw ones.
    verbose : int
        level of verbosity.
    Returns
    -------
    dict
        results dictionary with the raw and filtered average metrics.
    Notes
    -------
    This evaluation technique is inspired by the code at:
    https://github.com/ttrouill/complex/blob/master/efe/evaluation.py
    """
    # build indices for (sub, pred) and (pred, obj) pairs
    known_sub_triples = {}
    known_obj_triples = {}
    if known_triples is not None:
        for sub_id, rel_id, obj_id in known_triples:
            if (sub_id, rel_id) not in known_obj_triples:
                known_obj_triples[(sub_id, rel_id)] = [obj_id]
            elif obj_id not in known_obj_triples[(sub_id, rel_id)]:
                known_obj_triples[(sub_id, rel_id)].append(obj_id)
            if (rel_id, obj_id) not in known_sub_triples:
                known_sub_triples[(rel_id, obj_id)] = [sub_id]
            elif sub_id not in known_sub_triples[(rel_id, obj_id)]:
                known_sub_triples[(rel_id, obj_id)].append(sub_id)

    nb_test = len(test_triples)
    sub_ranks = np.zeros(nb_test, dtype=np.int32)
    sub_ranks_fl = np.zeros(nb_test, dtype=np.int32)
    obj_ranks = np.zeros(nb_test, dtype=np.int32)
    obj_ranks_fl = np.zeros(nb_test, dtype=np.int32)
    data_hash = hash(test_triples.data.tobytes())

    test_instances = enumerate(test_triples) if verbose == 0 else enumerate(tqdm(test_triples))
    for idx, (sub_id, rel_id, obj_id) in test_instances:

        # generate all possible subject corruption
        sub_corr = np.concatenate(
            [np.arange(nb_ents).reshape([-1, 1]), np.tile([rel_id, obj_id], [nb_ents, 1])], axis=1)

        # generate all possible object corruption
        obj_corr = np.concatenate(
            [np.tile([sub_id, rel_id], [nb_ents, 1]), np.arange(nb_ents).reshape([-1, 1])], axis=1)

        # evaluate the object corruptions
        sub_corr_scores = model(torch.Tensor(sub_corr).long().to(device)).cpu().numpy()
        sub_ranks[idx] = 1 + np.sum(sub_corr_scores > sub_corr_scores[sub_id])
        if known_triples is not None:
            sub_ranks_fl[idx] = sub_ranks[idx] - np.sum(
                sub_corr_scores[known_sub_triples[(rel_id, obj_id)]] > sub_corr_scores[sub_id])

        # evaluate the object corruptions
        obj_corr_scores = model(torch.Tensor(obj_corr).long().to(device)).cpu().numpy()
        obj_ranks[idx] = 1 + np.sum(obj_corr_scores > obj_corr_scores[obj_id])
        if known_triples is not None:
            obj_ranks_fl[idx] = obj_ranks[idx] - np.sum(
                obj_corr_scores[known_obj_triples[(sub_id, rel_id)]] > obj_corr_scores[obj_id])

    ranks_raw = np.concatenate([sub_ranks, obj_ranks], axis=0)
    mrr_raw = np.mean(1.0 / ranks_raw)
    ranks_fil = np.concatenate([sub_ranks_fl, obj_ranks_fl], axis=0)
    mrr_fil = np.mean(1.0 / ranks_fil)

    hits_at1 = (np.sum(ranks_fil <= 1) + 1e-10) / float(len(ranks_fil))
    hits_at3 = (np.sum(ranks_fil <= 3) + 1e-10) / float(len(ranks_fil))
    hits_at10 = (np.sum(ranks_fil <= 10) + 1e-10) / float(len(ranks_fil))

    if known_triples is None:
        result = {'hash': data_hash, 'raw': {'avg': dict()}}
    else:
        result = {'hash': data_hash, 'raw': {'avg': dict()}, 'filtered': {'avg': dict()}}

    result['raw']['avg']['mr'] = np.mean(ranks_raw)
    result['raw']['avg']['mrr'] = mrr_raw
    if known_triples is not None:
        result['filtered']['avg']['mr'] = np.mean(ranks_fil)
        result['filtered']['avg']['mrr'] = mrr_fil
        result['filtered']['avg']['hits@1'] = hits_at1
        result['filtered']['avg']['hits@3'] = hits_at3
        result['filtered']['avg']['hits@10'] = hits_at10
    return result


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', default=False,
                        action='store_true',
                        help='Whether to log run or not, default False')
    parser.add_argument('--gpu', default=False,
                        action='store_true', help="Use gpu or not.")
    parser.add_argument('--batch_size', default=PARAMS['batch_size'], type=int,
                        help='Batch size')
    parser.add_argument('--embed_dim', default=PARAMS['embed_dim'], type=int,
                        help="Output embedding size after last layer")
    parser.add_argument('--epoch', default=PARAMS['epoch'], type=int,
                        help="Number of epochs")
    parser.add_argument('--learning_rate', default=PARAMS['learning_rate'],
                        type=float, help="Learning rate for optimizer")
    parser.add_argument('--regularization', default=PARAMS['regularization'],
                        type=int,
                        help="Regularization constant")
    parser.add_argument('--print_progress_every', default=1, type=int,
                        help="Frequency (in epochs) of printing progress")
    parser.add_argument('--save_every', default=10, type=int,
                        help="Frequency (in epochs) of saving model" +
                             " and optimizer parameters")
    parser.add_argument('--seed', default=PARAMS['seed'],
                        type=int, help="Random seed")
    parser.add_argument('--metrics_separately', default=False,
                        action='store_true',
                        help="Calculate metrics separately" +
                             " for different edge types")
    parser.add_argument('--use_proteins', default=False,
                        action='store_true', help='Use proteins for train')
    parser.add_argument('--random_val_neg_sampler', default=False,
                        action='store_true',
                        help='Use random or honest sampler on validation')
    parser.add_argument('--val_regenerate', default=False, action='store_true',
                        help='Generate val neg samples after each usage')
    parser.add_argument('--neptune_project', default='Pollutants/sandbox',
                        type=str, help='Name of neptune project')
    parser.add_argument('--experiment_name', default='TriVec', type=str, 
                        help='Name of experiment for neptune')
    parser.add_argument("--reversed", default=False,
                        action='store_true',
                        help="Whether to add reversed drug-drug edges to KG")
    return parser
