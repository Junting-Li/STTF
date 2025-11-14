import torch
import numpy as np
import argparse
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--nodes',          type=int,   default=10537,
                    help='porto=10537, chengdu=39503')
parser.add_argument('--datapath',        type=str,   default='../data/porto/fusion/',
                    help='po=porto, cd=chengdu')
parser.add_argument('--dataset',        type=str,   default='porto',
                    help='po=porto, cd=chengdu')
parser.add_argument('--transhpath',        type=str,   default='../KGE/log/porto-transh.ckpt_final',)
parser.add_argument('--k',        type=int,   default=10,
                    help='bj=10, ny=30')
args = parser.parse_args()


if __name__ == '__main__':
    emb = torch.load(args.transhpath)
    poi_emb = emb['model_state_dict']['ent_embeddings.weight'].cpu().numpy()
    rel_emb = emb['model_state_dict']['rel_embeddings.weight'].cpu().numpy()
    KG = np.loadtxt(args.datapath + args.dataset + '_KG_graph.txt', dtype=int)

    M = np.zeros((args.nodes, args.nodes))
    
    for i in range(args.nodes):
        M[i] = np.exp(-np.linalg.norm(poi_emb[i] - poi_emb, ord=2, axis=1))

    for edge in KG:
        edge_type = edge[2]
        start = edge[0]
        end = edge[1]
        l2 = np.exp(-np.linalg.norm(poi_emb[start] + rel_emb[edge_type] - poi_emb[end], ord=2))
        M[start, end] = l2

    edge = np.argsort(M)[:, -args.k - 1:-1]
    print(edge.shape)
    poi = np.arange(M.shape[0])
    del M

    np.savez('../data/porto/porto_trash_poi_10', poi=poi, neighbors=edge)
    from_edge_ids = np.repeat(np.arange(len(edge)), edge.shape[1])

    to_edge_ids = edge.flatten()

    df = pd.DataFrame({
        'from_edge_id': from_edge_ids,
        'to_edge_id': to_edge_ids
    })


    df.to_csv(args.datapath + 'edge_rn.csv', index=False)

    print("over")
