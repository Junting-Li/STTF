import random
from datetime import datetime

import numpy as np
import torch
import os.path as osp
from torch.utils.data import DataLoader, Dataset
from config.config import Config
import argparse

parser = argparse.ArgumentParser()
args = parser.parse_args()
parser.add_argument('--nodes',          type=int,   default=10537,
                    help='porto=10537, Beijing=28342')


def load_poi_neighbors(poi_file):
    data = np.load(poi_file, allow_pickle=True)
    neighbors = data['neighbors']
    return neighbors

class TrajDataset(Dataset):
    def __init__(self, traj_data):
        self.traj_data = traj_data

    def __len__(self):
        return len(self.traj_data)

    def __getitem__(self, idx):
        return self.traj_data.iloc[idx]

def continuous_mask_sequence(sequence, mask_percentage, mask_length):
    num_elements = len(sequence)
    num_elements_to_mask = int(num_elements * mask_percentage)
    masked_sequence = sequence.clone()

    num_continuous_intervals = num_elements - mask_length + 1

    num_intervals_to_mask = num_elements_to_mask // mask_length

    max_interval = num_continuous_intervals - num_intervals_to_mask

    start_indices = set()
    # TODO: An infinite loop could happen here, but this rarely happens because the masking rate is too small,
    #       and we can exit by setting the number of loops
    # num = 0
    while len(start_indices) < num_intervals_to_mask:
        start_idx = random.randint(0, max_interval)
        if all(start_idx < s or start_idx > s + mask_length for s in start_indices):
            start_indices.add(start_idx)
        # num += road

    for start_idx in start_indices:
        masked_sequence[start_idx:start_idx+mask_length] = Config.road_special_tokens['mask_token']

    return masked_sequence


class TrajDataLoader:
    def __init__(self):
        self.batch_size = Config.batch_size
        self.num_workers = 8

    def get_data_loader(self, traj_data, is_shuffle=False):
        dataset = TrajDataset(traj_data=traj_data)

        data_loader = DataLoader(dataset,
                                 batch_size=self.batch_size,
                                 shuffle=is_shuffle,
                                 num_workers=self.num_workers,
                                 collate_fn=self._collate_func)
        return data_loader

    def _collate_func(self, data_df):
        bz = len(data_df)

        path = 'data/{}'.format(Config.dataset)
        edge_file = osp.join(path, 'porto_trash_poi_10.npz')
        neighbors = load_poi_neighbors(edge_file)
        road_traj_list = [traj.road_traj for traj in data_df]

        road_traj_list_pos = []

        for traj in road_traj_list:
            pos = []
            for poi in traj:
                pos_id = np.random.randint(len(neighbors[poi]))
                pos.append(neighbors[poi][pos_id])
            road_traj_list_pos.append(pos)

        road_temporal_list = [traj.ptime for traj in data_df]
        road_lens = [len(path) for path in road_traj_list]
        max_road_len = max(road_lens)
        road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_traj_inputs_pos = torch.zeros((bz, max_road_len + 1), dtype=torch.long)

        mask_road_traj_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_type_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_week_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)
        road_minute_inputs = torch.zeros((bz, max_road_len + 1), dtype=torch.long)


        for i in range(bz):
            path = road_traj_list[i]
            path_pos = road_traj_list_pos[i]
            road_len = len(path)

            road_type = data_df[i]['road_type']
            road_type_inputs[i, 1:road_len + 1] = torch.LongTensor(road_type) + 1

            road_shift_with_tokens = torch.LongTensor(path) + len(Config.road_special_tokens)
            road_shift_with_tokens_pos = torch.LongTensor(path_pos) + len(Config.road_special_tokens)

            road_traj_inputs[i, 1:road_len + 1] = road_shift_with_tokens
            road_traj_inputs_pos[i, 1:road_len + 1] = road_shift_with_tokens_pos

            road_traj_inputs[i, 0] = Config.road_special_tokens['cls_token']
            road_traj_inputs_pos[i, 0] = Config.road_special_tokens['cls_token']

            mask_road_traj = continuous_mask_sequence(road_shift_with_tokens,Config.mask_ratio,Config.mask_length)

            mask_road_traj_inputs[i, 1:road_len + 1] = mask_road_traj
            mask_road_traj_inputs[i, 0] = Config.road_special_tokens['cls_token']

            road_temporal = road_temporal_list[i]
            road_date = [datetime.fromtimestamp(t) for t in road_temporal]
            road_weeks = [d.weekday() + 1 for d in road_date]
            road_minutes = [d.minute + 1 + d.hour * 60 for d in road_date]

            road_week_inputs[i, 1:road_len + 1] = torch.LongTensor(road_weeks)
            road_week_inputs[i, 0] = road_weeks[0]
            road_minute_inputs[i, 1:road_len + 1] = torch.LongTensor(road_minutes)
            road_minute_inputs[i, 0] = road_minutes[0]

        mask_road_index = torch.where(mask_road_traj_inputs == Config.road_special_tokens['mask_token'])

        road_data = {
            'road_traj': road_traj_inputs,
            'road_traj_pos': road_traj_inputs_pos,
            'mask_road_index': mask_road_index,
            'road_type': road_type_inputs,
            'road_weeks': road_week_inputs,
            'road_minutes': road_minute_inputs,
        }

        return road_data
