# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import h5py, os
import numpy as np
import json


class VideoData(Dataset):
    def __init__(self, mode, val_size, video_type, split_index=0):
        """ Custom Dataset class wrapper for loading the frame features and ground truth importance scores.

        :param str mode: The mode of the model, train or test.
        :param str video_type: The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode = mode
        self.name = video_type.lower()
        self.val_size = val_size
        self.datasets = ['../../summarization_dataset/inception_v3_summe.h5',
                         '../../summarization_dataset/inception_v3_tvsum.h5',
                         '/home/jihoon/data/projects/summarization/summarization_dataset/yt8m_sum_all.h5']
        
        self.splits_filename = ['data/datasets/splits/' + self.name + '_split_27892_' + str(self.val_size) + "_" +  str(self.val_size) + '.json']
        
        self.split_index = split_index

        if 'summe' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'tvsum' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        elif 'yt8m' in self.splits_filename[0]:
            self.filename = self.datasets[2]    
        
        self.video_data = h5py.File(self.filename, 'r')
        self.list_frame_features, self.list_gtscores = [], []

        with open(self.splits_filename[0]) as f:
            self.data = json.loads(f.read())
            for i, split in enumerate(self.data):
                if i == self.split_index:
                    self.split = split
                    break


    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    def __getitem__(self, index):
        """ Function to be called for the index operator of `VideoData` Dataset.
        train mode returns: frame_features and gtscores
        test mode returns: frame_features and video name

        :param int index: The above-mentioned id of the data.
        """
        video_name = self.split[self.mode + '_keys'][index]
        d = {}
        d['video_name'] = video_name
        d['features'] = torch.Tensor(np.array(self.video_data[video_name + '/features']))
        d['gtscore'] = torch.Tensor(np.array(self.video_data[video_name + '/gtscore']))

        # cps, n_frames, nfps, positions
        if 'user_summary' in self.video_data[video_name]:
            d['gt_summary'] = np.array(self.video_data[video_name + '/user_summary'])
            d['n_frame_per_seg'] = np.array(self.video_data[video_name + '/n_frame_per_seg'])
            d['change_points'] = np.array(self.video_data[video_name + '/change_points'])
            d['picks'] = np.array(self.video_data[video_name + '/picks'])
            d['n_frames'] = np.array(self.video_data[video_name + '/n_frames'])
        else:
            n_frames = d['features'].shape[0]
            cps = np.array(self.video_data[video_name + '/change_points'])
            cps = np.append(cps, np.array([[cps[-1][0], n_frames-1]]), axis=0)
            d['n_frames'] = np.array(n_frames)
            d['picks'] = np.array([i for i in range(n_frames)])
            d['change_points'] = cps
            d['n_frame_per_seg'] = np.array([cp[1]-cp[0] for cp in cps])
            d['gt_summary'] = np.expand_dims(np.array(self.video_data[video_name + '/gt_summary']), axis=0)
        
        return d

def get_loader(mode, val_size, batch_size, video_type):
    """ Loads the `data.Dataset` of the `split_index` split for the `video_type` Dataset.
    Wrapped by a Dataloader, shuffled and `batch_size` = 1 in train `mode`.

    :param str mode: The mode of the model, train or test.
    :param str video_type: The Dataset being used, SumMe or TVSum.
    :param int split_index: The index of the Dataset split being used.
    :return: The Dataset used in each mode.
    """
    vd = VideoData(mode, val_size, video_type)
    return DataLoader(vd, batch_size=batch_size, shuffle=True, collate_fn=BatchCollator())

class BatchCollator(object):
    def __call__(self, batch):
        video_name, features, gtscore= [],[],[]
        cps, nseg, n_frames, picks, gt_summary = [], [], [], [], []

        try:
            for data in batch:
                video_name.append(data['video_name'])
                features.append(data['features'])
                gtscore.append(data['gtscore'])
                cps.append(data['change_points'])
                nseg.append(data['n_frame_per_seg'])
                n_frames.append(data['n_frames'])
                picks.append(data['picks'])
                gt_summary.append(data['gt_summary'])
        except:
            print('Error in batch collator')
        lengths = torch.LongTensor(list(map(lambda x: x.shape[0], features)))
        # print(lengths)
        max_len = max(list(map(lambda x: x.shape[0], features)))
        # print(max_len)

        mask = torch.arange(max_len)[None, :] < lengths[:, None]
        
        frame_feat = pad_sequence(features, batch_first=True)
        gtscore = pad_sequence(gtscore, batch_first=True)

        # batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask}
        batch_data = {'video_name' : video_name, 'features' : frame_feat, 'gtscore':gtscore, 'mask':mask, \
                      'n_frames': n_frames, 'picks': picks, 'n_frame_per_seg': nseg, 'change_points': cps, \
                        'gt_summary': gt_summary}
        return batch_data
    
if __name__ == '__main__':
    pass
