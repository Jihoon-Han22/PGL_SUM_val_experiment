# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from generate_summary import generate_summary
from evaluation_metrics import evaluate_summary
from layers.summarizer import PGL_SUM
import os
from os import listdir
from os.path import isfile, join
import h5py
import json
import argparse
from data_loader import get_loader
from tqdm import tqdm, trange
from utils import TensorboardWriter

import ast


def evaluate(model, device, eval_metric='avg', dataloader=None):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        # if model == None : model = self.model
        # if dataloader == None : dataloader = self.val_loader
        model.eval()
        
        fscore_history = []
        mse_loss_history = []
        mae_loss_history = []
        tau_history = []
        rho_history = []

        val_logits = {}

        dataloader = iter(dataloader)
        criterion = nn.MSELoss(reduction='none').to(device)
        mae_criterion = nn.L1Loss(reduction='none').to(device)
        for data in tqdm(dataloader):
        # for data in tqdm(dataloader, desc='Evaluate', ncols=80, leave=False):
            video_name = data['video_name']
            frame_features = data['features'].to(device)
            target = data['gtscore'].to(device)

            if len(frame_features.shape) == 2:
                seq = seq.unsqueeze(0)
            if len(target.shape) == 1:
                target = target.unsqueeze(0)

            B = frame_features.shape[0]
            mask=None
            if 'mask' in data:
                mask = data['mask'].to(device)

            with torch.no_grad():
                scores, attn_weights = model(frame_features, mask=mask)
                for i in range(B):
                    val_logits[video_name[i]] = scores[i].cpu()

                val_mse_loss = criterion(scores[mask], target[mask]).mean()
                val_mae_loss = mae_criterion(scores[mask], target[mask]).mean()
                
                mse_loss_history.append(val_mse_loss.item())
                mae_loss_history.append(val_mae_loss.item())

            scores = scores.squeeze().cpu()
            gt_summary = data['gt_summary'][0]
            sb = data['change_points'][0]
            n_frames = data['n_frames']
            nfps = data['n_frame_per_seg'][0].tolist()
            positions = data['picks'][0]
            
            machine_summary = generate_summary(scores, sb, n_frames, nfps, positions)

            f_score, kTau, sRho = evaluate_summary(machine_summary, gt_summary, eval_metric)
            
            fscore_history.append(f_score)
            tau_history.append(kTau)
            rho_history.append(sRho)
 
        val_mse_loss = np.mean(mse_loss_history)
        val_mae_loss = np.mean(mae_loss_history)
        final_f_score = np.mean(fscore_history)
        kendal_tau = np.mean(tau_history)
        spearman_rho = np.mean(rho_history)

        return final_f_score, val_mse_loss, val_mae_loss, val_logits, kendal_tau, spearman_rho


# def test(trained_model, device, mode, epoch_i, writer, dataloader, eval_metric, dataset_name, save_dir):

#     test_fscore, test_mse_loss, test_mae_loss, test_logits, test_kendal_tau, test_spearman_rho = \
#                 evaluate(trained_model, device=device, eval_metric = eval_metric, dataloader=dataloader)
#     if (mode=="val"):
#         writer.update_loss(test_fscore, epoch_i, 'val/f1_epoch(yt8m)')
#         writer.update_loss(test_mse_loss, epoch_i, 'val/mse_loss(yt8m)')
#         writer.update_loss(test_mae_loss, epoch_i, 'val/mae_loss(yt8m)')

#     if (mode=="test"):
#         writer.update_loss(test_fscore, epoch_i, 'test/f1_epoch(yt8m)')
#         writer.update_loss(test_mse_loss, epoch_i, 'test/mse_loss(yt8m)')
#         writer.update_loss(test_mae_loss, epoch_i, 'test/mae_loss(yt8m)')    
#     print("------------------------------------------------------")
#     print("   " + mode +" RESULT: ")
#     print('   YT8M F-score {0:0.5} kT {1:0.5} srho {2:0.5}'.format(test_fscore, test_kendal_tau, test_spearman_rho))
#     print("------------------------------------------------------")
#     f = open(str(save_dir) + '/results.txt', 'a+')
#     f.write("Testing on Model " + str(epoch_i) + '\n')
#     f.write(dataset_name +  '\tepoch ' + str(epoch_i) + ', Test MSE Loss ' + str(test_mse_loss) + '\n')
#     f.write(dataset_name +  '\tepoch ' + str(epoch_i) + ', Test F-score ' + str(test_fscore) + '\n')
#     f.write(dataset_name +  '\tepoch ' + str(epoch_i) + ', Test MAE Loss ' + str(test_mae_loss) + '\n')
#     f.write(dataset_name +  '\tepoch ' + str(epoch_i) + ', Test kendal Tau' + str(test_kendal_tau) + '\n')
#     f.write(dataset_name +  '\tepoch ' + str(epoch_i) + ', Test spearman Rho' + str(test_spearman_rho) + '\n')
#     f.flush()

if __name__ == "__main__":
    # arguments to run the script
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='yt8m', help="Dataset to be used. Supported: {SumMe, TVSum, yt8m}")
    parser.add_argument("--tuned_model_name", type=str, help="The name of the tuned model. i.e. LR5e5expG97Reg1e4_try1")
    parser.add_argument("--split_id", default=0, type=str, help="split_id of the splits")
    parser.add_argument("--val_size_list", type=str, help="List of validation sizes")

    parser.add_argument("--tag", default="dev", type=str, help="Tag")

    args = vars(parser.parse_args())
    dataset = args["dataset"]
    tuned_model_name = args["tuned_model_name"]
    split_id = args["split_id"]
    val_size_list = ast.literal_eval(args["val_size_list"])
    tag = args["tag"]

    # Experiment settings
    eval_metric = 'max' if dataset.lower() == 'summe' else 'avg'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_size = 27892
    #val_size_list = [25, 50, 400, 1000, 1500, 2000]

    # Paths
    model_path = f"Summaries/{tuned_model_name}/split0/models"
    model_files = [f for f in listdir(model_path) if (isfile(join(model_path, f)) and f.startswith("epoch") and f.endswith("pkl"))]
    model_files = sorted(model_files, key=lambda x: int(x.split('epoch-')[1].split('.pkl')[0]))
    dataset_path = f"/home/jihoon/data/projects/summarization/summarization_dataset/yt8m_sum_all.h5"

    
    # Inference through all validation set size
    for val_size in val_size_list:
        print("Validation set size:", val_size)

        save_dir = Path(f'Summaries/{tuned_model_name}/split' + str(split_id) + "/val_sizes" + "/val_size" + str(val_size))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        val_log_dir = save_dir.joinpath('val_logs')
        test_log_dir = save_dir.joinpath('test_logs')
        if not os.path.exists(val_log_dir):
            os.makedirs(val_log_dir)

        if not os.path.exists(test_log_dir):
            os.makedirs(test_log_dir)

        # Writer
        val_writer = TensorboardWriter(str(val_log_dir))
        test_writer = TensorboardWriter(str(test_log_dir))

        split_file = f"data/datasets/splits/{dataset}_split_{train_size}_{val_size}_{val_size}.json"
        trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute")
        val_loader = get_loader(mode='val', val_size = val_size, batch_size=1, video_type='yt8m')
        test_loader = get_loader(mode='test', val_size = val_size, batch_size=1, video_type='yt8m')
        for epoch_model in model_files:
            epoch_i = int(epoch_model.split('epoch-')[1].split('.pkl')[0])
            ckpt = join(model_path, epoch_model)
            trained_model = PGL_SUM(input_size=1024, output_size=1024, num_segments=4, heads=8,
                                fusion="add", pos_enc="absolute").to(device)
            # Model with the ckpt weight
            trained_model.load_state_dict(torch.load(ckpt))
            val_fscore, val_mse_loss, val_mae_loss, val_logits, val_kendal_tau, val_spearman_rho = \
                evaluate(trained_model, device=device, eval_metric = eval_metric, dataloader=val_loader)
            val_writer.update_loss(val_fscore, epoch_i, 'size_'+ str(val_size) + '_f1_epoch_yt8m')
            val_writer.update_loss(val_mse_loss, epoch_i, 'size_'+ str(val_size) + '_mse_loss_yt8m')
            val_writer.update_loss(val_mae_loss, epoch_i, 'size_'+ str(val_size) + '_mae_loss_yt8m')
            print(dataset +  '\tepoch ' + str(epoch_i) + ', val MSE Loss ' + str(val_mse_loss) + '\n')
            print(dataset +  '\tepoch ' + str(epoch_i) + ', val F-score ' + str(val_fscore) + '\n')
            print(dataset +  '\tepoch ' + str(epoch_i) + ', val MAE Loss ' + str(val_mae_loss) + '\n')
            
            test_fscore, test_mse_loss, test_mae_loss, test_logits, test_kendal_tau, test_spearman_rho = \
                evaluate(trained_model, device=device, eval_metric = eval_metric, dataloader=test_loader)
            test_writer.update_loss(test_fscore, epoch_i, 'size_'+ str(val_size) + '_f1_epoch_yt8m')
            test_writer.update_loss(test_mse_loss, epoch_i, 'size_'+ str(val_size) + '_mse_loss_yt8m')
            test_writer.update_loss(test_mae_loss, epoch_i, 'size_'+ str(val_size) + '_mae_loss_yt8m')

            print(dataset +  '\tepoch ' + str(epoch_i) + ', test MSE Loss ' + str(test_mse_loss) + '\n')
            print(dataset +  '\tepoch ' + str(epoch_i) + ', test F-score ' + str(test_fscore) + '\n')
            print(dataset +  '\tepoch ' + str(epoch_i) + ', test MAE Loss ' + str(test_mae_loss) + '\n')
            
            # test(trained_model, device=device, mode="val", epoch_i=epoch_i, writer=writer, dataloader=val_loader, eval_metric=eval_metric, dataset_name=dataset, save_dir=save_dir)
            # test(trained_model, device=device, mode="test", epoch_i=epoch_i, writer=writer, dataloader=test_loader, eval_metric=eval_metric, dataset_name=dataset, save_dir=save_dir)