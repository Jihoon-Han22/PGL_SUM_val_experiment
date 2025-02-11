# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys
import random
import json
import h5py
from tqdm import tqdm, trange
from layers.summarizer import PGL_SUM
from utils import TensorboardWriter
from evaluation_metrics import evaluate_summary
from generate_summary import generate_summary


class WarmupStepSchedule(torch.optim.lr_scheduler.LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, step_size, gamma, last_epoch=-1):

        def lr_lambda(step):
            if step < warmup_steps:
                return 0.99 * float(step) / float(max(1.0, warmup_steps)) + 0.01
            else:
                return max(gamma ** (step - warmup_steps), 0.02)
                # return gamma ** (int((step - warmup_steps) / step_size))

        super(WarmupStepSchedule, self).__init__(optimizer, lr_lambda, last_epoch=last_epoch)

class Solver(object):
    def __init__(self, config=None, train_loader=None, val_loader=None, test_loader=None):
        """Class that Builds, Trains and Evaluates PGL-SUM model"""
        # Initialize variables to None, to be safe
        self.model, self.optimizer, self.writer = None, None, None

        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.global_step = 0

        self.criterion = nn.MSELoss(reduction='none').to(self.config.device)
        self.mae_criterion = nn.L1Loss(reduction='none').to(self.config.device)

        # Set the seed for generating reproducible random numbers
        # if self.config.seed is not None:
        # import random
        # import time
        # seed = round(time.time())
        # random.seed(seed)
        # np.random.seed(seed)
        # torch.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)

    def build(self):
        """ Function for constructing the PGL-SUM model of its key modules and parameters."""
        # Model creation
        self.model = PGL_SUM(input_size=self.config.input_size,
                             output_size=self.config.output_size,
                             num_segments=self.config.n_segments,
                             heads=self.config.heads,
                             fusion=self.config.fusion,
                             pos_enc=self.config.pos_enc).to(self.config.device)
        if self.config.ckpt_path != '':
            print("Loading model weight from ", self.config.ckpt_path)
            self.model.load_state_dict(torch.load(self.config.ckpt_path), map_location=self.config.device)
        else:
            print("Model random initialize")
            if self.config.init_type is not None:
                self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

        print('----------------------------')
        all_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        all_params_Mb = all_params / 1024**2
        trainable_params_Mb = trainable_params / 1024**2
        print(f'# of Trainable parameters : {trainable_params}\t {round(trainable_params_Mb,2)}M')
        print(f'# of All parameters (including non-trainable) : {all_params}\t {round(all_params_Mb,2)}M')
        print('----------------------------')

        if self.config.mode == 'train':
            # Optimizer initialization
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.l2_req)
            self.scheduler = WarmupStepSchedule(self.optimizer, warmup_steps=self.config.warmup_steps, step_size=self.config.step_size, gamma=self.config.gamma)
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """ Initialize 'net' network weights, based on the chosen 'init_type' and 'init_gain'.

        :param nn.Module net: Network to be initialized.
        :param str init_type: Name of initialization method: normal | xavier | kaiming | orthogonal.
        :param float init_gain: Scaling factor for normal.
        """
        for name, param in net.named_parameters():
            if 'weight' in name and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))  # ReLU activation function
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))      # ReLU activation function
                else:
                    raise NotImplementedError(f"initialization method {init_type} is not implemented.")
            elif 'bias' in name:
                nn.init.constant_(param, 0.1)

    

    def train(self):
        """ Main function to train the PGL-SUM model. """
        best_f1_score = -1.0
        best_test_f1_score = -1.0
        best_model_epoch = 0
        best_ckpt_path = None
        # for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
        for epoch_i in range(self.config.n_epochs):
            print("[Epoch: {0:6}]".format(str(epoch_i)+"/"+str(self.config.n_epochs)))
            self.model.train()

            loss_history = []
            num_batches = int(len(self.train_loader))  # full-batch or mini batch
            iterator = iter(self.train_loader)

            # for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
            for _ in range(num_batches):
                # ---- Training ... ----#
                if self.config.verbose:
                    tqdm.write('Time to train the model...')

                self.optimizer.zero_grad()
                data = next(iterator)

                frame_features = data['features'].to(self.config.device)
                target = data['gtscore'].to(self.config.device)
                mask = data['mask'].to(self.config.device)

                # TODO: COME BACK FOR Attention!
                output, weights = self.model(frame_features, mask)
                loss = self.criterion(output[mask], target[mask]).mean()
                # self.writer.update_loss(loss, self.global_step, 'train/batch_loss')
                # self.global_step += 1

                if self.config.verbose:
                    tqdm.write(f'[{epoch_i}] loss: {loss.item()}')

                loss.backward()
                loss_history.append(loss.item())
                
                # Update model parameters every 'batch_size' iterations
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.optimizer.step()

            # Mean loss of each training step
            loss = np.mean(np.array(loss_history))

            # Plot
            if self.config.verbose:
                tqdm.write('Plotting...')

            self.writer.update_loss(loss, epoch_i, 'train/loss_epoch')
            self.writer.update_loss(self.scheduler.get_last_lr(), epoch_i, 'train/lr_epoch')
            self.scheduler.step()

            # Uncomment to save parameters at checkpoint
            save_ckpt_path = str(self.config.save_dir) + f'/epoch-{epoch_i}.pkl'
            # tqdm.write(f'Save parameters at {ckpt_path}')
            torch.save(self.model.state_dict(), save_ckpt_path)

            
            # NEW! Evaluate val dataset
            val_fscore, val_mse_loss, val_mae_loss, val_logits, val_kendal_tau, val_spearman_rho = \
                self.evaluate(dataloader=self.val_loader, epoch_i=epoch_i)
            # test_summe_fscore, test_summe_mse_loss, test_summe_mae_loss, test_summe_logits, test_summe_kendal_tau, test_summe_spearman_rho = \
            #     self.evaluate(dataloader=self.test_summe_loader, epoch_i=epoch_i, eval_metric = 'max')
            # test_tvsum_fscore, test_tvsum_mse_loss, test_tvsum_mae_loss, test_tvsum_logits, test_tvsum_kendal_tau, test_tvsum_spearman_rho = \
            #     self.evaluate(dataloader=self.test_tvsum_loader, epoch_i=epoch_i, eval_metric = 'avg')
            
            self.writer.update_loss(val_fscore, epoch_i, 'val/f1_epoch(yt8m)')
            self.writer.update_loss(val_mse_loss, epoch_i, 'val/mse_loss(yt8m)')
            self.writer.update_loss(val_mae_loss, epoch_i, 'val/mae_loss(yt8m)')

            # self.writer.update_loss(test_summe_fscore, epoch_i, 'test/f1_epoch(summe)')
            # self.writer.update_loss(test_summe_mse_loss, epoch_i, 'test/mse_loss(summe)')
            # self.writer.update_loss(test_summe_mae_loss, epoch_i, 'test/mae_loss(summe)')

            # self.writer.update_loss(test_tvsum_fscore, epoch_i, 'test/f1_epoch(tvsum)')
            # self.writer.update_loss(test_tvsum_mse_loss, epoch_i, 'test/mse_loss(tvsum)')
            # self.writer.update_loss(test_tvsum_mae_loss, epoch_i, 'test/mae_loss(tvsum)')

            if best_f1_score <= val_fscore:
                best_f1_score = val_fscore
                best_model_epoch = epoch_i
                best_model_logit = val_logits
                best_ckpt_path = save_ckpt_path
                save_logit_path = str(self.config.save_dir) + f'/best_loss_logit.pt'
                torch.save(best_model_logit, save_logit_path)

            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('   YT8M--val  F-score {0:0.5} kT {1:0.5} srho {2:0.5} / Best F1 score {3:0.5f}'.format(val_fscore, val_kendal_tau, val_spearman_rho, best_f1_score))
            # print('   SumMe-test F-score {0:0.5} kT {1:0.5} srho {2:0.5}'.format(test_summe_fscore, test_summe_kendal_tau, test_summe_spearman_rho))
            # print('   TVSum-test F-score {0:0.5} kT {1:0.5} srho {2:0.5}'.format(test_tvsum_fscore, test_tvsum_kendal_tau, test_tvsum_spearman_rho))
            
            # NEW! Evaluate test dataset
            test_fscore, test_mse_loss, test_mae_loss, test_logits, test_kendal_tau, test_spearman_rho = \
                self.evaluate(dataloader=self.test_loader, epoch_i=epoch_i)
            
            self.writer.update_loss(test_fscore, epoch_i, 'test/f1_epoch(yt8m)')
            self.writer.update_loss(test_mse_loss, epoch_i, 'test/mse_loss(yt8m)')
            self.writer.update_loss(test_mae_loss, epoch_i, 'test/mae_loss(yt8m)')

            if best_test_f1_score <= test_fscore:
                best_test_f1_score = test_fscore
                best_model_epoch = epoch_i
                best_model_logit = test_logits
                best_ckpt_path = save_ckpt_path
                save_logit_path = str(self.config.save_dir) + f'/best_test_loss_logit.pt'
                torch.save(best_model_logit, save_logit_path)

            print("   [Epoch {0}] Train loss: {1:.05f}".format(epoch_i, loss))
            print('   YT8M--test  F-score {0:0.5} kT {1:0.5} srho {2:0.5} / Best Test F1 score {3:0.5f}'.format(test_fscore, test_kendal_tau, test_spearman_rho, best_test_f1_score))

        print('   Best Val F1 score {0:0.5f} @ epoch{1} (range:[0, n_epochs-1])'.format(float(best_f1_score), best_model_epoch))

        return best_ckpt_path

    def evaluate(self, epoch_i, eval_metric='max', dataloader=None, model=None):
        """ Saves the frame's importance scores for the test videos in json format.

        :param int epoch_i: The current training epoch.
        """
        # if model == None : model = self.model
        # if dataloader == None : dataloader = self.val_loader
        self.model.eval()
        
        fscore_history = []
        mse_loss_history = []
        mae_loss_history = []
        tau_history = []
        rho_history = []

        val_logits = {}

        dataloader = iter(dataloader)

        for data in tqdm(dataloader):
        # for data in tqdm(dataloader, desc='Evaluate', ncols=80, leave=False):
            video_name = data['video_name']
            frame_features = data['features'].to(self.config.device)
            target = data['gtscore'].to(self.config.device)

            if len(frame_features.shape) == 2:
                seq = seq.unsqueeze(0)
            if len(target.shape) == 1:
                target = target.unsqueeze(0)

            B = frame_features.shape[0]
            mask=None
            if 'mask' in data:
                mask = data['mask'].to(self.config.device)

            with torch.no_grad():
                scores, attn_weights = self.model(frame_features, mask=mask)
                for i in range(B):
                    val_logits[video_name[i]] = scores[i].cpu()

                val_mse_loss = self.criterion(scores[mask], target[mask]).mean()
                val_mae_loss = self.mae_criterion(scores[mask], target[mask]).mean()
                
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


    def test(self, ckpt_path):
        if ckpt_path != None:
            print("Testing Model ", ckpt_path)
            print(self.config.device)
            self.model.load_state_dict(torch.load(ckpt_path))
        
        test_fscore, test_mse_loss, test_mae_loss, test_logits, test_kendal_tau, test_spearman_rho = self.evaluate(dataloader=self.test_loader, epoch_i=0, eval_metric='avg')

        print("------------------------------------------------------")
        print("   TEST RESULT: ")
        print('   TEST YT8M F-score {0:0.5} kT {1:0.5} srho {2:0.5}'.format(test_fscore, test_kendal_tau, test_spearman_rho))
        print("------------------------------------------------------")
        f = open(str(self.config.root_dir) + '/results.txt', 'a+')
        f.write("Testing on Model " + ckpt_path + '\n')
        f.write(self.config.video_type +  '\tsplit' + str(self.config.split_index) + ', Test F-score ' + str(test_fscore) + '\n')
        f.write(self.config.video_type +  '\tsplit' + str(self.config.split_index) + ', Test MSE Loss ' + str(test_mse_loss) + '\n')
        f.write(self.config.video_type +  '\tsplit' + str(self.config.split_index) + ', Test MAE Loss ' + str(test_mae_loss) + '\n')
        f.write(self.config.video_type +  '\tsplit' + str(self.config.split_index) + ', Test kendal Tau' + str(test_kendal_tau) + '\n')
        f.write(self.config.video_type +  '\tsplit' + str(self.config.split_index) + ', Test spearman Rho' + str(test_spearman_rho) + '\n')
        f.flush()


if __name__ == '__main__':
    pass
