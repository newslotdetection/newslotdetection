import torch
import torch.nn.functional as F
import numpy as np
import os
import copy
import logging

from sklearn.metrics import accuracy_score
from tqdm import trange, tqdm

from losses import loss_map
from utils.functions import save_model



class PretrainModelManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)

        args.num_labels = data.n_known_cls
        self.logger.info('num_labels when pretraining: %s', args.num_labels)
        self.logger.info('num_pretrain_epochs: %s', args.num_pretrain_epochs)

        self.model = model.set_model(args, data, 'bert')
        self.optimizer = model.set_optimizer(self.model, len(data.dataloader.train_labeled_loader.dataset), args.train_batch_size, \
            args.num_pretrain_epochs, args.lr_pre, args.warmup_proportion)

        self.device = model.device

        self.train_dataloader = data.dataloader.train_labeled_loader  # the parser_gt data
        self.eval_dataloader = data.dataloader.eval_loader  # the slu data
        # self.test_dataloader = data.dataloader.test_loader  # the parser data

        self.loss_fct = loss_map[args.loss_fct]

    def train(self, args, data, label_mask):

        wait = 0
        best_model = None
        best_eval_score = 0

        for epoch in trange(int(args.num_pretrain_epochs), desc="Pretraining Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Pretraining Iteration")):
                #print('step: ', step)
                batch = tuple(t.to(self.device) for t in batch)
                #input_ids, input_mask, segment_ids, label_ids = batch
                input_ids, input_mask, segment_ids, bin_label_ids, labels_ori, labels_slu = batch
                with torch.set_grad_enabled(True):

                    if args.backbone == "bert_MultiTask":

                        logits_slu, logits_ori = self.model(input_ids, segment_ids, input_mask, mode = "train", loss_fct = self.loss_fct, bin_label_ids = bin_label_ids)

                        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(logits_slu.shape[0], 1).to(self.device)

                        logits_slu = logits_slu * label_masks

                        loss_slu = self.loss_fct(logits_slu, labels_slu)
                        loss_ori = self.loss_fct(logits_ori, labels_ori)

                        loss = (1 - args.alpha) * loss_slu + args.alpha * loss_ori

                    elif args.backbone == "bert":

                        logits_slu = self.model(input_ids, segment_ids, input_mask, mode = "train", loss_fct = self.loss_fct, bin_label_ids = bin_label_ids)

                        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(logits_slu.shape[0], 1).to(self.device)

                        logits_slu = logits_slu * label_masks

                        loss= self.loss_fct(logits_slu, labels_slu)

                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred = self.get_outputs(args, mode = 'eval')
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score':best_eval_score,
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch + 1))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
            
            if eval_score > best_eval_score:
                
                best_model = copy.deepcopy(self.model)
                wait = 0
                best_eval_score = eval_score

            elif eval_score > 0:

                wait += 1
                self.logger.info("***** Wait: %s:  *****", str(wait))

                if wait >= args.wait_patient:
                    break
                
        self.model = best_model

        if args.save_model:
            pretrained_model_dir = os.path.join(args.method_output_dir, 'pretrain')
            if not os.path.exists(pretrained_model_dir):
                os.makedirs(pretrained_model_dir)
            save_model(self.model, pretrained_model_dir)

    def get_outputs(self, args, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader

        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        
        for batch in tqdm(dataloader, desc="Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            #input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
            
            with torch.set_grad_enabled(False):
                pooled_output, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                
                total_labels = torch.cat((total_labels,label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits = torch.cat((total_logits, logits))

        if get_feats:  
            feats = total_features.cpu().numpy()
            return feats 

        else:
            total_probs = F.softmax(total_logits.detach(), dim=1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

 