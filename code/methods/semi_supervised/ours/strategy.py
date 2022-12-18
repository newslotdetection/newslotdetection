import argparse
import copy
import json
#from data.utils.json_utils import *
import operator
# import transformers
import os
import random
import time
from collections import Counter, OrderedDict
from re import S

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.optimization import \
    WarmupLinearSchedule as warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from sklearn.metrics import accuracy_score
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm, trange
from utils import *
from utils.functions import (eval_and_save, restore_model, save_model,
                             save_results, seed_torch)


class Strategy(object):

    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt):
        """
        current_model:
            for training, it is the init model for training
            for retraining, the model has load the ckpt of last iter
        labeled_data: the labeled data list for this iter
        unlabeled_data: the unlabeled data lsit for this iter
        """
        self.logger = logger
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = current_model.to(self.device)
        self.train_labeled_Feature = labeled_data # parser_gt
        self.unlabeled_Feature_parser = unlabeled_data # parser
        self.unlabeled_Feature_parser_gt = unlabeled_data_parser_gt # parser
        self.batch_size = args.batch_size
        #self.tokenizer = BertTokenizer.from_pretrained(self.args.pretrained_model, do_lower_case=self.args.do_lower_case)
        self.label_map = label_map
        self.label2name = {ids:names for names, ids in self.label_map.items() }
        self.num_labels = len(label_map)

    def query(self, n):
        pass

    def feature2loader(self, all_unlabeled_Feature):
        datatensor = TensorDataset(
            all_unlabeled_Feature[0], 
            all_unlabeled_Feature[1], 
            all_unlabeled_Feature[2],
            all_unlabeled_Feature[3],
            all_unlabeled_Feature[4],
            all_unlabeled_Feature[5])
        sampler = SequentialSampler(datatensor)
        dataloader = DataLoader(datatensor, sampler=sampler, batch_size = self.args.train_batch_size) 
        return dataloader

    def update_dataset(self, select_size, selected_idx, n_unlabeled):
        '''
        samps_filter: list
        add the samps into the labeled data
        remove the samps in the unlabeled data
        '''
        # get the ground truth label of the selected values,
        '''
        final_select_idx = []
        already_select_idx = []
        for idx in selected_idx:
            i=0
            already_select_idx.append(idx)
            label_parser = self.unlabeled_Feature_parser[5][idx]
            if label_parser!=-1 :
                i+=1
                final_select_idx.append(idx)
            if len(final_select_idx)==select_size:
                break
        
        
        none_selected_idx = list( set(list(range(n_unlabeled))) - set(already_select_idx) ) 

        selected_feature=[]
        for i in range(6):
            feat = self.unlabeled_Feature_parser_gt[i][final_select_idx]
            selected_feature.append(feat)
        '''
        none_selected_idx = list( set(list(range(n_unlabeled))) - set(selected_idx) ) 
        selected_feature=[]
        for i in range(6):
            feat = self.unlabeled_Feature_parser_gt[i][selected_idx]
            selected_feature.append(feat)
        self.train_labeled_Feature = [torch.cat([x,y]) for x, y in zip(self.train_labeled_Feature, selected_feature)] 
        #self.unlabeled_Feature_parser = [torch.cat([x,y]) for x, y in zip(self.unlabeled_Feature_parser, none_selected_feature)] 
        #self.unlabeled_Feature_parser = none_selected_feature
        #self.labeled_dataloader = self.feature2loader(self.train_labeled_Feature)
        
        self.unlabeled_Feature_parser = [self.unlabeled_Feature_parser[i][none_selected_idx] for i in range(6)]
        self.unlabeled_Feature_parser_gt = [self.unlabeled_Feature_parser_gt[i][none_selected_idx] for i in range(6)]

        return selected_idx, none_selected_idx
            #self.unlabeled_dataloader_parser = self.feature2loader(self.unlabeled_Feature_parser)
        
    def evaluate(self, dataloader, dataloader_gt, iter_ratio):
        """
        type:
        test: only the test set
        unlabel: all the unlabeled data
        """      

        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.args.bert_model, do_lower_case=self.args.do_lower_case)
                
        results = {}
        slot_value_hyp = {}
        slot_value_gt = {}
        self.logger.info('get_results_samp......')

        results,  slot_value_hyp, slot_value_gt = self.get_results_samp(results, slot_value_hyp, slot_value_gt, dataloader, mode = 'hyp', data_type='unlabel' )
        results,  slot_value_hyp, slot_value_gt = self.get_results_samp(results, slot_value_hyp, slot_value_gt, dataloader_gt, mode = 'gt', data_type='unlabel')
        self.logger.info('Results saved in %s', str(self.args.result_dir))
        #id2slot = {str(i):slot_name for i, slot_name in enumerate(self.Data.all_label_list)}
        #eval_and_save(args, self.results, self.slot_value_hyp, self.slot_value_gt, data_type) #, slot_map = id2slot)
        #self.results['iter'] = str(iter)

        # save specific results in specific folders
        weighted_result = eval_and_save(self.args, results,  slot_value_hyp, slot_value_gt, "test", iter_i=str(iter_ratio)) #, slot_map = id2slot)
        #ratio = np.round(100*(self.args.labeled_ratio+(iter+1)*self.args.select_ratio), 2)

        # append results to csv
        save_results(self.args, weighted_result, iter_ratio)
        
    def get_outputs(self, dataloader, get_feats = False):
        
        self.model.eval()

        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)
        
        total_features = torch.empty((0,self.args.feat_dim)).to(self.device)
        total_logits = torch.empty((0, self.num_labels)).to(self.device)
        #total_logits = torch.empty((0, len(self.known_label_map))).to(self.device)
        
        for batch in tqdm(dataloader, desc="OURS-Get_Outputs-Iteration"):

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
            total_maxprobs, total_preds = total_probs.max(dim=1)
            
            y_pred = total_preds.cpu().numpy()
            y_true = total_labels.cpu().numpy()
            feats = total_features.cpu().numpy()
            
            return y_true, y_pred, total_probs, feats


    def get_results_samp(self, results,  slot_value_hyp, slot_value_gt , dataloader, mode='gt', data_type='unlabel', write_result=True):

        if mode=='hyp':
            _, y_pred, total_logits, feats = self.get_outputs(dataloader)
            #self.results['y_pred'] = y_pred  # cpu numpy
            #self.results['feats'] = feats  # cpu numpy
            total_maxprobs, total_preds = total_logits.max(dim = 1)
            total_maxprobs = total_maxprobs.cpu().numpy()
        samp_n = 0
        for batch in tqdm(dataloader, desc="OURS-get_results_samp-Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
            
            input_ids = input_ids.cpu().numpy()
            bin_label_ids = bin_label_ids.cpu().numpy()
            #slot_label_ids = slot_label_ids.cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            
            for i in range(input_ids.shape[0]):
                
                uttr_list = [self.tokenizer.ids_to_tokens[k] for k in input_ids[i]]

                #self.logger.info('uttr_list: %s', uttr_list)
                
                SEP_ind = uttr_list.index('[SEP]')
                uttr = ' '.join(uttr_list[1:SEP_ind])
                
                bin_label = np.array(bin_label_ids[i])
                indices = np.nonzero(bin_label)[0]
                value = ' '.join([uttr_list[int(ind)] for ind in indices])
                
                #self.logger.info('uttr: %s', uttr)
                #self.logger.info('value: %s', value)
                
                if mode=='hyp':
                    slot = y_pred[samp_n]
                    #slot = str(slot)
                    #prob = total_maxprobs[samp_n]
                else:
                    slot = label_ids[i] # slot id

                id2slot = {i:slot_name for slot_name, i in self.label_map.items()}
                
                slot = id2slot[slot]     # slot name
                if write_result:
                    if not uttr in results.keys():
                        results[uttr] = {'hyp': {}, 'gt': {}}
                    
                    results[uttr][mode][value] = slot
                
                if mode=='hyp':
                    if not slot in slot_value_hyp.keys():
                        slot_value_hyp[slot] = [] # set()
                    # self.slot_value_hyp[slot].add(value)
                    slot_value_hyp[slot].append(value)
                else:
                    if not slot in slot_value_gt.keys():
                        slot_value_gt[slot] = [] # set()
                    # self.slot_value_gt[slot].add(value)
                    slot_value_gt[slot].append(value)
                del value
                samp_n +=1 
        return results, slot_value_hyp, slot_value_gt
    
    def set_optimizer(self, model, num_train_examples, train_batch_size, num_train_epochs, lr, warmup_proportion):

        num_train_optimization_steps = int(num_train_examples / train_batch_size) * num_train_epochs

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = BertAdam(optimizer_grouped_parameters,
                        lr = lr,
                        warmup = warmup_proportion,
                        t_total = num_train_optimization_steps)  
        return optimizer        

    def train(self, train_dataloader, dev_dataloader, optimizer, loss_fct, label_mask):
        
        # train
        wait = 0
        best_model = None
        best_eval_score = 0


        optimizer = self.set_optimizer(self.model, len(train_dataloader.dataset), self.args.train_batch_size, \
                self.args.fine_tune_epoch, self.args.lr, 0) 

        #self.evaluate(test_dataloader, test_dataloader_slu, ind_iter)
        for epoch in trange(self.args.fine_tune_epoch, desc="Finetune Epoch"):
            
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="OURS-finetune-Training")):
                #print('step: ', step)
                batch = tuple(t.to(self.device) for t in batch)
                #input_ids, input_mask, segment_ids, label_ids = batch
                input_ids, input_mask, segment_ids, bin_label_ids, labels_ori, labels_slu = batch
                with torch.set_grad_enabled(True):

                    if self.args.backbone == "bert_MultiTask":
                        logits_slu, logits_ori = self.model(input_ids, segment_ids, input_mask, mode = "train", loss_fct = loss_fct, bin_label_ids = bin_label_ids)
                        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(logits_slu.shape[0], 1).to(self.device)
                        logits_slu = logits_slu * label_masks
                        loss_slu = loss_fct(logits_slu, labels_slu)
                        loss_ori = loss_fct(logits_ori, labels_ori)
                        loss = (1 - self.args.alpha) * loss_slu + self.args.alpha * loss_ori

                    elif self.args.backbone == "bert":
                        logits_slu = self.model(input_ids, segment_ids, input_mask, mode = "train", loss_fct = loss_fct, bin_label_ids = bin_label_ids)
                        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(logits_slu.shape[0], 1).to(self.device)
                        logits_slu = logits_slu * label_masks
                        loss = loss_fct(logits_slu, labels_slu)

                    loss.backward()
                    tr_loss += loss.item()
                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

                    optimizer.step()
                    optimizer.zero_grad()
            
            loss = tr_loss / nb_tr_steps
            
            y_true, y_pred, _, _ = self.get_outputs(dev_dataloader)
            eval_score = round(accuracy_score(y_true, y_pred) * 100, 2)

            eval_results = {
                'train_loss': loss,
                'eval_score': eval_score,
                'best_score': best_eval_score,
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
                if wait >= self.args.wait_patient:
                    break

        self.model = best_model                    
        self.logger.info('finish training...')

    def predict_prob(self):
        self.model.eval()
        total_logits = torch.empty((0, self.model.num_labels)).to(self.device)
        dataloader = self.feature2loader(self.unlabeled_Feature_parser)
        for batch in tqdm(dataloader, desc="predict_prob-Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
            with torch.set_grad_enabled(False):
                _, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                total_logits = torch.cat((total_logits, logits))
        probs = F.softmax(total_logits.detach(), dim=1)
        return probs


    def predict_prob_dropout_split(self, n_drop=10):

        #probs = self.model.predict_prob_dropout_split(data, n_drop=n_drop)
        # n_drop : the number of repeat for dropout
        self.model.train()  ## dropout as variational Bayesian approximation
        dataloader = self.feature2loader(self.unlabeled_Feature_parser)
        probs = torch.zeros([n_drop, len(dataloader.dataset), self.model.num_labels]).to(self.device)
        
        for k in range(n_drop):
            seed_torch(seed=k)
            i=0
            for batch in tqdm(dataloader, desc="predict_prob-Iteration"):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
                with torch.set_grad_enabled(False):
                    _, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                    prob = F.softmax(logits, dim=-1)
                      
                    if (i + 1) * dataloader.batch_size <= len(dataloader.dataset):
                        probs[k][dataloader.batch_size * i:dataloader.batch_size * (i + 1)] = prob
                    else:
                        probs[k][dataloader.batch_size * i:] = prob
                    i+=1        
        
        return probs


    def get_outputs_bothbranch(self, dataloader, get_feats=False):

        self.model.eval()

        total_labels_ori = torch.empty(0, dtype=torch.long).to(self.device)

        total_features = torch.empty((0, self.args.feat_dim)).to(self.device)
        total_logits_slu = torch.empty((0, self.num_labels)).to(self.device)
        total_logits_ori = torch.empty((0, self.args.num_labels_parser)).to(self.device)
        # total_logits = torch.empty((0, len(self.known_label_map))).to(self.device)

        for batch in tqdm(dataloader, desc="OURS-Get_Outputs-Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                pooled_output, logits_slu, logits_ori = self.model(input_ids, segment_ids, input_mask, mode="bothbranch", bin_label_ids=bin_label_ids)

                total_labels_ori = torch.cat((total_labels_ori, slot_label_ids))
                total_features = torch.cat((total_features, pooled_output))
                total_logits_slu = torch.cat((total_logits_slu, logits_slu))
                total_logits_ori = torch.cat((total_logits_ori, logits_ori))

        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        else:
            y_true_ori = total_labels_ori.cpu().numpy()
            total_probs_slu = F.softmax(total_logits_slu.detach(), dim=1)
            total_probs_ori = F.softmax(total_logits_ori.detach(), dim=1)
            feats = total_features.cpu().numpy()

            return total_probs_slu, total_probs_ori, feats, y_true_ori


class RandomSampling(Strategy):  ## return selected samples (n pieces) for current iteration
    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier ):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def query(self, n, n_unlabeled, label_mask):
        return np.random.choice(n_unlabeled, n, replace=False)


class MarginSampling(Strategy):

    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def query(self, n, n_unlabeled, label_mask):
        probs = self.predict_prob()
        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(probs.shape[0], 1).to(self.device)
        probs = probs * label_masks
        probs_sorted, idxs = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 0] - probs_sorted[:,1]
        _, top_n_index = torch.topk(uncertainties, n, largest=False)  #small to large
        return top_n_index.tolist()


class EntropySampling(Strategy):

    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def query(self, n, n_unlabeled, label_mask):
        probs = self.predict_prob()
        log_probs = torch.log(probs)
        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(probs.shape[0], 1).to(self.device)
        uncertainties = (probs*log_probs*label_masks).sum(1)
        _, top_n_index = torch.topk(uncertainties, n, largest=False)  # small to large
        return top_n_index.tolist()
        


class BALDDropout(Strategy):
    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)
        self.n_drop = 5

    def query(self, n, n_unlabeled, label_mask):
        probs = self.predict_prob_dropout_split(n_drop=5)
        label_masks_2d = torch.Tensor(label_mask).unsqueeze(0).repeat(probs.shape[1], 1).to(self.device)
        label_masks_3d = torch.Tensor(label_mask).reshape(1, 1, -1).repeat(probs.shape[0], probs.shape[1], 1).to(self.device)
        pb = probs.mean(0)
        entropy1 = (-pb*torch.log(pb)*label_masks_2d).sum(1)
        entropy2 = (-probs*torch.log(probs)*label_masks_3d).sum(2).mean(0)
        uncertainties = entropy2 - entropy1
        _, top_n_index = torch.topk(uncertainties, n, largest=False)  #small to large
        return top_n_index.tolist()


class MaximalMarginalRelevance(Strategy):
    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def get_cos_similar_matrix(self, m1, m2):
        num = np.dot(m1, m2.T)
        denom = np.linalg.norm(m1, axis=1).reshape(-1, 1) * np.linalg.norm(m2, axis=1)
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res

    def query(self, n, n_unlabeled, label_mask):

        # Score_Si = beta * (UNCERTAINTY(U_Si)) + (1 - beta) * {max[DISTINCTION(Labeled_S, U_Si)]}
        # => Score_Si = beta * (UNCERTAINTY(U_Si)) - (1 - beta) * {max[SIMILARITY(Labeled_S, U_Si)]}

        labeled_dataloader = self.feature2loader(self.train_labeled_Feature)
        unlabeled_dataloader = self.feature2loader(self.unlabeled_Feature_parser)

        _, _, probs, unlabeled_feats = self.get_outputs(unlabeled_dataloader)
        log_probs = torch.log(probs)
        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(probs.shape[0], 1).to(self.device)
        uncertainties = -1 * (probs * log_probs * label_masks).sum(1)  # Calculate UNCERTAINTY(U_Si), better rescale to 0-1

        labeled_feats = self.get_outputs(labeled_dataloader, get_feats=True)

        # (N1, 768) * (768, N2) = (N1, N2)
        sim_matrix = self.get_cos_similar_matrix(labeled_feats, unlabeled_feats)
        sim_scores = np.max(sim_matrix, axis=0) # (N2)
        sim_scores = torch.from_numpy(sim_scores).to(self.device)

        final_scores = self.args.beta * uncertainties - (1 - self.args.beta) * sim_scores

        _, top_n_index = torch.topk(final_scores, n, largest=True)  # large to small
        return top_n_index.tolist()


class MMR_Margin(Strategy):
    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def get_cos_similar_matrix(self, m1, m2):
        num = np.dot(m1, m2.T)
        denom = np.linalg.norm(m1, axis=1).reshape(-1, 1) * np.linalg.norm(m2, axis=1)
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res

    def query(self, n, n_unlabeled, label_mask):

        # Score_Si = beta * (UNCERTAINTY(U_Si)) + (1 - beta) * {max[DISTINCTION(Labeled_S, U_Si)]}
        # => Score_Si = beta * (UNCERTAINTY(U_Si)) - (1 - beta) * {max[SIMILARITY(Labeled_S, U_Si)]}

        labeled_dataloader = self.feature2loader(self.train_labeled_Feature)
        unlabeled_dataloader = self.feature2loader(self.unlabeled_Feature_parser)

        # get margin scores
        _, _, probs, unlabeled_feats = self.get_outputs(unlabeled_dataloader)
        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(probs.shape[0], 1).to(self.device)
        probs = probs * label_masks
        probs_sorted, _ = probs.sort(descending=True)
        uncertainties = probs_sorted[:, 1] - probs_sorted[:, 0]

        # get sim_scores
        labeled_feats = self.get_outputs(labeled_dataloader, get_feats=True)
        # (N1, 768) * (768, N2) = (N1, N2)
        sim_matrix = self.get_cos_similar_matrix(labeled_feats, unlabeled_feats)
        sim_scores = np.max(sim_matrix, axis=0) # (N2)
        sim_scores = torch.from_numpy(sim_scores).to(self.device)

        final_scores = self.args.beta * uncertainties - (1 - self.args.beta) * sim_scores

        _, top_n_index = torch.topk(final_scores, n, largest=True)  # large to small
        return top_n_index.tolist()


class MMR_Margin_bothbranch(Strategy):
    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def get_cos_similar_matrix(self, m1, m2):
        num = np.dot(m1, m2.T)
        denom = np.linalg.norm(m1, axis=1).reshape(-1, 1) * np.linalg.norm(m2, axis=1)
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res

    def query(self, n, n_unlabeled, label_mask):

        # Score_Si = beta * (UNCERTAINTY(U_Si)) + (1 - beta) * {max[DISTINCTION(Labeled_S, U_Si)]}
        # => Score_Si = beta * (UNCERTAINTY(U_Si)) - (1 - beta) * {max[SIMILARITY(Labeled_S, U_Si)]}

        labeled_dataloader = self.feature2loader(self.train_labeled_Feature)
        unlabeled_dataloader = self.feature2loader(self.unlabeled_Feature_parser)

        # get margin scores
        probs_slu, probs_ori, unlabeled_feats, _ = self.get_outputs_bothbranch(unlabeled_dataloader)
        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(probs_slu.shape[0], 1).to(self.device)
        probs_slu = probs_slu * label_masks
        probs_sorted_slu, _ = probs_slu.sort(descending=True)
        probs_sorted_ori, _ = probs_ori.sort(descending=True)
        uncertainties_slu = probs_sorted_slu[:, 1] - probs_sorted_slu[:, 0]
        uncertainties_ori = probs_sorted_ori[:, 1] - probs_sorted_ori[:, 0]

        # get sim_scores
        labeled_feats = self.get_outputs_bothbranch(labeled_dataloader, get_feats=True)
        # (N1, 768) * (768, N2) = (N1, N2)
        sim_matrix = self.get_cos_similar_matrix(labeled_feats, unlabeled_feats)
        sim_scores = np.max(sim_matrix, axis=0) # (N2)
        sim_scores = torch.from_numpy(sim_scores).to(self.device)

        final_scores_slu = self.args.beta * uncertainties_slu - (1 - self.args.beta) * sim_scores
        final_scores_ori = self.args.beta * uncertainties_ori - (1 - self.args.beta) * sim_scores
        final_scores = self.args.alpha * final_scores_ori + (1 - self.args.alpha) * final_scores_slu

        _, top_n_index = torch.topk(final_scores, n, largest=True)  # large to small
        return top_n_index.tolist()


class TwoStepSampling(Strategy):
    def __init__(self, logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier):
        super().__init__(logger, args, current_model, labeled_data, unlabeled_data, label_map, classifier)

    def get_cos_similar_matrix(self, m1, m2):
        num = np.dot(m1, m2.T)
        denom = np.linalg.norm(m1, axis=1).reshape(-1, 1) * np.linalg.norm(m2, axis=1)
        res = num / denom
        res[np.isneginf(res)] = 0
        return 0.5 + 0.5 * res

    def query(self, n, n_unlabeled, label_mask):

        # Score_Si = beta * (UNCERTAINTY(U_Si)) + (1 - beta) * {max[DISTINCTION(Labeled_S, U_Si)]}
        # => Score_Si = beta * (UNCERTAINTY(U_Si)) - (1 - beta) * {max[SIMILARITY(Labeled_S, U_Si)]}

        labeled_dataloader = self.feature2loader(self.train_labeled_Feature)
        unlabeled_dataloader = self.feature2loader(self.unlabeled_Feature_parser)

        # get margin scores
        probs_slu, probs_ori, unlabeled_feats, y_ori = self.get_outputs_bothbranch(unlabeled_dataloader)
        label_masks = torch.Tensor(label_mask).unsqueeze(0).repeat(probs_slu.shape[0], 1).to(self.device)
        probs_slu = probs_slu * label_masks
        probs_sorted_slu, _ = probs_slu.sort(descending=True)
        probs_sorted_ori, _ = probs_ori.sort(descending=True)
        uncertainties_slu = probs_sorted_slu[:, 1] - probs_sorted_slu[:, 0]
        uncertainties_ori = probs_sorted_ori[:, 1] - probs_sorted_ori[:, 0]

        # step 1
        _, top_2n_index = torch.topk(uncertainties_slu, 2*n, largest=True)  # large to small

        # step 2
        top_2n_index = top_2n_index.cpu().numpy()
        uncertainties_ori = uncertainties_ori.cpu().numpy()

        top_2n_y_ori = y_ori[top_2n_index]
        top_2n_uncertainties_ori = uncertainties_ori[top_2n_index]
        y_set = list(set(top_2n_y_ori.tolist()))

        selected_positions = []
        spare_positions = []

        # print(top_2n_index)
        # print(top_2n_y_ori)
        # print(top_2n_uncertainties_ori)

        for y in y_set:
            positions_in_top_2n_index = np.argwhere(top_2n_y_ori == y)
            # print(positions_in_top_2n_index)
            positions_in_top_2n_index = positions_in_top_2n_index.reshape(-1)
            # print(positions_in_top_2n_index)
            uncertainty_scores = top_2n_uncertainties_ori[positions_in_top_2n_index]
            # print(uncertainty_scores)
            sorted_index_in_positions = np.argsort(-uncertainty_scores) # sort uncertainty scores from big to small
            # print(sorted_index_in_positions)
            first_half_index_in_positions = sorted_index_in_positions[:len(sorted_index_in_positions)//2]
            second_half_index_in_positions = sorted_index_in_positions[len(sorted_index_in_positions)//2:]
            # print(positions_in_top_2n_index)
            # print(first_half_index_in_positions)
            # print(second_half_index_in_positions)
            selected_positions += positions_in_top_2n_index[first_half_index_in_positions].tolist()
            spare_positions.append(positions_in_top_2n_index[second_half_index_in_positions].tolist())
            # print(positions_in_top_2n_index[first_half_index_in_positions].tolist())
            # print(positions_in_top_2n_index[second_half_index_in_positions].tolist())
            # print(top_2n_index[selected_positions])

        if len(selected_positions) < n:
            curr_len = len(selected_positions)
            random.shuffle(spare_positions)
            for ids in spare_positions:  # one loop is enough
                selected_positions.append(ids[0])
                curr_len += 1
                if curr_len == n:
                    break
        assert len(selected_positions) == n

        final_indices = top_2n_index[selected_positions]

        return final_indices
