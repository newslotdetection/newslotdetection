import copy
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from losses import loss_map
from pytorch_pretrained_bert.tokenization import BertTokenizer
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             silhouette_score)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from utils.functions import (eval_and_save, restore_model, restore_model_as_dict, save_model,
                             save_results)
from utils.metrics import clustering_score

from .pretrain import PretrainModelManager
from .pretrain_parser_ori import PretrainModelManagerParserOri
#from .query_strategies import *
from .strategy import *
import sys

class ModelManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)
        self.model = model.model
        self.optimizer = model.optimizer
        self.device = model.device
        self.args = args
        self.select_size = int(data.dataloader.num_train_examples * self.args.select_ratio)

        # get all types of data
        self.Data = data
        self.eval_dataloader = data.dataloader.eval_loader # slu
        self.test_dataloader_parser = data.dataloader.test_loader_parser # parser
        self.test_dataloader_slu = data.dataloader.test_loader_slu # slu
        self.train_labeled_Feature = data.dataloader.train_labeled_Feature  # parser_gt

        self.unlabeled_Feature_parser = data.dataloader.train_unlabeled_Feature_parser # parser
        self.unlabeled_Feature_parser_gt = data.dataloader.train_unlabeled_Feature_parser_gt # parser_gt

        self.train_labeled_loader = data.dataloader.train_labeled_loader # parser_gt
        self.train_unlabeled_loader_parser = data.dataloader.train_unlabeled_loader_parser # parser

        # set the loss function
        self.loss_fct = loss_map[args.loss_fct]

        # get the slu label map
        self.slu_label_map = {}
        for i, label in enumerate(data.known_label_list): # slu labels
            self.slu_label_map[label] = i
        # self.label_map = self.known_label_map

        # self.classifier_num_label = len(self.slu_label_map)

        # get the known slu labels from train_labeled_loader
        self.known_label_set = set()
        for step, batch in enumerate(self.train_labeled_loader):
            _, _, _, _, _, label_ids = batch
            self.known_label_set.update(label_ids.numpy().tolist())
        self.label_mask = [1 if idx in self.known_label_set else 0 for idx in range(len(self.slu_label_map))]

        self.pretrain_manager_parser = PretrainModelManagerParserOri(args, data, model)  
        pretrain_manager_parser = self.pretrain_manager_parser
        
        if args.pre_train_parser:
            self.logger.info('Pre-training parser start...')
            pretrain_manager_parser.train(args, data)
            self.logger.info('Pre-training parser finished...')
            self.model = pretrain_manager_parser.model

        if args.pre_train:

            '''
            self.logger.info('Pre-training parser start...')
            pretrain_manager_parser.train(args, data)
            self.logger.info('Pre-training parser finished...')
            self.model = pretrain_manager_parser.model
            '''
            self.logger.info('restore_model pre-training model......')

            # instantiate a new model with the same structure and data as previous
            self.pretrain_manager = PretrainModelManager(args, data, model)  
            pretrain_manager = self.pretrain_manager

            self.logger.info('Pre-training on labeled data start...')

            # # load all the parameters except the classification layer in pre_train_parser to classification model
            # self.logger.info('load_pretrained_model without the cls layer......')
            # pretrained_dict = restore_model_as_dict(pretrain_manager_parser.model, os.path.join(args.method_output_dir, 'pretrain_parser'))
            # classifier_params = ['classifier.weight', 'classifier.bias', 'classifier_slu.weight', 'classifier_slu.bias', 'classifier_ori.weight', 'classifier_ori.bias']
            # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
            # pretrain_manager.model.load_state_dict(pretrained_dict, strict=False)

            # self.num_labels = len(self.slu_label_map)
            self.logger.info('Known labels when pre-training: %s/%s', len(self.known_label_set), len(self.slu_label_map))

            pretrain_manager.train(args, data, self.label_mask)
            self.logger.info('Pre-training finished...')
            self.model = pretrain_manager.model

            self.logger.info('Pre-training eval...')

            tagging_model = self.model 
            labeled_data = data.dataloader.train_labeled_Feature # parser_gt
            unlabeled_data = data.dataloader.train_unlabeled_Feature_parser # parser
            unlabeled_data_parser_gt = data.dataloader.train_unlabeled_Feature_parser # parser
                
            label_map = self.slu_label_map
            sampler = RandomSampling(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)           
            sampler.evaluate(self.test_dataloader_parser, self.test_dataloader_slu, self.args.labeled_ratio*100)            
                

        if args.train:
            self.pretrain_manager = PretrainModelManager(args, data, model)  
            pretrain_manager=self.pretrain_manager
            self.logger.info('restore_model pre-training model...')
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            # self.num_labels = len(self.slu_label_map)
            self.logger.info('Known labels when training (starting): %s/%s', len(self.known_label_set), len(self.slu_label_map))
            self.model = self.pretrained_model  # with the cls parm
            # self.load_pretrained_model(self.pretrained_model)  # omit the cls param
        # else:
        #    self.logger.info('restore_model...')
        #    self.model = restore_model(self.model, args.model_output_dir)

        if args.pre_train_parser_multitask:
            self.logger.info('MULTI-TASK: Pre-training parser start...')
            pretrain_manager_parser.train(args, data)
            self.logger.info('MULTI-TASK: Pre-training parser finished...')
            self.model = pretrain_manager_parser.model

        if args.pre_train_multitask:

            self.logger.info('MULTI-TASK: restore_model pre-training model......')

            # instantiate a new model with the same structure and data as previous
            self.pretrain_manager = PretrainModelManager(args, data, model)
            pretrain_manager = self.pretrain_manager

            self.logger.info('MULTI-TASK: Pre-training model on labeled data start...')

            # # load all the parameters except the classification layer in pre_train_parser to classification model
            # self.logger.info('MULTI-TASK: load_pretrained_model without the cls layer......')
            # pretrained_dict = restore_model_as_dict(pretrain_manager_parser.model,
            #                                  os.path.join(args.method_output_dir, 'pretrain_parser'))
            # classifier_params = ['classifier.weight', 'classifier.bias', 'classifier_slu.weight', 'classifier_slu.bias', 'classifier_ori.weight', 'classifier_ori.bias']
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
            # pretrain_manager.model.load_state_dict(pretrained_dict, strict=False)

            # self.num_labels = len(self.slu_label_map)
            self.logger.info('MULTI-TASK: Known labels when pre-training: %s/%s', len(self.known_label_set), len(self.slu_label_map))

            pretrain_manager.train(args, data, self.label_mask)
            self.logger.info('MULTI-TASK: Pre-training finished...')
            self.model = pretrain_manager.model

            self.logger.info('MULTI-TASK: Pre-training eval...')

            tagging_model = self.model
            labeled_data = data.dataloader.train_labeled_Feature  # parser_gt
            unlabeled_data = data.dataloader.train_unlabeled_Feature_parser  # parser
            unlabeled_data_parser_gt = data.dataloader.train_unlabeled_Feature_parser  # parser

            label_map = self.slu_label_map
            sampler = RandomSampling(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map,
                                     unlabeled_data_parser_gt)
            sampler.evaluate(self.test_dataloader_parser, self.test_dataloader_slu, self.args.labeled_ratio * 100)

        if args.train_multitask:
            self.pretrain_manager = PretrainModelManager(args, data, model)
            pretrain_manager = self.pretrain_manager
            self.logger.info('MULTI-TASK: restore model pre-training model...')
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            # self.num_labels = len(self.slu_label_map)
            self.logger.info('MULTI-TASK: Known labels when training (starting): %s/%s', len(self.known_label_set), len(self.slu_label_map))
            self.model = self.pretrained_model  # with the cls parm

        if args.resume:
            self.pretrain_manager = PretrainModelManager(args, data, model)
            pretrain_manager = self.pretrain_manager
            pretrained_model = pretrain_manager.model
            self.logger.info('restore trained model...')

            trained_model_dir = os.path.join(args.method_output_dir, 'train')
            trained_model_path = os.path.join(trained_model_dir, "trained.bin")
            pretrained_model.load_state_dict(torch.load(trained_model_path))

            self.model = pretrained_model  # with the cls parm

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

    def train(self, args, data):

        tagging_model = self.model

        # if self.args.resume:
        #     resume_dir = os.path.join(args.method_output_dir, "train")
        #     labeled_data = torch.load(os.path.join(resume_dir, "labeled.pt"))
        #     unlabeled_data = torch.load(os.path.join(resume_dir, "unlabeled.pt"))
        #     unlabeled_data_parser_gt = torch.load(os.path.join(resume_dir, "unlabeled.pt"))
        # else:
        #     labeled_data = data.dataloader.train_labeled_Feature # parser_gt
        #     unlabeled_data = data.dataloader.train_unlabeled_Feature_parser # parser
        #     unlabeled_data_parser_gt = data.dataloader.train_unlabeled_Feature_parser # parser

        labeled_data = data.dataloader.train_labeled_Feature  # parser_gt
        unlabeled_data = data.dataloader.train_unlabeled_Feature_parser  # parser
        unlabeled_data_parser_gt = data.dataloader.train_unlabeled_Feature_parser  # parser
              
        label_map = self.slu_label_map
        
        method_name = self.args.strategy
        if method_name == 'EntropySampling':
            sampler = EntropySampling(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)
        elif method_name == 'MarginSampling':
            sampler = MarginSampling(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)
        elif method_name == 'BALDDropout':
            sampler = BALDDropout(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)
        elif method_name == 'RandomSampling':
            sampler = RandomSampling(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)
        elif method_name == 'MaximalMarginalRelevance':
            sampler = MaximalMarginalRelevance(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)
        elif method_name == 'MMR_Margin':
            sampler = MMR_Margin(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map, unlabeled_data_parser_gt)
        elif method_name == 'MMR_Margin_bothbranch':
            sampler = MMR_Margin_bothbranch(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map,
                                 unlabeled_data_parser_gt)
        elif method_name == 'TwoStepSampling':
            sampler = TwoStepSampling(self.logger, args, tagging_model, labeled_data, unlabeled_data, label_map,
                                 unlabeled_data_parser_gt)


        #sampler.evaluate(self.test_dataloader_parser, self.test_dataloader_slu, self.args.labeled_ratio*100)
        #sys.exit()
        for iter_i in range(100):

            # 1. select data
            self.logger.info('--------------1. select data')
            train_unlabeled_loader_parser = self.feature2loader(sampler.unlabeled_Feature_parser)
            #train_unlabeled_loader_parser_gt = self.feature2loader(sampler.unlabeled_data_parser_gt)
            n_unlabeled = len(train_unlabeled_loader_parser.dataset)
            
            if n_unlabeled>self.select_size:
                selected_idx = sampler.query(self.select_size, n_unlabeled, self.label_mask)
                
                #selected_idx = sampler.query(n_unlabeled, n_unlabeled)
                none_selected_idx = list( set(list(range(n_unlabeled))) - set(selected_idx) ) 
            else:
                selected_idx = list(range(n_unlabeled))
                none_selected_idx = []
                self.logger.info('All in')
                train_unlabeled_loader_parser = None            
            
            
            # 2. update data, label the unlabeled data, using self.known_label_map
            self.logger.info('--------------2. update data, label the unlabeled data,')
                # after update, the self.train_labeled_data, self.train_unlabeled data are changed
            self.logger.info('select size = %s', self.select_size)
            final_select_idx, none_selected_idx = sampler.update_dataset(self.select_size, selected_idx, n_unlabeled )
            self.logger.info('final_select_idx: %s', str(len(final_select_idx)))
            self.logger.info('none_select_idx: %s', str(len(none_selected_idx)))
            
            train_dataloader=self.feature2loader(sampler.train_labeled_Feature)
            iter_ratio = np.round(100*(len(train_dataloader.dataset)/data.dataloader.num_train_examples), 2)
            self.logger.info('iter_ratio: %s', str(iter_ratio))
            # eval_dataloader here serves as the validation set, providing gt label id of each sample

            # update known labels
            for step, batch in enumerate(train_dataloader):
                _, _, _, _, _, label_ids = batch
                self.known_label_set.update(label_ids.numpy().tolist())
            self.label_mask = [1 if idx in self.known_label_set else 0 for idx in range(len(self.slu_label_map))]
            self.logger.info('Known labels when training (iter %s): %s/%s', iter_ratio, len(self.known_label_set), len(self.slu_label_map))

            sampler.train(train_dataloader, self.eval_dataloader, self.optimizer, self.loss_fct, self.label_mask)

            sampler.evaluate(self.test_dataloader_parser, self.test_dataloader_slu, iter_ratio)
            
            num_labeled_examples = len(train_dataloader.dataset)
            self.logger.info('num_labeled_examples = %s', num_labeled_examples)
            
            try:
                num_unlabeled_examples = len(train_unlabeled_loader_parser.dataset) 
                self.logger.info('num_unlabeled_examples = %s',num_unlabeled_examples)
            except:
                self.logger.info('num_unlabeled_examples = %s', 0)
            
            if iter_ratio>30:
                # self.save_model_and_data(args, sampler.model, sampler.train_labeled_Feature, sampler.unlabeled_Feature_parser)
                break            
            if len(none_selected_idx)==0:
                break

    def save_model_and_data(self, args, trained_model, labeled_feature, unlabeled_feature):
        trained_dir = os.path.join(args.method_output_dir, 'trained')
        if not os.path.exists(trained_dir):
            os.makedirs(trained_dir)
        torch.save(trained_model.state_dict(), os.path.join(trained_dir, "trained.bin"))
        torch.save(labeled_feature, os.path.join(trained_dir, "labeled.pt"))
        torch.save(unlabeled_feature, os.path.join(trained_dir, "unlabeled.pt"))
