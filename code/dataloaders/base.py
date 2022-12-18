import numpy as np
import os   
import random
import torch
import logging
import tensorflow as tf
from collections import Counter

from .__init__ import max_seq_lengths, backbone_loader_map, benchmark_labels

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

class DataManager:
    
    def __init__(self, args, logger_name = 'Discovery'):

        self.logger = logging.getLogger(logger_name)

        # args.max_seq_length = max_seq_lengths[args.dataset]
        args.max_seq_length = 256
        print(args.data_dir)
        self.data_dir = os.path.join(args.data_dir, args.dataset)

        # get slu labels
        self.all_label_list_ori = self.get_labels(args.dataset,'slu') # the labels in train_slu data
        cc = Counter(self.all_label_list_ori)
        sorted_cc = cc.most_common()
        self.logger.info('The counter of labels in train_slu: ')        

        for item in sorted_cc:
            self.logger.info("  %s : %s", item[0], str(item[1]))

        self.all_label_set = sorted(set(self.all_label_list_ori), key = self.all_label_list_ori.index)
        self.logger.info('origin all_label_set is  %s', len(self.all_label_set))        

        self.all_label_list = list(self.all_label_set)
        self.all_label_list.insert(0, 'noneslot')
        self.logger.info('all_label_list is   %s', len(self.all_label_list))
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
       
        self.known_label_list = self.all_label_list # all slots are known slots

        self.logger.info('The number of known slot is %s', self.n_known_cls)
        self.logger.info('The number of all slot is %s', len(self.all_label_list))
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list))
        self.logger.info('Lists of all labels are: %s', str(self.all_label_list))

        args.num_labels = self.num_labels = int(len(self.all_label_list) * args.cluster_num_factor)

        self.logger.info('num_labels int(len(self.all_label_list) * args.cluster_num_factor): %s', args.num_labels)
        
        # get parser_ori labels
        self.all_label_list_parser_ori = self.get_labels(args.dataset,'parser_ori') # the labels in train_parser_ori data
        cc = Counter(self.all_label_list_parser_ori)
        sorted_cc = cc.most_common()
        self.logger.info('The counter of labels in parser_ori: ')        

        for item in sorted_cc:
            self.logger.info("  %s : %s", item[0], str(item[1]))

        self.all_label_set_parser = sorted(set(self.all_label_list_parser_ori), key = self.all_label_list_parser_ori.index)
        self.logger.info('origin all_label_set is  %s', len(self.all_label_set_parser))        

        self.all_label_list_parser = list(self.all_label_set_parser)
        self.logger.info('all_label_list_parser is   %s', len(self.all_label_list_parser))
        self.n_known_cls_parser = round(len(self.all_label_list_parser) * args.known_cls_ratio)

        self.known_label_list_parser = self.all_label_list_parser # all slots are known slots

        self.logger.info('The number of known slot is %s', self.n_known_cls_parser)
        self.logger.info('The number of all slot is %s', len(self.all_label_list_parser))
        self.logger.info('Lists of known labels are: %s', str(self.known_label_list_parser))
        self.logger.info('Lists of all labels are: %s', str(self.all_label_list_parser))

        args.num_labels_parser = self.num_labels_parser = int(len(self.all_label_list_parser) * args.cluster_num_factor)
        self.logger.info('num_labels int(len(self.all_label_list_parser) * args.cluster_num_factor): %s', args.num_labels_parser)
        
        
        self.dataloader = self.get_loader(args, self.get_attrs())



    def get_labels(self, dataset, data_type):
        '''
        if dataset=='snips':
            labels = self._get_labels_slot(mode='train')
        else:
            labels = benchmark_labels[dataset]
        '''
        labels = self._get_labels_slot('train', data_type)

        return labels

    def _get_labels_slot(self, mode, data_type):
        #slot_dict = {}
        #slot_num=0
        labels = []
        with open(os.path.join(self.data_dir, data_type, mode, 'seq.out'), "r") as f:
            for i, line_out in enumerate(f):
                line_out = line_out.strip()  
                l2_list = line_out.split()
                
                for l in l2_list:
                    if "B" in l:
                        slot_name = l.split("-")[1]
                        labels.append(slot_name)
                        #if slot_name not in labels:
                        #    labels.append(slot_name)
                        #if slot_name not in slot_dict.keys():
                        #    slot_dict[slot_name]=int(slot_num)
                        #    slot_num = slot_num+1
        return labels

    def get_loader(self, args, attrs):
        
        dataloader = backbone_loader_map[args.backbone](args, attrs)

        return dataloader
    
    def get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs



