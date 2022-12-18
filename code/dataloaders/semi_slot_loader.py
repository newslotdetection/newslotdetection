import random
import numpy as np
import torch
import os
import csv
import sys
import logging
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import math
import spacy
import tensorflow as tf
# python3 -m spacy download en_core_web_sm
nlp = spacy.load('en_core_web_sm')

def is_number(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    n = n.strip().lower()
    return n.isdigit() or n in known_nos

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)

class SLOT_Loader:
    
    def __init__(self, args, base_attrs, logger_name = 'Discovery'):
        set_seed(args.seed)
        self.args = args
        self.logger = logging.getLogger(logger_name)
        self.logger.info("dataset %s", self.args.dataset)

        self.train_examples, self.train_labeled_examples_slu, self.train_unlabeled_examples, self.labeled_uttrs  = get_examples(args, base_attrs, 'train', "slu")
        # self.train_examples: all original training samp
        # train_unlabeled_example already del the samp in labeled_uttr
        self.eval_examples = get_examples(args, base_attrs, 'eval', "slu")
        #self.test_examples = get_examples(args, base_attrs, 'test', "slu")
        self.logger.info("Number of labeled training samples = %s", str(len(self.train_labeled_examples_slu)))
        self.logger.info("Number of unlabeled training samples = %s", str(len(self.train_unlabeled_examples)))
        self.logger.info("Number of evaluation samples = %s", str(len(self.eval_examples)))
        #self.logger.info("Number of testing samples = %s", str(len(self.test_examples)))
        
        processor = DatasetProcessor()
        train_examples_parser = processor.get_examples(base_attrs['data_dir'], "train", "parser")

        # # add examples with "noslot" from examples_parser
        self.train_labeled_examples_parser = self.refine_labeled_samples_parser(train_examples_parser, self.labeled_uttrs)
        self.logger.info("train_labeled_examples_parser = %s", str(len(self.train_labeled_examples_parser)))
        

        # combine self.train_labeled_examples and self.train_labeled_examples_parser
        self.train_labeled_loader, self.train_labeled_Feature = get_concat_loader(self.train_labeled_examples_slu, self.train_labeled_examples_parser, base_attrs['known_label_list'], args)
        #self.train_labeled_loader, self.train_labeled_Feature = get_loader(self.train_labeled_examples, args, base_attrs['known_label_list'], 'train_labeled')
        self.logger.info("Number of self.train_labeled_loader = %s", str(len(self.train_labeled_loader.dataset)))

        #self.train_unlabeled_loader, self.train_unlabeled_Feature_slu = get_loader(self.train_unlabeled_examples, args, base_attrs['all_label_list'], 'train_unlabeled')
        self.train_unlabeled_loader, self.train_unlabeled_Feature_slu = get_loader(self.train_unlabeled_examples, args, base_attrs['all_label_list'], 'labeled')
        self.train_loader, self.train_input_ids, self.train_input_mask, self.train_segment_ids, self.train_bin_label_ids, self.train_slot_label_ids= \
            get_semi_loader([self.train_labeled_examples_slu, self.train_labeled_examples_parser], self.train_unlabeled_examples, base_attrs, args)
        # train_loader: semi-loader, labeled: label_id, unlabled: label_id=-1

        self.eval_loader, _ = get_loader(self.eval_examples, args, base_attrs['known_label_list'], 'eval')
        self.eval_loader_all_label, _ = get_loader(self.eval_examples, args, base_attrs['all_label_list'], 'eval')
        #self.test_loader_slu, self.test_Feature_slu = get_loader(self.test_examples, args, base_attrs['all_label_list'], 'test')
        self.num_train_examples = len(self.train_examples) # all train samp

        #self.all_unlabeled_loader_slu,_ = get_concat_loader(self.train_unlabeled_examples, self.test_examples, base_attrs['all_label_list'], args)
        self.all_unlabeled_loader_slu = self.train_unlabeled_loader

        self.train_unlabeled_examples_parser = [example for example in train_examples_parser if not ' '.join(example.words) in self.labeled_uttrs.keys() ]
        #self.test_examples_parser = get_examples(args, base_attrs, 'test', "parser")
        #self.test_loader_parser, self.test_Feature_parser = get_loader(self.test_examples_parser, args, base_attrs['all_label_list'], 'test')
        self.train_unlabeled_loader_parser, self.train_unlabeled_Feature_parser = get_loader(self.train_unlabeled_examples_parser, args, base_attrs['all_label_list'], 'train_unlabeled')
        
        #self.test_loader = self.test_loader_parser
        self.all_unlabeled_loader_parser = self.train_unlabeled_loader_parser
        self.all_unlabeled_Feature_parser = self.train_unlabeled_Feature_parser
        #self.all_unlabeled_loader_parser,_ = get_concat_loader(self.train_unlabeled_examples_parser, self.test_examples_parser, base_attrs['all_label_list'], args)
        #self.all_unlabeled_Feature_parser = [torch.cat([x,y]) for x, y in zip(self.train_unlabeled_Feature_parser, self.test_Feature_parser)]
    
        self.logger.info("Number of all_unlabeled_loader_parser = %s", str(len(self.all_unlabeled_loader_parser.dataset)))
        #self.logger.info("Number of test_loader_parser = %s", str(len(self.test_loader_parser.dataset)))

        # there is no nee for test_loader
        self.test_loader = self.eval_loader


    def refine_labeled_samples_parser(self, train_examples_parser, labeled_uttrs):
        examples = []
        values_none = []
        values_slu = list(set([v for value in labeled_uttrs.values() for v in value]))
        values_slu.extend(['cheap', 'expensive', 'number', 'numbers'])
        self.logger.info('list of values_slu is %s', str(list(set(values_slu))))

        for example in train_examples_parser:
            uttr = ' '.join(example.words)
            value = ' '.join(example.words_value)
            #if uttr in labeled_uttrs.keys() and not value in labeled_uttrs[uttr]:
            if uttr in labeled_uttrs.keys() and not value in values_slu:
                #print(example.instance_labels)
                if self.args.dataset=="atis":
                    try:
                        ent_label = nlp(value).ents[0].label_
                    except:
                        ent_label="none"
                    if is_number(value) or ent_label=="DATE" or "pm" in value or "am" in value:
                        continue
                    else:
                        example.instance_labels = 'noslot'
                        values_none.append(value)
                        examples.append(example) 
                else:
                    example.instance_labels = 'noslot'
                    values_none.append(value)
                    examples.append(example)                        
        self.logger.info('list of values_none is %s', str(list(set(values_none))))
        return examples


def get_examples(args, base_attrs, mode, data_type):

    processor = DatasetProcessor()
    ori_examples = processor.get_examples(base_attrs['data_dir'], mode, data_type)
    
    ori_examples_filter = []
    for example in ori_examples: 
        if (example.instance_labels in base_attrs['all_label_list']):
            ori_examples_filter.append(example)        
    ori_examples = ori_examples_filter


    if mode == 'train':
        labeled_uttrs = {} # {'uttr': [value1, value2...]}
        train_labels = np.array([example.instance_labels for example in ori_examples])
        train_labeled_ids = []

        train_known_label_num = {}
        for label in base_attrs['known_label_list']:
            #num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
            num = math.ceil(len(train_labels[train_labels == label]) * args.labeled_ratio)
            pos = list(np.where(train_labels == label)[0])
            train_labeled_ids.extend(random.sample(pos, num))
            train_known_label_num[label] = num
        print('train_known_label_num : ', train_known_label_num )

        '''
        train_all_label_num = {}
        for label in base_attrs['all_label_list']:
            #num = round(len(train_labels[train_labels == label]) * args.labeled_ratio)
            num = math.ceil(len(train_labels[train_labels == label]) * args.labeled_ratio)
            pos = list(np.where(train_labels == label)[0])
            train_labeled_ids.extend(random.sample(pos, num))
            train_all_label_num[label] = num
        print('train_all_label_num : ', train_all_label_num )
        '''
        
        labeled_examples, unlabeled_examples = [], []
        for idx, example in enumerate(ori_examples):
            if idx in train_labeled_ids:
                labeled_examples.append(example)
                uttr = ' '.join(example.words)
                value = ' '.join(example.words_value)
                if not uttr in labeled_uttrs:
                    labeled_uttrs[uttr] = []
                labeled_uttrs[uttr].append(value)
            else:
                #print(example.words)
                if not ' '.join(example.words) in labeled_uttrs.keys():
                    unlabeled_examples.append(example)

        return ori_examples, labeled_examples, unlabeled_examples, labeled_uttrs

    elif mode == 'eval':

        examples = []
        for example in ori_examples:
            if (example.instance_labels in base_attrs['known_label_list']):
                examples.append(example)
        
        return examples
    
    elif mode == 'test':
        return ori_examples

def get_loader(examples, args, label_list, mode):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
    features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    bin_label_ids = torch.tensor([f.bin_label_ids for f in features], dtype=torch.long)
    slot_label_ids = torch.tensor([f.slot_label_ids for f in features], dtype=torch.long)

    if mode == 'train_unlabeled':
        label_ids = torch.tensor([-1 for f in features], dtype=torch.long)
    else:
        label_ids = torch.tensor([f.instance_label_ids for f in features], dtype=torch.long)

    datatensor = TensorDataset(input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids)

    if mode == 'train_labeled':  
        sampler = RandomSampler(datatensor)
        dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size)    

    else:
        sampler = SequentialSampler(datatensor)

        if mode == 'train_unlabeled':
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size)    

        elif mode == 'eval':
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.eval_batch_size)    
        
        elif mode == 'test':
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.test_batch_size) 
        else:
            dataloader = DataLoader(datatensor, sampler=sampler, batch_size = args.train_batch_size) 


    return dataloader, [input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids]

def get_semi_loader(labeled_examples_list, unlabeled_examples, base_attrs, args):

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
    labeled_features_list = [convert_examples_to_features(labeled_examples, base_attrs['known_label_list'], args.max_seq_length, tokenizer) for labeled_examples in labeled_examples_list]
    unlabeled_features = convert_examples_to_features(unlabeled_examples, base_attrs['all_label_list'], args.max_seq_length, tokenizer)

    unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
    unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
    unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
    unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)     
    unlabeled_bin_label_ids = torch.tensor([f.bin_label_ids for f in unlabeled_features], dtype=torch.long)
    unlabeled_slot_label_ids = torch.tensor([f.slot_label_ids for f in unlabeled_features], dtype=torch.long)


    labeled_input_ids = []
    labeled_input_mask = []
    labeled_segment_ids = []
    labeled_label_ids = []
    labeled_bin_label_ids = []
    labeled_slot_label_ids = []

    for labeled_features in labeled_features_list:
        labeled_input_ids_i = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
        labeled_input_mask_i = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
        labeled_segment_ids_i = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
        labeled_label_ids_i = torch.tensor([f.instance_label_ids for f in labeled_features], dtype=torch.long)      
        labeled_bin_label_ids_i = torch.tensor([f.bin_label_ids for f in labeled_features], dtype=torch.long)
        labeled_slot_label_ids_i = torch.tensor([f.slot_label_ids for f in labeled_features], dtype=torch.long)

        labeled_input_ids.append(labeled_input_ids_i)
        labeled_input_mask.append(labeled_input_mask_i)
        labeled_segment_ids.append(labeled_segment_ids_i)
        labeled_label_ids.append(labeled_label_ids_i.unsqueeze(1))
        labeled_bin_label_ids.append(labeled_bin_label_ids_i)
        labeled_slot_label_ids.append(labeled_slot_label_ids_i)

    semi_input_ids = torch.cat([torch.vstack(labeled_input_ids), unlabeled_input_ids])
    semi_input_mask = torch.cat([torch.vstack(labeled_input_mask), unlabeled_input_mask])
    semi_segment_ids = torch.cat([torch.vstack(labeled_segment_ids), unlabeled_segment_ids])
    semi_label_ids = torch.cat([torch.vstack(labeled_label_ids), unlabeled_label_ids.unsqueeze(1)]).squeeze()
    semi_bin_label_ids = torch.cat([torch.vstack(labeled_bin_label_ids), unlabeled_bin_label_ids])
    semi_slot_label_ids = torch.cat([torch.vstack(labeled_slot_label_ids), unlabeled_slot_label_ids])

    semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_bin_label_ids, semi_slot_label_ids, semi_label_ids)
    semi_sampler = SequentialSampler(semi_data)
    semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.train_batch_size) 

    return semi_dataloader, semi_input_ids, semi_input_mask, semi_segment_ids, semi_bin_label_ids, semi_slot_label_ids

def get_concat_loader(labeled_examples, labeled_examples2, label_list, args):
    
        
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)    
    labeled_features = convert_examples_to_features(labeled_examples, label_list, args.max_seq_length, tokenizer)
    unlabeled_features = convert_examples_to_features(labeled_examples2, label_list, args.max_seq_length, tokenizer)

    labeled_input_ids = torch.tensor([f.input_ids for f in labeled_features], dtype=torch.long)
    labeled_input_mask = torch.tensor([f.input_mask for f in labeled_features], dtype=torch.long)
    labeled_segment_ids = torch.tensor([f.segment_ids for f in labeled_features], dtype=torch.long)
    labeled_bin_label_ids = torch.tensor([f.bin_label_ids for f in labeled_features], dtype=torch.long)
    labeled_slot_label_ids = torch.tensor([f.slot_label_ids for f in labeled_features], dtype=torch.long)
    
    labeled_label_ids = torch.tensor([f.instance_label_ids for f in labeled_features], dtype=torch.long)      
    '''
    if mode=='slu':
        labeled_label_ids = torch.tensor([f.instance_label_ids for f in labeled_features], dtype=torch.long)      
    else:
        labeled_label_ids = torch.tensor([-1 for f in labeled_features], dtype=torch.long)      
    '''

    unlabeled_input_ids = torch.tensor([f.input_ids for f in unlabeled_features], dtype=torch.long)
    unlabeled_input_mask = torch.tensor([f.input_mask for f in unlabeled_features], dtype=torch.long)
    unlabeled_segment_ids = torch.tensor([f.segment_ids for f in unlabeled_features], dtype=torch.long)
    unlabeled_bin_label_ids = torch.tensor([f.bin_label_ids for f in unlabeled_features], dtype=torch.long)
    unlabeled_slot_label_ids = torch.tensor([f.slot_label_ids for f in unlabeled_features], dtype=torch.long)
    
    unlabeled_label_ids = torch.tensor([f.instance_label_ids for f in unlabeled_features], dtype=torch.long)     

    '''
    if mode=='slu':
        unlabeled_label_ids = torch.tensor([f.instance_label_ids for f in unlabeled_features], dtype=torch.long)     
    else:
        unlabeled_label_ids = torch.tensor([-1 for f in unlabeled_features], dtype=torch.long)     
    '''
    semi_input_ids = torch.cat([labeled_input_ids, unlabeled_input_ids])
    semi_input_mask = torch.cat([labeled_input_mask, unlabeled_input_mask])
    semi_segment_ids = torch.cat([labeled_segment_ids, unlabeled_segment_ids])
    semi_label_ids = torch.cat([labeled_label_ids, unlabeled_label_ids])
    semi_bin_label_ids = torch.cat([labeled_bin_label_ids, unlabeled_bin_label_ids])
    semi_slot_label_ids = torch.cat([labeled_slot_label_ids, unlabeled_slot_label_ids])

    semi_data = TensorDataset(semi_input_ids, semi_input_mask, semi_segment_ids, semi_bin_label_ids, semi_slot_label_ids, semi_label_ids)
    semi_sampler = SequentialSampler(semi_data)
    semi_dataloader = DataLoader(semi_data, sampler=semi_sampler, batch_size = args.test_batch_size) 

    return semi_dataloader, [semi_input_ids, semi_input_mask, semi_segment_ids, semi_bin_label_ids, semi_slot_label_ids, semi_label_ids]


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, words, words_value, bin_labels, slot_labels, instance_labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.words_value = words_value
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.bin_labels = bin_labels
        self.slot_labels = slot_labels
        self.instance_labels = instance_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, instance_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.bin_label_ids = bin_label_ids
        self.slot_label_ids = slot_label_ids
        self.instance_label_ids = instance_label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines
    
    @classmethod
    def _read_seq(cls, input_file, quotechar=None):
        """ 
        Read seq.in seq.out BIO file
        type: out_slu, out_parser
        """
        out_lists = []
        with open(os.path.join(input_file, 'seq.in'), "r") as f_in, open(os.path.join(input_file, 'seq.out'), "r") as f_out:
            for line_in, line_out in zip(f_in, f_out):
                line_in = line_in.strip()  
                line_out = line_out.strip()  
                words = line_in.split()
                l2_list = line_out.split()
                #print('words: ', words)
                #print('l2_list: ', l2_list)

                assert len(words)==len(l2_list), 'len(words)!=len(l2_list)'
                #utter_list.append(tokens)
                #y2_list.append(l2_list)

                bin_labels = []
                slot_labels = []
                words_value = []
                for i, l in enumerate(l2_list):
                    if "B" in l:
                        words_value.append(words[i])
                        bin_labels.append("B")
                        labels = l.split("-")[1] 
                        slot_labels.append(labels)
                        # add instance_label
                        instance_labels = labels
                    elif "I" in l:
                        words_value.append(words[i])
                        labels = l.split("-")[1] 
                        slot_labels.append(labels)                        
                        bin_labels.append("I")
                    else:
                        slot_labels.append("O")                        
                        bin_labels.append("O")                    
                
                out_lists.append([words, words_value, bin_labels, slot_labels, instance_labels])    
                
                '''
                if "B" in bin_labels or "I" in bin_labels:
                    out_lists.append([words, words_value, bin_labels, slot_labels, instance_labels])    
                    #out_lists.append([words_value, bin_labels, slot_labels, instance_labels])    
                    assert len(words)==len(bin_labels), "len(bin_labels) must=len(words"
                else:
                    print('words without B: ', words)
                '''
        return out_lists

class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode, data_type):
        if mode == 'train':
            return self._create_examples(
                self._read_seq(os.path.join(data_dir, data_type, "train")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_seq(os.path.join(data_dir, data_type, "valid")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_seq(os.path.join(data_dir, data_type, "test")), "test")

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            words = line[0]
            words_value = line[1]
            bin_labels = line[2]
            slot_labels = line[3]
            instance_labels = line[-1]
            
            examples.append(InputExample(
                guid = guid, 
                words = words, 
                words_value = words_value,
                bin_labels = bin_labels, 
                slot_labels = slot_labels,
                instance_labels = instance_labels)               
                )
        return examples    


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    bin_label_map = {"O":0, "B":1, "I":2}
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = ['[CLS]']
        predict_mask = [0]
        slot_label_ids = [0]
        bin_label_ids = [0]

        for i, w in enumerate(example.words):
            if w == '[SEP]':
                sub_words = ['[SEP]']
            else:
                sub_words = tokenizer.tokenize(w)

            if not sub_words:
                sub_words = ['[UNK]']
            # tokenize_count.append(len(sub_words))
            #print('sub_words: ', sub_words)
            tokens.extend(sub_words)
            for j in range(len(sub_words)):
                if sub_words[0] == '[SEP]':
                    predict_mask.append(0)
                else:
                    predict_mask.append(1)
                if example.slot_labels[i]=="O":
                    slot_label_ids.append(0)
                else:
                    try:
                        slot_label_ids.append(label_map[example.slot_labels[i]])
                    except:
                        # rewrite the label of the unlabeled data to be [-1, -2, -3]
                        slot_label_ids.append(-1)
                        #slot_label_ids.append(example.slot_labels[i])

            bin_label = example.bin_labels[i]
            if bin_label=="B":
                bin_label_ids.append(bin_label_map[bin_label])
                bin_label_ids.extend([bin_label_map["I"]]*(len(sub_words)-1))
            else:
                bin_label_ids.extend([bin_label_map[bin_label]]*len(sub_words))

        # truncate
        if len(tokens) > max_seq_length - 1:
            print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens),
                                                                                    max_seq_length))
            tokens = tokens[0:(max_seq_length - 1)]
            #tokens_sentence = tokens_sentence[0:(max_seq_length - 1)]
            predict_mask = predict_mask[0:(max_seq_length - 1)]
            bin_label_ids = bin_label_ids[0:(max_seq_length - 1)]
            slot_label_ids = slot_label_ids[0:(max_seq_length - 1)]

        tokens.append('[SEP]')
        #tokens_sentence.append('[SEP]')
        predict_mask.append(0)
        bin_label_ids.append(0)
        slot_label_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        #segment_ids = [0] * (first_sep + 1) + [1] * (len(input_ids) - first_sep - 1)
        segment_ids = [0] * len(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        input_mask = [1] * len(input_ids)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        bin_label_ids += padding 
        slot_label_ids += padding 

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        #print('label_map: ', label_map)
        #print('example.instance_labels: ', example.instance_labels)
        #print('example.words: ', example.words)
        #instance_label_ids = label_map[example.instance_labels]
        
        try:
            instance_label_ids = label_map[example.instance_labels]
            
        except:
            #print('label_map: ', label_map)

            #print('533example.instance_labels: ', example.instance_labels)
            instance_label_ids = -1

        #if example.instance_labels =='none':
        #    print(example.instance_labels)
        #    print(instance_label_ids)
        
        # if ex_index < 5:
        #     logger.info("*** Example ***")
        #     logger.info("guid: %s" % (example.guid))
        #     logger.info("tokens: %s" % " ".join(
        #         [str(x) for x in tokens]))
        #     logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #     logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #     logger.info(
        #         "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #     logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          bin_label_ids = bin_label_ids,
                          slot_label_ids = slot_label_ids,
                          instance_label_ids=instance_label_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
