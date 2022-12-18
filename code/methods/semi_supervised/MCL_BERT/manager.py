import torch
import logging
import copy
import torch.nn.functional as F

from tqdm import trange, tqdm
from sklearn.metrics import confusion_matrix

from losses import loss_map
from utils.metrics import clustering_score
from utils.functions import restore_model, save_model, eval_and_save
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np

class MCLManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        self.Data = data
        self.logger = logging.getLogger(logger_name)
        self.num_labels = data.num_labels   
        self.model = model.model
        #self.optimizer = model.optimizer
        self.device = model.device

        self.train_dataloader = data.dataloader.train_loader      
        #self.eval_dataloader = data.dataloader.eval_loader
        self.eval_dataloader = data.dataloader.eval_loader_all_label
        self.test_dataloader = data.dataloader.test_loader 
        self.train_labeled_loader = data.dataloader.train_labeled_loader        
        
        self.optimizer = model.set_optimizer(self.model, len(self.train_dataloader.dataset), args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)
        self.loss_fct = loss_map[args.loss_fct]

        if not args.train:
            self.model = restore_model(self.model, args.model_output_dir)
        
        self.logger.info('model.num_labels is %s', self.model.num_labels)


    def train(self, args, data): 

        best_model = None
        wait = 0
        best_eval_score = 0
        eval_score_is_0 = 0
        for epoch in trange(int(args.num_train_epochs), desc="MCL-Epoch"):  
            
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()

            for batch in tqdm(self.train_dataloader, desc="MCL-Training(All)"):

                batch = tuple(t.to(self.device) for t in batch)
                #input_ids, input_mask, segment_ids, label_ids = batch
                #loss = self.model(input_ids, segment_ids, input_mask, label_ids, mode = 'train', loss_fct = self.loss_fct)
                input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, instance_label_ids = batch
                loss = self.model(input_ids, segment_ids, input_mask, instance_label_ids, mode = 'train', loss_fct = self.loss_fct, bin_label_ids = bin_label_ids)

                loss.backward()

                tr_loss += loss.item() 
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()
            
            tr_loss = tr_loss / nb_tr_steps

            y_true, y_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(y_true, y_pred)['NMI']
            eval_results = {
                'train_loss': tr_loss,
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
            #else:
                wait += 1
                if wait >= args.wait_patient:
                    break
            '''
            elif eval_score==0:
                eval_score_is_0+=1
                if eval_score_is_0>=10:
                    break  
            '''
        self.model = best_model
        
        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def get_outputs(self, args, mode = 'eval', get_feats = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader

        self.model.eval()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="MCL-Iteration"):
            batch = tuple(t.to(self.device) for t in batch)
            #input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

            with torch.set_grad_enabled(False):
                #features, logits = self.model(input_ids, segment_ids, input_mask)  
                features, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)  
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))
                total_features = torch.cat((total_features, features))

        if get_feats:

            feats = total_features.cpu().numpy()
            return feats
        
        else:
            total_probs = F.softmax(total_logits.detach(), dim = 1)
            total_maxprobs, total_preds = total_probs.max(dim = 1)

            y_true = total_labels.cpu().numpy()
            y_pred = total_preds.cpu().numpy()

            return y_true, y_pred

    def test(self, args, data):

        y_true, y_pred = self.get_outputs(args, mode = 'test')
        test_results = clustering_score(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred
        
        return test_results

    def evaluator(self, args, data, data_type = "test"):
        """
        type:
        test: only the test set
        unlabel: all the unlabeled data
        """
        if data_type=="test":
            dataloader =   data.dataloader.test_loader
            dataloader_gt = data.dataloader.test_loader_slu
        else:
            dataloader =   data.dataloader.all_unlabeled_loader_parser
            dataloader_gt = data.dataloader.all_unlabeled_loader_slu            

        self.model.eval()
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
                
        self.results = {}
        self.slot_value_hyp = {}
        self.slot_value_gt = {}
        self.get_results_samp(args, dataloader, mode = 'hyp', data_type='unlabel' )
        self.get_results_samp(args, dataloader_gt, mode = 'gt', data_type='unlabel')
        
        self.get_results_samp(args, self.train_labeled_loader, mode='gt', data_type='label',write_result=False)
        self.logger.info('Results saved in %s', str(args.result_dir))
        #id2slot = {str(i):slot_name for i, slot_name in enumerate(self.Data.all_label_list)}
        #eval_and_save(args, self.results, self.slot_value_hyp, self.slot_value_gt, data_type) #, slot_map = id2slot)
        weighted_result = eval_and_save(args, self.results, self.slot_value_hyp, self.slot_value_gt, data_type) #, slot_map = id2slot)
        return weighted_result        
    def get_results_samp(self, args, dataloader, mode='gt', data_type='unlabel', write_result=True):

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_logits = torch.empty((0, args.num_labels)).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        self.logger.info('num_labels: %s', args.num_labels)

        for batch in tqdm(dataloader, desc="MCL-Iteration of prediction"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

            if mode=='hyp':
                with torch.set_grad_enabled(False):

                    features, logits = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                    #total_labels = torch.cat((total_labels, label_ids))
                    total_logits = torch.cat((total_logits, logits))

                total_probs = F.softmax(total_logits.detach(), dim=1)
                total_maxprobs, total_preds = total_probs.max(dim = 1)

                total_maxprobs = total_maxprobs.cpu().numpy()
                #y_true = total_labels.cpu().numpy()
                y_pred = total_preds.cpu().numpy()

            input_ids = input_ids.cpu().numpy()
            bin_label_ids = bin_label_ids.cpu().numpy()
            slot_label_ids = slot_label_ids.cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            
            for i in range(input_ids.shape[0]):
                uttr_list = [self.tokenizer.ids_to_tokens[k] for k in input_ids[i]]
                uttr = ' '.join(uttr_list)
                bin_label = np.array(bin_label_ids[i])
                indices = np.nonzero(bin_label)[0]
                value = ' '.join([uttr_list[int(ind)] for ind in indices])
                
                if mode=='hyp':
                    slot = y_pred[i]
                    slot = str(slot)
                    prob = total_maxprobs[i]
                else:
                    slot = label_ids[i]
                    #id2slot = {i:slot_name for i, slot_name in enumerate(self.Data.all_label_list)}
                    if data_type=='unlabel':
                        id2slot = {i:slot_name for i, slot_name in enumerate(self.Data.all_label_list)}
                    else:
                        id2slot = {i:slot_name for i, slot_name in enumerate(self.Data.known_label_list)}
                    slot = id2slot[slot]
                if write_result:
                    if not uttr in self.results.keys():
                        self.results[uttr] = {'hyp': {}, 'gt': {}}
                    if mode=='hyp':
                        self.results[uttr][mode][value] = [slot, str(prob)]
                    else:
                        self.results[uttr][mode][value] = slot
                        
                if mode=='hyp':
                    if not slot in self.slot_value_hyp.keys():
                        self.slot_value_hyp[slot] = [] #set()
                    #self.slot_value_hyp[slot].add(value)
                    self.slot_value_hyp[slot].append(value)
                else:
                    if not slot in self.slot_value_gt.keys():
                        self.slot_value_gt[slot] = [] #set()
                    #self.slot_value_gt[slot].add(value)
                    self.slot_value_gt[slot].append(value)