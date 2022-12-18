import logging
import copy
import torch
import torch.nn.functional as F
from .pretrain import PretrainDTCManager
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from tqdm import trange, tqdm
from utils.metrics import clustering_score
from utils.functions import save_model, restore_model
import os
from utils.functions import eval_and_save
from pytorch_pretrained_bert.tokenization import BertTokenizer
import numpy as np

def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

class DTCManager:
    
    def __init__(self, args, data, model, logger_name = 'Discovery'):
        
        self.Data = data
        self.logger = logging.getLogger(logger_name)
        pretrain_manager = PretrainDTCManager(args, data, model)  

        self.device = model.device

        #self.train_dataloader = data.dataloader.train_unlabeled_loader
        self.train_dataloader = data.dataloader.all_unlabeled_loader_parser
        #from dataloaders.slot_loader import get_loader
        #self.eval_dataloader, _ = get_loader(data.dataloader.eval_examples, args, data.all_label_list, 'eval')
        self.eval_dataloader = data.dataloader.eval_loader
        self.test_dataloader = data.dataloader.test_loader 

        self.train_labeled_loader = data.dataloader.train_labeled_loader        


        if args.train:
            
            num_train_examples = len(self.train_dataloader.dataset)

            self.logger.info('Pre-raining start with known_slot_num...')
            pretrain_manager.train(args, data)
            self.logger.info('pretrain_manager.model.num_labels is %s', pretrain_manager.model.num_labels)

            self.logger.info('Pre-training finished...')

            args.num_labels = data.num_labels
            self.model = model.set_model(args, data, 'bert')
            self.load_pretrained_model(pretrain_manager.model)
            self.logger.info('model.num_labels is %s', self.model.num_labels)
            self.eval_dataloader = data.dataloader.eval_loader_all_label
            self.initialize_centroids(args)
            
            self.warmup_optimizer = model.set_optimizer(self.model, num_train_examples, args.train_batch_size, \
                args.num_warmup_train_epochs, args.lr, args.warmup_proportion)

            self.logger.info('WarmUp Training start...')
            self.p_target = self.warmup_train(args)
            self.logger.info('WarmUp Training finished...')

            self.optimizer = model.set_optimizer(self.model, num_train_examples, args.train_batch_size, \
                args.num_train_epochs, args.lr, args.warmup_proportion)

        else:
            self.pretrained_model = restore_model(pretrain_manager.model, os.path.join(args.method_output_dir, 'pretrain'))
            args.num_labels = data.num_labels
            self.model = model.set_model(args, data, 'bert')
            self.load_pretrained_model(pretrain_manager.model)
            self.model = restore_model(self.model, args.model_output_dir)


    def initialize_centroids(self, args):

        self.logger.info("Initialize centroids with Kmeans on self.train_dataloader...")
        self.logger.info('n_clusters=args.num_labels %s', args.num_labels)

        feats = self.get_outputs(args, mode = 'train', get_feats = True)
        #km = KMeans(n_clusters=args.num_labels, n_jobs=-1, random_state=args.seed)
        km = KMeans(n_clusters=args.num_labels, random_state=args.seed)
        km.fit(feats)
        self.logger.info("Initialization finished...")
        self.model.cluster_layer.data = torch.tensor(km.cluster_centers_).to(self.device)

    def warmup_train(self, args):
        
        probs = self.get_outputs(args, mode = 'train', get_probs = True)
        p_target = target_distribution(probs)

        for epoch in trange(int(args.num_warmup_train_epochs), desc="Warmup_Epoch"):

            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Warmup_Training")):

                batch = tuple(t.to(self.device) for t in batch)
                #input_ids, input_mask, segment_ids, label_ids = batch
                input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

                logits, q = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                loss = F.kl_div(q.log(),torch.Tensor(p_target[step * args.train_batch_size: (step+1) * args.train_batch_size]).to(self.device))

                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.warmup_optimizer.step()
                self.warmup_optimizer.zero_grad()       

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            eval_results = {
                'loss': tr_loss, 
                'eval_score': round(eval_score, 2)
            }
            self.logger.info("***** Epoch: %s: Eval results *****", str(epoch))
            for key in sorted(eval_results.keys()):
                self.logger.info("  %s = %s", key, str(eval_results[key]))
        
        return p_target
    

    def get_outputs(self, args, mode = 'eval', get_feats = False, get_probs = False):
        
        if mode == 'eval':
            dataloader = self.eval_dataloader
        elif mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'train':
            dataloader = self.train_dataloader

        self.model.eval()
        total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        total_features = torch.empty((0, args.num_labels)).to(self.device)
        total_probs = torch.empty((0, args.num_labels)).to(self.device)

        for batch in tqdm(dataloader, desc="DTC-Iteration"):

            batch = tuple(t.to(self.device) for t in batch)
            #input_ids, input_mask, segment_ids, label_ids = batch
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch
            
            with torch.set_grad_enabled(False):
                logits, probs  = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                total_labels = torch.cat((total_labels, label_ids))
                total_features = torch.cat((total_features, logits))
                total_probs = torch.cat((total_probs, probs))


        if get_feats:
            feats = total_features.cpu().numpy()
            return feats

        elif get_probs:
            return total_probs.cpu().numpy()

        else:
            total_preds = total_probs.argmax(1)
            y_pred = total_preds.cpu().numpy()

            y_true = total_labels.cpu().numpy()

            return y_true, y_pred

    def train(self, args, data): 

        #ntrain = len(data.dataloader.train_unlabeled_examples)
        ntrain = len(self.train_dataloader.dataset)
        Z = torch.zeros(ntrain, args.num_labels).float().to(self.device)        # intermediate values
        z_ema = torch.zeros(ntrain, args.num_labels).float().to(self.device)        # temporal outputs
        z_epoch = torch.zeros(ntrain, args.num_labels).float().to(self.device)  # current outputs

        best_model = None
        best_eval_score = 0
        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):  

            # Fine-tuning with auxiliary distribution
            tr_loss, nb_tr_examples, nb_tr_steps = 0, 0, 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):

                batch = tuple(t.to(self.device) for t in batch)
                #input_ids, input_mask, segment_ids, label_ids = batch
                input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

                logits, q = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                z_epoch[step * args.train_batch_size: (step+1) * args.train_batch_size, :] = q
                kl_loss = F.kl_div(q.log(), torch.tensor(self.p_target[step * args.train_batch_size: (step+1) * args.train_batch_size]).to(self.device))
                
                kl_loss.backward() 
                tr_loss += kl_loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad() 
            
            z_epoch = torch.tensor(self.get_outputs(args, mode = 'train', get_probs = True)).to(self.device)
            Z = args.alpha * Z + (1. - args.alpha) * z_epoch
            z_ema = Z * (1. / (1. - args.alpha ** (epoch + 1)))

            if epoch % args.update_interval == 0:
                self.logger.info('updating target ...')
                self.p_target = target_distribution(z_ema).float().to(self.device) 
                self.logger.info('updating finished ...')

            eval_true, eval_pred = self.get_outputs(args, mode = 'eval')
            eval_score = clustering_score(eval_true, eval_pred)['NMI']
            train_loss = tr_loss / nb_tr_steps

            eval_results = {
                'train_loss': train_loss, 
                'best_eval_score': best_eval_score,
                'eval_score': round(eval_score, 2),
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
                if wait >= args.wait_patient:
                    break 
        
        self.model = best_model

        if args.save_model:
            save_model(self.model, args.model_output_dir)

    def test(self, args, data):
    
        y_true, y_pred = self.get_outputs(args,mode = 'test')
        test_results = clustering_score(y_true, y_pred) 
        cm = confusion_matrix(y_true,y_pred) 
        
        self.logger.info
        self.logger.info("***** Test: Confusion Matrix *****")
        self.logger.info("%s", str(cm))
        self.logger.info("***** Test results *****")
        
        for key in sorted(test_results.keys()):
            self.logger.info("  %s = %s", key, str(test_results[key]))

        test_results['y_true'] = y_true
        test_results['y_pred'] = y_pred

        return test_results

    def load_pretrained_model(self, pretrained_model):
    
        pretrained_dict = pretrained_model.state_dict()
        classifier_params = ['cluster_layer', 'classifier.weight','classifier.bias']
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k not in classifier_params}
        self.model.load_state_dict(pretrained_dict, strict=False)


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

        #total_labels = torch.empty(0,dtype=torch.long).to(self.device)
        total_preds = torch.empty(0,dtype=torch.long).to(self.device)

        #total_features = torch.empty((0, args.num_labels)).to(self.device)
        total_probs = torch.empty((0, args.num_labels)).to(self.device)
        self.logger.info('num_labels: %s', args.num_labels)


        for batch in tqdm(dataloader, desc="DTC-Iteration of prediction"):

            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, bin_label_ids, slot_label_ids, label_ids = batch

            if mode=='hyp':
                with torch.set_grad_enabled(False):

                    logits, probs  = self.model(input_ids, segment_ids, input_mask, bin_label_ids = bin_label_ids)
                    #total_labels = torch.cat((total_labels, label_ids))
                    #total_features = torch.cat((total_features, logits))
                    total_probs = torch.cat((total_probs, probs))
                total_maxprobs, total_preds = total_probs.max(dim = 1)
                #total_preds = total_probs.argmax(1)
                total_maxprobs = total_maxprobs.cpu().numpy()
                y_pred = total_preds.cpu().numpy()

            #y_true = total_labels.cpu().numpy()
            
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
                #print('slot: ', slot)
                #self.logger.info('slot: %s', slot)
                if write_result:
                #print('slot: ', slot)
                    if not uttr in self.results.keys():
                        self.results[uttr] = {'hyp': {}, 'gt': {}}
                    #self.results[uttr][mode][value] = slot
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