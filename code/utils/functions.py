import os
import torch
import numpy as np
import pandas as pd
from pytorch_pretrained_bert.modeling import WEIGHTS_NAME, CONFIG_NAME
import json
import copy
import torch
import random
from prettytable import PrettyTable


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #transformers.set_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def normalize_num(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    dct = {x: str(y) for y, x in enumerate(known_nos)}
    if n.lower() in dct:
        return dct[n.lower()]
    return n


def is_number(n):
    known_nos = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    n = n.strip().lower()
    return n.isdigit() or n in known_nos

def save_npy(npy_file, path, file_name):
    npy_path = os.path.join(path, file_name)
    np.save(npy_path, npy_file)

def load_npy(path, file_name):
    npy_path = os.path.join(path, file_name)
    npy_file = np.load(npy_path)
    return npy_file

def save_model(model, model_dir):

    save_model = model.module if hasattr(model, 'module') else model  
    model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model_config_file = os.path.join(model_dir, CONFIG_NAME)
    torch.save(save_model.state_dict(), model_file)
    with open(model_config_file, "w") as f:
        f.write(save_model.config.to_json_string())

def restore_model(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    model.load_state_dict(torch.load(output_model_file))
    return model

def restore_model_as_dict(model, model_dir):
    output_model_file = os.path.join(model_dir, WEIGHTS_NAME)
    loaded = torch.load(output_model_file)
    return loaded

def save_results(args, test_results, ratio):

    '''
    pred_labels_path = os.path.join(args.method_output_dir, 'y_pred.npy')
    np.save(pred_labels_path, test_results['y_pred'])
    true_labels_path = os.path.join(args.method_output_dir, 'y_true.npy')
    np.save(true_labels_path, test_results['y_true'])

    del test_results['y_pred']
    del test_results['y_true']
    '''
    
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    if args.backbone == 'bert_MultiTask':
        alpha_v = args.alpha
    else:
        alpha_v = 0

    if args.strategy == "MaximalMarginalRelevance" or args.strategy == "MMR_Margin" or args.strategy == "MMR_Margin_bothbranch":
        beta_v = args.beta
    else:
        beta_v = 'NA'

    var = [ratio, args.strategy, args.select_ratio, args.fine_tune_epoch, args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed, alpha_v, beta_v]
    names = ['ratio', 'strategy', 'select_ratio', 'ft_epo', 'dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed', 'alpha', 'beta']
    vars_dict = {k:v for k,v in zip(names, var) }
    results = dict(test_results,**vars_dict)
    keys = list(results.keys())
    values = list(results.values())
    results_file_name = '_'.join(['results_ours', args.strategy])
    results_path = os.path.join(args.result_dir, results_file_name+'.csv')
    
    if not os.path.exists(results_path) or os.path.getsize(results_path) == 0:
        ori = []
        ori.append(values)
        df1 = pd.DataFrame(ori,columns = keys)
        df1.to_csv(results_path,index=False)
    else:
        df1 = pd.read_csv(results_path)
        new = pd.DataFrame(results,index=[1])
        df1 = df1.append(new,ignore_index=True)
        df1.to_csv(results_path,index=False)
    data_diagram = pd.read_csv(results_path)
    
    print('test_results', data_diagram)

def eval_and_save(args, test_results, slot_value_hyp, slot_value_gt, data_type, slot_map=None, iter_i=0):

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    var = [iter_i, args.strategy, args.dataset, args.method, args.backbone, args.known_cls_ratio, args.labeled_ratio, args.cluster_num_factor, args.seed]
    names = ['iter', 'strategy', 'dataset', 'method', 'backbone', 'known_cls_ratio', 'labeled_ratio', 'cluster_num_factor', 'seed']

    results_folder_name = ""
    for k, v in zip(names, var):
        results_folder_name += str(k) + '-' + str(v) + '-'

    results_path = os.path.join(args.result_dir, results_folder_name, data_type)
    print('results_path: ', results_path)
    if not os.path.exists(results_path):
        os.makedirs(results_path)    
    # save
    '''
    pred_labels_path = os.path.join(results_path, 'y_pred.npy')
    feats_path = os.path.join(results_path, 'feats.npy')
    try:
        np.save(pred_labels_path, test_results['y_pred'])
        del test_results['y_pred']
        np.save(feats_path, test_results['feats'])
        del test_results['feats']
    except:
        pass
    '''
    with open (os.path.join(results_path, 'samp_results.json'), 'w') as f:
        f.write(json.dumps(test_results, indent=4))

    with open (os.path.join(results_path, 'slot_value_hyp.json'), 'w') as f:
        f.write(json.dumps(slot_value_hyp, indent=4))
    
    with open (os.path.join(results_path, 'slot_value_gt.json'), 'w') as f:
        f.write(json.dumps(slot_value_gt, indent=4))

    slot_value_hyp_set = {slot:list(set(value)) for slot, value in slot_value_hyp.items()}
    slot_value_gt_set = {slot:list(set(value)) for slot, value in slot_value_gt.items()}
    with open (os.path.join(results_path, 'slot_value_hyp_set.json'), 'w') as f:
        f.write(json.dumps(slot_value_hyp_set, indent=4))
    with open (os.path.join(results_path, 'slot_value_gt_set.json'), 'w') as f:
        f.write(json.dumps(slot_value_gt_set, indent=4))
    
    slot_names = list(slot_value_gt_set.keys())
    #slot_names.remove('noneslot')
    model_evaluator = GenericEvaluator('NN MODEL', slot_map, slot_names)
    
    recall_value_all = []
    samp_result_final = {}
    for uttr, result in test_results.items():
        #turn_slu = {v:k for k, v in result['gt'].items()}
        #print('uttr: ',uttr)
        #print('result: ', result)
        # one slot may have many value
        # value: slot ---> slot value
        turn_slu = {}
        slu_hyp = {}
        samp_result_final[uttr] = {}

        for value, slot in result['gt'].items():
            if not slot in turn_slu.keys():
                turn_slu[slot] = []
            turn_slu[slot].append(value.lower())          
        
        for value, slot in result['hyp'].items():
            if not slot in slu_hyp.keys():
                slu_hyp[slot] = []
            slu_hyp[slot].append(value.lower())  
        #m_tps,m_fps,m_fns = model_evaluator.add_turn(turn_slu, slu_hyp)
        slu_hyp_copy = copy.deepcopy(slu_hyp)
        turn_slu_copy = copy.deepcopy(turn_slu)
        samp_result_final[uttr] ['hyp'] = slu_hyp_copy
        samp_result_final[uttr] ['gt'] = turn_slu_copy

        if len(turn_slu.values())>0:
            m_tps,m_fps,recall_value = model_evaluator.add_turn(turn_slu, slu_hyp)
        #test_results[uttr]['hyp'] = {k: slot_map[v] for k, v in result['hyp'].items()}
            recall_value_all.append(recall_value)
    print('recall_value: ', np.mean(np.array(recall_value_all)))
    with open (os.path.join(results_path, 'samp_result_final.json'), 'w') as f:
        f.write(json.dumps(samp_result_final, indent=4))

    with open(os.path.join(results_path, 'result_metrics.json'), 'wt') as of:
        weighted_result = model_evaluator.eval(of)
    return weighted_result


class SlotEvaluator:
    def __init__(self, name='dummy'):
        self.tp = 0.000001
        self.fp = 0.000001
        self.tn = 0
        self.fn = 0.000001

    @property
    def precision(self):
        return round(self.tp) / (self.tp + self.fp)

    @property
    def recall(self):
        return round(self.tp) / (self.tp + self.fn)

    @property
    def f1(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + .000000000000001)

    @property
    def instances(self):
        return self.tp + self.fn


class GenericEvaluator:

    def __init__(self, name, eval_mapping, slot_names):
        self.name = name
        self.eval_mapping = eval_mapping
        self.slot_evaluators = {x: SlotEvaluator() for x in slot_names}

    def add_turn(self, turn_slu, slu_hyp):
        tps = 0
        fps = 0
        tns = 0
        fns = 0
        

        gold_values = [v for slot, value in turn_slu.items() for v in value]
        hyp_values = [v for slot, value in slu_hyp.items() for v in value]

        recall_value = len(set(gold_values) & set(hyp_values))/len(set(gold_values)) # recall rate of candidate value extraction

        for gold_slot, gold_value in list(turn_slu.items()):
                
            if gold_slot not in self.slot_evaluators:
                continue
            
            if gold_slot not in slu_hyp:
                self.slot_evaluators[gold_slot].fn += 1
                fns += 1
                continue
            #if slu_hyp[gold_slot].lower() in gold_value.lower() or gold_value.lower() == 'slot':
            sub_gold_value = [v for value in gold_value for v in value.split(' ')]
            if len(set(slu_hyp[gold_slot]) & set(gold_value))>0 or len(set(slu_hyp[gold_slot]) & set(sub_gold_value))>0:
            
                self.slot_evaluators[gold_slot].tp += 1
                tps += 1
                del slu_hyp[gold_slot]
                continue
            else:
                self.slot_evaluators[gold_slot].fp += 1
                fps += 1
                del slu_hyp[gold_slot]
        for predicted_slot, predicted_value in slu_hyp.items():
            if predicted_slot not in self.slot_evaluators:
                continue
            self.slot_evaluators[predicted_slot].fp += 1
            fps += 1
        #return tps, fps, fns
        return tps, fps, recall_value 

    def eval(self, result):
        print(self.name, file=result)
        mean_precision = mean_recall = mean_f1 = 0
        w_mean_precision = w_mean_recall = w_mean_f1 = 0
        onto_w_mean_precision = onto_w_mean_recall = onto_w_mean_f1 = 0
        nt_mean_precision = nt_mean_recall = nt_mean_f1 = 0
        onto_mean_precision = onto_mean_recall = onto_mean_f1 = 0

        for name, evaluator in self.slot_evaluators.items():
            print(name, evaluator.precision, evaluator.recall, evaluator.f1, file=result)
            if name=="nt":
                nt_mean_precision += evaluator.precision
                nt_mean_recall += evaluator.recall
                nt_mean_f1 += evaluator.f1                
            else:
                onto_mean_precision += evaluator.precision
                onto_mean_recall += evaluator.recall
                onto_mean_f1 += evaluator.f1  

            mean_precision += evaluator.precision
            mean_recall += evaluator.recall
            mean_f1 += evaluator.f1
            
            total_instances = sum([evaltr.instances for evaltr in self.slot_evaluators.values()])
            w_mean_precision += evaluator.precision * (evaluator.instances) / total_instances
            w_mean_recall += evaluator.recall * (evaluator.instances) / total_instances
            w_mean_f1 += evaluator.f1 * (evaluator.instances) / total_instances
            
            onto_total_instances = sum([evaltr.instances for slot_name, evaltr in self.slot_evaluators.items() if slot_name!="nt"])
            if name!="nt":
                onto_w_mean_precision += evaluator.precision * (evaluator.instances) / onto_total_instances
                onto_w_mean_recall += evaluator.recall * (evaluator.instances) / onto_total_instances
                onto_w_mean_f1 += evaluator.f1 * (evaluator.instances) / onto_total_instances
                

        
        print('mean', mean_precision / len(self.slot_evaluators), mean_recall / len(self.slot_evaluators), mean_f1 / len(self.slot_evaluators), file=result)
        print('weighted-mean', w_mean_precision, w_mean_recall, w_mean_f1, file=result)
        print('mean: ', [mean_precision / len(self.slot_evaluators), mean_recall / len(self.slot_evaluators), mean_f1 / len(self.slot_evaluators)])
        print('weighted-mean: ', [w_mean_precision, w_mean_recall, w_mean_f1])
                
        print('-' * 80, file=result)
        w_result = {
                'P': w_mean_precision,
                'R': w_mean_recall,
                'F1': w_mean_f1,
                'nt_P': nt_mean_precision,
                'nt_R': nt_mean_recall,
                'nt_F1': nt_mean_f1,
                'onto_P': onto_w_mean_precision,
                'onto_R': onto_w_mean_recall,
                'onto_F1': onto_w_mean_f1,
                }

        # w_result = PrettyTable()
        # w_result.add_column('P', [w_mean_precision])
        # w_result.add_column('R', [w_mean_recall])
        # w_result.add_column('F1', [w_mean_f1])
        # w_result.add_column('nt_P', [nt_mean_precision])
        # w_result.add_column('nt_R', [nt_mean_recall])
        # w_result.add_column('nt_F1', [nt_mean_f1])
        # w_result.add_column('onto_P', [onto_w_mean_precision])
        # w_result.add_column('onto_R', [onto_w_mean_recall])
        # w_result.add_column('onto_F1', [onto_w_mean_f1])

        return w_result


def compute_ap(frames, sorted_list):
    spoted = 0
    i = 0
    precision_sum = 0
    cut_off = 20
    for fr_name in sorted_list:
        i += 1
        spoted_any = False
        for fr in fr_name.split('-'):
            if fr in frames:
                spoted += 1
                print(fr, 'adding {}/{}'.format(spoted, i))
                precision_sum += spoted / i
                i += 1
        if spoted == len(frames):
            break
    # return precision_sum
    return precision_sum / len(frames)


from collections import Counter

def slot_mapping(slot_value_gt, slot_value_hyp):
    
    del slot_value_gt['noslot']
    sort_gt = sorted(slot_value_gt.items(), key=lambda x: len(x[1]), reverse=True)
    value2slot_hyp = {}
    #print('hyp...........')
    slot_in_hyp = []
    for slot, value in slot_value_hyp.items():
        slot_in_hyp.append(slot)
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        for v in value:
            if v not in value2slot_hyp.keys():
                value2slot_hyp[v] = []
            value2slot_hyp[v].append(slot)

    #print('gt...........')
    slot_map = {}
    candidate_id = [str(i) for i in range(len(slot_value_gt))]
    #print( 'candidate_id: ', candidate_id  )
    for slot, value in sort_gt:
        #print('slot: ', slot)
        #print('value: ', Counter(value))
        #print('len_value: ', len(value))

        slot_id_in_hyp = []
        for v in value:
            try:
                slot_id_in_hyp.extend(value2slot_hyp[v])
            except:
                pass
        counter = Counter(slot_id_in_hyp)
        
        #print('len(slot_id_in_hyp): ', len(slot_id_in_hyp))
        #print('Counter: ', counter)
        #print ([k for k,v in counter.most_common()])
        for k, v in counter.most_common():
            if k in candidate_id:
                slot_map[k] = slot
                #print('slot_map: ', slot_map)
                candidate_id.remove(k)
                break
            else:
                if len(candidate_id)==1:
                    slot_map[candidate_id[0]] = slot
                #print('len(candidate_id)',len(candidate_id))
                #print('candidate_id',candidate_id)
                #if len(candidate_id)==2:
                #    slot_map[candidate_id[0]] = slot
                #    candidate_id.remove(candidate_id[0])
                #print('slot_map: ', slot_map)    
    for slot in slot_in_hyp:
        if not slot in slot_map.keys():
            slot_map[slot]="noslot"
            
    return slot_map


def value_mapping(slot_value_gt, slot_value_hyp):
    value2slot_gt = {}
    for slot, value in slot_value_gt.items():
        for v in value:
            value2slot_gt[v]=slot
    
    value_map = {}
    for slot, value in slot_value_hyp.items():
        for v in value:
            if v in value2slot_gt.keys():
                value_map[v] = value2slot_gt[v]
            else:
                value_map[v] = 'no'
    return value_map
