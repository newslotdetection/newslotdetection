from configs.base import ParamManager
from dataloaders.base import DataManager
from backbones.base import ModelManager
from methods import method_map
from utils.functions import save_results
import logging
import argparse
import sys
import os
import datetime

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--type', type=str, default='open_intent_discovery', help="Type for methods")

    parser.add_argument('--logger_name', type=str, default='Discovery', help="Logger name for open intent discovery.")

    parser.add_argument('--log_dir', type=str, default='logs', help="Logger directory.")

    parser.add_argument("--dataset", default='banking', type=str, help="The name of the dataset to train selected")

    parser.add_argument("--known_cls_ratio", default=1.0, type=float, help="The number of known classes for validation")
    
    parser.add_argument("--labeled_ratio", default=0.1, type=float, help="The ratio of labeled samples in the training set")
    
    parser.add_argument("--cluster_num_factor", default=1.0, type=float, help="The factor (magnification) of the number of clusters K.")

    parser.add_argument("--method", type=str, default='DeepAligned', help="which method to use")

    parser.add_argument("--train", action="store_true", help="Whether train the model")

    parser.add_argument("--save_model", action="store_true", help="save trained-model for open intent detection")

    parser.add_argument("--backbone", type=str, default='bert', help="which backbone to use")

    parser.add_argument('--setting', type=str, default='semi_supervised', help="Type for clustering methods.")

    parser.add_argument("--config_file_name", type=str, default='DeepAligned.py', help = "The name of the config file.")

    parser.add_argument('--seed', type=int, default=0, help="random seed for initialization")

    parser.add_argument("--gpu_id", type=str, default='0', help="Select the GPU id")

    parser.add_argument("--pipe_results_path", type=str, default='pipe_results', help="the path to save results of pipeline methods")
    
    #parser.add_argument("--data_dir", default = sys.path[0]+'/../data', type=str,
    #                    help="The input data dir. Should contain the .csv files (or other data files) for the task.")

    parser.add_argument("--data_dir", default = '../data', type=str,
                        help="The input data dir.")
 
    parser.add_argument("--output_dir", default= './outputs', type=str, 
                        help="The output directory where all train data will be written.") 

    parser.add_argument("--model_dir", default='models', type=str, 
                        help="The output directory where the model predictions and checkpoints will be written.") 

    parser.add_argument("--result_dir", type=str, default = 'results', help="The path to save results")

    parser.add_argument("--results_file_name", type=str, default = 'results.csv', help="The file name of all the results.")

    parser.add_argument("--save_results", action="store_true", help="save final results for open intent detection")
    parser.add_argument("--do_lower_case", default=True)

    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for training.")

    parser.add_argument("--context", default=True, help="whether concat the context embedding for encoder")
    parser.add_argument("--value_enc_type", type=str, default = 'bert', help="bert or lstm, if bert, there is no lstm for value encoder")
    parser.add_argument("--thr", type=float, default = 0.9, help="the threshold for selecting condident samples")
    parser.add_argument("--pre_train", action="store_true", help="Whether train the model")
    parser.add_argument("--pre_train_parser", action="store_true", help="Whether train the model")

    parser.add_argument("--pre_train_parser_multitask", action="store_true", help="Pre-train parser in a multitask way")
    parser.add_argument("--pre_train_multitask", action="store_true", help="Pre-train in a multitask way")
    parser.add_argument("--train_multitask", action="store_true", help="Train in a multitask way")

    # strategy
    parser.add_argument("--strategy", type=str, default = 'RandomSampling', help="the strategy for active learning")
    parser.add_argument("--select_ratio", type=float, default = 0.05, help="the select_ratio for active learning")
    parser.add_argument("--fine_tune_epoch", type=int, default = 2, help="the fine_tune_epoch for active learning")

    # contribution of loss_ori
    parser.add_argument("--alpha", type=float, default=0.05, help="contribution of loss_ori")
    parser.add_argument("--beta", type=float, default=0.5, help="proportion of uncertainty in mmr")
    parser.add_argument("--resume", action="store_true", help="if to resume from the last training status (model and data selection)")

    args = parser.parse_args()

    return args


def set_logger(args):

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    file_name = f"{args.method}_{args.dataset}_{args.backbone}_{args.known_cls_ratio}_{args.labeled_ratio}_{args.select_ratio}_{args.strategy}_{time}.log"
    
    logger = logging.getLogger(args.logger_name)
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(os.path.join(args.log_dir, file_name))
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
    fh.setFormatter(fh_formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def run(args, data, model, logger):

    method_manager = method_map[args.method]
    method = method_manager(args, data, model, logger_name = args.logger_name)
    
    if args.train or args.train_multitask:
        
        logger.info('Training Begin...')
        method.train(args, data)
        logger.info('Training Finished...')
    
    logger.info('Evaluate begin...')
    #method.evaluator(args, data, data_type = "test")
    # save each iter

    #outputs = method.evaluator(args, data, data_type = "all_unlabel")
    #logger.info('Evaluate finished...')
    #if args.save_results:
    #    logger.info('Results saved in %s', str(os.path.join(args.result_dir, args.results_file_name)))
    #    save_results(args, outputs)    

if __name__ == '__main__':
    
    sys.path.append('.')
    args = parse_arguments()
    logger = set_logger(args)
    
    logger.info('Open Value Discovery Begin...')
    logger.info('Parameters Initialization...')
    param = ParamManager(args)
    args = param.args

    logger.debug("="*30+" Params "+"="*30)
    for k in args.keys():
        logger.debug(f"{k}:\t{args[k]}")
    logger.debug("="*30+" End Params "+"="*30)

    logger.info('Data and Model Preparation...')
    data = DataManager(args)
    model = ModelManager(args, data)

    run(args, data, model, logger)
    logger.info('Open Value Discovery Finished...')
    

