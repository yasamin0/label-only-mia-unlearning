import logging
import os, random
import numpy as np
import torch
import action

from parameter_parser import parameter_parser


def config_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(levelname)s:%(asctime)s: - %(name)s - : %(message)s')
    ch.setFormatter(formatter)

    logger.addHandler(ch)


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    if args.action == 'model_train':
        if args.unlearning_method == 'scratch':
            action.ActionModelTrainScratch(args)
        else:
            action.ActionModelTrainSisa(args)
    elif args.action == 'attack':
        if args.unlearning_method == 'scratch':
            action.ActionAttackScratch(args)
        else:
            action.ActionAttackSisa(args)
    else:
        raise Exception(f'Invalid action: No {args.action}')


if __name__ == '__main__':
    args = parameter_parser().parse_args()
    set_seed(args.seed)   
    config_logger()
    main(args)
