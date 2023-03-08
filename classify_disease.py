# Copyright 2021-2022, The Johns Hopkins University Applied Physics Laboratory LLC
# All rights reserved.

import argparse
from disease_classification.data_utils import gen_preliminary_data
from disease_classification.learn_utils import train_model, eval_model
import disease_classification.clf_config as cfg

def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-masks", action="store_true", help="Generate masks for the data and calculate ITA.")
    parser.add_argument("--train", action="store_true", help="Train the model.")
    parser.add_argument("--test", action="store_true", help="Run evaluation on the best performing model.")
    parser.add_argument("--model-type", type=str, default="baseline", help="Model type to train/test. Options: baseline (default), masked, masked+AD, AD")
    parser.add_argument("--gpu", type=str, default="0", help="GPU number.")
    parser.add_argument("--edgemixup", default=False, description="set True to deploy EdgeMixup")
    return parser

def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    cfg.CLF_ASSETS.mkdir(parents=True, exist_ok=True)
    cfg.CLF_RAW_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CLF_SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    if args.generate_masks:
        gen_preliminary_data()
    
    if args.train:
        train_model(args)
    
    if args.test:
        eval_model(args)


if __name__ == '__main__':
    main()
