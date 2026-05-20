import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.pipelines.prepare import prepare_all_datasets
from core.pipelines.train import main as train_main
from core.pipelines.evaluate import main as evaluate_main
from core.pipelines.generate import main as generate_main

def main():
    parser = argparse.ArgumentParser(description="GAN Texture Expansion Pipeline CLI Manager")
    parser.add_argument(
        "action", 
        choices=["prepare", "train", "evaluate", "generate"], 
        help="Pipeline stage to execute"
    )
    
    args = parser.parse_args()

    if args.action == "prepare":
        prepare_all_datasets()
        
    elif args.action == "train":
        train_main()
        
    elif args.action == "evaluate":
        evaluate_main()
        
    elif args.action == "generate":
        generate_main()

if __name__ == "__main__":
    main()