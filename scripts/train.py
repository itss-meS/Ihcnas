import os
import sys
import argparse
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.trainer import Trainer


def load_config(config_name):
    path = os.path.join('configs', config_name)
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', type=str, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config_name)
    trainer = Trainer(cfg)

    resume_path = None
    if args.resume_from is not None:
        resume_path = args.resume_from
    elif args.resume:
        resume_path = trainer.checkpointer.find_latest()

    if resume_path is not None and os.path.exists(resume_path):
        trainer.resume(resume_path)

    trainer.fit()


if __name__ == '__main__':
    main()
