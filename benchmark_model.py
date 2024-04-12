import argparse
import torch
import numpy as np
import random
from datetime import datetime

from experiments.control_experiment import ControlExperiment


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(args.device)
    exp = ControlExperiment(args)
    exp.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env related
    parser.add_argument(
        "--env", type=str, default="cheetah", help="Environment to train on"
    )

    # model related
    parser.add_argument("--model", type=str, default="td3", choices=["ddpg", "sac", "td3"])
    
    # model hyperparameters
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    
    # episodic/step related
    parser.add_argument(
        "--learn_frequency", type=int, default=1, help="How often to learn (in steps)"
    )
    parser.add_argument(
        "--max_iter", type=int, default=1, help="max iterations on learning"
    )
    parser.add_argument(
        "--report_frequency",
        type=int,
        default=100,
        help="How often to report (in steps)",
    )

    # train related
    parser.add_argument("--max_episodes", type=int, default=99999999)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=3e-4)

    # eval related
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=10000,
        help="How often to evaluate (in steps)",
    )

    # experience replay related
    parser.add_argument("--buffer_size", type=int, default=1000000)


    # threads related
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runid", type=int, default=999)
    parser.add_argument("--uniqueid", default=False)

    # reporter related
    parser.add_argument("--result_path", type=str, default="_logs")
    parser.add_argument("--reporter_save_frequency", type=int, default=5000)
    parser.add_argument("--reporter_average_last_k", type=int, default=1)
    parser.add_argument(
        "--render", default=False, type=lambda x: (str(x).lower() == "true")
    )
    parser.add_argument("--render_frequency", type=int, default=10)

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(10_000)
        print(f"INFO: No seed specified. Sampled {args.seed}")

    if args.uniqueid:
        args.runid = str(datetime.now()).replace(" ", "_")

    if args.num_threads != -1:
        torch.set_num_threads(args.num_threads)
    main(args)
