import argparse
from datetime import datetime

import numpy as np
import torch
from sympy.integrals.meijerint_doc import category

from experiments.control_experiment import ControlExperiment

import warnings


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"INFO: Using device = {args.device}")
    exp = ControlExperiment(args)

    exp.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # env related
    parser.add_argument(
        "--env", type=str, default="ant"
    )  # noisycartpoleswingup  noisypendulum  noisyMountainCar  noisyacrobot

    # model related
    parser.add_argument("--model", type=str, default="sac")
    parser.add_argument("--n_hidden", type=int, default=256)
    parser.add_argument("--depth_actor", type=int, default=3)
    parser.add_argument("--width_actor", type=int, default=256)
    parser.add_argument("--no_norm_actor", default=False, action="store_true")
    parser.add_argument("--act_actor", type=str, default="crelu")
    parser.add_argument("--depth_critic", type=int, default=3)
    parser.add_argument("--width_critic", type=int, default=256)
    parser.add_argument("--no_norm_critic", default=False, action="store_true")
    parser.add_argument("--act_critic", type=str, default="crelu")

    parser.add_argument("--verbose", default=False, action="store_true")
    parser.add_argument("--progress", default=False, action="store_true")
    parser.add_argument("--tensorboard", default=False, action="store_true")
    parser.add_argument("--n_critics", type=int, default=2)  # 10
    parser.add_argument("--reduce", type=str, default="min")  # or "mean"
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=1.0)

    # parser.add_argument("--eps_loss", type=float, default=1.0)  # 1 / sqrt(2)
    parser.add_argument("--delta", type=float, default=0.25)  # 0.05 good enough
    parser.add_argument("--policy_delay", type=int, default=1)  # 1 good enough
    parser.add_argument("--validationrounds", default=False, action="store_true")
    parser.add_argument("--saveparams", default=False, action="store_true")

    # train related
    parser.add_argument("--max_steps", type=int, default=300000)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=256)  # 256
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--bootstrap_rate", type=float, default=0.05)
    parser.add_argument("--posterior_sampling_rate", type=int, default=5)
    parser.add_argument("--max_episodes", type=int, default=99999999)
    parser.add_argument("--prior_variance", type=float, default=1.0)
    parser.add_argument("--prior_mean", type=str, default="min")  # or "redq"
    parser.add_argument("--prior_scale", type=float, default=5.0)

    # exploration related
    parser.add_argument(
        "--exploration_type", type=str, default="boltzmann"
    )  # "thompson", "ofu"
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--bound_type", type=str, default="catoni")  # or "mcallester"
    parser.add_argument("--explore_noise", type=float, default=0.1)
    parser.add_argument("--complexity_coef", type=float, default=0.01)
    parser.add_argument("--num_shooting", type=int, default=500)

    # -----------------------------------------------------------------------

    # drnd related
    parser.add_argument("--n_targets_drnd", type=float, default=10)

    # bootdqn related
    parser.add_argument("--dqn_l2_reg", type=float, default=1.0)

    # episodic/step related
    parser.add_argument(
        "--reset_frequency",
        type=int,
        default=0,
        help="How often to reset the critics (in steps)",
    )
    parser.add_argument(
        "--learn_frequency", type=int, default=1, help="How often to learn (in steps)"
    )
    parser.add_argument(
        "--max_iter", type=int, default=5, help="max iterations on learning"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=0,
        help="how many epochs will the replay buffer be visited, if >0 max_iter will be overridden",
    )
    parser.add_argument(
        "--report_frequency",
        type=int,
        default=100,
        help="How often to report (in steps)",
    )

    # eval related
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument(
        "--eval_frequency",
        type=int,
        default=20000,
        help="How often to evaluate (in steps)",
    )

    # experience replay related
    parser.add_argument("--buffer_size", type=int, default=100000)

    # planning related
    # parser.add_argument("--planning_horizon", type=int, default=50)
    # parser.add_argument("--ssm_horizon", type=int, default=1)

    # bofucb related
    # parser.add_argument("--bofucb_gamma", type=float, default=0.9999)

    # threads related
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--runid", type=int, default=999)
    parser.add_argument("--uniqueid", default=False, action="store_true")

    # reporter related
    parser.add_argument("--result_path", type=str, default="_logs")
    parser.add_argument("--reporter_save_frequency", type=int, default=5000)
    parser.add_argument("--reporter_average_last_k", type=int, default=1)
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--render_frequency", type=int, default=10)

    # noisy actions and observations
    parser.add_argument("--noisy_act", type=float, default=0.0)
    parser.add_argument("--noisy_obs", type=float, default=0.0)
    # sparse
    # parser.add_argument("--healthy_reward", type=float, default=0.0)
    # parser.add_argument("--ctrl_cost_weight", type=float, default=0.001)
    # parser.add_argument("--threshold", type=float, default=2)
    parser.add_argument("--no_deprecate", default=False, action="store_true")

    args = parser.parse_args()

    if args.seed == -1:
        args.seed = np.random.randint(10_000)
        print(f"INFO: No seed specified. Sampled {args.seed}")

    # Make unique names
    if args.uniqueid:
        args.runid = f"{args.env}_{args.model}_{args.seed}_" + str(
            datetime.now()
        ).replace(" ", "_")

    if args.num_threads != -1:
        torch.set_num_threads(args.num_threads)

    if args.n_epochs > 0:
        print(f"INFO: max_iter option is overwritten by n_epochs > 0")

    if not args.no_deprecate:
        warnings.filterwarnings("default", category=DeprecationWarning)

    main(args)
