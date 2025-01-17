from collections import defaultdict
from pathlib import Path

import click
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th


def collect_reward_types(dict_list):
    reward_types = defaultdict(float)

    for dic in dict_list.values():
        for dictionary in dic:
            for key, value in dictionary.items():
                if "reward" in key:
                    # Extract the reward type from the key
                    reward_types[key] += value

    for key in reward_types.keys():
        reward_types[key] /= len(dict_list)

    return reward_types


def get_time_data(path):
    data_paths = list(p for p in path.iterdir() if "infos" in p.name)

    time = th.zeros(len(data_paths))
    data = dict()
    for t, file in enumerate(data_paths):
        # This can probably look nicer
        time[t] = int(file.name.split("_")[1].split(".")[0])
        data[t] = collect_reward_types(th.load(file))
    return data, time


def get_coll_timedata(src_path):
    p = Path(src_path)
    coll_data = list()
    coll_time = list()
    for i, child in enumerate(p.iterdir()):
        data, time = get_time_data(child)
        coll_data.append(data)
        coll_time.append(time)
    return coll_data, coll_time


def plot_rewards(coll_data, coll_time, plot_median=True, save_path=None):
    ref_time = th.sort(coll_time[0])[0]
    rewards = coll_data[0][0].keys()
    for reward in rewards:
        results = []
        for data, time in zip(coll_data, coll_time):
            sorttime, ids = th.sort(time)
            assert all(ref_time.eq(sorttime))
            y_data = th.tensor([data[reward] for data in data.values()])
            results.append(y_data[ids])
        results = th.stack(results, -1)
        if plot_median:
            plt.plot(ref_time, results.median(-1)[0])
            plt.fill_between(
                ref_time,
                results.quantile(0.25, -1),
                results.quantile(0.75, -1),
                alpha=0.2,
            )
        else:
            resmean = results.mean(-1)
            resstd = results.std(-1)
            plt.plot(ref_time, results.mean(-1))
            plt.fill_between(ref_time, resmean - resstd, resmean + resstd, alpha=0.2)

        plt.title(f"{reward}")
        sns.despine()
        plt.xlim(0)
        if save_path is not None:
            plt.savefig(save_path / f"{reward}.png")
            plt.close()
        else:
            plt.show()


@click.command()
@click.option("--src_dir")
@click.option("--plot_mean", default=False, is_flag=True)
@click.option("--save_path", default=None)
def vis_rewards(src_dir, plot_mean=False, save_path=None):
    coll_data, coll_time = get_coll_timedata(src_dir)
    assert len(th.cat(coll_time)) > 0, "We don't have any reward times for this src_dir"

    if save_path is not None:
        save_path = Path(src_dir) / save_path
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving figures to {save_path}")
    plot_rewards(coll_data, coll_time, plot_median=not plot_mean, save_path=save_path)


if __name__ == "__main__":
    vis_rewards()
