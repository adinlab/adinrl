import os.path
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch as th


#
def get_raw(src_dir, exclude=()):
    p = Path(src_dir)

    if os.path.exists(p / "combined_raw.npy"):
        return np.load(p / "combined_raw.npy", allow_pickle=True).item()
    res = dict()
    for child in p.iterdir():
        print(child.name)
        if ".npy" in child.name:
            continue
        if ".png" in child.name:
            continue
        name = child.relative_to(p).name
        if any(exl in name for exl in exclude):
            continue
        res[name] = {}
        if child.is_dir():
            for grandchild in child.iterdir():
                name_grandchild = grandchild.relative_to(child).name
                res[name][name_grandchild] = np.load(
                    grandchild / "eval_results.npy", allow_pickle=True
                ).item()

    np.save(p / "combined_raw.npy", res)
    return res


def check_complete(data):
    time_steps = data[0].keys()
    print(time_steps)
    return all(d.keys() == time_steps for d in data)


def collect(data):
    order = np.argsort(list(data.keys()))
    return np.array(list(data.values()))[order]


def prep_full(raw):
    # Dictionary of methods, each of which is seeds x steps x n_eval
    res = {}
    time_steps = None
    for name in raw:
        tmp = raw[name]
        tmp = [tmp[k] for k in sorted(tmp.keys())]
        # Check for consistency within a model
        assert check_complete(tmp), f"{name} is broken"
        # Check for consistency between models
        if time_steps is None:
            time_steps = tmp[0].keys()
        else:
            assert tmp[0].keys() == time_steps

        tmp = np.stack([collect(df) for df in tmp])
        res[name] = tmp
    order = np.argsort(list(time_steps))
    return res, np.array(list(time_steps))[order]


def get_ind_data(model, q=0.25):
    collect = {}
    for seed in model.keys():
        for tmp in model[seed].keys():
            if tmp not in collect.keys():
                collect[tmp] = [model[seed][tmp]]
            else:
                collect[tmp].append(model[seed][tmp])

    temp = []
    mean = []
    std = []
    median = []
    quant_lower = []
    quant_upper = []
    for key, item in collect.items():
        temp.append(key)
        tmp = th.stack(item).mean(1)
        mean.append(tmp.mean())
        std.append(tmp.std())
        median.append(tmp.quantile(0.5))
        quant_lower.append(tmp.quantile(q))
        quant_upper.append(tmp.quantile(1 - q))

    values, indices = th.tensor(temp).sort()
    return (
        values,
        th.tensor(mean)[indices],
        th.tensor(std)[indices],
        th.tensor(median)[indices],
        th.tensor(quant_lower)[indices],
        th.tensor(quant_upper)[indices],
    )


def compute_iqm(data):
    if len(data.shape) == 1:
        lower, upper = np.quantile(data, [0.25, 0.75])
        mask = (lower <= data) & (data <= upper)
        return np.sum(data * mask) / mask.sum(), lower, upper
        pass
    else:
        data = data.mean(-1)
        lower, upper = np.quantile(data, [0.25, 0.75], 0, keepdims=True)
        mask = (lower <= data) & (data <= upper)
        return np.sum(data * mask, 0) / mask.sum(0), lower[0], upper[0]


def plot_with_confidence(x_data, y_data, title=None, quantiles=None, ours=False):
    plt.plot(
        x_data, y_data, label=title if title is not None else "", lw=3 if ours else 1
    )
    if quantiles is not None:
        plt.fill_between(x_data, quantiles[0], quantiles[1], alpha=0.2)


@click.command()
@click.option("--src_dir")
@click.option("--plot_mean", default=False, is_flag=True)
@click.option("--plot_std", default=False, is_flag=True)
@click.option("--plot_iqm", default=False, is_flag=True)
@click.option("--title", default=None)
@click.option("--exclude", default=None)
@click.option("--save", default=False, is_flag=True)
@click.option("--dont_show", default=False, is_flag=True)
@click.option("--prepare_full", default=False, is_flag=True)
def vis_eval(
    src_dir,
    plot_mean=False,
    plot_std=False,
    plot_iqm=False,
    title=None,
    save=False,
    exclude=None,
    dont_show=False,
    prepare_full=False,
):
    exclude = exclude.split(",") if exclude is not None else []
    raw = get_raw(src_dir, exclude)
    if prepare_full:
        data, time = prep_full(raw)
        np.save(f"{src_dir}/combined_prepared.npy", data)
        np.save(f"{src_dir}/combined_timepoints.npy", time)
        print("INFO: Saved collected data")
    # Custom order
    if plot_iqm:
        raw = np.load(f"{src_dir}/combined_prepared.npy", allow_pickle=True).item()
        temp = np.load(f"{src_dir}/combined_timepoints.npy", allow_pickle=True)
        print(temp)
    for name in raw.keys():
        if plot_iqm:
            data, lower, upper = compute_iqm(raw[name])
            mean = data
            interval = [lower, upper]
            plot_mean = True
        else:
            model = raw[name]
            temp, mean, std, median, quant_lower, quant_upper = get_ind_data(model)
            interval = (
                [mean - std, mean + std] if plot_std else [quant_lower, quant_upper]
            )
        plot_with_confidence(
            temp,
            mean if plot_mean else median,
            title=name,
            quantiles=interval,
            ours=name == "pbac_det",
        )

    sns.despine()
    plt.legend(fontsize=15)
    plt.xlim(0)
    plt.xlabel("Episode", fontsize=15)
    plt.ylabel("Reward", fontsize=15)
    if title is not None:
        plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.grid(True)
    if save:
        plt.savefig(f"{src_dir}/result-{title}.pdf")
        print("INFO: Saved")
    if dont_show:
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    vis_eval()
