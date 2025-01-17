import datetime
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def read_data(data_path, step_len, threshold, occurence_threshold, smoothing_window=1):
    mat = np.load(data_path, allow_pickle=True)

    cur_ep = 0
    rew = 0

    y = []
    smoothed_y = []
    cumulative_regret = []
    cumulative_reward = []

    steps = [0]

    occurence_count = 0
    dedicated_episode = None

    smoother = deque(maxlen=smoothing_window)
    for i in range(step_len):
        if mat[i][0] == cur_ep:
            rew += mat[i][2]
        else:
            smoother.append(rew)
            avg_rev = np.mean(smoother)

            if dedicated_episode is None:
                if rew > threshold:
                    occurence_count += 1
                    if occurence_count == occurence_threshold:
                        dedicated_episode = cur_ep - occurence_threshold
                else:
                    occurence_count = 0

            y.append(rew)
            smoothed_y.append(avg_rev)

            if cur_ep == 0:
                cumulative_regret.append(max(0, threshold - rew))
                cumulative_reward.append(rew)
            else:
                cumulative_regret.append(
                    cumulative_regret[-1] + max(0, threshold - rew)
                )
                cumulative_reward.append(cumulative_reward[-1] + rew)

            steps.append(i)

            rew = mat[i][2]
            cur_ep = mat[i][0]

    smoother.append(rew)
    avg_rev = np.mean(smoother)

    y.append(rew)
    smoothed_y.append(avg_rev)

    y = np.stack(y)
    smoothed_y = np.stack(smoothed_y)

    cumulative_regret = np.stack(cumulative_regret)
    cumulative_reward = np.stack(cumulative_reward)

    best_rew = y.max()
    best_rew_step = y.argmax()

    if dedicated_episode is None:
        dedicated_episode = cur_ep

    dedicated_episode += 1

    scores = {
        "AUC": np.mean(y),
        "BEST": best_rew,
        "FINAL": y[-1],
        "BEST_EPISODE": best_rew_step,
        "DEDICATED_EPISODE": dedicated_episode,
        "REGRET": (
            np.ones_like(y[:dedicated_episode]) * threshold - y[:dedicated_episode]
        )
        .clip(min=0)
        .sum(),
    }

    return y, smoothed_y, cumulative_reward, cumulative_regret, scores


def read_eval_results(data_path):
    data = np.load(data_path, allow_pickle=True).item()

    if isinstance(data[0], float):
        x = list(int(key) for key in data.keys())
        y = list(map(lambda x: data[x], data))
    else:  # isinstance(data[0], np.ndarray):
        x = list(data.keys())
        y = list(map(lambda x: data[x].mean(), data))
    return x, y


def read_eval_Q_results(data_path):
    data = np.load(data_path, allow_pickle=True).item()
    x = list(data.keys())
    y = list(map(lambda x: data[x], data))
    return x, y


def read_over_results(data_path):
    data = np.load(data_path, allow_pickle=True).item()
    x = list(data.keys())
    y = list(map(lambda x: data[x], data))
    return x, y


def read_data_single_episode(data_path, step_len, threshold):
    mat = np.load(data_path, allow_pickle=True)

    cur_ep = 0
    rew = 0

    y = []
    cumulative_reward = []
    cumulative_regret = []

    steps = [0]

    for i in range(step_len):
        rew = mat[i][2]
        y.append(rew)

        if i == 0:
            cumulative_regret.append(max(0, threshold - rew))
            cumulative_reward.append(rew)
        else:
            cumulative_regret.append(cumulative_regret[-1] + max(0, threshold - rew))
            cumulative_reward.append(cumulative_reward[-1] + rew)

        steps.append(i)

    y = np.stack(y)
    cumulative_reward = np.stack(cumulative_reward)
    cumulative_regret = np.stack(cumulative_regret)

    scores = {
        "AUC": np.mean(y),
        "REGRET": threshold - np.mean(y),
    }

    return y, cumulative_reward, cumulative_regret, scores


def plot_env_results(
    learning_curves,
    eval_curves,
    eval_Q_curves,
    over_curves,
    env_name,
    model_names_dict,
    output_dir,
    # ttest_model,
):
    fig = plt.figure(figsize=(12, 8))
    ax0 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax1 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    colors = np.array(sns.color_palette("bright", len(learning_curves)))
    # title

    fig.suptitle(env_name, fontweight="bold", fontsize=14)

    ax0.set_xlabel("Episodes", fontsize=12)
    ax0.set_ylabel("Per Episode Reward", fontsize=12)
    ax1.set_xlabel("Episodes", fontsize=12)
    ax1.set_ylabel("Per Episode Reward", fontsize=12)
    ax1.set_title("Smoothed", fontsize=12)

    # for i, model in enumerate(learning_curves.keys()):
    #     model_name = model_names_dict[model].get('name', model)

    #     y = np.stack(learning_curves[model]["y"])
    #     smoothed_y = np.stack(learning_curves[model]["smoothed_y"])

    #     mean_y = y.mean(axis=0)
    #     std_y = y.std(axis=0)/np.sqrt(y.shape[0])

    #     mean_smoothed_y = smoothed_y.mean(axis=0)
    #     std_smoothed_y = smoothed_y.std(axis=0)/np.sqrt(smoothed_y.shape[0])

    #     x = np.arange(1, len(mean_y) + 1)
    #     ax0.plot(x, mean_y, label=f'{model_name} ({mean_y.mean():.2f})', color=colors[i], linewidth=2.5)
    #     ax0.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2, color=colors[i])

    #     ax1.plot(x, mean_smoothed_y, label=f'{model_name}', color=colors[i], linewidth=2.5)
    #     ax1.fill_between(x, mean_smoothed_y - std_smoothed_y, mean_smoothed_y + std_smoothed_y, alpha=0.2, color=colors[i])

    # ax2.set_xlabel("Steps", fontsize=12)
    # ax2.set_ylabel("Reward", fontsize=12)
    # ax2.set_title("Evaluation", fontsize=12)
    # t_test_values = {}
    for i, model in enumerate(eval_curves.keys()):
        model_name = model_names_dict[model].get("name", model)

        y = np.stack(eval_curves[model]["y"])
        y_Q = np.stack(eval_Q_curves[model]["y"])

        o_x = np.stack(over_curves[model]["x"])[0]
        o_r = np.stack(over_curves[model]["y_r"])
        o_q = np.stack(over_curves[model]["y_q"])
        mean_o_r = o_r.mean(axis=0).mean(axis=1)
        std_o_r = o_r.std(axis=0).mean(axis=1) / np.sqrt(o_r.shape[0])
        mean_o_q = o_q.mean(axis=0).mean(axis=1)
        std_o_q = o_q.std(axis=0).mean(axis=1) / np.sqrt(o_q.shape[0])
        underestimation_gap = mean_o_r - mean_o_q

        mean_y = y.mean(axis=0)
        std_y = y.std(axis=0) / np.sqrt(y.shape[0])

        x = np.stack(eval_curves[model]["x"])[0]

        # -----------model selection----
        # y_Q_model = y_Q[:, 95:98]
        # mean_values = np.mean(y_Q_model[:, :, :5], axis=2)
        # idx = np.argmax(mean_values, axis=1)
        # final_rewards = np.array([y_Q_model[i, j, 10:] for i, j in enumerate(idx)])
        # ---------------
        final_rewards = y_Q[:, -1:]
        final_rewards = final_rewards.flatten()
        sorted = np.sort(final_rewards)
        q1 = np.percentile(sorted, 25)
        q3 = np.percentile(sorted, 75)
        iqmvalues = sorted[(sorted >= q1) & (sorted <= q3)]
        IQM_mean = iqmvalues.mean()
        iIQM_STD = iqmvalues.std()

        ax2.plot(
            x,
            mean_y,
            label=f"{model_name}  (final IQM reward: {IQM_mean:.2f}+/- {iIQM_STD:.2f})",
            color=colors[i],
            linewidth=2.5,
        )
        ax2.fill_between(x, mean_y - std_y, mean_y + std_y, alpha=0.2, color=colors[i])

        ax3.plot(
            o_x,
            mean_o_r,
            linestyle="--",
            linewidth=2.5,
        )
        ax3.plot(
            o_x,
            mean_o_q,
            label=f"{model_name} (Estimation_gap: {underestimation_gap.mean():.2f})",
            color=colors[i],
            linewidth=2.5,
        )
        ax3.fill_between(
            o_x, mean_o_r - std_o_r, mean_o_r + std_o_r, alpha=0.2, color=colors[i]
        )
        ax3.fill_between(
            o_x, mean_o_q - std_o_q, mean_o_q + std_o_q, alpha=0.2, color=colors[i]
        )

        # t_test_values[model_name] = [mean_y.mean(), std_y.mean(), len(mean_y)]

    # value = {}
    # for modelname in t_test_values.keys():
    #    nominator = t_test_values[ttest_model][0] - t_test_values[modelname][0]
    #    denominator = np.sqrt(
    #        (t_test_values[ttest_model][1] ** 2 / t_test_values[ttest_model][2])
    #        + (t_test_values[modelname][1] ** 2 / t_test_values[modelname][2])
    #    )
    #    value[ttest_model, modelname] = nominator / denominator
    ax2.legend(loc="lower right")
    ax2.grid()
    ax3.legend()
    ax3.grid()

    plt.tight_layout()
    now = datetime.datetime.now()

    plt.savefig(
        f"{output_dir}/results {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}.png"
    )
    plt.savefig(
        f"{output_dir}/results {now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}.pdf",
        format="pdf",
    )

    plt.close()
    # print(value)
