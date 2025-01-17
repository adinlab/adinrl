SCORES_TO_REPORT = ["AUC", "REGRET"]

ENV_INFORMATION_DICT = {
    "cartpole": {
        "name": "Cartpole Swingup",
        "threshold": 1000,
        "total_steps": 50000,
        "occurence_threshold": 5,
        "smoothing_window": 50,
    },
    "halfcheetah": {
        "name": "Half Cheetah",
        "threshold": 3000,
        "total_steps": 1000000,
        "occurence_threshold": 5,
        "smoothing_window": 50,
    },
    "ant": {
        "name": "Ant",
        "threshold": 5000,
        "total_steps": 3000000,
        "occurence_threshold": 5,
        "smoothing_window": 50,
    },
    "reacher": {
        "name": "Reacher",
        "threshold": 5000,
        "total_steps": 3000000,
        "occurence_threshold": 5,
        "smoothing_window": 50,
    },
}

MODELS_DICT = {
    "pacsac-ucb-1vb": {
        "name": "PAC4SAC with 1VB",
    },
    "sac": {
        "name": "SAC",
    },
    "oac": {
        "name": "OAC",
    },
    "td3": {
        "name": "TD3",
    },
    "ddpg": {
        "name": "DDPG",
    },
    "pactd3-2vb": {
        "name": "PAC4TD3 with 2VB",
    },
    "drnd": {"name": "DRND"},
}
