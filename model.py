#!/usr/bin/env python3
import sys

import torch
import torch.nn as nn

USAGE = """USAGE
    python model.py ALGO MODEL [EPISODES]

DESCRIPTION
    ALGO         algorithm of the saved model: dqn or ppo
    MODEL        path to the saved model file
    EPISODES     number of episodes to run (default: 1)"""


class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes=(128, 128)):
        super().__init__()
        layers = []
        in_size = obs_dim
        for size in hidden_sizes:
            layers += [nn.Linear(in_size, size), nn.ReLU()]
            in_size = size
        layers.append(nn.Linear(in_size, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def build_mlp(input_dim, hidden_sizes, output_dim):
    layers, last_dim = [], input_dim
    for size in hidden_sizes:
        layers += [nn.Linear(last_dim, size), nn.Tanh()]
        last_dim = size
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden_sizes):
        super().__init__()
        self.actor = build_mlp(obs_dim, hidden_sizes, n_actions)
        self.critic = build_mlp(obs_dim, hidden_sizes, 1)


def usage_error(message):
    print(f"model.py: {message}\n\n{USAGE}", file=sys.stderr)
    sys.exit(84)


def parse_args(argv):
    if len(argv) not in (2, 3):
        usage_error("invalid number of arguments")
    algo, path = argv[0], argv[1]
    if algo not in ("dqn", "ppo"):
        usage_error(f"invalid ALGO '{algo}': expected dqn or ppo")
    episodes = argv[2] if len(argv) == 3 else "1"
    if not episodes.isdigit() or int(episodes) < 1:
        usage_error(f"invalid EPISODES '{episodes}': expected a positive integer")
    return algo, path, int(episodes)


def load_policy(algo, path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    try:
        config = checkpoint["config"]
        saved_algo = checkpoint.get("algo", config.get("algo"))
        net = config.get("network", config)
        sizes = (int(net["obs_dim"]), int(net["n_actions"]), net["hidden_sizes"])
    except (KeyError, TypeError, AttributeError):
        raise ValueError(f"{path} is not a valid model file") from None
    if saved_algo != algo:
        raise ValueError(f"{path} is a {saved_algo} model, not a {algo} model")

    if algo == "dqn":
        network = QNetwork(*sizes)
        greedy = network
    else:
        network = ActorCritic(*sizes)
        greedy = network.actor
    network.load_state_dict(checkpoint["state_dict"])
    network.eval()

    def policy(obs):
        with torch.no_grad():
            return int(greedy(torch.as_tensor(obs, dtype=torch.float32)).argmax())

    return policy


def main(argv):
    if argv in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    algo, path, episodes = parse_args(argv)
    try:
        load_policy(algo, path)
    except Exception as error:
        print(f"model.py: {error}", file=sys.stderr)
        return 84
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
