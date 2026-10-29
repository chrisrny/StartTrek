#!/usr/bin/env python3
import sys

USAGE = """USAGE
    python model.py ALGO MODEL [EPISODES]

DESCRIPTION
    ALGO         algorithm of the saved model: dqn or ppo
    MODEL        path to the saved model file
    EPISODES     number of episodes to run (default: 1)"""


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


def main(argv):
    if argv in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    parse_args(argv)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
