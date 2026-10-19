# Start Trek — Lunar Lander

Agents d'apprentissage par renforcement (DQN et PPO) entraînés à poser le module de `LunarLander-v3` (Gymnasium).

## Installation

Python 3.12 recommandé (gymnasium ≥ 1.2 ne supporte plus Python 3.9).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install swig
pip install -r requirements.txt
```

`swig` doit être installé avant le reste : Box2D en a besoin pour compiler.

## Structure

| Fichier | Rôle |
| --- | --- |
| `dqn.ipynb` | Agent DQN : entraînement, expériences, analyse |
| `ppo.ipynb` | Agent PPO : entraînement, expériences, comparaison avec DQN |
| `models/` | Modèles finaux sauvegardés |
| `results/` | Résultats exportés (`dqn/`, `ppo/`) pour la comparaison |
| `model.py` | Évaluation d'un modèle sauvegardé |
| `bonus/` | Bonus |
| `soutenance/` | Support de soutenance |

## Évaluation d'un modèle

```bash
python model.py ALGO MODEL [EPISODES]
```

`ALGO` : `dqn` ou `ppo` — `MODEL` : chemin du modèle — `EPISODES` : nombre d'épisodes (défaut : 1).
