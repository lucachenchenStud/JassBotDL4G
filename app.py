# app.py
import json

from flask import Flask, request, jsonify
import logging
import torch

from jass.game.const import card_strings
from jass.game.game_observation import GameObservation

from trump_model import TrumpMLP
from nn_policy import NNCardPolicy
from card_model import CardPolicyNetwork

# Choose agent
from agent_nn_only import AgentNNOnly
# OR later:
from agent_nn_mcts import AgentNNMCTS

# ---------------------------------------------------------------------
# Flask setup
# ---------------------------------------------------------------------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------
# Device: PythonAnywhere = CPU only
# ---------------------------------------------------------------------
device = torch.device("cpu")

# ---------------------------------------------------------------------
# Load models ONCE (important)
# ---------------------------------------------------------------------
nn_card_policy = NNCardPolicy(
    model_class=CardPolicyNetwork,
    model_path="models/card_model.pt",
    device=device,
)

trump_model = TrumpMLP(model_path="models/trump_model.pt", device=device)


# If you later want MCTS:
agent = AgentNNOnly(
    trump_predictor=trump_model,
    policy=nn_card_policy
)

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/ready", methods=["GET"])
def ready():
    return jsonify({"status": "ready"}), 200


@app.route("/action_trump", methods=["POST"])
def play_trump():
    data = request.get_json()

    obs = GameObservation()
    obs.from_json(data)

    action = agent.action_trump(obs)
    return jsonify(int(action))

@app.route("/action_play_card", methods=["POST"])
def play_card():
    raw = request.get_json()
    obs = GameObservation.from_json(raw)

    card_id = agent.action_play_card(obs)

    # 🔑 FIX: convert to string
    card_str = card_strings[card_id]

    return jsonify({"card": card_str}), 200