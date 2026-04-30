"""
server.py  —  Flask bridge between the JS dashboard and the trained SAC agent.

Endpoints:
  GET  /health   →  {"status": "ok"}
  POST /predict  →  {"powers": [p1, p2, p3, p4]}

Run:  python server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import SAC
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the robust model (trained with σ=10m noise)
model = SAC.load("models/sac_noise_10.0")
print("Model loaded: models/sac_noise_10.0")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
      "aps":   [{x, y}, {x, y}, {x, y}, {x, y}],  // metres
      "users": [{x, y}, {x, y}],
      "eve":   {x, y}                               // perceived (already noisy)
    }
    Returns:
    {
      "powers": [p1, p2, p3, p4]                    // Watts, 0–1
    }
    """
    data = request.get_json(force=True)

    obs = []
    for ap in data["aps"]:
        obs += [ap["x"], ap["y"]]
    for u in data["users"]:
        obs += [u["x"], u["y"]]
    obs += [data["eve"]["x"], data["eve"]["y"]]

    obs_np = np.array(obs, dtype=np.float32)
    action, _ = model.predict(obs_np, deterministic=True)
    powers = np.clip(action, 0.0, 1.0).tolist()

    return jsonify({"powers": powers})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
