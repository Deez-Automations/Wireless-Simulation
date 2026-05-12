"""
server.py  —  Flask bridge between the JS dashboard and the trained UA-SAC agent.

Endpoints:
  GET  /health   →  {"status": "ok", "model": "uasac_robust"}
  POST /predict  →  {"powers": [p1, p2, p3, p4]}

State vector (15 elements — matches UA-SAC training exactly):
  s* = [AP_locs(8) | User_locs(4) | Eve_perceived(2) | ρ(1)] ∈ ℝ¹⁵
  ρ = σ / D_max  (sent by the dashboard from the noise slider)

Run:  python server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from stable_baselines3 import SAC
import numpy as np

app = Flask(__name__)
CORS(app)

D_MAX = 50.0   # metres — grid boundary, same as env

# Load UA-SAC robust model (trained with σ ~ U[0,10], augment_rho=True)
try:
    model = SAC.load("models/uasac_robust")
    MODEL_NAME = "uasac_robust"
    print(f"✓ Model loaded: models/uasac_robust  (UA-SAC, 15-element state)")
except Exception as e:
    print(f"⚠ UA-SAC model not found, falling back to sac_noise_10.0: {e}")
    model = SAC.load("models/sac_noise_10.0")
    MODEL_NAME = "sac_noise_10.0 (fallback)"
    print(f"  Loaded fallback: models/sac_noise_10.0  (14-element state, no ρ)")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
      "aps":   [{x, y}, {x, y}, {x, y}, {x, y}],  // metres
      "users": [{x, y}, {x, y}],
      "eve":   {x, y},                              // perceived (already noisy)
      "rho":   0.12                                 // σ / D_max  (optional, default 0)
    }
    Returns:
    {
      "powers": [p1, p2, p3, p4]                    // Watts, 0–1
    }

    State vector s*:
      [ap1_x, ap1_y, ap2_x, ap2_y, ap3_x, ap3_y, ap4_x, ap4_y,   (8)
       u1_x,  u1_y,  u2_x,  u2_y,                                  (4)
       eve_x, eve_y,                                                (2)
       rho]                                                         (1) = 15
    """
    data = request.get_json(force=True)

    obs = []
    for ap in data["aps"]:
        obs += [ap["x"], ap["y"]]
    for u in data["users"]:
        obs += [u["x"], u["y"]]
    obs += [data["eve"]["x"], data["eve"]["y"]]

    # Append ρ — the UA-SAC uncertainty signal
    rho = float(data.get("rho", 0.0))
    rho = max(0.0, min(1.0, rho))   # clamp to valid range

    # Only append ρ if this is the UA-SAC model (15 elements)
    if MODEL_NAME.startswith("uasac"):
        obs.append(rho)

    obs_np = np.array(obs, dtype=np.float32)
    action, _ = model.predict(obs_np, deterministic=True)
    powers = np.clip(action, 0.0, 1.0).tolist()

    return jsonify({"powers": powers, "rho": rho})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
