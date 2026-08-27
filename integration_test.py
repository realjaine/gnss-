"""
integration_test.py
----------------------
Dry run for tomorrow's GUI: simulates a short stream of GNSS epochs (a mix
of genuine/jammed/spoofed), runs each one through:

    model prediction  ->  anti-jamming response logic  ->  printed log

This proves the two pieces (gnss_model.pkl + response_logic.py) actually
work together end to end, which is exactly what the Streamlit app will do
tomorrow, just epoch-by-epoch as they'd arrive live instead of all at once.
"""

import numpy as np
import pandas as pd
import joblib

from response_logic import AntiJammingResponder, PositionEstimate
from gnss_pipeline import generate_synthetic_genuine, simulate_jamming, simulate_spoofing

FEATURES = ["cn0_dbhz", "num_satellites", "pseudorange_m", "pseudorange_std_m", "doppler_hz", "signal_dropout"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["signal_dropout"] = df["pseudorange_m"].isna().astype(int)
    df["pseudorange_m"] = df["pseudorange_m"].fillna(0)
    df["doppler_hz"] = df["doppler_hz"].fillna(0)
    return df


def build_mixed_stream(n_total: int = 25, seed: int = 7) -> pd.DataFrame:
    """
    Builds one short, ORDERED sequence of epochs meant to imitate a UAV
    flight: starts genuine, hits a patch of jamming, recovers, then gets
    hit by a spoofing attempt. This ordering (not shuffled) is what makes
    the integration test meaningful — we want to see the response logic
    react correctly to a realistic attack pattern over time, not just
    classify isolated rows.
    """
    rng = np.random.default_rng(seed)

    genuine = generate_synthetic_genuine(n_samples=n_total, seed=seed)
    jammed_all = simulate_jamming(genuine, seed=seed + 1)
    spoofed_all = simulate_spoofing(genuine, seed=seed + 2)

    # Build a scripted sequence: genuine -> jammed patch -> genuine -> spoofed patch -> genuine
    sequence_labels = (
        ["genuine"] * 6
        + ["jammed"] * 5
        + ["genuine"] * 4
        + ["spoofed"] * 5
        + ["genuine"] * 5
    )[:n_total]

    rows = []
    for i, label in enumerate(sequence_labels):
        if label == "genuine":
            rows.append(genuine.iloc[i])
        elif label == "jammed":
            rows.append(jammed_all.iloc[i])
        else:
            rows.append(spoofed_all.iloc[i])

    stream = pd.DataFrame(rows).reset_index(drop=True)
    return stream


def run_integration_test():
    bundle = joblib.load("gnss_model.pkl")
    model = bundle["model"]
    le = bundle["label_encoder"]
    features = bundle["features"]

    stream = build_mixed_stream(n_total=25)
    stream_feat = prepare_features(stream)
    X = stream_feat[features]

    preds_encoded = model.predict(X)
    preds = le.inverse_transform(preds_encoded)

    responder = AntiJammingResponder(
        initial_position=PositionEstimate(lat=13.0827, lon=80.2707, alt_m=80.0)
    )

    print("=" * 100)
    print(f"{'Epoch':<6}{'True':<9}{'Predicted':<11}{'Match':<7}{'Alert':<7}{'Action'}")
    print("=" * 100)

    correct = 0
    for i in range(len(stream)):
        true_label = stream.loc[i, "label"]
        pred_label = preds[i]
        match = "OK" if pred_label == true_label else "MISS"
        correct += int(pred_label == true_label)

        raw_lat = stream.loc[i, "lat"] if pd.notna(stream.loc[i, "lat"]) else None
        raw_lon = stream.loc[i, "lon"] if pd.notna(stream.loc[i, "lon"]) else None
        raw_alt = stream.loc[i, "alt_m"] if pd.notna(stream.loc[i, "alt_m"]) else None

        result = responder.handle(pred_label, raw_lat, raw_lon, raw_alt)

        print(f"{i:<6}{true_label:<9}{pred_label:<11}{match:<7}{str(result.alert):<7}{result.action_taken}")

    print("=" * 100)
    print(f"\nStream classification accuracy: {correct}/{len(stream)} ({correct/len(stream):.1%})")
    print("Integration test completed with no errors — model + response logic work end to end.")


if __name__ == "__main__":
    run_integration_test()
