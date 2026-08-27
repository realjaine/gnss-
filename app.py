"""
app.py
-------
Streamlit GUI that wraps everything built so far (model + anti-jamming
response logic + SHAP explainability) into one live demo you can run on
your laptop and show your supervisor.

Run it with:
    streamlit run app.py

Nothing here talks to real hardware or RF signals — every "sample" is
either a row picked from gnss_dataset_combined.csv or numbers you type in
by hand, exactly like the rest of this project.

v2 additions on top of the original:
  - A real interactive map (pydeck, via st.pydeck_chart) instead of the
    plain st.map dots — shows a connecting flight-path line, per-point
    tooltips, and auto-zooms to whatever's on screen (so an "impossible
    jump" spoof to Beijing is visibly dramatic, not just two dots you
    have to guess the distance between).
  - A session-wide "Flight Trail" tab that accumulates every position
    you've classified so far (across ALL tabs) into one map — the whole
    simulated UAV path, with every jamming dead-reckoning excursion and
    every rejected spoof jump plotted together. This is the single view
    that best sells the "map visualizing genuine vs. flagged spoofed
    positions" feature from the project brief.
  - A live KPI strip (epochs classified / genuine / jammed / spoofed /
    alerts raised) that updates from session history, plus a sidebar
    "mission control" panel with a legend and a one-click session reset.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import shap
import pydeck as pdk
import streamlit as st

from response_logic import AntiJammingResponder, PositionEstimate

FEATURES = ["cn0_dbhz", "num_satellites", "pseudorange_m", "pseudorange_std_m", "doppler_hz", "signal_dropout"]

LABEL_COLORS = {"genuine": "#2ecc71", "jammed": "#f39c12", "spoofed": "#e74c3c"}

# RGBA colors for the pydeck map layers (kept separate from LABEL_COLORS,
# which is hex, because pydeck wants [r, g, b, a] lists).
MAP_RGBA = {
    "genuine": [46, 204, 113, 210],
    "jammed": [243, 156, 18, 220],
    "spoofed": [231, 76, 60, 230],
}

DEFAULT_HOME = PositionEstimate(lat=13.0827, lon=80.2707, alt_m=80.0)  # Chennai-area default start fix


# ---------------------------------------------------------------------------
# CACHING: st.cache_resource / st.cache_data
# ---------------------------------------------------------------------------
# Streamlit reruns your ENTIRE script top to bottom every time the user
# interacts with anything (clicks a button, moves a slider, etc). Without
# caching, that would mean reloading the model file and the whole CSV from
# disk on every single click - slow, and pointless since these never
# change during a session. @st.cache_resource / @st.cache_data tell
# Streamlit "compute this once, then reuse it on every rerun."
#   - cache_resource is for things that aren't plain data (models, DB
#     connections, ML objects).
#   - cache_data is for plain data (dataframes, arrays).
@st.cache_resource
def load_model_bundle():
    return joblib.load("gnss_model.pkl")


@st.cache_data
def load_dataset():
    return pd.read_csv("gnss_dataset_combined.csv")


@st.cache_resource
def get_shap_explainer(_model):
    # Leading underscore on the argument name tells Streamlit's cache "don't
    # try to hash this object to decide if it's cached before" (some ML
    # objects aren't hashable) - it still only runs once per session.
    return shap.TreeExplainer(_model)


def prepare_features(row: pd.Series) -> pd.DataFrame:
    """Same feature engineering as training: signal_dropout flag + NaN fill."""
    signal_dropout = int(pd.isna(row.get("pseudorange_m")))
    pseudorange_m = 0.0 if pd.isna(row.get("pseudorange_m")) else row["pseudorange_m"]
    doppler_hz = 0.0 if pd.isna(row.get("doppler_hz")) else row["doppler_hz"]
    return pd.DataFrame([{
        "cn0_dbhz": row["cn0_dbhz"],
        "num_satellites": row["num_satellites"],
        "pseudorange_m": pseudorange_m,
        "pseudorange_std_m": row["pseudorange_std_m"],
        "doppler_hz": doppler_hz,
        "signal_dropout": signal_dropout,
    }])


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
# Because Streamlit reruns the whole script on every interaction, plain
# variables would be wiped and recreated every time the user does anything -
# so the UAV's "last trusted position", the running classification log, and
# the flight trail would never actually persist. st.session_state survives
# reruns within one browser session, so we initialize each of these ONCE.
if "responder" not in st.session_state:
    st.session_state.responder = AntiJammingResponder(initial_position=DEFAULT_HOME)
if "history" not in st.session_state:
    st.session_state.history = []       # one dict per classified sample (all tabs)
if "trail" not in st.session_state:
    st.session_state.trail = []         # accumulated map points, session-wide
if "epoch_counter" not in st.session_state:
    st.session_state.epoch_counter = 0


def reset_session():
    st.session_state.responder = AntiJammingResponder(initial_position=DEFAULT_HOME)
    st.session_state.history = []
    st.session_state.trail = []
    st.session_state.epoch_counter = 0


# ---------------------------------------------------------------------------
# PAGE SETUP / POLISH
# ---------------------------------------------------------------------------
# st.set_page_config MUST be the very first Streamlit command in the whole
# script (Streamlit enforces this) - so it comes before even the file-loading
# guards below, even though logically "load files" feels like it should come
# first.
st.set_page_config(page_title="GNSS Spoofing & Jamming Detection", layout="wide", page_icon="🛰️")

# Light-touch CSS polish — bigger metric numbers and a bit less dead space
# up top. Deliberately minimal so it doesn't fight Streamlit's own theme
# (light or dark) or break on Streamlit Community Cloud's default styling.
st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛰️ AI-Based GNSS Spoofing & Jamming Detection for UAV Navigation")
st.caption(
    "Software-based detection demo — no live RF signals, hardware, or SDR used. "
    "All samples are either recorded GNSS logs or synthetically generated in Python."
)

# ---------------------------------------------------------------------------
# ROBUSTNESS: fail loudly but CLEANLY, not with a raw Python traceback.
# ---------------------------------------------------------------------------
# If gnss_model.pkl or gnss_dataset_combined.csv aren't in the same folder as
# this script (e.g. you forgot to copy one over), the app would otherwise
# crash on import with a scary red traceback before anything even renders.
# In front of a supervisor, that looks like the whole project is broken,
# even if it's just a missing file. This catches that specific case and
# shows one clear sentence instead, then st.stop() halts execution cleanly
# (no further code runs, no additional errors cascade from it).
try:
    bundle = load_model_bundle()
except FileNotFoundError:
    st.error(
        "gnss_model.pkl not found in this folder. Make sure it's sitting "
        "next to app.py, then refresh this page."
    )
    st.stop()

try:
    dataset = load_dataset()
except FileNotFoundError:
    st.error(
        "gnss_dataset_combined.csv not found in this folder. Make sure it's "
        "sitting next to app.py, then refresh this page."
    )
    st.stop()

if len(dataset) == 0:
    st.error("gnss_dataset_combined.csv loaded but has 0 rows — check the file isn't empty/corrupted.")
    st.stop()

model = bundle["model"]
le = bundle["label_encoder"]
explainer = get_shap_explainer(model)


# ---------------------------------------------------------------------------
# SIDEBAR: mission control panel
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛰️ Mission Control")
    st.caption("Session-wide status — updates as you classify samples in any tab.")

    h = st.session_state.history
    st.metric("Epochs classified", len(h))
    c1, c2 = st.columns(2)
    c1.metric("Alerts raised", sum(1 for r in h if r["alert"]))
    c2.metric("Spoofed flagged", sum(1 for r in h if r["predicted"] == "spoofed"))

    st.markdown("---")
    st.markdown("**Map legend**")
    st.markdown("🟢 **Genuine** — trusted, position updated normally")
    st.markdown("🟠 **Jammed** — alert + dead-reckoning fallback")
    st.markdown("🔴 **Spoofed** — alert + reading rejected, position held")

    st.markdown("---")
    if st.button("🔄 Reset session", use_container_width=True):
        reset_session()
        st.rerun()

    st.markdown("---")
    st.caption("Project repo:")
    st.markdown("[github.com/realjaine/gnss-](https://github.com/realjaine/gnss-)")


# ---------------------------------------------------------------------------
# LIVE KPI STRIP (main panel) — reflects session history from any tab
# ---------------------------------------------------------------------------
hist = st.session_state.history
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Epochs classified", len(hist))
k2.metric("🟢 Genuine", sum(1 for r in hist if r["predicted"] == "genuine"))
k3.metric("🟠 Jammed", sum(1 for r in hist if r["predicted"] == "jammed"))
k4.metric("🔴 Spoofed", sum(1 for r in hist if r["predicted"] == "spoofed"))
k5.metric("⚠️ Alerts raised", sum(1 for r in hist if r["alert"]))
st.divider()


# ---------------------------------------------------------------------------
# MAP HELPERS (pydeck) — richer than plain st.map: real basemap tiles,
# a connecting path line, hover tooltips, and auto-zoom to fit the points.
# ---------------------------------------------------------------------------
def _auto_zoom(lats, lons) -> float:
    """Rough zoom heuristic so a local hop and a cross-continent 'impossible
    jump' spoof both render at a sensible scale without manual tuning."""
    span = max(max(lats) - min(lats), max(lons) - min(lons)) if len(lats) > 1 else 0.01
    if span > 60:
        return 1.3
    if span > 20:
        return 2.5
    if span > 5:
        return 4.2
    if span > 1:
        return 6.5
    if span > 0.1:
        return 9.5
    return 12.0


def _make_deck(points: list, path_coords: list = None) -> pdk.Deck:
    """points: list of dicts with lat, lon, color ([r,g,b,a]), radius, label."""
    layers = []

    if path_coords and len(path_coords) >= 2:
        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{"path": path_coords}],
                get_path="path",
                get_width=4,
                width_min_pixels=2,
                get_color=[100, 181, 246, 150],
            )
        )

    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=points,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=6,
            radius_max_pixels=45,
            stroked=True,
            get_line_color=[255, 255, 255, 200],
            line_width_min_pixels=1,
            pickable=True,
        )
    )

    lats = [p["lat"] for p in points]
    lons = [p["lon"] for p in points]
    view_state = pdk.ViewState(
        latitude=sum(lats) / len(lats),
        longitude=sum(lons) / len(lons),
        zoom=_auto_zoom(lats, lons),
        pitch=25,
    )
    tooltip = {
        "html": "<b>{label}</b><br/>lat: {lat}<br/>lon: {lon}",
        "style": {"backgroundColor": "#1e1e1e", "color": "white"},
    }
    return pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)


def render_map(result: dict):
    """Focused, single-step map: where the UAV was, and where this epoch's
    reading/estimate landed. Design:
      - genuine -> the new trusted point, with a line back to the previous one.
      - jammed  -> last trusted position + dead-reckoned estimate.
      - spoofed -> last trusted position + the RAW (rejected) claimed
                   position — deliberately shown even though the system
                   itself ignored it, so the fake jump is visible.
    """
    pred_label = result["pred_label"]
    prev = result["prev_trusted"]
    raw_lat, raw_lon = result["raw_lat"], result["raw_lon"]
    resp = result["response"]

    points, path_coords = [], None

    if pred_label == "genuine" and raw_lat is not None:
        points.append({"lat": raw_lat, "lon": raw_lon, "color": MAP_RGBA["genuine"],
                        "radius": 25000, "label": "Genuine position (now trusted)"})
        if prev is not None:
            points.append({"lat": prev.lat, "lon": prev.lon, "color": [46, 204, 113, 90],
                            "radius": 15000, "label": "Previous trusted position"})
            path_coords = [[prev.lon, prev.lat], [raw_lon, raw_lat]]
        st.caption("🟢 Genuine position (this is now the trusted position)")

    elif pred_label == "jammed":
        points.append({"lat": prev.lat, "lon": prev.lon, "color": MAP_RGBA["genuine"],
                        "radius": 20000, "label": "Last trusted position"})
        points.append({"lat": resp.position_used.lat, "lon": resp.position_used.lon,
                        "color": MAP_RGBA["jammed"], "radius": 22000, "label": "Dead-reckoned estimate"})
        path_coords = [[prev.lon, prev.lat], [resp.position_used.lon, resp.position_used.lat]]
        st.caption("🟢 Last trusted position → 🟠 Dead-reckoned estimate (jamming fallback)")

    elif pred_label == "spoofed" and raw_lat is not None:
        points.append({"lat": prev.lat, "lon": prev.lon, "color": MAP_RGBA["genuine"],
                        "radius": 20000, "label": "Last trusted position"})
        points.append({"lat": raw_lat, "lon": raw_lon, "color": MAP_RGBA["spoofed"],
                        "radius": 38000, "label": "⚠ Flagged as spoofed (rejected)"})
        path_coords = [[prev.lon, prev.lat], [raw_lon, raw_lat]]
        st.caption("🟢 Last trusted position   🔴 Flagged as spoofed (rejected — reading NOT used)")

    else:
        st.caption("No position data available for this sample.")
        return

    st.pydeck_chart(_make_deck(points, path_coords), use_container_width=True, height=380)


def update_trail(pred_label: str, raw_lat, raw_lon, resp, epoch: int):
    """Appends this epoch's point to the session-wide flight trail (used by
    the Flight Trail tab). Capped so a very long demo session doesn't blow
    up the page."""
    trail = st.session_state.trail

    if pred_label == "genuine" and raw_lat is not None:
        trail.append({"lat": raw_lat, "lon": raw_lon, "color": MAP_RGBA["genuine"],
                       "radius": 14000, "label": f"Genuine — epoch {epoch}", "epoch": epoch, "type": "genuine"})
    elif pred_label == "jammed":
        trail.append({"lat": resp.position_used.lat, "lon": resp.position_used.lon, "color": MAP_RGBA["jammed"],
                       "radius": 14000, "label": f"Dead-reckoned — epoch {epoch}", "epoch": epoch, "type": "jammed"})
    elif pred_label == "spoofed" and raw_lat is not None:
        trail.append({"lat": raw_lat, "lon": raw_lon, "color": MAP_RGBA["spoofed"],
                       "radius": 20000, "label": f"⚠ Spoofed (rejected) — epoch {epoch}", "epoch": epoch, "type": "spoofed"})

    if len(trail) > 150:
        st.session_state.trail = trail[-150:]


# ---------------------------------------------------------------------------
# CORE FUNCTION: classify one sample, run it through the response module,
# and log it into session history + the flight trail.
# ---------------------------------------------------------------------------
def classify_and_respond(row: pd.Series, source: str = "dataset"):
    X = prepare_features(row)
    pred_encoded = model.predict(X)[0]
    pred_proba = model.predict_proba(X)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    raw_lat = row["lat"] if "lat" in row and pd.notna(row["lat"]) else None
    raw_lon = row["lon"] if "lon" in row and pd.notna(row["lon"]) else None
    raw_alt = row["alt_m"] if "alt_m" in row and pd.notna(row["alt_m"]) else None

    # Capture the trusted position BEFORE this call, since the map/trail
    # need "where we were" alongside "what happened this epoch".
    prev_trusted = st.session_state.responder.last_trusted_position
    result = st.session_state.responder.handle(pred_label, raw_lat, raw_lon, raw_alt)

    # Live SHAP values for THIS specific prediction (not just the global
    # summary) — answers "why did it say spoofed?" for the exact row shown.
    shap_vals = explainer.shap_values(X)
    if shap_vals.ndim == 3:
        class_idx = int(pred_encoded)
        shap_for_pred_class = shap_vals[0, :, class_idx]
    else:
        shap_for_pred_class = shap_vals[0]

    st.session_state.epoch_counter += 1
    epoch = st.session_state.epoch_counter

    true_label = row["label"] if "label" in row and pd.notna(row.get("label")) else "-"

    st.session_state.history.append({
        "epoch": epoch,
        "source": source,
        "true_label": true_label,
        "predicted": pred_label,
        "alert": result.alert,
        "action": result.action_taken,
    })
    update_trail(pred_label, raw_lat, raw_lon, result, epoch)

    return {
        "pred_label": pred_label,
        "pred_proba": pred_proba,
        "response": result,
        "shap_values": shap_for_pred_class,
        "X": X,
        "raw_lat": raw_lat,
        "raw_lon": raw_lon,
        "prev_trusted": prev_trusted,
        "epoch": epoch,
    }


def render_result(result: dict):
    pred_label = result["pred_label"]
    color = LABEL_COLORS[pred_label]

    st.caption(f"Epoch #{result['epoch']} this session")
    col1, col2 = st.columns([1, 1.4])

    with col1:
        st.markdown(
            f"<div style='background-color:{color}; padding:18px; border-radius:10px; "
            f"text-align:center; color:white; font-size:26px; font-weight:bold;'>"
            f"{pred_label.upper()}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("**Confidence per class**")
        conf_df = pd.DataFrame({"class": le.classes_, "confidence": result["pred_proba"]})
        st.bar_chart(conf_df.set_index("class"))

    with col2:
        resp = result["response"]
        st.markdown("**Anti-jamming response triggered**")
        if resp.alert:
            st.error(resp.action_taken)
        else:
            st.success(resp.action_taken)
        st.markdown(
            f"- Position used: `lat={resp.position_used.lat:.6f}, "
            f"lon={resp.position_used.lon:.6f}, alt={resp.position_used.alt_m:.1f}m`\n"
            f"- Reading trusted: `{resp.trusted}`"
        )

        st.markdown("**Why this prediction? (live SHAP contribution)**")
        shap_df = pd.DataFrame({
            "feature": FEATURES,
            "shap_value": result["shap_values"],
        }).sort_values("shap_value", key=abs, ascending=False)
        st.bar_chart(shap_df.set_index("feature"))

    render_map(result)


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_dataset, tab_manual, tab_stream, tab_trail, tab_explain, tab_summary = st.tabs(
    ["📂 Pick from dataset", "✍️ Manual input", "▶️ Live stream demo",
     "🗺️ Flight Trail", "🔍 Explainability", "📊 Results Summary"]
)

with tab_dataset:
    st.subheader("Classify a row from gnss_dataset_combined.csv")
    idx = st.slider("Row index", 0, len(dataset) - 1, 0)
    row = dataset.iloc[idx]
    st.dataframe(row.to_frame().T, use_container_width=True)

    if st.button("Classify this row", key="classify_dataset_row"):
        result = classify_and_respond(row, source="dataset")
        render_result(result)

with tab_manual:
    st.subheader("Enter feature values manually")
    c1, c2, c3 = st.columns(3)
    with c1:
        cn0 = st.slider("C/N0 (dBHz)", 5.0, 55.0, 40.0, help="Signal strength. Lower = more jamming-like.")
        num_sats = st.slider("Visible satellites", 0, 14, 9)
    with c2:
        pseudorange = st.number_input("Pseudorange (m)", value=22_000_000.0, step=1000.0, format="%.1f")
        pr_std = st.slider("Pseudorange spread (std, m)", 0.0, 300.0, 15.0,
                            help="Higher = signals inconsistent across satellites. Spoofing/jamming raise this.")
    with c3:
        doppler = st.slider("Doppler shift (Hz)", -3000.0, 3000.0, 0.0)
        dropout = st.checkbox("Full signal dropout this epoch", value=False)

    # Optional position override -- lets you demo an extreme "impossible
    # jump" spoof (e.g. type in coordinates for Beijing) even though the
    # dataset's own synthetic spoofing only ever jumps ~50-500m, which is
    # too small to see clearly on a world map. This is purely for the map
    # visualization; it has no effect on the classifier itself, since lat/lon
    # aren't in FEATURES.
    include_position = st.checkbox(
        "Include a position for this sample (for the map demo)", value=False
    )
    manual_lat, manual_lon = None, None
    if include_position:
        pc1, pc2 = st.columns(2)
        with pc1:
            manual_lat = st.number_input("Latitude", value=15.87, format="%.4f",
                                          help="Default is roughly Kurnool. Try something far away (e.g. 39.9 for Beijing) to demo an extreme spoof jump.")
        with pc2:
            manual_lon = st.number_input("Longitude", value=78.03, format="%.4f",
                                          help="Try 116.4 (Beijing) alongside the latitude above for a dramatic 'impossible jump' demo.")

    manual_row = pd.Series({
        "cn0_dbhz": cn0, "num_satellites": num_sats,
        "pseudorange_m": np.nan if dropout else pseudorange,
        "pseudorange_std_m": pr_std,
        "doppler_hz": np.nan if dropout else doppler,
        "lat": manual_lat, "lon": manual_lon, "alt_m": 80.0 if include_position else None,
    })

    if st.button("Classify these values", key="classify_manual"):
        result = classify_and_respond(manual_row, source="manual")
        render_result(result)

with tab_stream:
    st.subheader("Live stream demo — simulated sequence of incoming epochs")
    st.write(
        "Runs a sequence of samples (like a live feed) through detection + response, "
        "updating the display each time. Resets the session (history, trail, trusted "
        "position) for a clean run. This is the main demo view."
    )
    n_samples = st.slider("Number of samples to stream", 5, 30, 20, key="stream_n")
    speed = st.slider("Delay between samples (seconds)", 0.1, 2.0, 0.6, key="stream_speed")

    if st.button("▶ Start stream", key="start_stream"):
        reset_session()

        sample_rows = dataset.sample(n=n_samples, random_state=None).reset_index(drop=True)

        # st.empty() creates a placeholder we can overwrite in a loop.
        # This is the trick that makes a "live" animation possible even
        # though Streamlit normally only redraws on rerun: within a single
        # script execution (this one button-click), we can update the same
        # placeholder repeatedly with time.sleep() between updates, and the
        # browser shows each frame as it happens.
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        log_placeholder = st.empty()

        for i in range(len(sample_rows)):
            row = sample_rows.iloc[i]
            result = classify_and_respond(row, source="stream")

            status_placeholder.markdown(f"### Streaming epoch {i + 1} / {len(sample_rows)}")
            with result_placeholder.container():
                render_result(result)

            log_df = pd.DataFrame(st.session_state.history)
            log_placeholder.dataframe(log_df, use_container_width=True)

            time.sleep(speed)

        status_placeholder.markdown("### Stream complete.")

with tab_trail:
    st.subheader("🗺️ Full mission flight trail")
    st.write(
        "Accumulates every position classified this session — across the dataset, manual, "
        "and stream tabs — into one map: genuine fixes (green) trace the actual flight path, "
        "jamming excursions (orange) show where dead-reckoning had to take over, and rejected "
        "spoof jumps (red) show every fake position the system refused to trust. This is the "
        "single view that best demonstrates the project's spoofing/jamming detection in context."
    )

    trail = st.session_state.trail
    if len(trail) == 0:
        st.info(
            "No positions logged yet this session. Classify a few rows in the Dataset or "
            "Manual tab (or run the Live stream demo) — every genuine/jammed/spoofed epoch "
            "with a position gets added here automatically."
        )
    else:
        genuine_path = [[p["lon"], p["lat"]] for p in trail if p["type"] == "genuine"]
        deck = _make_deck(trail, path_coords=genuine_path if len(genuine_path) >= 2 else None)
        st.pydeck_chart(deck, use_container_width=True, height=480)

        t1, t2, t3, t4 = st.columns(4)
        t1.metric("Total positions", len(trail))
        t2.metric("🟢 Genuine", sum(1 for p in trail if p["type"] == "genuine"))
        t3.metric("🟠 Dead-reckoned", sum(1 for p in trail if p["type"] == "jammed"))
        t4.metric("🔴 Spoofed (rejected)", sum(1 for p in trail if p["type"] == "spoofed"))

        with st.expander("Show trail as a table"):
            st.dataframe(
                pd.DataFrame(trail)[["epoch", "type", "lat", "lon"]],
                use_container_width=True,
            )

        if st.button("🧹 Clear trail only (keep history/position)"):
            st.session_state.trail = []
            st.rerun()

with tab_explain:
    st.subheader("Model explainability")
    st.write(
        "This model (XGBoost) was evaluated with SHAP to show which features drive each "
        "class prediction. This is the global picture (from Day 2's evaluation); the "
        "per-prediction SHAP bars in the other tabs show the same idea for one specific sample."
    )
    try:
        st.image("shap_summary_bar.png", caption="SHAP feature importance by class (from Day 2 evaluation)")
    except Exception:
        st.info("shap_summary_bar.png not found in this folder — upload it alongside app.py to show it here.")

    st.markdown(
        "**How to read this:** longer bars mean that feature had a bigger average influence "
        "on the model's decision for that class. `pseudorange_std_m` and `cn0_dbhz` dominate "
        "overall — matching the literature: jamming shows up mainly in signal strength, "
        "spoofing shows up mainly in cross-satellite consistency."
    )

with tab_summary:
    # -----------------------------------------------------------------
    # RESULTS SUMMARY -- doubles as a report-ready summary, not just a
    # live demo screen. Everything here is read from files this project
    # already produced (Day 2's model comparison, Day 3's generalization
    # test) — nothing new is computed here, this tab just presents it in
    # one place instead of scattered across CSVs.
    # -----------------------------------------------------------------
    st.subheader("Model performance (Day 2)")
    if os.path.exists("model_comparison_table.csv"):
        st.dataframe(pd.read_csv("model_comparison_table.csv"), use_container_width=True)
    else:
        st.info("model_comparison_table.csv not found in this folder — copy it here to show this table.")

    st.subheader("Generalization test (Day 3)")
    if os.path.exists("generalization_comparison_table.csv"):
        gen_table = pd.read_csv("generalization_comparison_table.csv")
        st.dataframe(gen_table, use_container_width=True)
        st.caption(
            "Accuracy on the 'shifted' set (different synthetic jamming/spoofing parameters "
            "than training) is lower than on the original test set — mainly because milder "
            "jamming was often missed. This is an honest, expected finding: it shows the model "
            "learned a real but narrow signature, not that the test was broken."
        )
    else:
        st.info("generalization_comparison_table.csv not found in this folder — copy it here to show this table.")

    st.subheader("This session's live results")
    if len(hist) == 0:
        st.info("No samples classified yet this session — try the Dataset, Manual, or Live stream tabs.")
    else:
        hist_df = pd.DataFrame(hist)
        known = hist_df[hist_df["true_label"] != "-"]
        if len(known) > 0:
            session_acc = (known["true_label"] == known["predicted"]).mean()
            st.metric("Session accuracy (rows with a known true label)", f"{session_acc:.1%}")
        st.dataframe(hist_df, use_container_width=True)

    st.subheader("Anti-jamming response logic")
    anti_jamming_paragraph = (
        "When the model classifies a reading as genuine, it is trusted and used to update the "
        "UAV's tracked position and velocity as normal. When a reading is classified as jammed, "
        "it is treated as unreliable rather than fake: the system raises an alert and falls back "
        "on dead reckoning, extrapolating a short distance forward from the last trusted position "
        "using the last known velocity, since jamming degrades signal quality without injecting "
        "false information. When a reading is classified as spoofed, it is discarded entirely: "
        "the system raises an alert and holds the last trusted position, because a convincing "
        "fake reading carries no genuine navigational information and extrapolating from it would "
        "actively mislead the UAV."
    )
    st.write(anti_jamming_paragraph)

    st.divider()
    # NOTE: deliberately using plain .to_string() here instead of
    # .to_markdown() - the latter needs the optional 'tabulate' package,
    # which isn't in this project's install list, so relying on it would
    # add a new, easy-to-hit failure mode on exactly the day we're supposed
    # to be reducing fragility, not adding it.
    model_table_txt = (
        pd.read_csv("model_comparison_table.csv").to_string(index=False)
        if os.path.exists("model_comparison_table.csv") else "(table not found)"
    )
    gen_table_txt = (
        pd.read_csv("generalization_comparison_table.csv").to_string(index=False)
        if os.path.exists("generalization_comparison_table.csv") else "(table not found)"
    )
    summary_md = (
        "GNSS Spoofing & Jamming Detection — Results Summary\n"
        "=====================================================\n\n"
        "Model performance (Day 2)\n--------------------------\n"
        + model_table_txt
        + "\n\nGeneralization test (Day 3)\n----------------------------\n"
        + gen_table_txt
        + "\n\nAnti-jamming response logic\n----------------------------\n"
        + anti_jamming_paragraph + "\n"
    )
    st.download_button(
        "Download this summary as a text file",
        data=summary_md,
        file_name="gnss_results_summary.md",
        mime="text/markdown",
    )
