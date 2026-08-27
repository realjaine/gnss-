"""
generate_dataset.py
--------------------
Runs the full pipeline end-to-end and produces gnss_dataset_combined.csv.

Today, since you likely don't have a GnssLogger recording yet, this uses
generate_synthetic_genuine() as the base. As SOON as you have a real
recording, change USE_REAL_DATA to True and set GNSSLOGGER_CSV_PATH below —
nothing else in the script needs to change.
"""

from gnss_pipeline import (
    load_gnsslogger_csv,
    generate_synthetic_genuine,
    simulate_jamming,
    simulate_spoofing,
    load_public_dataset,
    combine_and_save,
    PUBLIC_DATASET_NOTE,
)

# ----- CONFIG -----
USE_REAL_DATA = False
GNSSLOGGER_CSV_PATH = "gnsslogger_raw_export.csv"  # set this once you have a real recording

USE_PUBLIC_DATASET = False
PUBLIC_DATASET_PATH = "public_gps_spoofing_dataset.csv"

N_SYNTHETIC_GENUINE = 1000
OUT_PATH = "gnss_dataset_combined.csv"
# -------------------


def main():
    # STEP 2: get genuine data (real if available, else synthetic stand-in)
    if USE_REAL_DATA:
        genuine_df = load_gnsslogger_csv(GNSSLOGGER_CSV_PATH)
        print(f"Loaded {len(genuine_df)} real genuine epochs from {GNSSLOGGER_CSV_PATH}")
    else:
        genuine_df = generate_synthetic_genuine(n_samples=N_SYNTHETIC_GENUINE)
        print(f"Generated {len(genuine_df)} synthetic genuine epochs (no real recording yet)")

    # STEP 3 & 4: derive jammed / spoofed samples from genuine samples
    jammed_df = simulate_jamming(genuine_df)
    spoofed_df = simulate_spoofing(genuine_df)
    print(f"Simulated {len(jammed_df)} jammed epochs and {len(spoofed_df)} spoofed epochs")

    dfs_to_combine = [genuine_df, jammed_df, spoofed_df]

    # STEP 5: optionally merge in the public dataset
    if USE_PUBLIC_DATASET:
        try:
            public_df = load_public_dataset(PUBLIC_DATASET_PATH)
            dfs_to_combine.append(public_df)
            print(f"Merged in {len(public_df)} rows from public dataset")
        except FileNotFoundError:
            print(f"Public dataset not found at {PUBLIC_DATASET_PATH} — continuing without it.")
            print(PUBLIC_DATASET_NOTE)
    else:
        print(PUBLIC_DATASET_NOTE)

    # STEP 6: combine, save, sanity check
    combine_and_save(dfs_to_combine, out_path=OUT_PATH)


if __name__ == "__main__":
    main()
