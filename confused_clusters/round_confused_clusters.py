from pathlib import Path
import pandas as pd

# Original folder containing the 48 CSV files
INPUT_DIR = Path("confused_clusters")

# New folder for rounded copies
OUTPUT_DIR = Path("confused_clusters_rounded")
OUTPUT_DIR.mkdir(exist_ok=True)

# Process every CSV file in the original folder
for csv_file in sorted(INPUT_DIR.glob("*.csv")):

    # Read original CSV
    df = pd.read_csv(csv_file)

    # Round every column except sample_index
    for column in df.columns:
        if column != "sample_index":
            df[column] = df[column].round(2)

    # Save with the SAME filename in the new folder
    output_file = OUTPUT_DIR / csv_file.name
    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")

print("\n========================================")
print("Rounding complete.")
print(f"Rounded files saved in: {OUTPUT_DIR}/")
print("========================================")