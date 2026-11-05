from pathlib import Path
import pandas as pd
import sys

# ============================================================
# DIRECTORIES
# ============================================================

INPUT_DIR = Path("uncertainty_csv")
OUTPUT_DIR = Path("uncertainty_csv_rounded")

OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# FIND ALL CSV FILES
# ============================================================

input_files = sorted(INPUT_DIR.glob("*.csv"))

if not input_files:
    print("ERROR: No CSV files found in uncertainty_csv/")
    sys.exit(1)

print("=" * 70)
print("ROUNDING UNCERTAINTY CSV FILES")
print("=" * 70)
print(f"Input folder : {INPUT_DIR}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"CSV files    : {len(input_files)}")
print("=" * 70)

# ============================================================
# FIRST PASS:
# Validate EVERYTHING before writing anything
# ============================================================

print("\nSTEP 1: Validating input files...\n")

file_info = []

for csv_file in input_files:

    df = pd.read_csv(csv_file)

    # Basic information
    rows = len(df)
    columns = list(df.columns)

    # Check duplicate sample indices if sample_index exists
    duplicate_indices = 0
    if "sample_index" in df.columns:
        duplicate_indices = df["sample_index"].duplicated().sum()

    # Check missing values
    missing_values = df.isna().sum().sum()

    # Check empty dataframe
    if rows == 0:
        print(f"ERROR: {csv_file.name} contains 0 rows.")
        sys.exit(1)

    # Check missing values
    if missing_values > 0:
        print(
            f"ERROR: {csv_file.name} contains "
            f"{missing_values} missing values."
        )
        sys.exit(1)

    # Check duplicate sample indices
    if duplicate_indices > 0:
        print(
            f"ERROR: {csv_file.name} contains "
            f"{duplicate_indices} duplicate sample_index values."
        )
        sys.exit(1)

    file_info.append({
        "name": csv_file.name,
        "rows": rows,
        "columns": columns
    })

    print(
        f"OK  {csv_file.name:<45} "
        f"rows={rows:<8} columns={len(columns)}"
    )

print("\nAll input files passed validation.")

# ============================================================
# SECOND PASS:
# Create rounded copies
# ============================================================

print("\nSTEP 2: Creating rounded copies...\n")

for csv_file in input_files:

    df = pd.read_csv(csv_file)

    # Save original sample_index for final verification
    original_sample_index = None

    if "sample_index" in df.columns:
        original_sample_index = df["sample_index"].copy()

    # --------------------------------------------------------
    # Round every numeric column EXCEPT sample_index
    # --------------------------------------------------------

    for column in df.columns:

        if column == "sample_index":
            continue

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].round(2)

    # --------------------------------------------------------
    # Write with exactly 2 decimal places for floating values
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / csv_file.name

    df.to_csv(
        output_file,
        index=False,
        float_format="%.2f"
    )

    print(f"Created: {output_file}")

# ============================================================
# THIRD PASS:
# VERIFY EVERY OUTPUT FILE
# ============================================================

print("\n" + "=" * 70)
print("STEP 3: VERIFYING OUTPUT FILES")
print("=" * 70)

output_files = sorted(OUTPUT_DIR.glob("*.csv"))

# ------------------------------------------------------------
# Check file count
# ------------------------------------------------------------

if len(output_files) != len(input_files):

    print(
        f"ERROR: Input has {len(input_files)} files but "
        f"output has {len(output_files)} files."
    )

    sys.exit(1)

print(f"\nFile count check: {len(output_files)} input = {len(output_files)} output")

# ------------------------------------------------------------
# Verify every file
# ------------------------------------------------------------

for csv_file in input_files:

    original = pd.read_csv(csv_file)
    rounded_file = OUTPUT_DIR / csv_file.name
    rounded = pd.read_csv(rounded_file)

    # --------------------------------------------------------
    # 1. Row count
    # --------------------------------------------------------

    if len(original) != len(rounded):

        print(
            f"\nERROR: ROW COUNT MISMATCH in {csv_file.name}"
        )
        print(f"Original: {len(original)}")
        print(f"Rounded : {len(rounded)}")
        sys.exit(1)

    # --------------------------------------------------------
    # 2. Column count
    # --------------------------------------------------------

    if list(original.columns) != list(rounded.columns):

        print(
            f"\nERROR: COLUMN MISMATCH in {csv_file.name}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # 3. Sample index must be EXACTLY identical
    # --------------------------------------------------------

    if "sample_index" in original.columns:

        if not original["sample_index"].equals(
            rounded["sample_index"]
        ):

            print(
                f"\nERROR: sample_index changed in {csv_file.name}"
            )
            sys.exit(1)

    # --------------------------------------------------------
    # 4. No missing values introduced
    # --------------------------------------------------------

    if rounded.isna().sum().sum() != 0:

        print(
            f"\nERROR: Missing values found in rounded file "
            f"{csv_file.name}"
        )
        sys.exit(1)

    # --------------------------------------------------------
    # 5. Verify every numeric value was rounded correctly
    # --------------------------------------------------------

    for column in original.columns:

        if column == "sample_index":
            continue

        if pd.api.types.is_numeric_dtype(original[column]):

            expected = original[column].round(2)

            actual = rounded[column]

            if not expected.equals(actual):

                print(
                    f"\nERROR: Value mismatch in "
                    f"{csv_file.name}, column '{column}'"
                )

                sys.exit(1)

    print(
        f"VERIFIED  {csv_file.name:<45} "
        f"rows={len(original)}"
    )

# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(f"Input files : {len(input_files)}")
print(f"Output files: {len(output_files)}")
print()
print("Every row was preserved.")
print("Every column was preserved.")
print("Every sample_index was preserved exactly.")
print("No missing values were introduced.")
print("All numeric uncertainty values were rounded to 2 decimals.")
print()
print(f"Rounded files are in:")
print(f"    {OUTPUT_DIR}/")
print("=" * 70)