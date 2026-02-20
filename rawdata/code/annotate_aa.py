import os
import glob

# Change this to your target directory
directory = "."

for filepath in glob.glob(os.path.join(directory, "*.txt")):
    filename = os.path.basename(filepath)
    stem, ext = os.path.splitext(filename)

    # Skip already processed files (avoid infinite loop)
    if stem.endswith("_aa"):
        continue

    with open(filepath, "r") as f:
        numbers = f.read().split()

    if len(numbers) < 5:
        print(f"Skipping {filename}: not enough numbers")
        continue

    # Take first 5 numbers
    first_five = numbers[:5]

    new_filename = os.path.join(directory, f"{stem}_aa.txt")

    with open(new_filename, "w") as f:
        f.write(" ".join(first_five))

    print(f"Created {new_filename}")