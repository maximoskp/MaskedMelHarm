import os
from pathlib import Path
import music21 as m21
from tqdm import tqdm

mxl_folder = '/media/maindisk/data/hooktheory_midi_hr/CA_train'
out_folder = '/media/maindisk/data/hooktheory_midi_hr/all12_train'

# Define transposition intervals (-5 to +6)
transposition_intervals = range(-5, 7)  # Includes -5, -4, ..., 0, ..., +6

# Define input and output directories
input_root = Path(mxl_folder)  # Change this to your directory
output_root = Path(out_folder)        # Change this to where you want the results

# Ensure output directory exists
output_root.mkdir(parents=True, exist_ok=True)

# Find all MusicXML files
midi_files = [file for ext in ["*.mid", "*.midi"] for file in input_root.rglob(ext)]

# Setup progress bar
with tqdm(total=len(midi_files) * (len(transposition_intervals) - 1), desc="Processing Files") as pbar:
    for input_path in midi_files:
        rel_path = input_path.relative_to(input_root)  # Preserve subfolder structure
        
        try:
            score = m21.converter.parse(input_path)  # Load the MIDI file
        except Exception as e:
            print(f"❌ Error loading {input_path}: {e}")
            continue  # Skip to the next file

        # Process transpositions
        for interval in transposition_intervals:
            try:
                # Transpose and create output path
                transposed_score = score.transpose(interval)
                output_file = output_root / rel_path.with_stem(f"{input_path.stem}_tr_{interval}")
                output_file = output_file.with_suffix(".mid")
                
                # Create necessary subdirectories
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write to file
                transposed_score.write("midi", output_file)
                pbar.update(1)  # Update progress bar
                
            except Exception as e:
                print(f"❌ Error transposing {input_path} by {interval}: {e}")
