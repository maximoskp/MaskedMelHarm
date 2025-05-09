import pretty_midi
import numpy as np
import os
from tqdm import tqdm

def extend_chords(midi_file_path, output_path):
    midi = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Assume melody is in track 0 and chords are in track 1
    chord_instr = midi.instruments[1]

    # Collect start times of chords (grouped by simultaneous notes)
    chord_notes = sorted(chord_instr.notes, key=lambda n: n.start)
    extended_notes = []
    i = 0
    while i < len(chord_notes):
        chord_start = chord_notes[i].start
        chord_group = []
        
        # Gather all notes starting at the same time (the current chord)
        while i < len(chord_notes) and np.isclose(chord_notes[i].start, chord_start):
            chord_group.append(chord_notes[i])
            i += 1

        # Determine the end time: either start of the next chord or bar end
        if i < len(chord_notes):
            next_chord_start = chord_notes[i].start
        else:
            next_chord_start = midi.get_end_time()

        # Optionally snap to bar end instead
        beat_times = midi.get_beats()
        bar_ends = beat_times[::4]  # crude assumption: 4 beats per bar
        current_bar_end = next((b for b in bar_ends if b > chord_start), next_chord_start)

        new_end_time = min(next_chord_start, current_bar_end)

        for note in chord_group:
            extended_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=new_end_time
            )
            extended_notes.append(extended_note)

    # Replace chord notes with extended ones
    chord_instr.notes = extended_notes
    midi.write(output_path)
# end extend_chords

midis_in = '/media/maindisk/maximos/repos/MaskedMelHarm/MIDIs'

data_files = []
for dirpath, _, filenames in os.walk(midis_in):
    for file in filenames:
        if file.endswith('.mid'):
            full_path = os.path.join(dirpath, file)
            data_files.append(full_path)
print('total data_files:', len(data_files))

for f in tqdm(data_files):
    extend_chords(f,f)