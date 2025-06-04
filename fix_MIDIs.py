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

def extend_chords_and_trim_64_beats(midi_file_path, output_path, beat_limit=64):
    midi = pretty_midi.PrettyMIDI(midi_file_path)
    
    # Get the time of the 64th beat (or the end of file if fewer)
    beat_times = midi.get_beats()
    if len(beat_times) >= beat_limit:
        time_cutoff = beat_times[beat_limit - 1]
    else:
        time_cutoff = midi.get_end_time()

    # Process each part (assuming part 0 = melody, part 1 = chords)
    new_instruments = []
    for idx, instr in enumerate(midi.instruments):
        if idx == 1:
            # Extend chords in part 1
            notes = sorted(instr.notes, key=lambda n: n.start)
            extended_notes = []
            i = 0
            while i < len(notes):
                chord_start = notes[i].start
                if chord_start >= time_cutoff:
                    break

                chord_group = []
                while i < len(notes) and np.isclose(notes[i].start, chord_start):
                    chord_group.append(notes[i])
                    i += 1

                if i < len(notes):
                    next_chord_start = notes[i].start
                else:
                    next_chord_start = midi.get_end_time()

                # Try to extend until either next chord or time cutoff
                new_end_time = min(next_chord_start, time_cutoff)
                for note in chord_group:
                    extended_notes.append(pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=note.start,
                        end=new_end_time
                    ))

            instr.notes = extended_notes
        else:
            # For other parts: just keep notes starting before the cutoff
            instr.notes = [n for n in instr.notes if n.start < time_cutoff]

        new_instruments.append(instr)

    # Replace instruments and write to output
    midi.instruments = new_instruments
    midi.write(output_path)
# end extend_chords_and_trim_64_beats

midis_in = '/media/maindisk/maximos/repos/MaskedMelHarm/MIDIs'

data_files = []
for dirpath, _, filenames in os.walk(midis_in):
    for file in filenames:
        if file.endswith('.mid'):
            full_path = os.path.join(dirpath, file)
            data_files.append(full_path)
print('total data_files:', len(data_files))

for f in tqdm(data_files):
    extend_chords_and_trim_64_beats(f,f)