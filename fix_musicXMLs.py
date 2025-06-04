from music21 import converter, stream, chord, note
import os
from tqdm import tqdm

def chords_are_equal(c1, c2):
    return sorted(p.name for p in c1.pitches) == sorted(p.name for p in c2.pitches)

def process_chords(part):
    processed_part = stream.Part()
    
    for measure in part.getElementsByClass(stream.Measure):
        new_measure = stream.Measure(number=measure.number)
        
        current_chord = None
        accumulated_duration = 0.0
        last_offset = 0.0

        for element in measure:
            if isinstance(element, chord.Chord):
                if current_chord is not None and chords_are_equal(element, current_chord):
                    accumulated_duration += element.quarterLength
                else:
                    if current_chord:
                        new_chord = chord.Chord(current_chord)
                        new_chord.quarterLength = accumulated_duration
                        new_measure.insert(last_offset, new_chord)
                        last_offset += accumulated_duration

                    current_chord = element
                    accumulated_duration = element.quarterLength
            else:
                if current_chord:
                    new_chord = chord.Chord(current_chord)
                    new_chord.quarterLength = accumulated_duration
                    new_measure.insert(last_offset, new_chord)
                    last_offset += accumulated_duration
                    current_chord = None
                    accumulated_duration = 0.0

                new_measure.insert(element.offset, element)

        # Handle final chord in the measure
        if current_chord:
            new_chord = chord.Chord(current_chord)
            new_chord.quarterLength = accumulated_duration
            new_measure.insert(last_offset, new_chord)

        processed_part.append(new_measure)

    return processed_part
# end process_chords

# subfolders = ['jazz', 'testset']
subfolders = ['testset']
os.makedirs('musicXMLs_hr', exist_ok=True)

for subfolder in subfolders:
    os.makedirs('musicXMLs_hr/' + subfolder, exist_ok=True)
    file_names = os.listdir('musicXMLs/' + subfolder)
    print(subfolder)
    for file_name in tqdm(file_names):
        # Load your file
        try:
            score = converter.parse('musicXMLs/' + subfolder + '/' + file_name)
            # Assume part 1 is the chord part
            melody_part = score.parts[0]
            chord_part = score.parts[1]

            # Process the chord part
            processed_chord_part = process_chords(chord_part)

            # Replace the original part and save
            new_score = stream.Score()
            new_score.insert(0, melody_part)
            new_score.insert(0, processed_chord_part)

            # Save or show the result
            new_score.write("musicxml", fp='musicXMLs_hr/' + subfolder + '/' + file_name)
        except:
            print('problem with piece:', file_name)
