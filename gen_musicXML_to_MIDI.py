import os
import subprocess
from tqdm import tqdm

def num_of_filenames_with_pattern(folder, pattern):
    result = subprocess.run(
        ['bash', '-c', f'find "{folder}" -maxdepth 1 -type f | grep -c "{pattern}"'],
        capture_output=True,
        text=True
    )
    return int(result.stdout.strip())
# end num_of_filenames_with_pattern


# subfolders = ['jazz']
subfolders = ['testset']
os.makedirs('MIDIs_hr', exist_ok=True)

for subfolder in subfolders:
    print(subfolder)
    os.makedirs('MIDIs_hr/' + subfolder, exist_ok=True)
    mxl_file_names = os.listdir('musicXMLs/' + subfolder)
    for file_name in tqdm(mxl_file_names):
        pattern = file_name.split('_')[0] + '_'
        c = num_of_filenames_with_pattern('musicXMLs/' + subfolder, pattern)
        # only process those that have all competing implementations
        if c == 7:
            mxl_file_name = 'musicXMLs/' + subfolder + '/' + file_name
            midi_file_name = 'MIDIs_hr/' + subfolder + '/' + file_name + '.mid'
            os.system(f'QT_QPA_PLATFORM=offscreen mscore -o {midi_file_name} {mxl_file_name}')