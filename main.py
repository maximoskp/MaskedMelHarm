# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
import shutil
import uuid
import os
import torch
import torch.nn as nn
from models import GridMLMMelHarm
from GridMLM_tokenizers import CSGridMLMTokenizer
from data_utils import CSGridMLMDataset, CSGridMLM_collate_fn
from torch.utils.data import DataLoader
from train_utils import apply_masking
from generate_utils import generate_files_with_base2, generate_files_with_base2, load_model

tokenizer = CSGridMLMTokenizer(fixed_length=256)

# curriculum_type = 'base2'
subfolder = 'all12'
device_name = 'cuda:2'
model_base2 = load_model(
    curriculum_type='base2',
    subfolder=subfolder,
    device_name='cuda:2',
    tokenizer=tokenizer,
    pianoroll_dim=100
)
model_random = load_model(
    curriculum_type='random',
    subfolder=subfolder,
    device_name='cuda:2',
    tokenizer=tokenizer,
    pianoroll_dim=100
)

app = FastAPI(
    title='Melody harmonization API',
    description='Allows uploading a MIDI or musicXML \
        file with a melody and optional constraints and returns a harmonized version.',
    version='0.0.1'
)

UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)
DOWNLOAD_DIR = "downloads/"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)


@app.post("/process")
async def base2_harmonization(file: UploadFile = File(..., description='MIDI or musicXML')):
    '''
    Melodic harmonization using the midpoint doubling method.
    Upload a melody plus chord constraints MIDI or musicXML file.

    MIDI file: should inlude a first part with the melody and an optional second part
    with the chord constraints.

    musicXML file: should include a single part with the melody and optional chord symbols
    with the chord constraints.
    '''
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the file
    # output_path = process_file(input_path)
    mxl_folder = DOWNLOAD_DIR + 'base2/musicXML/'
    midi_folder = DOWNLOAD_DIR + 'base2/MIDI/'
    output_path = midi_folder + 'gen_' + f"{file_id}_{file.filename}"
    os.makedirs(mxl_folder, exist_ok=True)
    os.makedirs(midi_folder, exist_ok=True)

    _, _, _, _ = generate_files_with_base2(
        model=model_base2,
        tokenizer=tokenizer,
        input_f=input_path,
        mxl_folder=mxl_folder,
        midi_folder=midi_folder,
        name_suffix=f"{file_id}_{os.path.splitext(file.filename)[0]}",
        use_constraints=True
    )
    print(input_path)
    print(output_path)

    # Return processed file as a downloadable response
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')
