# uvicorn main:app --host 0.0.0.0 --port 3052 --reload

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
from generate_utils import generate_files_with_base2, generate_files_with_random, load_model

tokenizer = CSGridMLMTokenizer(fixed_length=256)

device_name = 'cuda:0'
model_all12_base2 = load_model(
    curriculum_type='base2',
    subfolder='all12',
    device_name=device_name,
    tokenizer=tokenizer,
    pianoroll_dim=100
)
model_all12_random = load_model(
    curriculum_type='random',
    subfolder='all12',
    device_name=device_name,
    tokenizer=tokenizer,
    pianoroll_dim=100
)
model_CA_base2 = load_model(
    curriculum_type='base2',
    subfolder='CA',
    device_name=device_name,
    tokenizer=tokenizer,
    pianoroll_dim=100
)
model_CA_random = load_model(
    curriculum_type='random',
    subfolder='CA',
    device_name=device_name,
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

@app.get("/download_example_input_files", summary="Download example input ZIP")
async def download_example_input_files():
    """
    Download a ZIP file containing example input files.
    """
    file_path = "data/example_inputs.zip"  # Relative to your app's working directory
    if not os.path.isfile(file_path):
        return {"error": "File not found."}
    return FileResponse(
        path=file_path,
        filename=os.path.basename("data/example_inputs.zip"),
        media_type="application/zip"
    )
# end download_example_input_files

@app.post("/base2_all12_harmonization")
async def base2_all12_harmonization(file: UploadFile = File(..., description='MIDI or musicXML')):
    '''
    Melodic harmonization using the midpoint doubling method, 
    trained on all 12 transpositions of the dataset.
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
    if file.filename.endswith('.mid') or file.filename.endswith('.midi'):
        output_path = midi_folder + 'gen_' + f"{file_id}_{file.filename}"
    elif file.filename.endswith('.xml') or file.filename.endswith('.mxl') or file.filename.endswith('.musicxml'):
        output_path = mxl_folder + 'gen_' + f"{file_id}_{file.filename}"
    else:
        print('ERROR: unknow file extension: ', file.filename)
    os.makedirs(mxl_folder, exist_ok=True)
    os.makedirs(midi_folder, exist_ok=True)

    _, _, _, _ = generate_files_with_base2(
        model=model_all12_base2,
        tokenizer=tokenizer,
        input_f=input_path,
        mxl_folder=mxl_folder,
        midi_folder=midi_folder,
        name_suffix=f"{file_id}_{os.path.splitext(file.filename)[0]}",
        use_constraints=True,
        normalize_tonality=False
    )
    print(input_path)
    print(output_path)

    # Return processed file as a downloadable response
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')
# end base2_all12_harmonization

@app.post("/random_all12_harmonization")
async def random_all12_harmonization(file: UploadFile = File(..., description='MIDI or musicXML')):
    '''
    Melodic harmonization using the midpoint doubling method,
    trained on all 12 transpositions of the dataset.
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
    mxl_folder = DOWNLOAD_DIR + 'random/musicXML/'
    midi_folder = DOWNLOAD_DIR + 'random/MIDI/'
    if file.filename.endswith('.mid') or file.filename.endswith('.midi'):
        output_path = midi_folder + 'gen_' + f"{file_id}_{file.filename}"
    elif file.filename.endswith('.xml') or file.filename.endswith('.mxl') or file.filename.endswith('.musicxml'):
        output_path = mxl_folder + 'gen_' + f"{file_id}_{file.filename}"
    else:
        print('ERROR: unknow file extension: ', file.filename)
    os.makedirs(mxl_folder, exist_ok=True)
    os.makedirs(midi_folder, exist_ok=True)

    _, _, _, _ = generate_files_with_random(
        model=model_all12_random,
        tokenizer=tokenizer,
        input_f=input_path,
        mxl_folder=mxl_folder,
        midi_folder=midi_folder,
        name_suffix=f"{file_id}_{os.path.splitext(file.filename)[0]}",
        use_constraints=True,
        normalize_tonality=False
    )
    print(input_path)
    print(output_path)

    # Return processed file as a downloadable response
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')
# end random_all12_harmonization

@app.post("/base2_CA_harmonization")
async def base2_CA_harmonization(file: UploadFile = File(..., description='MIDI or musicXML')):
    '''
    Melodic harmonization using the midpoint doubling method, 
    trained on Cmaj-Amin transpositions of the dataset.
    The input pieces are transposed back and forth to be harmonized.
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
    if file.filename.endswith('.mid') or file.filename.endswith('.midi'):
        output_path = midi_folder + 'gen_' + f"{file_id}_{file.filename}"
    elif file.filename.endswith('.xml') or file.filename.endswith('.mxl') or file.filename.endswith('.musicxml'):
        output_path = mxl_folder + 'gen_' + f"{file_id}_{file.filename}"
    else:
        print('ERROR: unknow file extension: ', file.filename)
    os.makedirs(mxl_folder, exist_ok=True)
    os.makedirs(midi_folder, exist_ok=True)

    _, _, _, _ = generate_files_with_base2(
        model=model_CA_base2,
        tokenizer=tokenizer,
        input_f=input_path,
        mxl_folder=mxl_folder,
        midi_folder=midi_folder,
        name_suffix=f"{file_id}_{os.path.splitext(file.filename)[0]}",
        use_constraints=True,
        normalize_tonality=True
    )
    print(input_path)
    print(output_path)

    # Return processed file as a downloadable response
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')
# end base2_CA_harmonization

@app.post("/random_CA_harmonization")
async def random_CA_harmonization(file: UploadFile = File(..., description='MIDI or musicXML')):
    '''
    Melodic harmonization using the midpoint doubling method,
    trained on Cmaj-Amin transpositions of the dataset.
    The input pieces are transposed back and forth to be harmonized.
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
    mxl_folder = DOWNLOAD_DIR + 'random/musicXML/'
    midi_folder = DOWNLOAD_DIR + 'random/MIDI/'
    if file.filename.endswith('.mid') or file.filename.endswith('.midi'):
        output_path = midi_folder + 'gen_' + f"{file_id}_{file.filename}"
    elif file.filename.endswith('.xml') or file.filename.endswith('.mxl') or file.filename.endswith('.musicxml'):
        output_path = mxl_folder + 'gen_' + f"{file_id}_{file.filename}"
    else:
        print('ERROR: unknow file extension: ', file.filename)
    os.makedirs(mxl_folder, exist_ok=True)
    os.makedirs(midi_folder, exist_ok=True)

    _, _, _, _ = generate_files_with_random(
        model=model_CA_random,
        tokenizer=tokenizer,
        input_f=input_path,
        mxl_folder=mxl_folder,
        midi_folder=midi_folder,
        name_suffix=f"{file_id}_{os.path.splitext(file.filename)[0]}",
        use_constraints=True,
        normalize_tonality=True
    )
    print(input_path)
    print(output_path)

    # Return processed file as a downloadable response
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')
# end random_CA_harmonization