# main.py
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse
import shutil
import uuid
import os

# from process_file import process_file  # Your function

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <head><title>File Processor</title></head>
        <body>
            <h1>Upload a File</h1>
            <form action="/process" enctype="multipart/form-data" method="post">
                <input name="file" type="file">
                <input type="submit" value="Upload and Process">
            </form>
        </body>
    </html>
    """

@app.post("/process")
async def process(file: UploadFile = File(...)):
    # Save uploaded file
    file_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Process the file
    # output_path = process_file(input_path)
    print('input_path: ', input_path)
    output_path = 'examples_MIDI/all12/base2/gen_constr_test.mid'

    # Return processed file as a downloadable response
    return FileResponse(output_path, filename=os.path.basename(output_path), media_type='application/octet-stream')
