import gradio as gr
import music21 as m21
import tempfile, base64, html, textwrap, os, uuid, shutil

from GridMLM_tokenizers import CSGridMLMTokenizer
from gen_gradio_funcs import (
    generate_files_with_base2,
    generate_files_with_random,
    load_model    
)
from local_gradio_funcs import make_srcdoc, render_original, load_example

# ------------------------------------------------------------------
# 1)  Load tokenizer & models 
# ------------------------------------------------------------------
print("Loading tokenizer + models …")
tokenizer = CSGridMLMTokenizer(fixed_length=256)

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" #for huggingface free

models = {
    # "base2 • all 12 keys":  load_model("base2",  "all12", DEVICE, tokenizer),
    # "random • all 12 keys": load_model("random", "all12", DEVICE, tokenizer),
    "base2 • Cmaj/Amin":    load_model("base2",  "CA",    DEVICE, tokenizer, total_stages=10),
    "random10 • Cmaj/Amin":   load_model("random10", "CA",    DEVICE, tokenizer, total_stages=10),
    "random20 • Cmaj/Amin":   load_model("random20", "CA",    DEVICE, tokenizer, total_stages=20),
}
print("✅ Models ready")

# ------------------------------------------------------------------
# 3)  Harmonisation worker
# ------------------------------------------------------------------
def harmonise(file_path: str,
              variant: str,
              constraints: bool):

    # ---------------- choose model & generation routine ----------
    is_base2 = variant.startswith("base2")
    gen_fn   = generate_files_with_base2 if is_base2 else generate_files_with_random
    model    = models[variant]
    need_norm = "Cmaj/Amin" in variant

    # ---------------- run generation -----------------------------
    name_sfx = f"{uuid.uuid4().hex}_{os.path.basename(file_path)}"
    out_root = tempfile.mkdtemp()
    mxl_dir  = os.path.join(out_root, "mxl") ; os.makedirs(mxl_dir)
    mid_dir  = os.path.join(out_root, "mid") ; os.makedirs(mid_dir)

    _, _, gen_score, _ = gen_fn(
        model              = model,
        tokenizer          = tokenizer,
        input_f            = file_path,
        mxl_folder         = mxl_dir,
        midi_folder        = mid_dir,
        name_suffix        = name_sfx,
        use_constraints    = constraints,
        normalize_tonality = need_norm,
    )


    # paths returned for download buttons (keep the .mxl/.mid MuseScore likes)
    mxl_path  = os.path.join(mxl_dir,  f"gen_{name_sfx}.mxl")
    midi_path = os.path.join(mid_dir,  f"gen_{name_sfx}.mid")

    # ---------------- build viewer iframe ------------------------
    # write PLAIN (un-compressed) MusicXML just for the viewer
    xml_tmp = os.path.join(out_root, "viewer.musicxml")
    gen_score.write("musicxml", fp=xml_tmp)
    gen_score.write("midi", fp=midi_path)

    with open(xml_tmp, "r", encoding="utf-8") as f:
        xml_str = f.read()

    with open(midi_path, "rb") as f:
        midi_b64 = base64.b64encode(f.read()).decode()

    iframe_html = make_srcdoc(xml_str, midi_b64)

    return iframe_html, mxl_path, midi_path

''' Gradio code '''

css = """
  /* shrink any component with class "small-file" */
  .small-file {
    max-height: 60px !important;
    overflow: hidden !important;
  }
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown("## 🎶 Melodic Harmonisation Demo")

    with gr.Row():
        # ---------- LEFT COLUMN : input & controls ----------
        with gr.Column():
            gr.Markdown("### 1. Upload melody and preview")

            file_in      = gr.File(type="filepath", label="MIDI or MusicXML")
            orig_viewer  = gr.HTML()                 # preview of the melody
            # hidden state to hold the preprocessed MusicXML path
            preproc_xml = gr.State()  
            # new “Clear” button
            clear_btn = gr.Button("Clear Input", variant="secondary", visible=False)
            # when clicked, wipe both the file input and the preview
            clear_btn.click(
                fn=lambda: (None, "", None, gr.update(visible=False)), 
                inputs=None, 
                outputs=[file_in, orig_viewer, preproc_xml, clear_btn]
            )

            file_in.change(render_original,   inputs=file_in, outputs=[orig_viewer, preproc_xml])
            # whenever file_in changes, show the clear button if there's a path
            file_in.change(
                fn=lambda fp: gr.update(visible=bool(fp)),
                inputs=file_in,
                outputs=clear_btn
            )            
            # load examples
            example_dd = gr.Dropdown(
                choices=sorted(os.listdir("example_inputs")),
                label="Or load a built‑in example…",
                value=None,
            )
            example_dd.change(
                fn=load_example,
                inputs=example_dd,
                outputs=[file_in, orig_viewer, preproc_xml, clear_btn],
            )

            gr.Markdown("### 2. Choose model")
            variant = gr.Dropdown(
                choices=list(models.keys()),
                value="base2 • all 12 keys",
                label="Model variant"
            )
            constraints = gr.Checkbox(label="Respect chord-constraints", value=True)
            run_btn = gr.Button("Harmonise 🎹", variant="primary")

        # ---------- RIGHT COLUMN : generated result ----------
        with gr.Column():
            gr.Markdown("### 3. Generated harmonisation")
            gen_viewer = gr.HTML()
            mxl_out    = gr.File(label="Download MusicXML", elem_classes="small-file")
            mid_out    = gr.File(label="Download MIDI", elem_classes="small-file")

    # wiring
    run_btn.click(
        fn=harmonise,
        inputs=[preproc_xml, variant, constraints],
        outputs=[gen_viewer, mxl_out, mid_out],
    )

demo.launch()