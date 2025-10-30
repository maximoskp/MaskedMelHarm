import gradio as gr
import torch
import music21 as m21
import tempfile, base64, html, textwrap, os, uuid, shutil

from GridMLM_tokenizers import CSGridMLMTokenizer, CSGridMLMTokenizerold
from gen_gradio_funcs import (
    generate_files_with_base2,
    generate_files_with_random,
    generate_files_with_greedy,
    generate_files_with_beam,
    generate_files_with_nucleus,
    load_model,
    load_SE_model,
    load_SE_Modular,
    load_model_no_stage,
    load_DE_model
)
from local_gradio_funcs import make_srcdoc, render_original, load_example


# ------------------------------------------------------------------
# helpers 
# ------------------------------------------------------------------
class _TokShim:
    def __init__(self, tok):
        self._tok = tok
    def __getattr__(self, name):
        return getattr(self._tok, name)
    def encode(self, *args, **kwargs):
        out = self._tok.encode(*args, **kwargs)
        if isinstance(out, dict) and "input_ids" not in out and "harmony_ids" in out:
            out = dict(out)
            out["input_ids"] = out["harmony_ids"]
            if "harmony_tokens" in out and "input_tokens" not in out:
                out["input_tokens"] = out["harmony_tokens"]
        return out

def _parse_grid_length(name: str, default: int = 256) -> int:
    try:
        if "_L" in name:
            after = name.split("_L", 1)[1]
            num = ''.join(ch for ch in after if ch.isdigit())
            return int(num) if num else default
    except Exception:
        pass
    return default

def _make_tokenizer_from_subfolder(subfolder: str) -> CSGridMLMTokenizer:
    grid_len = _parse_grid_length(subfolder)
    tok = CSGridMLMTokenizer(
        fixed_length=grid_len,
        quantization="16th" if "Q16" in subfolder else "4th",
        intertwine_bar_info=("bar" in subfolder),
        trim_start=False,
        use_pc_roll=("PC" in subfolder),
        use_full_range_melody=("FR" in subfolder),
    )
    return tok

def _ensure_tupled_metrics(tok):
    if hasattr(tok, "compute_harmonic_rhythm_density"):
        _orig_r = tok.compute_harmonic_rhythm_density
        def _r(ids, _o=_orig_r, _t=tok):
            out = _o(ids)
            if isinstance(out, tuple) and len(out) == 2:
                return out
            val = float(out)
            return val, _t.to_category(val, [1.0001, 1.5556])
        tok.compute_harmonic_rhythm_density = _r
    if hasattr(tok, "compute_harmonic_complexity"):
        _orig_c = tok.compute_harmonic_complexity
        def _c(tokens, _o=_orig_c, _t=tok):
            out = _o(tokens)
            if isinstance(out, tuple) and len(out) == 2:
                return out
            val = float(out)
            return val, _t.to_category(val, [1.8225, 1.9254])
        tok.compute_harmonic_complexity = _c
    return tok

def build_se_modular_models_from_root(root_dir: str, device: str):
    out = {}
    if not os.path.isdir(root_dir):
        return out
    for sub_name in sorted(os.listdir(root_dir)):
        sub_dir = os.path.join(root_dir, sub_name)
        if not os.path.isdir(sub_dir):
            continue
        for fn in sorted(os.listdir(sub_dir)):
            if not fn.lower().endswith((".pt", ".pth")):
                continue
            curriculum = os.path.splitext(fn)[0]
            ckpt_path = os.path.join(sub_dir, fn)
            try:
                checkpoint = torch.load(ckpt_path, map_location="cpu")
            except Exception:
                continue
            tok = _make_tokenizer_from_subfolder(sub_name)
            if "melody_proj.weight" in checkpoint:
                tok.pianoroll_dim = checkpoint["melody_proj.weight"].shape[1]
            tok = _ensure_tupled_metrics(tok)
            shim_tok = _TokShim(tok)
            grid_len = _parse_grid_length(sub_name)
            condition_dim = None if "bar" in sub_name else 16
            num_stages = checkpoint["stage_embedding.weight"].shape[0] if "stage_embedding.weight" in checkpoint else None
            tpos = any(k.endswith("full_pos") for k in checkpoint.keys())
            model = load_SE_Modular(
                curriculum_type=curriculum,
                subfolder=sub_name,
                device_name=device,
                tokenizer=tok,
                grid_length=grid_len,
                condition_dim=condition_dim,
                unmasking_stages=num_stages,
                trainable_pos_emb=tpos,
                nvis=None,
            )
            key = f"SE • {sub_name} • {curriculum}"
            out[key] = (model, shim_tok)
    return out


# ------------------------------------------------------------------
# 1)  Load tokenizer & models 
# ------------------------------------------------------------------
print("Loading tokenizer + models …")
tokenizer = CSGridMLMTokenizerold(fixed_length=256)

# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE = "cpu" #for huggingface free

models = {
    # "base2 • all 12 keys":  load_model("base2",  "all12", DEVICE, tokenizer),
    # "random • all 12 keys": load_model("random", "all12", DEVICE, tokenizer),
    # "base2 • Cmaj/Amin":      (load_model("base2",   "CA", DEVICE, tokenizer, total_stages=10), tokenizer),
    # "random10 • Cmaj/Amin":   (load_model("random10","CA", DEVICE, tokenizer, total_stages=10), tokenizer),
    # "random20 • Cmaj/Amin":   (load_model("random20","CA", DEVICE, tokenizer, total_stages=20), tokenizer),
}

# add SE models found under this folder automatically
# SE_PARENT = "/media/maindisk/tsamis/repos/MaskedMelHarm/saved_models/SE"
SE_PARENT = "saved_models/SE"
models.update(build_se_modular_models_from_root(SE_PARENT, DEVICE))

print("✅ Models ready")


# ------------------------------------------------------------------
# 3)  Harmonisation worker
# ------------------------------------------------------------------
def harmonise(file_path: str,
              variant: str,
              unmasking_order: str,
              constraints: bool):
    # ---------------- choose model & generation routine ----------
    if variant.startswith("base2"):
        gen_fn = generate_files_with_base2
    elif variant.startswith("random"):
        gen_fn = generate_files_with_random
    else:
        gen_fn = generate_files_with_nucleus
    model, tok = models[variant]
    need_norm = True # "Cmaj/Amin" in variant

    # ---------------- run generation -----------------------------
    name_sfx = f"{uuid.uuid4().hex}_{os.path.basename(file_path)}"
    out_root = tempfile.mkdtemp()
    mxl_dir  = os.path.join(out_root, "mxl") ; os.makedirs(mxl_dir)
    mid_dir  = os.path.join(out_root, "mid") ; os.makedirs(mid_dir)

    _, _, gen_score, _ = gen_fn(
        model              = model,
        tokenizer          = tok,
        input_f            = file_path,
        mxl_folder         = mxl_dir,
        midi_folder        = mid_dir,
        name_suffix        = name_sfx,
        use_constraints    = constraints,
        intertwine_bar_info='bar' in variant,
        normalize_tonality = need_norm,
        temperature=0.2,
        p=0.9,
        unmasking_order=unmasking_order,
        num_stages=10,
        use_conditions=None if 'bar' in variant else 16,
        create_gen=True,
        create_real=False
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
            # preproc_xml = gr.State()
            preproc_midi = gr.State()
            # new “Clear” button
            clear_btn = gr.Button("Clear Input", variant="secondary", visible=False)
            # when clicked, wipe both the file input and the preview
            clear_btn.click(
                fn=lambda: (None, "", None, gr.update(visible=False)), 
                inputs=None, 
                outputs=[file_in, orig_viewer, preproc_midi, clear_btn]
                # outputs=[file_in, orig_viewer, preproc_xml, clear_btn]
            )

            # file_in.change(render_original,   inputs=file_in, outputs=[orig_viewer, preproc_xml])
            file_in.change(render_original,   inputs=file_in, outputs=[orig_viewer, preproc_midi])
            # whenever file_in changes, show the clear button if there's a path
            file_in.change(
                fn=lambda fp: gr.update(visible=bool(fp)),
                inputs=file_in,
                outputs=clear_btn
            )            
            # load examples
            example_dd = gr.Dropdown(
                choices=sorted(os.listdir("example_inputs")),
                label="Or load a built-in example…",
                value=None,
            )
            example_dd.change(
                fn=load_example,
                inputs=example_dd,
                outputs=[file_in, orig_viewer, preproc_midi, clear_btn],
                # outputs=[file_in, orig_viewer, preproc_xml, clear_btn],
            )

            gr.Markdown("### 2. Choose model")
            variant = gr.Dropdown(
                choices=list(models.keys()),
                value="SE • Q4_L80_bar_PC • f2f",
                label="Model variant"
            )
            gr.Markdown("### 3. Choose unmasking order")
            unmasking_order = gr.Dropdown(
                choices=['start', 'end', 'random', 'certain', 'uncertain'],
                value="certain",
                label="Unmasking order"
            )
            constraints = gr.Checkbox(label="Respect chord-constraints", value=True)
            # constraints = False
            run_btn = gr.Button("Harmonise 🎹", variant="primary")

        # ---------- RIGHT COLUMN : generated result ----------
        with gr.Column():
            gr.Markdown("### 4. Generated harmonisation")
            gen_viewer = gr.HTML()
            mxl_out    = gr.File(label="Download MusicXML", elem_classes="small-file")
            mid_out    = gr.File(label="Download MIDI", elem_classes="small-file")

    # wiring
    run_btn.click(
        fn=harmonise,
        inputs=[preproc_midi, variant, unmasking_order, constraints],
        outputs=[gen_viewer, mxl_out, mid_out],
    )

demo.launch()