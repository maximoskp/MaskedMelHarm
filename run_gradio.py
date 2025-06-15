import gradio as gr
import music21 as m21
import tempfile, base64, html, textwrap

# ---------- server-side helpers ---------- #

def make_srcdoc(xml_str: str, midi_b64: str) -> str:
    xml_b64 = base64.b64encode(xml_str.encode("utf-8")).decode("ascii")
    doc = textwrap.dedent(f"""
    <!DOCTYPE html><html><head><meta charset="utf-8"/>
      <title>Verovio + Tone.js Playback w/ Highlight</title>
      <!-- Verovio WASM -->
      <script src="https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js" defer></script>
      <!-- Tone.js + MIDI -->
      <script src="https://cdn.jsdelivr.net/npm/tone@15.2.6/build/Tone.js" defer></script>
      <script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.27/build/Midi.js" defer></script>

      <style>
        body {{ margin:0; font-family:sans-serif; background:#fafafa; }}
        #score {{ width:100%; height:60vh; overflow:auto; border-bottom:1px solid #ccc; }}
        #controls {{ padding:8px; }}
        #seek {{ width:50%; vertical-align:middle; }}
        button {{ margin-right:8px; }}
        .highlight {{ fill:orange !important; stroke:orange !important; }}
      </style>
    </head><body>
      <div id="score"></div>
      <div id="controls">
        <button id="play">▶️ Play</button>
        <button id="pause">⏸️ Pause</button>
        <input id="seek" type="range" min="0" max="1" step="0.001" value="0">
        <span id="time">0.00 / 0.00</span>
      </div>

      <script defer>
      document.addEventListener("DOMContentLoaded", () => {{
        verovio.module.onRuntimeInitialized = async () => {{
          // 1) Instantiate toolkit & load MusicXML
          const vrv = new verovio.toolkit();

          // 1a) Dynamically size to the Gradio “canvas” container:
          const container = document.getElementById("score");
          const containerWidth  = container.clientWidth;
          const containerHeight = container.clientHeight;   

          vrv.setOptions({{
            pageWidth: containerWidth,
            // you can also set scale here if you want a % zoom:
            scale: 50,
            adjustPageHeight: true
          }});             
          
          const xmlData = atob("{xml_b64}");
          vrv.loadData(xmlData);

          // 2) Generate the internal timemap (needed for getTimeForElement) :contentReference[oaicite:2]{{index=2}}
          vrv.renderToMIDI({{ breaks: "none" }});

          container.innerHTML = vrv.renderToSVG(1);;

          // 4) Grab all <g class="note" id="…"> and map to their onset times
          const noteEls = Array.from(document.querySelectorAll("g.note[id]"));
          const noteTimes = noteEls.map(el => {{
            return {{ id: el.id, time: vrv.getTimeForElement(el.id) / 1000 }};
          }});

          // 5) Decode your external MIDI & prep Tone.js
          const midi = new Midi(
            Uint8Array.from(atob("{midi_b64}"),
              c => c.charCodeAt(0)).buffer
          );
          Tone.Transport.bpm.value = midi.header.tempos[0]?.bpm || 120;
          const totalDur = midi.duration;
          const piano = new Tone.Sampler({{
            urls: {{ 
              A0: "A0.mp3", C1: "C1.mp3", "D#1": "Ds1.mp3", "F#1": "Fs1.mp3",
              A1: "A1.mp3", C2: "C2.mp3", "D#2": "Ds2.mp3", "F#2": "Fs2.mp3",
              A2: "A2.mp3", C3: "C3.mp3", "D#3": "Ds3.mp3", "F#3": "Fs3.mp3",
              A3: "A3.mp3", C4: "C4.mp3", "D#4": "Ds4.mp3", "F#4": "Fs4.mp3",
              A4: "A4.mp3", C5: "C5.mp3", "D#5": "Ds5.mp3", "F#5": "Fs5.mp3",
              A5: "A5.mp3", C6: "C6.mp3", "D#6": "Ds6.mp3", "F#6": "Fs6.mp3",
              A6: "A6.mp3", C7: "C7.mp3", "D#7": "Ds7.mp3", "F#7": "Fs7.mp3",
              A7: "A7.mp3", C8: "C8.mp3"
            }},
            release: 1,
            baseUrl: "https://tonejs.github.io/audio/salamander/"
          }}).toDestination();
          await piano.loaded;

          // 6) Schedule playback + per-note highlighting
          const notes = midi.tracks.flatMap(t => t.notes);
          let part = null;
          function schedule(start = 0) {{
            if (part) part.dispose();
            part = new Tone.Part((time, note) => {{
              piano.triggerAttackRelease(note.name, note.duration, time, note.velocity);
              Tone.Draw.schedule(() => {{
                // clear old highlights
                noteEls.forEach(el => el.classList.remove("highlight"));
                // find the note whose onset ≃ note.time
                const match = noteTimes.find(n => Math.abs(n.time - note.time) < 0.02);
                if (match) document.getElementById(match.id).classList.add("highlight");
              }}, time);
            }}, notes.filter(n => n.time >= start)).start(0);
          }}

          // 7) Wire up Play/Pause/Seek
          document.getElementById("play").onclick = async () => {{
            await Tone.start();
            if (Tone.Transport.state !== "started") schedule(Tone.Transport.seconds);
            Tone.Transport.start();
          }};
          document.getElementById("pause").onclick = () => Tone.Transport.pause();

          const seek = document.getElementById("seek"),
                timeLbl = document.getElementById("time");
          seek.oninput = () => {{
            Tone.Transport.pause();
            Tone.Transport.seconds = parseFloat(seek.value) * totalDur;
            schedule(Tone.Transport.seconds);
          }};
          Tone.Transport.scheduleRepeat(() => {{
            const t = Tone.Transport.seconds;
            timeLbl.textContent = `${{t.toFixed(2)}} / ${{totalDur.toFixed(2)}}`;
            seek.value = t / totalDur;
            if (t >= totalDur) {{ Tone.Transport.stop(); seek.value = 0; }}
          }}, "16n");
        }};
      }});
      </script>
    </body></html>
    """)
    return (
      '<iframe sandbox="allow-scripts allow-same-origin" '
      'style="width:100%;height:700px;border:none;" '
      f'srcdoc="{html.escape(doc)}"></iframe>'
    )

def process_upload(file_path: str):
    score = m21.converter.parse(file_path)       # ← add your harmonizer here if you like

    # MusicXML string
    xml_tmp = tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False).name
    score.write("musicxml", fp=xml_tmp)
    with open(xml_tmp) as f:
        xml_str = f.read()

    # MIDI -> base64
    midi_tmp = tempfile.NamedTemporaryFile(suffix=".mid", delete=False).name
    score.write("midi", fp=midi_tmp)
    with open(midi_tmp, "rb") as f:
        midi_b64 = base64.b64encode(f.read()).decode()

    return make_srcdoc(xml_str, midi_b64)

# ---------- Gradio UI ---------- #
with gr.Blocks() as demo:
    gr.Markdown("## 🎼 Interactive Score + MIDI Playback (OSMD + Tone.js)")
    file_in  = gr.File(type="filepath", label="Upload MIDI or MusicXML")
    render   = gr.Button("Render & Play")
    viewer   = gr.HTML()                             # plain HTML (iframe string)

    render.click(process_upload, inputs=file_in, outputs=viewer)

    # quick example
    gr.Examples(["test.mxl"], inputs=file_in, outputs=viewer,
                fn=process_upload)

demo.launch()
