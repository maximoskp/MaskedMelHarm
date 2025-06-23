import music21 as m21
import tempfile, base64, html, textwrap


def make_srcdoc(xml_str: str, midi_b64: str) -> str:
    import base64, html, textwrap
    xml_b64 = base64.b64encode(xml_str.encode("utf-8")).decode("ascii")
    doc = textwrap.dedent(f"""
<!DOCTYPE html><html><head><meta charset="utf-8"/>
<title>Score + Playback</title>

<!-- libraries -->
<script src="https://www.verovio.org/javascript/latest/verovio-toolkit-wasm.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/tone@15.2.6/build/Tone.js" defer></script>
<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.27/build/Midi.js" defer></script>

<style>
  body {{ margin:0; font-family:sans-serif; background:#fafafa; }}

  /*── White box ────────────────────────────────────────────────*/
  /* Change *only* these two values to tune the overall height */
  #score {{  min-height:25vh;    /* default height for tiny scores  */
             max-height:90vh;    /* cap before scrolling appears    */
             overflow:auto;      /* scroll *inside* this area only  */
             width:100%;
             border-bottom:1px solid #ccc; }}

  /* responsive width: never upscale SVG */
  #score svg {{ height:auto; max-width:100%; }}

  /* controls always visible just below the score */
  #controls {{ padding:8px; }}
  #seek     {{ width:50%; vertical-align:middle; }}
  button    {{ margin-right:8px; }}

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
    /* -------- render notation -------- */
    const vrv = new verovio.toolkit();
    vrv.setOptions({{ scale:50, adjustPageHeight:true }});
    vrv.loadData(atob("{xml_b64}"));
    vrv.renderToMIDI({{breaks:"none"}});
    document.getElementById("score").innerHTML = vrv.renderToSVG(1);

    /* -------- prepare Tone.js playback -------- */
    const midi = new Midi(
      Uint8Array.from(atob("{midi_b64}"), c => c.charCodeAt(0)).buffer
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
      baseUrl:"https://tonejs.github.io/audio/salamander/"
    }}).toDestination();
    await piano.loaded;

    /* note ↔ SVG mapping for highlighting */
    const notes   = midi.tracks.flatMap(t => t.notes);
    const noteEls = Array.from(document.querySelectorAll("g.note[id]"));
    const map     = noteEls.map(el => {{
      return {{ id: el.id, time: vrv.getTimeForElement(el.id)/1000 }};
    }});

    // helper: clear all, then highlight every element whose time ≈ t
    function updateHighlight(t) {{
        noteEls.forEach(e => e.classList.remove("highlight"));
        map
        .filter(m => Math.abs(m.time - t) < 0.02)
        .forEach(m => document.getElementById(m.id).classList.add("highlight"));
    }}

    let part = null;
    function schedule(start=0) {{
      if (part) part.dispose();
      part = new Tone.Part((t, n) => {{
        piano.triggerAttackRelease(n.name, n.duration, t, n.velocity);
        // schedule *all* matching notes at once
        Tone.Draw.schedule(() => updateHighlight(n.time), t);
      }}, notes.filter(n => n.time >= start)).start(0);
    }}

    /* transport buttons */
    document.getElementById("play").onclick = async () => {{
      await Tone.start();
      if (Tone.Transport.state !== "started") schedule(Tone.Transport.seconds);
      Tone.Transport.start();
    }};
    document.getElementById("pause").onclick = () => Tone.Transport.pause();

    /* seek slider */
    const seek = document.getElementById("seek");
    const timeLbl = document.getElementById("time");
    seek.oninput = () => {{
      Tone.Transport.pause();
      const newT = parseFloat(seek.value) * totalDur;
      Tone.Transport.seconds = newT;
      schedule(newT);
      updateHighlight(newT);      // <-- highlight immediately on scrub
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
      'style="width:100%; height:55vh; border:none;" '
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


def score_to_srcdoc(score: m21.stream.Score):
    """Utility: music21 Score → iframe html."""
    xml_str = score.write("musicxml")
    midi_b = score.write("midi")
    with open(midi_b, "rb") as f:
        midi_b64 = base64.b64encode(f.read()).decode()
    return make_srcdoc(xml_str, midi_b64)


def render_original(file_path: str):

    # guard against “clear” clicks on the File widget
    if not file_path:
        # returning an empty string clears the HTML preview
      return ""
    score = m21.converter.parse(file_path)
    
    # export a plain-text MusicXML + a MIDI
    xml_tmp  = tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False).name
    midi_tmp = tempfile.NamedTemporaryFile(suffix=".mid",        delete=False).name
    score.write("musicxml", fp=xml_tmp)
    score.write("midi",     fp=midi_tmp)

    with open(xml_tmp, "r", encoding="utf-8") as f:
        xml_str = f.read()
    with open(midi_tmp, "rb") as f:
        midi_b64 = base64.b64encode(f.read()).decode()

    return make_srcdoc(xml_str, midi_b64)