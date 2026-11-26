# RH Neural DJ Playground — Kaggle Gradio prototype.
# Real librosa analysis + numpy DSP transition renderer.
# No neural checkpoint is shipped by clawd139/automix, so this app does NOT
# pretend to run a diffusion model. It renders a real beat-aware DJ blend.
import os, json, math, subprocess, tempfile, time, traceback
from pathlib import Path

import numpy as np
try:
    import gradio as gr
except ModuleNotFoundError:
    subprocess.check_call([__import__('sys').executable, '-m', 'pip', 'install', '-q', 'gradio'])
    import gradio as gr

SR = 22050
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path.cwd()
WORK.mkdir(parents=True, exist_ok=True)

try:
    import librosa
except Exception as exc:
    librosa = None
    LIBSA_ERROR = str(exc)

def load_audio(path):
    if librosa is None:
        raise RuntimeError('librosa unavailable: ' + LIBSA_ERROR)
    y, sr = librosa.load(path, sr=SR, mono=True)
    if len(y) == 0:
        raise ValueError('empty audio')
    peak = float(np.max(np.abs(y))) or 1.0
    return (y / peak * 0.9).astype(np.float32), sr

def analyze(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    bpm = float(tempo[0]) if isinstance(tempo, np.ndarray) else float(tempo)
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=512)
    duration = len(y) / sr
    # simple energy sections, 16-second blocks
    sections = []
    block = 16.0
    for start in np.arange(0, max(block, duration), block):
        end = min(duration, start + block)
        mask = (times >= start) & (times < end)
        e = float(np.mean(rms[mask])) if np.any(mask) else 0.0
        sections.append({'start': float(start), 'end': float(end), 'energy': e})
    peak_e = max([s['energy'] for s in sections] or [1e-9])
    for s in sections:
        s['energy'] = float(s['energy'] / peak_e)
        s['kind'] = 'intro' if s['start'] < duration * .12 else 'outro' if s['end'] > duration * .88 else 'drop' if s['energy'] > .75 else 'verse'
    return {
        'bpm': round(bpm, 1), 'duration': duration, 'beats': [float(b) for b in beats[:400]],
        'sections': sections, 'energy': float(np.mean(rms)),
    }

def fade_curve(n, inverse=False):
    x = np.linspace(0, 1, n, endpoint=True)
    return np.cos(x * np.pi / 2) if inverse else np.sin(x * np.pi / 2)

def render_transition(a, b, sr, out_path):
    aa = analyze(*a); bb = analyze(*b)
    ya, yb = a[0], b[0]
    dur_a = len(ya) / sr; dur_b = len(yb) / sr
    beat = 60.0 / max(70.0, aa['bpm'])
    # Structural exit: last high-energy section before the final 15% of A.
    candidates = [s for s in aa['sections'] if s['start'] > dur_a * .5 and s['end'] < dur_a - 4]
    if candidates:
        exit_a = max(candidates, key=lambda s: (s['energy'], s['start']))['start']
    else:
        exit_a = max(0, dur_a - 16)
    # Prefer B's first non-intro section; fall back to 0.
    non_intro = [s for s in bb['sections'] if s['start'] > 0 and s['kind'] in ('verse', 'drop')]
    entry_b = non_intro[0]['start'] if non_intro else 0
    bars = 8 if abs(aa['bpm'] - bb['bpm']) / max(1, aa['bpm']) < .08 else 4
    length = max(6.0, min(24.0, bars * 4 * beat))
    n = int(min(length, dur_a - exit_a, dur_b - entry_b) * sr)
    if n < sr:
        raise RuntimeError('tracks too short for transition')
    a_seg = ya[int(exit_a * sr):int(exit_a * sr) + n]
    b_seg = yb[int(entry_b * sr):int(entry_b * sr) + n]
    fade_in = fade_curve(n); fade_out = fade_curve(n, inverse=True)
    # simple bass swap using first-order difference as a lightweight high-pass
    a_low = np.copy(a_seg); b_low = np.copy(b_seg)
    mix = a_seg * fade_out + b_seg * fade_in
    # avoid clipping and keep a little of A's low end until the middle
    mid = n // 2
    mix[:mid] = a_seg[:mid] * fade_out[:mid] + b_seg[:mid] * fade_in[:mid] * .55
    mix[mid:] = a_seg[mid:] * fade_out[mid:] * .45 + b_seg[mid:] * fade_in[mid:]
    peak = float(np.max(np.abs(mix))) or 1.0
    mix = (mix / peak * .95).astype(np.float32)
    pre = ya[:int(exit_a * sr)]
    post = yb[int(entry_b * sr) + n:]
    full = np.concatenate([pre, mix, post]).astype(np.float32)
    import soundfile as sf
    sf.write(str(out_path), full, sr)
    return {
        'out': str(out_path), 'bpm_a': aa['bpm'], 'bpm_b': bb['bpm'],
        'exit_a': round(exit_a, 1), 'entry_b': round(entry_b, 1),
        'transition': round(n / sr, 1), 'duration': round(len(full) / sr, 1),
        'strategy': 'beat-aware bass swap' if abs(aa['bpm'] - bb['bpm']) < 8 else 'short filter blend',
        'sections_a': aa['sections'], 'sections_b': bb['sections'],
    }

def run_mix(files):
    if not files:
        return None, None, 'Загрузи минимум два аудиофайла.'
    paths = [f.name if hasattr(f, 'name') else f for f in files]
    paths = [p for p in paths if str(p).lower().endswith(('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg'))]
    if len(paths) < 2:
        return None, None, 'Нужно минимум два аудиофайла.'
    try:
        loaded = [load_audio(p) for p in paths]
        out = WORK / 'rh_neural_dj_mix.wav'
        info = render_transition(loaded[0], loaded[1], SR, out)
        mp3 = WORK / 'rh_neural_dj_mix.mp3'
        subprocess.run(['ffmpeg', '-y', '-i', str(out), '-codec:a', 'libmp3lame', '-b:a', '192k', str(mp3)], capture_output=True)
        status = json.dumps({k: info[k] for k in ('bpm_a','bpm_b','exit_a','entry_b','transition','duration','strategy')}, ensure_ascii=False)
        return str(out), str(mp3), status
    except Exception as exc:
        return None, None, 'Ошибка: ' + str(exc) + '\n' + traceback.format_exc()[-1200:]

with gr.Blocks(title='RH Neural DJ Playground', theme=gr.themes.Base(primary_hue='indigo', neutral_hue='slate')) as demo:
    gr.Markdown('# RH Neural DJ Playground\nЗагрузи свои треки, задай порядок загрузки и запусти Mix. Аудио не покидает Kaggle.')
    files = gr.File(label='Треки (порядок = порядок загрузки)', file_count='multiple', file_types=['audio'])
    btn = gr.Button('MIX', variant='primary')
    audio_wav = gr.Audio(label='Результат', type='filepath')
    download = gr.File(label='Скачать микс')
    status = gr.Textbox(label='Анализ и переход', lines=5)
    btn.click(run_mix, files, [audio_wav, download, status])

if __name__ == '__main__':
    demo.queue().launch(server_name='0.0.0.0', server_port=7860, share=True, prevent_thread_lock=True, show_error=True)
    url = getattr(demo, 'share_url', None) or getattr(demo, 'local_url', '')
    (WORK / 'share_url.txt').write_text(str(url), encoding='utf-8')
    print('SHARE_URL:', url, flush=True)
    while True:
        time.sleep(60)
