"""Edit six generated scenes, mix the original score, and export a 90s film."""
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
plan = json.loads((ROOT / 'production.json').read_text())
work = ROOT / 'edit'
work.mkdir(exist_ok=True)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_format', '-show_streams', '-of', 'json', str(path)]))


def run(cmd):
    with (work / 'ffmpeg.log').open('a') as log:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL, stderr=log)


paths = []
for s in plan['shots']:
    source = ROOT / 'clips' / (s['id'] + '.mp4')
    m = probe(source)
    assert abs(float(m['format']['duration']) - 15) < .3, f'Unexpected length: {source}'
    assert any(x['codec_type'] == 'audio' for x in m['streams']), f'Missing generated audio: {source}'
    target = work / (s['id'] + '.mp4')
    if not target.exists() or target.stat().st_mtime < source.stat().st_mtime:
        run(['ffmpeg', '-y', '-v', 'warning', '-i', str(source), '-vf',
             'scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps=24,tpad=stop_mode=clone:stop_duration=0.3,trim=duration=15,setpts=PTS-STARTPTS',
             '-af', 'aresample=48000,apad,atrim=duration=15,asetpts=PTS-STARTPTS,afade=t=in:d=0.08,afade=t=out:st=14.88:d=0.12',
             '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-t', '15', str(target)])
    paths.append(target)
    print('Normalized ' + s['id'], flush=True)

concat = work / 'concat.txt'
concat.write_text(''.join("file '" + str(x).replace("'", "'\\''") + "'\n" for x in paths))
picture = work / 'picture-assembly.mp4'
run(['ffmpeg', '-y', '-v', 'warning', '-f', 'concat', '-safe', '0', '-i', str(concat), '-c', 'copy', '-t', '90', str(picture)])

score = ROOT / 'audio/score-and-effects.wav'
assert score.exists(), 'Run compose_soundtrack.py before assembling.'
mix = work / 'mix-before-master.wav'
run(['ffmpeg', '-y', '-v', 'warning', '-i', str(picture), '-i', str(score), '-filter_complex',
     '[0:a]highpass=f=45,loudnorm=I=-30:TP=-8:LRA=11[native];'
     '[1:a]highpass=f=35,loudnorm=I=-20:TP=-3:LRA=14[score];'
     '[native][score]amix=inputs=2:duration=longest:normalize=0,afade=t=in:d=0.5,afade=t=out:st=88:d=2[mix]',
     '-map', '[mix]', '-ar', '48000', '-ac', '2', '-c:a', 'pcm_f32le', '-t', '90', str(mix)])

# Two-pass loudness measurement keeps the final musical dynamics predictable.
measure = subprocess.run(['ffmpeg', '-hide_banner', '-i', str(mix), '-af', 'loudnorm=I=-16:TP=-1.5:LRA=14:print_format=json', '-f', 'null', '-'], capture_output=True, text=True, check=True)
stats = json.JSONDecoder().raw_decode(measure.stderr[measure.stderr.rfind('{'):])[0]
norm = ('loudnorm=I=-16:TP=-1.5:LRA=14:linear=true:'
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:measured_LRA={stats['input_lra']}:"
        f"measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}")
master_audio = ROOT / 'audio/final-soundtrack.wav'
run(['ffmpeg', '-y', '-v', 'warning', '-i', str(mix), '-af', norm, '-ar', '48000', '-ac', '2', '-c:a', 'pcm_s24le', '-t', '90', str(master_audio)])

clean = ROOT / 'jatayu-the-last-flight-clean.mp4'
run(['ffmpeg', '-y', '-v', 'warning', '-i', str(picture), '-i', str(master_audio), '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '256k', '-t', '90', '-movflags', '+faststart', str(clean)])
print('Mixed original score, designed effects and quiet native ambience.', flush=True)

# Short phrases leave the puppet action visible.
cues = [
    (3.5, 7.5, 'When Ravana carried Sita away,'),
    (7.7, 12.5, 'an old guardian heard her cry.'),
    (16.2, 20.7, 'Jatayu stood in his path.'),
    (21.0, 28.5, 'Age had weakened his body, but not his resolve.'),
    (32.0, 38.5, 'He broke the king’s bow.'),
    (39.0, 44.3, 'He shattered the chariot.'),
    (46.0, 51.5, 'But the struggle exhausted him.'),
    (53.5, 58.5, 'Ravana struck, and Jatayu fell.'),
    (61.0, 65.8, 'Ravana escaped with Sita.'),
    (66.1, 73.1, 'Jatayu held on long enough to bear witness.'),
    (76.1, 80.7, 'He named Sita’s captor.'),
    (81.0, 84.7, 'Rama bowed beside him.'),
    (85.0, 89.3, 'His wings fell. His courage did not.'),
]


def stamp(seconds):
    n = round(seconds * 1000)
    return f'{n//3600000:02}:{n//60000%60:02}:{n//1000%60:02},{n%1000:03}'


(ROOT / 'narration.srt').write_text('\n\n'.join(f'{i}\n{stamp(a)} --> {stamp(b)}\n{text}' for i, (a, b, text) in enumerate(cues, 1)) + '\n')
target = ROOT / 'jatayu-the-last-flight.mp4'
font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 29)
inputs = []
filters = []
current = '0:v'
for i, (start, stop, text) in enumerate(cues, 1):
    canvas = Image.new('RGBA', (1280, 86), (0, 0, 0, 0))
    d = ImageDraw.Draw(canvas)
    width = d.textlength(text, font=font)
    assert width < 1200, 'Caption is too wide'
    d.rounded_rectangle(((1280-width)/2-20, 12, (1280+width)/2+20, 64), radius=7, fill=(15, 12, 10, 218))
    d.text((640, 37), text, font=font, fill=(255, 245, 225), anchor='mm')
    png = work / f'caption-{i}.png'
    canvas.save(png)
    inputs += ['-loop', '1', '-i', str(png)]
    nxt = f'v{i}'
    filters.append(f"[{current}][{i}:v]overlay=x=0:y=H-h:enable='between(t,{start},{stop})'[{nxt}]")
    current = nxt

title = Image.new('RGBA', (1280, 720), (0, 0, 0, 0))
d = ImageDraw.Draw(title)
serif = ImageFont.truetype('/System/Library/Fonts/Supplemental/Georgia.ttf', 63)
small = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 22)
d.text((850, 130), 'JATAYU', font=serif, fill=(42, 24, 13, 245), anchor='mm')
d.text((850, 185), 'THE LAST FLIGHT', font=small, fill=(42, 24, 13, 245), anchor='mm')
d.line((720, 215, 980, 215), fill=(65, 38, 18, 180), width=1)
png = work / 'opening-title.png'
title.save(png)
inputs += ['-loop', '1', '-i', str(png)]
idx = len(cues) + 1
filters.append(f'[{idx}:v]format=rgba,fade=t=in:st=0.5:d=0.6:alpha=1,fade=t=out:st=2.8:d=0.6:alpha=1[title]')
filters.append(f'[{current}][title]overlay=0:0:shortest=1[withtitle]')
filters.append('[withtitle]fade=t=in:d=0.45,fade=t=out:st=89.1:d=0.9[out]')
run(['ffmpeg', '-y', '-v', 'warning', '-i', str(clean), *inputs, '-filter_complex', ';'.join(filters), '-map', '[out]', '-map', '0:a', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-t', '90', '-movflags', '+faststart', str(target)])

meta = probe(target)
assert abs(float(meta['format']['duration']) - 90) < .15
result = {'file': target.name, 'duration_seconds': float(meta['format']['duration']), 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'streams': meta['streams'], 'source_clips': [x.name for x in paths], 'score': 'audio/soundtrack-manifest.json', 'loudness_pre_master': stats, 'captions': 'narration.srt', 'spoken_narration': False}
(ROOT / 'master-verification.json').write_text(json.dumps(result, indent=2) + '\n')
print(target, flush=True)
