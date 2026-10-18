"""Check the delivered media, not just the editing implementation."""
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
movie=ROOT/'jatayu-the-last-flight.mp4'
meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(movie)]))
video=next(x for x in meta['streams'] if x['codec_type']=='video')
audio=next(x for x in meta['streams'] if x['codec_type']=='audio')
assert video['width']==1280 and video['height']==720
assert video['r_frame_rate']=='24/1'
assert int(video['nb_read_frames'])==2160
assert audio['channels']==2 and audio['sample_rate']=='48000'
assert abs(float(meta['format']['duration'])-90)<.1
check=subprocess.run(['ffmpeg','-hide_banner','-nostats','-xerror','-i',str(movie),'-vf','blackdetect=d=0.35:pix_th=0.025','-af','loudnorm=I=-16:TP=-1.5:LRA=14:print_format=json','-f','null','-'],capture_output=True,text=True,check=True)
loudness=json.JSONDecoder().raw_decode(check.stderr[check.stderr.rfind('{'):])[0]
assert -18<float(loudness['input_i'])<-14, loudness
assert float(loudness['input_tp']) < -.5, loudness
black=[x for x in check.stderr.splitlines() if 'black_start' in x]
data=json.loads((ROOT/'master-verification.json').read_text())
data.update(technical_qc={'full_decode':'passed','frame_count':2160,'resolution':'1280x720','fps':24,'duration_seconds':float(meta['format']['duration']),'audio_channels':2,'audio_sample_rate':48000,'integrated_lufs':float(loudness['input_i']),'true_peak_dbtp':float(loudness['input_tp']),'loudness_range_lu':float(loudness['input_lra']),'black_intervals':black,'visual_review':'Sampled frames inspected across every scene; caption/title frames checked separately.','subjective_audio_audition':False})
(ROOT/'master-verification.json').write_text(json.dumps(data,indent=2)+'\n')
for name,at in [('opening-title',1.8),('long-caption',25),('fall-caption',55),('closing-line',87)]:
    subprocess.run(['ffmpeg','-y','-v','error','-ss',str(at),'-i',str(movie),'-frames:v','1',str(ROOT/'edit/qa'/f'final-{name}.jpg')],check=True)
print(json.dumps(data['technical_qc'],indent=2))
