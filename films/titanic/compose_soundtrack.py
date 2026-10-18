"""An original, deterministic theatre-inspired score and synchronized Foley.

No recorded performance or third-party music samples are used. Instrument
names describe timbral inspiration, not authentic sampled instruments.
"""
import hashlib
import json
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audio'
OUT.mkdir(exist_ok=True)
SR = 48000
DURATION = 90
RNG = np.random.default_rng(25118)
N = SR * DURATION
music = np.zeros((N, 2), np.float32)
foley = np.zeros_like(music)
events = []


def noise(n, low=80, high=8000):
    """Smooth band-limited noise without external sample libraries."""
    size = 1 << (int(n) - 1).bit_length()
    freq = np.fft.rfftfreq(size, 1 / SR)
    spectrum = np.fft.rfft(RNG.normal(0, 1, size))
    response = (1 - np.exp(-(freq / low) ** 4)) * np.exp(-(freq / high) ** 4)
    out = np.fft.irfft(spectrum * response, size)[:n]
    return out / max(np.std(out), 1e-6)


def place(track, start, sample, gain=1, pan=0, room=True):
    idx = int(start * SR)
    if idx < 0:
        sample = sample[-idx:]
        idx = 0
    end = min(N, idx + len(sample))
    if end <= idx:
        return
    mono = sample[:end-idx] * gain
    track[idx:end, 0] += mono * np.sqrt((1 - pan) / 2)
    track[idx:end, 1] += mono * np.sqrt((1 + pan) / 2)
    if room:
        for seconds, amp, flip in [(0.043, .15, False), (.081, .11, True), (.137, .075, False), (.213, .045, True), (.337, .025, False)]:
            off = idx + round(seconds * SR)
            stop = min(N, off + len(mono))
            if stop > off:
                p = -pan if flip else pan
                track[off:stop, 0] += mono[:stop-off] * amp * np.sqrt((1-p)/2)
                track[off:stop, 1] += mono[:stop-off] * amp * np.sqrt((1+p)/2)


def drum(kind, velocity=1):
    duration = .65 if kind == 'bass' else .38
    t = np.arange(int(duration * SR)) / SR
    fundamental = 112 if kind == 'bass' else 226
    out = np.zeros(len(t))
    for ratio, amp, decay in [(1, 1, .17), (1.51, .53, .11), (2.15, .29, .075), (2.67, .19, .055), (3.4, .1, .04)]:
        phase = 2*np.pi*fundamental*ratio*(t + .006*(1-np.exp(-t/.014)))
        out += amp * np.sin(phase) * np.exp(-t/decay)
    slap = noise(len(t), 650, 6900) * np.exp(-t/(.018 if kind=='bass' else .032))
    out = (out*.7 + slap*(.6 if kind=='bass' else 1.25)) * (1-np.exp(-t/.0007))
    out *= np.minimum(1, (duration-t)/.025)
    return out * velocity * .22


def metal(duration=1.6, gong=False):
    t = np.arange(int(duration*SR))/SR
    base = 103 if gong else 1290
    ratios = [1, 1.47, 1.87, 2.63, 3.22, 4.17, 5.43, 6.79]
    out = np.zeros(len(t))
    for i, ratio in enumerate(ratios):
        envelope = np.exp(-t/(duration/(2.5+i*.5)))
        out += np.sin(2*np.pi*base*ratio*t + .1*np.sin(2*np.pi*4.5*t)) * envelope/(1+i*.5)
    out += noise(len(t), 1400, 11000) * np.exp(-t/.14)*.24
    return out * (1-np.exp(-t/.002)) * np.minimum(1,(duration-t)/.1) * .10


def reed(hz, duration, expressive=1):
    t = np.arange(int((duration+.25)*SR))/SR
    # Continuous pitch bend, breath and a restrained vibrato; no quantized beep.
    pitch = hz * (1+.0035*np.sin(2*np.pi*5.1*t)*np.minimum(t,1))
    pitch *= 1-.016*np.exp(-t/.065)*expressive
    phase = 2*np.pi*np.cumsum(pitch)/SR
    out = np.zeros(len(t))
    for harmonic in range(1, 11):
        weight = np.exp(-harmonic/4.2)/(harmonic**.8)
        out += weight*np.sin(harmonic*phase + .11*harmonic)
    breath = noise(len(t), 900, 5000)*.025
    env = (1-np.exp(-t/.16))*np.exp(-np.maximum(t-duration+.32,0)/.16)
    env *= np.clip((duration+.25-t)/.12,0,1)
    return (out + breath)*env*.085


def strike(at, kind, level, pan=0):
    place(music, at, drum(kind, 1+RNG.uniform(-.07,.07)), level, pan)


# A newly composed five-note motif and a breathing tonic/fifth drone.
t = np.arange(N)/SR
bed_envelope = np.interp(t,[0,3,15,28,33,39,41,45,58,60,74,77,86,90],[0,.5,.4,.45,.5,.35,.08,.25,.35,.08,.14,.25,.25,0])
drone = np.zeros(N)
for hz,gain in [(73.416,.019),(110,.009),(146.832,.007),(220,.003)]:
    drone += gain*np.sin(2*np.pi*hz*t+.12*np.sin(2*np.pi*.16*t))*(.85+.15*np.sin(2*np.pi*.11*t))
place(music,0,drone*bed_envelope,room=False)
# Arrival and romance: roomy low drums with a small-cymbal pulse.
for at in np.arange(3,29,3):
    strike(at,'bass',.38,-.12)
    if at>=15:
        strike(at+1.5,'stick',.18,.12)
        place(music,at,metal(.8),.15,.22)
melody=[(3.3,293.665,1.8),(5.7,349.228,1.3),(7.5,391.995,2.0),(10.2,440,1.5),(12.2,391.995,1.8),
        (16.0,293.665,2.0),(18.6,349.228,1.3),(20.4,391.995,2.1),(23.2,440,1.6),(25.4,349.228,2.4)]
for at,hz,dur in melody:
    place(music,at,reed(hz,dur),.65,.10)
# Approach: a measured acceleration; hush after the collision.
for i,at in enumerate(np.arange(30.5,38.8,.65)):
    strike(at,'bass' if i%3==0 else 'stick',.25+.035*i,-.1 if i%3==0 else .13)
place(music,39.0,metal(3.5,True),.55,0)
events.append({'time':39,'event':'Collision accent, then percussion falls away'})
# Lifeboats: dry, urgent articulated rhythm, stopping before the water scene.
for bar in np.arange(45.3,58.0,1.8):
    for off,kind,level in [(0,'bass',.8),(.45,'stick',.4),(.9,'bass',.6),(1.125,'stick',.4),(1.35,'stick',.55),(1.575,'stick',.3)]:
        strike(bar+off+RNG.uniform(-.008,.008),kind,level,-.13 if kind=='bass' else .13)
    place(music,bar,metal(.6),.21,.2)
for at,hz,dur in [(46.1,440,1.0),(49.0,391.995,1.5),(52.6,349.228,1.4),(56.0,293.665,1.7)]:
    place(music,at,reed(hz,dur),.48,.08)
# Loss and remembrance: the motif returns lower, slower and unaccompanied.
for at,hz,dur,level in [(62,293.665,2.6,.50),(66.1,261.626,2.4,.46),(70.3,220,3.0,.48),
                         (76.7,293.665,2.6,.55),(80.1,349.228,2.0,.50),(83.0,391.995,1.8,.46),(85.5,293.665,3.0,.52)]:
    place(music,at,reed(hz,dur),level,.08)
place(music,76,metal(4.5,True),.26,0)
events.append({'time':76,'event':'Soft gong and slower reprise for remembrance'})

# Original synthesized sea, breeze and practical theatre cues.
sea=noise(N,50,2400)
sea_env=np.interp(t,[0,3,28,39,45,57,61,72,76,87,90],[0,.6,.45,.65,.8,1,.6,.5,.3,.25,0])
swell=(.58+.42*np.sin(2*np.pi*.16*t-.8)**2)
place(foley,0,sea*sea_env*swell*.008,pan=-.1,room=False)
wind=noise(N,120,3400)
wind_env=np.interp(t,[0,15,28,43,59,61,74,85,90],[0,.4,.6,.5,.7,.35,.2,.1,0])
place(foley,0,wind*wind_env*.004,pan=.2,room=False)
for at,level in [(1.8,.24),(8.2,.17),(46,.28),(48.4,.20)]:
    place(foley,at,metal(2.2),level,-.25)
    events.append({'time':at,'event':'Distant ship bell'})
# Low scraping metal at the scripted contact, without an explosive impact.
z=np.arange(int(3.8*SR))/SR
env=np.sin(np.pi*z/3.8)**2
scrape=noise(len(z),180,5800)*(.45+.55*np.sin(2*np.pi*21*z)**2)*.018
scrape+=(np.sin(2*np.pi*147*z)+.3*np.sin(2*np.pi*389*z))*.012
place(foley,36.7,scrape*env,1.2,.15)
events.append({'time':36.7,'event':'Hull scrapes the iceberg'})
# Strained ropes and a small splash as the lifeboat reaches the water.
for at in [46.2,48.4,50.5]:
    z=np.arange(int(1.2*SR))/SR
    creak=np.sin(2*np.pi*(210*z+22*z*z)+.3*np.sin(2*np.pi*9*z))*np.sin(np.pi*z/1.2)**2*.012
    creak+=noise(len(z),350,2400)*np.sin(np.pi*z/1.2)**2*.003
    place(foley,at,creak,.8,.35)
for at,level,pan in [(56.1,.7,.4),(71.6,.4,-.3),(77.8,.28,.4),(81.3,.24,.3),(85,.20,.2)]:
    z=np.arange(int(1.1*SR))/SR
    splash=noise(len(z),100,4800)*(1-np.exp(-z/.03))*np.exp(-z/.29)*.025
    place(foley,at,splash,level,pan)
    events.append({'time':at,'event':'Soft water and oar movement'})

def finish(data):
    # Smooth both endpoints; leave headroom for the later native-audio mix.
    data[:SR] *= np.linspace(0,1,SR,dtype=np.float32)[:,None]
    data[-2*SR:] *= np.linspace(1,0,2*SR,dtype=np.float32)[:,None]
    return data


def save(name,data):
    path=OUT/name
    with wave.open(str(path),'wb') as f:
        f.setnchannels(2); f.setsampwidth(2); f.setframerate(SR)
        f.writeframes((np.clip(data,-1,1)*32767).astype('<i2').tobytes())
    return {'file':name,'duration_seconds':DURATION,'sample_rate':SR,'channels':2,'peak_dbfs':round(float(20*np.log10(max(np.max(np.abs(data)),1e-9))),2),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


music=finish(music);foley=finish(foley)
combined=music+foley
gain=.74/max(np.max(np.abs(combined)),.74)
files=[save('original-score.wav',music*gain),save('designed-effects.wav',foley*gain),save('score-and-effects.wav',combined*gain)]
manifest={'title':'Titanic: Shadows on the Sea — original theatre-inspired score','duration_seconds':90,'method':'New algorithmic composition and modal/noise synthesis; no third-party recorded samples. FLUX 3 native generated clip ambience is mixed separately.','inspiration':'Kerala Tholpavakoothu: ezhupara/maddalam-like cylindrical drums, ilathalam-like small cymbals, kuzhal-like reed and chengila-like gong. Synthesized timbres are an interpretation, not a traditional troupe recording.','source':'https://tholpavakoothu.in/about-us/about-performance/','seed':25118,'events':sorted(events,key=lambda x:x['time']),'files':files}
(OUT/'soundtrack-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'audio_stems':files},indent=2))
