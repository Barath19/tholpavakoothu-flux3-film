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
RNG = np.random.default_rng(24618)
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


# A quiet, breathing tonic/fifth bed ties together the separately rendered shots.
t = np.arange(N)/SR
bed_envelope = np.interp(t, [0,2,14,30,48,54,58,61,74,84,88,90], [0,.40,.40,.50,.50,.18,.12,.20,.28,.30,.16,0])
drone = np.zeros(N)
for hz, gain in [(73.416,.018),(110,.009),(146.832,.008),(220,.003)]:
    drone += gain*np.sin(2*np.pi*hz*t+.12*np.sin(2*np.pi*.16*t))*(.85+.15*np.sin(2*np.pi*.11*t))
place(music, 0, drone*bed_envelope, room=False)

# Opening: sparse strokes; challenge: alternating low drum and dry stick strokes.
for at in [2.6,6.1,9.6,12.5]:
    strike(at,'bass',.58,-.13)
for at,hz,dur in [(3.2,293.665,2.2),(7.1,349.228,1.8),(10.4,391.995,2.4)]:
    place(music, at, reed(hz,dur), .60, .12)
for bar in np.arange(15.25,29.7,2.5):
    for off,kind,level in [(0,'bass',.85),(.625,'stick',.48),(1.25,'bass',.66),(1.5625,'stick',.44),(1.875,'stick',.59),(2.1875,'stick',.38)]:
        strike(bar+off+RNG.uniform(-.009,.009),kind,level,-.12 if kind=='bass' else .16)
    place(music,bar,metal(.65),.27,.25)

# Combat: tighter rhythms and short fills, ending at the decisive stroke.
for bar in np.arange(30.1,52.7,2.0):
    strength=.75+.3*(bar-30.1)/23
    for j in range(8):
        kind='bass' if j in [0,3,4] else 'stick'
        level=(.85 if j in [0,4] else .53)*strength
        strike(bar+j*.25+RNG.uniform(-.009,.009),kind,level,-.14 if kind=='bass' else .16)
    place(music,bar,metal(.8),.29,.2)
    if bar>43:
        strike(bar+1.625,'stick',.34,.1)
        strike(bar+1.875,'stick',.43,.1)
for at,hz,dur in [(31.2,440,1.3),(33.0,391.995,.9),(35.0,349.228,1.2),(40.3,440,1.8),(44.8,523.251,1.2),(46.4,440,1.5)]:
    place(music,at,reed(hz,dur),.56,.08)
for at in [52.9,53.05,53.2,53.35]:
    strike(at,'stick',.7,.12)
strike(53.6,'bass',1.35,0)
place(music,53.6,metal(3.8,True),.72,-.04)
events.append({'time':53.6,'event':'Final battle accent; percussion stops'})

# Lament: a sparse reed motif that resolves as Rama bows.
for at,hz,dur,gain in [(61.9,391.995,2.5,.56),(65.1,349.228,2.1,.52),(68.4,293.665,3.5,.60),(75.6,293.665,2.2,.63),(78.2,349.228,1.8,.62),(80.6,391.995,1.8,.58),(83.0,349.228,1.2,.52),(84.9,293.665,3.4,.65)]:
    place(music,at,reed(hz,dur),gain,.08)
place(music,85.0,metal(4.9,True),.50,0)
events.append({'time':85,'event':'Soft bronze gong at Rama bow; reed resolves to tonic'})


def flutter(at, duration, level, pan):
    count=int(duration*SR)
    z=np.arange(count)/SR
    env=np.sin(np.pi*np.minimum(z/duration,1))**2
    sample=noise(count,95,2300)*env*(.52+.48*np.sin(2*np.pi*7*z)**2)*.025
    place(foley,at,sample,level,pan)


def crack(at, level, pan):
    z=np.arange(int(.7*SR))/SR
    sig=noise(len(z),500,8800)*np.exp(-z/.019)
    sig+=.65*np.sin(2*np.pi*680*z)*np.exp(-z/.05)
    sig+=.5*np.sin(2*np.pi*178*z)*np.exp(-z/.10)
    place(foley,at,sig*.13,level,pan)
    for off,vol in [(.13,.30),(.28,.22),(.45,.12)]:
        place(foley,at+off,drum('stick'),vol*level,pan)


# Soft air underneath, and deliberate movement cues tied to visible impacts.
wind = noise(N, 70, 1600)
wind_env = np.interp(t,[0,3,29,53,60,75,87,90],[0,.9,.4,.4,1.0,.65,.4,0])
place(foley,0,wind*(.004+.002*np.sin(2*np.pi*.13*t))*wind_env,room=False)
for at,dur,level,pan in [(5.4,3.5,1.3,.35),(10.9,1.1,1.7,-.35),(12.4,1.1,1.9,-.25),(20.8,1.0,2.4,-.3),(23.5,1,2,-.2),(26.1,.9,2,-.15),(31.4,.8,2.6,-.2),(34.2,.8,2.2,-.05),(38.3,.75,2.5,-.1),(41.1,1.1,2.7,0),(48.2,1.2,1.5,-.2),(53.25,.4,2.1,.2),(54.1,2.2,1.8,-.15),(68.8,2,.45,0)]:
    flutter(at,dur,level,pan)
for at,level,pan in [(38.6,.95,.15),(41.6,1.2,.3),(43.0,.4,.2),(54.1,.32,-.2)]:
    crack(at,level,pan)
    events.append({'time':at,'event':'Dry wood/leather impact and clatter'})


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
manifest={'title':'Jatayu — original theatre-inspired score','duration_seconds':90,'method':'New algorithmic composition and modal/noise synthesis; no third-party recorded samples. FLUX 3 native generated clip ambience is mixed separately.','inspiration':'Kerala Tholpavakoothu: ezhupara/maddalam-like cylindrical drums, ilathalam-like small cymbals, kuzhal-like reed and chengila-like gong. Synthesized timbres are an interpretation, not a traditional troupe recording.','source':'https://tholpavakoothu.in/about-us/about-performance/','seed':24618,'events':sorted(events,key=lambda x:x['time']),'files':files}
(OUT/'soundtrack-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'audio_stems':files},indent=2))
