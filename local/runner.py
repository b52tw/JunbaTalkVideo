"""MuseTalk ONNX, OpenVINO execution. No cloud services or torch runtime.

Model exports include the VAE 0.18215 scale and unscale operations.
Whisper mel filters: OpenAI Whisper (MIT). MuseTalk: TMElyralab (MIT).
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
# Offline worker: disable optional OpenVINO analytics before runtime imports.
# Windows embedded Python ._pth excludes this sibling script directory.
# Use the worker location, never the current working directory or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules["openvino_telemetry"] = None
import time
import traceback
from runtime_support import process_policy,report_peak

def emit(**obj):
    print(json.dumps(obj, ensure_ascii=False), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", action="store_true")
    p.add_argument("--config")
    p.add_argument("--check-imports", action="store_true")
    a = p.parse_args()
    # Check app-local imports before compiling any large neural model.
    from quality import rect
    if a.check_imports:
        import quality
        import cv2, numpy, scipy, openvino
        emit(stage="checked", module="quality", path=str(Path(quality.__file__).resolve()), isolated=bool(sys.flags.isolated))
        return
    import openvino as ov
    core = ov.Core()
    if a.probe:
        devices = []
        for d in core.available_devices:
            try:
                name = str(core.get_property(d, "FULL_DEVICE_NAME"))
            except Exception:
                name = d
            devices.append({"id": d, "name": name})
        emit(devices=devices, runtime=ov.__version__)
        return
    import cv2
    import numpy as np
    from scipy.signal import resample_poly
    from scipy.io import wavfile
    cv2.setNumThreads(2)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    process_policy(cfg)
    threads = 2 if cfg.get("low_impact", True) else 4
    root = Path(cfg["models"])
    requested = cfg.get("device", "AUTO")
    available = core.available_devices
    if requested == "AUTO":
        device = next((d for d in available if d.startswith("GPU")), "CPU")
    else:
        device = requested
        if device not in available:
            raise RuntimeError("指定裝置不可用：" + device + "；可用：" + ", ".join(available))
    actual = {}
    def compile_model(name, shapes=None, small=False):
        emit(stage="compile", message="正在編譯模型：" + name + " → " + device)
        model = core.read_model(str(root / (name + "_fp16.onnx")))
        if shapes:
            model.reshape(shapes)
        target = "CPU" if small else device
        opts = {"PERFORMANCE_HINT": "LATENCY"}
        if target == "CPU":
            opts["INFERENCE_NUM_THREADS"] = threads
            opts["NUM_STREAMS"] = "1"
        try:
            c = core.compile_model(model, target, opts)
        except Exception as e:
            # AUTO is the only mode that permits a CPU fallback. Explicit GPU/NPU
            # failures remain errors, so the UI cannot misreport acceleration.
            if requested != "AUTO" or target == "CPU":
                raise RuntimeError(name + " 無法在 " + target + " 編譯：" + str(e)) from e
            emit(stage="fallback", message=name + " 無法使用 GPU，改用 CPU：" + str(e)[:350])
            c = core.compile_model(model, "CPU", {"PERFORMANCE_HINT":"LATENCY", "INFERENCE_NUM_THREADS":threads, "NUM_STREAMS":"1"})
        actual[name] = [str(x) for x in c.get_property("EXECUTION_DEVICES")]
        emit(stage="devices", actual=actual)
        return c
    # Static shapes improve Intel GPU/NPU compatibility. The tiny positional
    # encoding graph runs on CPU; listing it explicitly avoids misleading totals.
    pe = compile_model("positional_encoding", {"audio_features":[1,50,384]}, small=True)
    whisper = compile_model("whisper_encoder", {"input_features":[1,80,3000]})
    sr, audio = wavfile.read(cfg["audio"])
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        g = math.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    duration = len(audio) / 16000
    mel = np.load(Path(__file__).with_name("mel_filters.npz"))["mel_80"]
    window = np.hanning(401)[:-1].astype(np.float32)
    features = []
    emit(stage="audio", message="本機分析語音特徵")
    for start in range(0, len(audio), 480000):
        chunk = audio[start:start+480000]
        size = len(chunk)
        chunk = np.pad(chunk, (0, 480000-size))
        x = np.pad(chunk, (200,200), mode="reflect")
        frames = np.lib.stride_tricks.sliding_window_view(x,400)[::160][:-1]
        power = np.abs(np.fft.rfft(frames*window,axis=1)).astype(np.float32)**2
        logs = np.log10(np.maximum(mel @ power.T, 1e-10))
        logs = (np.maximum(logs, logs.max()-8.0)+4.0)/4.0
        result = whisper([logs[None].astype(np.float32)])[0]
        features.append(result[0,:max(1,math.ceil(size/320))].copy())
    features = np.concatenate(features, axis=0)
    del whisper
    enc = compile_model("vae_encoder")
    fps = cfg.get("fps",12)
    total = max(1,math.ceil(duration*fps))
    photo = cfg.get("kind") == "photo"
    decoder = None
    flags = subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    if photo:
        first = cv2.imdecode(np.fromfile(cfg["source"],dtype=np.uint8),cv2.IMREAD_COLOR)
    else:
        # FFmpeg handles Unicode Windows paths and additional containers more
        # consistently than OpenCV's bundled VideoCapture backend.
        preview = subprocess.run([cfg["ffmpeg"],"-hide_banner","-loglevel","error","-nostdin","-i",cfg["source"],"-frames:v","1","-f","image2pipe","-vcodec","png","-threads","2","pipe:1"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,creationflags=flags)
        if preview.returncode:
            raise RuntimeError("無法讀取素材影片："+preview.stderr.decode("utf-8",errors="replace")[-600:])
        first = cv2.imdecode(np.frombuffer(preview.stdout,dtype=np.uint8),cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError("無法讀取素材圖片")
    h0,w0 = first.shape[:2]
    scale = min(1.0,cfg.get("max_side",720)/max(h0,w0))
    w,h = max(2,int(w0*scale)//2*2), max(2,int(h0*scale)//2*2)
    box = cfg.get("bbox",[0.25,0.18,0.5,0.5])
    x,y,bw,bh = [float(v) for v in box]
    x1,y1,x2,y2 = rect(box,w,h)
    if x2-x1<24 or y2-y1<24:
        raise RuntimeError("臉部選取範圍太小；請重新框選包含眼睛、嘴巴與下巴的臉部")
    def latent(frame):
        face = cv2.resize(frame[y1:y2,x1:x2],(256,256))[:,:,::-1].copy()
        ref = (face.astype(np.float32)/127.5-1).transpose(2,0,1)[None]
        masked = ref.copy()
        masked[:,:,128:,:] = -1
        return np.concatenate([enc([masked])[0],enc([ref])[0]],axis=1)
    base = cv2.resize(first,(w,h),interpolation=cv2.INTER_AREA)
    ref_latent = latent(base) if photo else None
    if photo:
        del enc
        import gc
        gc.collect()
    dec = compile_model("vae_decoder")
    unet = compile_model("unet")
    target = Path(cfg["output"])
    cache=target.with_name("mouth-cache")
    cache.mkdir(exist_ok=True)
    started=time.monotonic()
    decode_log=open(target.with_name("local-decode.log"),"wb")
    if not photo:
        decoder = subprocess.Popen([cfg["ffmpeg"],"-hide_banner","-loglevel","error","-nostdin","-stream_loop","-1","-i",cfg["source"],"-an","-vf",f"fps={fps},scale={w}:{h}","-f","rawvideo","-pix_fmt","bgr24","-threads","2","pipe:1"],stdout=subprocess.PIPE,stderr=decode_log,creationflags=flags)
    frame = base
    try:
        for i in range(total):
            if decoder:
                raw = decoder.stdout.read(w*h*3)
                if len(raw)!=w*h*3:
                    raise RuntimeError("讀取素材影片中斷，請查看 local-decode.log")
                frame = np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3)
                z = latent(frame)
            else:
                frame = base.copy()
                z = ref_latent
            center = int(i*50/fps)
            indices = np.clip(np.arange(center-4,center+6),0,len(features)-1)
            prompt = features[indices].reshape(1,50,384).astype(np.float32)
            prompt = pe([prompt])[0]
            generated = dec([unet([z,prompt])[0]])[0][0]
            generated = np.clip((generated.transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)[:,:,::-1]
            # Lossless small patches allow the heavy neural workers to exit
            # before the optional detail model and full-resolution compositor.
            ok,png=cv2.imencode(".png",generated)
            if not ok:raise RuntimeError("無法暫存嘴型貼片")
            (cache/f"{i:05d}.png").write_bytes(png.tobytes())
            elapsed = time.monotonic()-started
            emit(stage="render", frame=i+1,total=total,eta=round(elapsed/(i+1)*(total-i-1)),message=f"第 1 階段：嘴型推論 {i+1}/{total} 格",actual=actual)
        metadata={"actual":actual,"frames":total,"fps":fps,"width":w,"height":h}
        target.with_name("mouth-cache.json").write_text(json.dumps(metadata),encoding="utf-8")
    finally:
        if decoder:
            decoder.stdout.close()
            if decoder.poll() is None:
                decoder.kill()
            decoder.wait()
        decode_log.close()
    emit(stage="cached", actual=actual, message="嘴型推論完成，釋放模型後合成原圖與細節")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        emit(stage="error", message=str(e))
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    finally:report_peak("standard")
