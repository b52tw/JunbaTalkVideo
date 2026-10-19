"""Offline OpenVoice V2 tone-color cloning via OpenVINO (no torch/cloud).

Fixed 256-frame graphs allow Intel GPU/NPU compilation attempts. Text is spoken
by the caller's local TTS before this worker; prosody follows that source audio.
Upstream weights: myshell-ai/OpenVoiceV2, MIT. ONNX export: TigreGotico.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys
# Offline worker: disable optional OpenVINO analytics before runtime imports.
# Windows embedded Python ._pth excludes this sibling script directory.
# Use the worker location, never the current working directory or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules["openvino_telemetry"] = None
import time
import traceback

def emit(**data):
    print(json.dumps(data,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);args=p.parse_args()
    import numpy as np
    import openvino as ov
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    cfg=json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
    core=ov.Core();available=core.available_devices
    requested=cfg.get("device","AUTO");actual={};fallbacks=[]
    frames=256;hop=256;block=frames*hop;overlap=32*hop;stride=block-overlap;sr=22050
    def compile_part(name,shapes):
        model=core.read_model(str(Path(cfg["models"])/(name+".onnx")))
        model.reshape(shapes)
        if requested=="AUTO":
            gpu=[d for d in available if d.startswith("GPU")]
            # The small reference encoder may use NPU while the converter uses
            # GPU. Query actual supported operations before compiling on NPU.
            order=(["NPU"] if name=="tone_ref_encoder" and "NPU" in available else [])+gpu
            if "NPU" in available and "NPU" not in order:order.append("NPU")
            order.append("CPU")
        else:
            if requested not in available:raise RuntimeError("指定裝置不可用："+requested)
            order=[requested]
        for device in order:
            emit(stage="compile",message=name+"：正在嘗試 "+device,actual=actual,fallbacks=fallbacks)
            try:
                if device=="NPU":
                    supported=core.query_model(model,"NPU")
                    missing=[op.get_friendly_name() for op in model.get_ops() if op.get_friendly_name() not in supported]
                    if missing:raise RuntimeError("NPU 尚不支援此模型的運算："+", ".join(missing[:6]))
                opts={"PERFORMANCE_HINT":"LATENCY"}
                if device=="CPU":opts.update({"INFERENCE_NUM_THREADS":2,"NUM_STREAMS":"1"})
                compiled=core.compile_model(model,device,opts)
                # A real inference is required before declaring acceleration.
                smoke={port.get_any_name():np.zeros(list(port.shape),dtype=port.element_type.to_dtype()) for port in compiled.inputs}
                if "spec_lengths" in smoke:smoke["spec_lengths"][:]=frames
                compiled(smoke)
                actual[name]=[str(x) for x in compiled.get_property("EXECUTION_DEVICES")]
                emit(stage="devices",actual=actual,fallbacks=fallbacks,message=name+" 已通過推論測試")
                return compiled
            except Exception as e:
                reason=name+" / "+device+"："+str(e)[:400]
                if requested!="AUTO" or device=="CPU":raise RuntimeError(reason) from e
                fallbacks.append(reason);emit(stage="fallback",message=reason,actual=actual,fallbacks=fallbacks)
        raise RuntimeError("沒有可執行聲音模型的裝置")
    def load_audio(path):
        rate,a=wavfile.read(path)
        if np.issubdtype(a.dtype,np.integer):
            scale=max(abs(np.iinfo(a.dtype).min),np.iinfo(a.dtype).max);a=a.astype(np.float32)/scale
        else:a=a.astype(np.float32)
        if a.ndim==2:a=a.mean(axis=1)
        if rate!=sr:
            g=math.gcd(rate,sr);a=resample_poly(a,sr//g,rate//g).astype(np.float32)
        if len(a)<sr:raise RuntimeError("音訊至少需要 1 秒")
        if not np.isfinite(a).all() or np.sqrt(np.mean(a*a))<0.0002:raise RuntimeError("音訊接近無聲，請使用清楚的單人錄音")
        return a
    source=load_audio(cfg["source"])
    ref=compile_part("tone_ref_encoder",{"spec":[1,frames,513]})
    window=np.hanning(1025)[:-1].astype(np.float32)
    def spec(a):
        a=np.pad(a,(384,384),mode="reflect")
        chunks=np.lib.stride_tricks.sliding_window_view(a,1024)[::hop]
        fft=np.fft.rfft(chunks*window,axis=-1)
        return np.sqrt(fft.real**2+fft.imag**2+1e-6).astype(np.float32)
    def embedding(audio):
        # Prefer energetic 3-second windows; do not train or contact any service.
        starts=list(range(0,max(1,len(audio)-block+1),block))
        if len(audio)>block:starts.append(len(audio)-block)
        starts=sorted(set(starts),key=lambda i:float(np.mean(audio[i:i+block]**2)),reverse=True)[:3]
        vectors=[]
        for i in starts:
            a=audio[i:i+block];a=np.pad(a,(0,block-len(a)))
            vectors.append(ref({"spec":spec(a)[None]})[0].copy())
        return np.mean(vectors,axis=0).astype(np.float32)[:,:,None]
    emit(stage="source_embedding",message="從基礎朗讀擷取音色特徵",actual=actual)
    source_g=embedding(source)
    cache=cfg.get("target_embedding","")
    if cache:
        target_g=np.load(cache,allow_pickle=False)
        if target_g.shape!=(1,256,1) or not np.isfinite(target_g).all():raise RuntimeError("儲存的音色特徵無效，請重新匯入樣本")
        emit(stage="target_embedding",message="沿用已儲存的本機音色",actual=actual)
    else:
        emit(stage="target_embedding",message="從錄音樣本擷取您的音色特徵",actual=actual)
        target_g=embedding(load_audio(cfg["reference"]))
    target_g=target_g.astype(np.float32)
    if cfg.get("save_embedding"):np.save(cfg["save_embedding"],target_g,allow_pickle=False)
    converter=compile_part("tone_converter",{"spec":[1,513,frames],"spec_lengths":[1],"src_g":[1,256,1],"tgt_g":[1,256,1]})
    starts=list(range(0,len(source),stride));output=np.zeros(len(source)+block,np.float32);weights=np.zeros_like(output)
    start_time=time.monotonic()
    for number,i in enumerate(starts):
        a=source[i:i+block];a=np.pad(a,(0,block-len(a)))
        result=converter({"spec":spec(a).T[None],"spec_lengths":np.array([frames],np.int64),"src_g":source_g,"tgt_g":target_g})[0].reshape(-1)
        if len(result)<block:result=np.pad(result,(0,block-len(result)))
        result=result[:block]
        if not np.isfinite(result).all():raise RuntimeError("模型輸出無效，請改用 CPU 或其他錄音樣本")
        w=np.ones(block,np.float32)
        if number>0:w[:overlap]=np.linspace(0,1,overlap)
        if number<len(starts)-1:w[-overlap:]=np.linspace(1,0,overlap)
        output[i:i+block]+=result*w;weights[i:i+block]+=w
        elapsed=time.monotonic()-start_time
        emit(stage="convert",message=f"本機聲音複製 {number+1}/{len(starts)} 段",frame=number+1,total=len(starts),eta=round(elapsed/(number+1)*(len(starts)-number-1)),actual=actual,fallbacks=fallbacks)
    result=output[:len(source)]/np.maximum(weights[:len(source)],1e-7)
    peak=float(np.max(np.abs(result)))
    if peak>0.98:result*=0.98/peak
    if float(np.sqrt(np.mean(result**2)))<1e-5:raise RuntimeError("模型輸出接近無聲，請換一段清晰樣本")
    temp=Path(cfg["output"]).with_suffix(".tmp.wav")
    wavfile.write(temp,sr,(np.clip(result,-1,1)*32767).astype(np.int16));os.replace(temp,cfg["output"])
    emit(stage="done",message="本機聲音複製完成，請試聽音色相似程度",actual=actual,fallbacks=fallbacks,duration=len(result)/sr)

if __name__=="__main__":
    try:main()
    except Exception as e:
        emit(stage="error",message=str(e));traceback.print_exc(file=sys.stderr);sys.exit(1)
