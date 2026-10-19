"""One neural stage per process. Intermediate arrays stay on disk/memory maps.
UNet, Whisper, VAE encoder and VAE decoder are never resident together.
"""
import argparse,json,math,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.modules['openvino_telemetry']=None
from runtime_support import emit,process_policy,report_peak
STAGE='startup'

def main():
    global STAGE
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',required=True,choices=['cache','audio','encode','infer','decode']);a=p.parse_args();STAGE=a.stage
    import numpy as np,cv2
    from quality import rect
    from model_cache import cached_path,prepare
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));process_policy(cfg);cv2.setNumThreads(2)
    fallback_path=Path(cfg['output']).parent/'backend-fallback.json'
    if STAGE=='cache':
        if fallback_path.exists():fallback_path.unlink()
        try:prepare(cfg['models'],emit)
        except Exception as e:
            from inference_backend import fault
            fault(cfg,emit,'IR_cache',e,'IR 轉換程序結束後改直接 ONNX CPU；下階段另開程序')
            fallback_path.write_text(json.dumps(dict(run_id=cfg.get('run_id','direct'),runtime='openvino_onnx_cpu',device='CPU',npu_assist=False)))
        return
    if fallback_path.exists():
        fallback=json.loads(fallback_path.read_text())
        if fallback['run_id']==cfg.get('run_id','direct'):cfg.update(fallback)

    folder=Path(cfg['output']).parent;cache=folder/'stage-cache';cache.mkdir(exist_ok=True)
    state_path=cache/'state.json';run_id=cfg.get('run_id','direct')
    if STAGE=='audio':state={'run_id':run_id,'actual':{}}
    else:
        state=json.loads(state_path.read_text(encoding='utf-8'))
        if state.get('run_id')!=run_id:raise RuntimeError('分段暫存與本次工作不符，請重新生成')
    from inference_backend import make_model
    actual=state['actual'];root=Path(cfg['models'])
    def compile_model(name,shapes=None,small=False):return make_model(root,name,cfg,emit,actual,shapes,small)
    def save_state():state_path.write_text(json.dumps(state),encoding='utf-8')
    started=time.monotonic()
    def progress(index,total,label):
        elapsed=time.monotonic()-started
        emit(stage='render',stage_name=STAGE,frame=index,total=total,eta=round(elapsed/max(1,index)*(total-index)),actual=actual,message=f'{label} {index}/{total}；完成後釋放本階段模型')
    if STAGE=='audio':
        from scipy.io import wavfile
        from scipy.signal import resample_poly
        sr,audio=wavfile.read(cfg['audio'])
        if np.issubdtype(audio.dtype,np.integer):audio=audio.astype(np.float32)/max(abs(np.iinfo(audio.dtype).min),np.iinfo(audio.dtype).max)
        else:audio=audio.astype(np.float32)
        if audio.ndim==2:audio=audio.mean(axis=1)
        if sr!=16000:
            g=math.gcd(sr,16000);audio=resample_poly(audio,16000//g,sr//g).astype(np.float32)
        state['duration']=len(audio)/16000;state['fps']=cfg.get('fps',25);state['frames']=max(1,math.ceil(state['duration']*state['fps']))
        whisper=compile_model('whisper_encoder',{'input_features':[1,80,3000]})
        mel=np.load(Path(__file__).with_name('mel_filters.npz'))['mel_80'];window=np.hanning(401)[:-1].astype(np.float32);features=[]
        chunks=math.ceil(len(audio)/480000)
        for count,start in enumerate(range(0,len(audio),480000)):
            chunk=audio[start:start+480000];size=len(chunk);chunk=np.pad(chunk,(0,480000-size));x=np.pad(chunk,(200,200),mode='reflect')
            frames=np.lib.stride_tricks.sliding_window_view(x,400)[::160][:-1];power=np.abs(np.fft.rfft(frames*window,axis=1)).astype(np.float32)**2
            logs=np.log10(np.maximum(mel@power.T,1e-10));logs=(np.maximum(logs,logs.max()-8.0)+4.0)/4.0
            result=whisper([logs[None].astype(np.float32)])[0]
            features.append(result[0,:max(1,math.ceil(size/320))].copy());progress(count+1,chunks,'1/5 語音特徵')
        np.save(cache/'audio-features.npy',np.concatenate(features,axis=0));save_state()
    elif STAGE=='encode':
        flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0;photo=cfg.get('kind')=='photo'
        if photo:first=cv2.imdecode(np.fromfile(cfg['source'],dtype=np.uint8),cv2.IMREAD_COLOR)
        else:
            p=subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-i',cfg['source'],'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,creationflags=flags)
            if p.returncode:raise RuntimeError('無法讀取原始影片')
            first=cv2.imdecode(np.frombuffer(p.stdout,dtype=np.uint8),cv2.IMREAD_COLOR)
        if first is None:raise RuntimeError('無法讀取原始素材')
        h0,w0=first.shape[:2];scale=min(1.,cfg.get('max_side',1920)/max(h0,w0));w,h=max(2,int(w0*scale)//2*2),max(2,int(h0*scale)//2*2)
        state.update(width=w,height=h,photo=photo);box=cfg['bbox'];x1,y1,x2,y2=rect(box,w,h)
        if min(x2-x1,y2-y1)<24:raise RuntimeError('臉部範圍太小，請重新框選')
        from contour_preflight import prepare_source
        from semantic_mouth import prepare_face
        contour_report=prepare_source(cv2.resize(first,(w,h),interpolation=cv2.INTER_AREA),cfg,folder)
        state['style_profile']=contour_report['profile']
        emit(stage='mask_preview',message='已輸出 mouth-overlay.png；嚴格唇線模式，失敗不採用 5 點粗框')
        encoder=compile_model('vae_encoder');n=1 if photo else state['frames'];z=np.lib.format.open_memmap(cache/'reference-latents.npy',mode='w+',dtype=np.float32,shape=(n,1,8,32,32));decoder=None;log=open(folder/'stage-source-decode.log','wb')
        if not photo:decoder=subprocess.Popen([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-stream_loop','-1','-i',cfg['source'],'-an','-vf',f'fps={state["fps"]},scale={w}:{h}','-f','rawvideo','-pix_fmt','bgr24','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=log,creationflags=flags)
        try:
            for i in range(n):
                if photo:frame=cv2.resize(first,(w,h),interpolation=cv2.INTER_AREA)
                else:
                    raw=decoder.stdout.read(w*h*3)
                    if len(raw)!=w*h*3:raise RuntimeError('讀取影片中斷')
                    frame=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3)
                face=prepare_face(cv2.resize(frame[y1:y2,x1:x2],(256,256)),state['style_profile'])[:,:,::-1].copy();ref=(face.astype(np.float32)/127.5-1).transpose(2,0,1)[None];masked=ref.copy();masked[:,:,128:,:]=-1
                z[i]=np.concatenate([encoder([masked])[0],encoder([ref])[0]],axis=1);progress(i+1,n,'2/5 臉部編碼')
            z.flush();save_state()
        finally:
            if decoder:
                decoder.stdout.close()
                if decoder.poll() is None:decoder.kill()
                decoder.wait()
            log.close()
    elif STAGE=='infer':
        pe=compile_model('positional_encoding',{'audio_features':[1,50,384]},small=True);unet=compile_model('unet')
        features=np.load(cache/'audio-features.npy',mmap_mode='r');z=np.load(cache/'reference-latents.npy',mmap_mode='r');n=state['frames'];fps=state['fps']
        predictions=np.lib.format.open_memmap(cache/'predicted-latents.npy',mode='w+',dtype=np.float32,shape=(n,1,4,32,32))
        for i in range(n):
            center=int(i*50/fps);indices=np.clip(np.arange(center-4,center+6),0,len(features)-1);prompt=features[indices].reshape(1,50,384).astype(np.float32);prompt=pe([prompt])[0]
            predictions[i]=unet([z[0 if state['photo'] else i],prompt])[0];progress(i+1,n,'3/5 嘴型推論')
        predictions.flush();save_state()
    else:
        decoder=compile_model('vae_decoder');predictions=np.load(cache/'predicted-latents.npy',mmap_mode='r');mouth=folder/'mouth-cache';mouth.mkdir(exist_ok=True);n=state['frames']
        for i in range(n):
            generated=decoder([predictions[i]])[0][0];generated=np.clip((generated.transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)[:,:,::-1]
            ok,data=cv2.imencode('.png',generated)
            if not ok:raise RuntimeError('嘴部影像暫存失敗')
            (mouth/f'{i:05d}.png').write_bytes(data.tobytes());progress(i+1,n,'4/5 嘴型影像解碼')
        meta={k:state[k] for k in ['actual','frames','fps','width','height']};(folder/'mouth-cache.json').write_text(json.dumps(meta),encoding='utf-8');save_state()
    emit(stage='completed',stage_name=STAGE,actual=actual,message=STAGE+' 完成，釋放模型')
if __name__=='__main__':
    try:main()
    except __import__('inference_backend').CPUFallbackRequired as e:emit(stage='fallback',message='退出原後端，等待新 CPU 程序：'+str(e));sys.exit(75)
    except Exception as e:emit(stage='error',message=str(e));traceback.print_exc(file=sys.stderr);sys.exit(1)
    finally:report_peak(STAGE)
