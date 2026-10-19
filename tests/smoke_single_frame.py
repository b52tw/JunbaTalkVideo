"""One synthetic image + one micro Conv/ReLU per installed backend. No real models."""
import argparse,ast,json,sys,tempfile,wave
from pathlib import Path
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local'));sys.modules['openvino_telemetry']=None

def main():
    import cv2,numpy as np
    from quality import blend_mouth,mouth_mask,rect,photo_motion,blink_starts
    from visual_polish import lab_transfer,lid_shadow
    from body_motion import TorsoMotion,audio_drive
    from micro_graph import create
    from inference_backend import NativeModel,OrtModel
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();checks={}
    # Exactly one synthetic source frame, no frame sequences or video encoding.
    h,w=192,160;yy,xx=np.mgrid[:h,:w];frame=np.stack([100+xx*.2,130+yy*.1,165+xx*.1],2).clip(0,255).astype(np.uint8);face=[.25,.10,.5,.45];mouth=[.37,.39,.24,.08]
    patch=frame[19:106,40:120].copy();patch=np.clip(patch.astype(np.int16)+[12,-5,-10],0,255).astype(np.uint8)
    for mode in ('feather','poisson'):
        out=blend_mouth(frame,patch,face,mouth,blend=mode,color_transfer=True)
        x1,y1,x2,y2=rect(face,w,h);mask=np.zeros((h,w),bool);mask[y1:y2,x1:x2]=mouth_mask((y2-y1,x2-x1),face,mouth)[:,:,0]>0
        assert np.array_equal(out[~mask],frame[~mask]);assert out.dtype==np.uint8;checks[mode+'_outside_unchanged']=True
    eye=[.36,.26,.12,.06];shadow=lid_shadow(frame,eye,1);assert np.array_equal(lid_shadow(frame,eye,0),frame);assert np.all(shadow<=frame);assert np.max(frame.astype(float)-shadow)<255*.15+1;checks['shadow_bounded']=True
    time=blink_starts()[0]+.1;blink=photo_motion(frame,face,time,[eye],blink=True,shadow=True);assert blink.shape==frame.shape;checks['blink_single_frame']=True
    alpha=np.zeros((h,w),np.float32);cv2.ellipse(alpha,(80,123),(40,64),0,0,360,1,-1);torso=TorsoMotion(frame,face,drive=np.array([3.],np.float32),alpha=alpha);moved=torso.apply(1.0);assert torso.ready;assert np.array_equal(moved[:int((face[1]+face[3]*.92)*h)],frame[:int((face[1]+face[3]*.92)*h)]);assert np.array_equal(moved[:,0],frame[:,0]);checks['torso_head_and_canvas_locked']=True
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp);micro=create(folder/'micro.onnx');assert micro.stat().st_size<1048576;checks['micro_graph_bytes']=micro.stat().st_size
        x=np.linspace(-1,1,48,dtype=np.float32).reshape(1,3,4,4)
        for name,cls in [('openvino',NativeModel),('ort_cpu',OrtModel)]:
            try:
                actual={};m=cls(micro,'smoke_conv_relu',dict(device='CPU',output=str(folder/'test.mp4')),lambda **kw:None,actual);y=m([x])[0];assert np.allclose(y,np.maximum(x,0),atol=.005,rtol=.001);checks[name]=actual;del m
            except ImportError:checks[name]='not installed in this source-only environment'
        wav=folder/'sample.wav'
        with wave.open(str(wav),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes((np.sin(np.arange(640)*2*np.pi*160/16000)*4000).astype('<i2').tobytes())
        drive=audio_drive(wav);assert np.all(np.isfinite(drive)) and np.all((drive>=0)&(drive<=5));checks['short_dummy_audio_damping']=True
    report=dict(ok=True,real_model_weights_loaded=False,synthetic_frames=1,checks=checks)
    target=Path(a.output);target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
