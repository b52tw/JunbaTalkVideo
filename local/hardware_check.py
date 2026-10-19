"""Isolated provider discovery; optional micro-graph checks, metadata-only model audit."""
import argparse,hashlib,json,subprocess,sys,tempfile,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.modules['openvino_telemetry']=None
from runtime_support import peak_mb

def one(backend,device,mode,folder):
    result={'backend':backend,'available':False,'devices':[],'providers':[],'checks':[]}
    try:
        if backend=='openvino':
            import openvino as ov
            core=ov.Core();core.set_property({'ENABLE_MMAP':True});result.update(available=True,version=ov.__version__,devices=[{'id':d,'name':str(core.get_property(d,'FULL_DEVICE_NAME'))} for d in core.available_devices])
        else:
            import onnxruntime as ort
            result.update(available=True,version=ort.__version__,providers=ort.get_available_providers())
        if mode!='enumerate':
            import numpy as np
            from micro_graph import create
            from inference_backend import NativeModel,OrtModel
            actual={};events=[];cfg=dict(device=device,runtime=backend,output=str(Path(folder)/'micro.mp4'),low_impact=True)
            path=create(Path(folder)/'micro.onnx');m=(NativeModel if backend=='openvino' else OrtModel)(path,'micro_conv_relu',cfg,lambda **x:events.append(x),actual)
            x=np.linspace(-1,1,48,dtype=np.float32).reshape(1,3,4,4);out=m([x])[0]
            if not np.allclose(out,np.maximum(x,0),atol=.005,rtol=.001):raise RuntimeError('Micro graph numerical mismatch')
            result['checks'].append(dict(requested=device,actual=actual,ok=True,graph_bytes=path.stat().st_size));result['events']=events
    except Exception as e:result.update(error=str(e),traceback=traceback.format_exc(),events=events if 'events' in locals() else [])
    result['peak_mib']=peak_mb();return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--one',choices=['openvino','ort']);p.add_argument('--device',default='CPU');a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));folder=Path(cfg['report']).parent;folder.mkdir(parents=True,exist_ok=True)
    if a.one:print(json.dumps(one(a.one,a.device,cfg.get('mode','enumerate'),folder),ensure_ascii=False));return
    report=dict(time=time.time(),micro_only=True,real_weights_loaded=False,backends=[],files=[],memory_target_mib=200)
    for backend in ('openvino','ort'):
        devices=['CPU']
        if cfg.get('mode')=='full' and cfg.get('device','AUTO')!='CPU':devices.append(cfg.get('device') if cfg.get('device')!='AUTO' else 'GPU')
        for device in dict.fromkeys(['CPU' if backend=='ort' and d=='NPU' else d for d in devices]):
            c=None
            try:
                c=subprocess.run([sys.executable,'-I','-u',str(Path(__file__)),'--config',a.config,'--one',backend,'--device',device],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=35)
                parsed=next(json.loads(s) for s in reversed(c.stdout.splitlines()) if s.startswith('{'))
                if c.returncode:parsed['child_exit_code']=c.returncode
            except Exception as e:parsed=dict(backend=backend,requested=device,available=False,error=str(e),stderr=c.stderr[-5000:] if c is not None else '')
            report['backends'].append(parsed)
    for item in cfg.get('files',[]):
        path=Path(item['path']);row={'name':item['name'],'present':path.is_file(),'hash_checked':False}
        if path.is_file():
            row['size_ok']=path.stat().st_size==item['size']
            if cfg.get('full_hash'):
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(1048576),b''):h.update(block)
                row.update(hash_checked=True,sha256_ok=h.hexdigest()==item['sha256'])
        report['files'].append(row)
    def cpu_ok(backend):return any(b['backend']==backend and any(c.get('ok') and all('CPU' in str(v) for v in c['actual'].values()) for c in b.get('checks',[])) for b in report['backends'])
    report['cpu_verified']=cpu_ok('openvino') or cpu_ok('ort');accelerated=any(any(any(('GPU' in str(v) or 'DmlExecutionProvider' in str(v)) for v in c['actual'].values()) for c in b.get('checks',[]) if c.get('ok')) for b in report['backends'])
    report['recommended']={'runtime':'openvino' if cpu_ok('openvino') else 'directml','device':'CPU'} if report['cpu_verified'] and not accelerated else None
    report['message']='已列出可用裝置；僅微型圖可證明 DLL／算子可用。未載入真實模型。'
    Path(cfg['report']).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
