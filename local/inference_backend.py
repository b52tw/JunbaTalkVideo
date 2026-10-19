"""Bounded, sequential inference with explicit CPU fallback and fault logs."""
import gc,json,time,traceback
from pathlib import Path

def fault(cfg,emit,name,error,action):
    entry={'time':time.time(),'model':name,'error':str(error),'traceback':traceback.format_exc(),'action':action}
    folder=Path(cfg['output']).parent;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'backend-faults.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(entry,ensure_ascii=False)+'\n')
    emit(stage='fallback',message=f'{name}：{action}；{str(error)[:260]}；詳見 backend-faults.jsonl')

class CPUFallbackRequired(RuntimeError):pass

def request_cpu(cfg,emit,name,error,runtime=None):
    fault(cfg,emit,name,error,'此程序退出後，再由新程序以 CPU 重試；不在原 C++ arena 疊加模型')
    folder=Path(cfg['output']).parent;fallback=dict(run_id=cfg.get('run_id','direct'),runtime=runtime or cfg.get('runtime','openvino'),device='CPU',npu_assist=False,fallback_attempt=int(cfg.get('fallback_attempt',0))+1)
    pending=folder/'backend-fallback.pending.json';pending.write_text(json.dumps(fallback),encoding='utf-8');pending.replace(folder/'backend-fallback.json')
    raise CPUFallbackRequired(str(error))

class NativeModel:
    def __init__(self,path,name,cfg,emit,actual,shapes=None,small=False):
        import openvino as ov
        self.path=Path(path);self.name=name;self.cfg=cfg;self.emit=emit;self.actual=actual;self.shapes=shapes;self.reported=False
        core=ov.Core();core.set_property({'ENABLE_MMAP':True});available=core.available_devices;request=cfg.get('device','AUTO')
        target='CPU' if small else next((d for d in available if d.startswith('GPU')),'CPU') if request=='AUTO' else request
        if cfg.get('npu_assist') and name=='whisper_encoder' and 'NPU' in available:target='NPU'
        try:self._load(core,target)
        except Exception as e:
            if target=='CPU':raise
            request_cpu(cfg,emit,name,e)
    def _load(self,core,target):
        opts={'PERFORMANCE_HINT':'LATENCY'}
        supported={str(p) for p in core.get_property(target,'SUPPORTED_PROPERTIES')}
        if 'COMPILATION_NUM_THREADS' in supported:opts['COMPILATION_NUM_THREADS']=1
        if target!='NPU':opts['NUM_STREAMS']='1'
        if target=='CPU':opts['INFERENCE_NUM_THREADS']=1 if self.cfg.get('low_impact',True) else 2
        if target=='CPU' and self.name=='sad_generator':opts['INFERENCE_PRECISION_HINT']='f32'
        if target.startswith('GPU') and self.cfg.get('low_impact',True):
            if 'INFERENCE_PRECISION_HINT' in supported:opts['INFERENCE_PRECISION_HINT']='f16'
            if 'GPU_ENABLE_LOOP_UNROLLING' in supported:opts['GPU_ENABLE_LOOP_UNROLLING']=False
        model=str(self.path)
        if self.shapes:
            model=core.read_model(model);model.reshape(self.shapes)
        self.emit(stage='compile',message=f'{self.name} → OpenVINO {target}；ENABLE_MMAP=True；低平行度編譯')
        self.model=core.compile_model(model,target,opts);self.devices=[str(x) for x in self.model.get_property('EXECUTION_DEVICES')]
    def __call__(self,inputs):
        try:result=self.model(inputs)
        except Exception as e:
            if self.devices==['CPU']:raise
            request_cpu(self.cfg,self.emit,self.name,e)
        if not self.reported:self.actual[self.name]=self.devices;self.emit(stage='devices',actual=self.actual);self.reported=True
        return result

class OrtModel:
    def __init__(self,path,name,cfg,emit,actual,small=False):
        import onnxruntime as ort
        self.ort=ort;self.path=path;self.name=name;self.cfg=cfg;self.emit=emit;self.actual=actual;self.reported=False
        cpu=small or cfg.get('device')=='CPU'
        if not cpu and 'DmlExecutionProvider' not in ort.get_available_providers():
            request_cpu(cfg,emit,name,RuntimeError('DmlExecutionProvider 不可用'),'directml')
        try:self._load(cpu)
        except Exception as e:
            if cpu:raise
            request_cpu(cfg,emit,name,e,'directml')
    def _load(self,cpu):
        ort=self.ort;options=ort.SessionOptions();options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_mem_pattern=False;options.enable_cpu_mem_arena=False;options.intra_op_num_threads=2;options.inter_op_num_threads=1
        if cpu and self.name in ('unet','sad_generator'):
            options.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC;options.add_session_config_entry('session.disable_prepacking','1')
        folder=Path(self.cfg['output']).parent/'provider-profiles';folder.mkdir(parents=True,exist_ok=True)
        options.enable_profiling=True;options.profile_file_prefix=str(folder/self.name)
        providers=['CPUExecutionProvider'] if cpu else [('DmlExecutionProvider',{'device_id':str(self.cfg.get('dml_adapter',0))}),'CPUExecutionProvider']
        self.emit(stage='compile',message=self.name+' → '+('ORT CPU' if cpu else 'ORT DirectML；首格後回報節點'))
        self.session=ort.InferenceSession(str(self.path),sess_options=options,providers=providers);self.session.disable_fallback();self.cpu=cpu
    def __call__(self,inputs):
        import numpy as np
        def run():
            types={'tensor(float16)':np.float16,'tensor(float)':np.float32,'tensor(int64)':np.int64}
            feed={m.name:np.asarray(x,dtype=types.get(m.type)) for m,x in zip(self.session.get_inputs(),inputs)}
            return self.session.run(None,feed)
        try:result=run()
        except Exception as e:
            if self.cpu:raise
            request_cpu(self.cfg,self.emit,self.name,e,'directml')
        if not self.reported:
            counts={}
            for e in json.loads(Path(self.session.end_profiling()).read_text(encoding='utf-8')):
                p=e.get('args',{}).get('provider')
                if p:counts[p]=counts.get(p,0)+1
            self.actual[self.name]=[f'{p} ({n} 節點)' for p,n in counts.items()] or ['已推論，無 profiling 節點'];self.emit(stage='devices',actual=self.actual);self.reported=True
        return result

def make_model(root,name,cfg,emit,actual,shapes=None,small=False):
    from model_cache import cached_path
    root=Path(root);onnx=root/(name+'_fp16.onnx');runtime=cfg.get('runtime','openvino')
    if runtime in ('directml','ort-cpu') and not (cfg.get('npu_assist') and name=='whisper_encoder'):
        try:return OrtModel(onnx,name,cfg,emit,actual,small)
        except CPUFallbackRequired:raise
        except Exception as e:
            if int(cfg.get('fallback_attempt',0))>=2:raise
            request_cpu(cfg,emit,name,e,'openvino_onnx_cpu')
    path=cached_path(root,name) if runtime!='openvino_onnx_cpu' else None
    if path is None:path=onnx
    try:return NativeModel(path,name,cfg,emit,actual,shapes,small)
    except CPUFallbackRequired:raise
    except Exception as e:
        if int(cfg.get('fallback_attempt',0))>=2:raise
        request_cpu(cfg,emit,name,e,'ort-cpu')
