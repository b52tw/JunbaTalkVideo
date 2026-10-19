"""Isolated SadTalker prepare/audio/export/infer/compose stages, one frame at a time."""
import argparse,contextlib,gc,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.modules['openvino_telemetry']=None
from runtime_support import emit,process_policy,report_peak

def child(script,config,*extra):subprocess.run([sys.executable,'-I','-u',str(Path(__file__).with_name('stage_dispatch.py')),'--script',script,'--config',str(Path(config).resolve()),*extra],check=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['prepare','audio','keypoints','infer']);a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));work=Path(cfg['work']);work.mkdir(exist_ok=True,parents=True);process_policy(cfg)
    if a.stage is None:
        for stage in ('prepare','audio'):emit(stage='compile',message='SadTalker 分段：'+stage);child('sad_worker.py',a.config,'--stage',stage)
        if cfg.get('runtime','openvino')!='torch':
            try:
                child('sad_export.py',a.config,'--stage','export')
                if cfg.get('runtime','openvino')=='openvino':child('sad_export.py',a.config,'--stage','cache')
            except subprocess.CalledProcessError as e:
                from inference_backend import fault
                fault(cfg,emit,'sad_conversion',e,'短命轉換程序已退出；改獨立 PyTorch CPU 渲染')
                cfg.update(runtime='torch',device='CPU');path=work/'cpu-fallback-config.json';path.write_text(json.dumps(cfg));a.config=str(path)
        child('sad_worker.py',a.config,'--stage','keypoints');child('sad_worker.py',a.config,'--stage','infer');child('renderer.py',str(work/'compose-config.json'));return
    import numpy as np,cv2
    if a.stage!='infer' or cfg.get('runtime')=='torch':
        import torch
        torch.set_num_threads(min(8,max(1,int(cfg.get('threads',2)))));torch.set_num_interop_threads(1);torch.manual_seed(1247)
    cv2.setNumThreads(2)
    sys.path.insert(0,cfg['root']);os.chdir(cfg['root'])
    def read(path):return cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
    def write(path,image):
        ok,data=cv2.imencode(Path(path).suffix,image)
        if not ok:raise RuntimeError('影像寫入失敗')
        data.tofile(path)
    cv2.imread=lambda path,flags=cv2.IMREAD_COLOR:cv2.imdecode(np.fromfile(path,dtype=np.uint8),flags)
    cv2.imwrite=lambda path,image,*args:(write(path,image) is None)
    from src.utils.init_path import init_path
    paths=init_path(str(Path(cfg['root'])/'checkpoints'),str(Path(cfg['root'])/'src/config'),512,False,'full');state_path=work/'state.json'
    if a.stage=='prepare':
        from src.utils.preprocess import CropAndExtract
        image=read(cfg['source'])
        if image is None:raise RuntimeError('無法讀取照片')
        scale=min(1.,cfg.get('max_side',1920)/max(image.shape[:2]));h=max(2,int(image.shape[0]*scale)//2*2);w=max(2,int(image.shape[1]*scale)//2*2);image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA);reference=work/'reference.png';write(str(reference),image)
        folder=work/'coeff';folder.mkdir(exist_ok=True);engine=CropAndExtract(paths,'cpu');coeff,crop,info=engine.generate(str(reference),str(folder),'full',source_image_flag=True,pic_size=512)
        if not coeff:raise RuntimeError('找不到人臉')
        clx,cly,crx,cry=info[1];lx,ly,rx,ry=map(int,info[2]);x1,y1,x2,y2=clx+lx,cly+ly,clx+rx,cly+ry
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise RuntimeError('臉部回貼範圍不完整')
        points=np.loadtxt(folder/'reference_landmarks.txt').reshape(-1,68,2)[0]
        def box(subset,pad):
            lo=subset.min(0);hi=subset.max(0);delta=(hi-lo)*pad;lo=np.maximum(lo-delta,0);hi=np.minimum(hi+delta,512)
            return [(x1+lo[0]/512*(x2-x1))/w,(y1+lo[1]/512*(y2-y1))/h,(hi[0]-lo[0])/512*(x2-x1)/w,(hi[1]-lo[1])/512*(y2-y1)/h]
        state={'eye_boxes':[box(points[36:42],[.15,.55]),box(points[42:48],[.15,.55])],'coeff':coeff,'crop':crop,'crop_info':info,'width':w,'height':h,'face':[x1/w,y1/h,(x2-x1)/w,(y2-y1)/h],'mouth':cfg.get('mouth_box') or box(points[48:68],[.12,.4])};state_path.write_text(json.dumps(state),encoding='utf-8')
    elif a.stage=='audio':
        import src.generate_batch as batch_module
        from quality import blink_amount
        from src.test_audio2coeff import Audio2Coeff
        state=json.loads(state_path.read_text());batch_module.generate_blink_seq_randomly=lambda n:np.array([[blink_amount(i/25,cfg.get('blink_profile','bezier'))] for i in range(n)]);batch=batch_module.get_data(state['coeff'],cfg['audio'],'cpu',None);state['motion']=Audio2Coeff(paths,'cpu').generate(batch,str(work),cfg.get('pose_style',0),None);state_path.write_text(json.dumps(state))
    elif a.stage=='keypoints':
        from types import SimpleNamespace
        import yaml,safetensors.torch
        from src.facerender.modules.keypoint_detector import KPDetector
        from src.facerender.modules.mapping import MappingNet
        from src.facerender.modules.make_animation import keypoint_transformation
        from src.generate_facerender_batch import get_facerender_data
        state=json.loads(state_path.read_text());config=yaml.safe_load(Path(paths['facerender_yaml']).read_text())
        kp=KPDetector(**config['model_params']['kp_detector_params'],**config['model_params']['common_params']).eval().requires_grad_(False)
        mapping=MappingNet(**config['model_params']['mapping_params']).eval().requires_grad_(False)
        with safetensors.safe_open(paths['checkpoint'],framework='pt',device='cpu') as f:weights={k[len('kp_extractor.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('kp_extractor.')}
        kp.load_state_dict(weights);del weights
        saved=torch.load(paths['mappingnet_checkpoint'],map_location='cpu',weights_only=True);mapping.load_state_dict(saved['mapping']);del saved;gc.collect()
        engine=SimpleNamespace(kp_extractor=kp,mapping=mapping)

        data=get_facerender_data(state['motion'],state['crop'],state['coeff'],cfg['audio'],1,expression_scale=cfg.get('expression_scale',1),still_mode=cfg.get('composition_mode','mouth')!='full',preprocess='full',size=512)
        image=data['source_image'].float();targets=data['target_semantics_list'].float();total=data['frame_num']
        with torch.inference_mode():
            canonical=engine.kp_extractor(image);source=keypoint_transformation(canonical,engine.mapping(data['source_semantics'].float()))['value']
            values=np.lib.format.open_memmap(work/'driving.npy',mode='w+',dtype=np.float32,shape=(total,1,15,3))
            for i in range(total):values[i]=keypoint_transformation(canonical,engine.mapping(targets[:,i]))['value'].numpy()
            values.flush();np.save(work/'source-keypoints.npy',source.numpy());np.save(work/'image.npy',image.numpy())
        state['frames']=total;state_path.write_text(json.dumps(state))
    else:
        state=json.loads(state_path.read_text());total=state['frames'];actual={};runtime=cfg.get('runtime','openvino');model=None
        image=np.load(work/'image.npy',mmap_mode='r');source=np.load(work/'source-keypoints.npy',mmap_mode='r');driving_points=np.load(work/'driving.npy',mmap_mode='r')
        if runtime!='torch':
            from sad_export import paths_for
            from inference_backend import NativeModel,OrtModel,fault,CPUFallbackRequired,request_cpu
            _,onnx,xml,_=paths_for(cfg['root'])
            try:model=(OrtModel if runtime=='directml' else NativeModel)(onnx if runtime=='directml' else xml,'sad_generator',cfg,emit,actual)
            except CPUFallbackRequired:raise
            except Exception as e:
                request_cpu(cfg,emit,'sad_generator',e,'torch')
        if runtime=='torch':
            import torch,yaml,safetensors.torch
            torch.set_num_threads(min(4,max(1,int(cfg.get('threads',2)))))
            from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
            config=yaml.safe_load((Path(cfg['root'])/'src/config/facerender_still.yaml').read_text())
            generator=OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],**config['model_params']['common_params']).eval().requires_grad_(False)
            with safetensors.safe_open(str(Path(cfg['root'])/'checkpoints/SadTalker_V0.0.2_512.safetensors'),framework='pt',device='cpu') as f:weights={k[len('generator.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('generator.')}
            generator.load_state_dict(weights);del weights;gc.collect();actual['sad_generator']=['PyTorch CPU fallback']
        folder=work/'mouth-cache';folder.mkdir(exist_ok=True);started=time.monotonic()
        with (torch.inference_mode() if runtime=='torch' else contextlib.nullcontext()):
            for i in range(total):
                driving=driving_points[i]
                prediction=generator(torch.from_numpy(np.array(image)),kp_source={'value':torch.from_numpy(np.array(source))},kp_driving={'value':torch.from_numpy(np.array(driving))})['prediction'].cpu().numpy() if model is None else model([image,source,driving])[0]
                patch=np.clip(np.rint(prediction[0].transpose(1,2,0)*255),0,255).astype(np.uint8)[:,:,::-1];write(str(folder/f'{i:05d}.png'),patch)
                emit(stage='render',frame=i+1,total=total,eta=round((time.monotonic()-started)/(i+1)*(total-i-1)),actual=actual,message=f'SadTalker 512：{i+1}/{total}')
        (work/'mouth-cache.json').write_text(json.dumps({'actual':actual,'frames':total,'fps':25,'width':state['width'],'height':state['height']}));compose=dict(cfg);compose.update(models=cfg.get('detail_models',''),source=str(work/'reference.png'),kind='photo',bbox=state['face'],mouth_box=state['mouth'],eye_boxes=cfg.get('eye_boxes') or state['eye_boxes'],runtime='openvino')
        if cfg.get('composition_mode','mouth')=='full':compose.update(blink=False,gaze=False,body_motion='')
        (work/'compose-config.json').write_text(json.dumps(compose),encoding='utf-8')
    report_peak('sad_'+a.stage)
if __name__=='__main__':
    try:main()
    except __import__('inference_backend').CPUFallbackRequired as e:emit(stage='fallback',message='已退出失敗後端，等待新 CPU 程序：'+str(e));sys.exit(75)
    except Exception as e:emit(stage='error',message=str(e));traceback.print_exc();sys.exit(1)
