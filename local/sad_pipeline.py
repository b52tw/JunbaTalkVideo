"""Managed SadTalker: isolated preparation, audio motion, then one frame at a time.
The pinned upstream model/code stays unmodified. No downloaded code runs at startup.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.modules['openvino_telemetry']=None
from runtime_support import emit,process_policy,report_peak

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['prepare','audio','render']);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));process_policy({'low_impact':True})
    work=Path(cfg['work']);work.mkdir(exist_ok=True,parents=True)
    if a.stage is None:
        for stage in ['prepare','audio','render']:
            emit(stage='compile',stage_name='sad_'+stage,message='SadTalker 分階段：'+stage+'；前一程序結束後才載入下一組模型')
            subprocess.run([sys.executable,'-I','-u',str(Path(__file__).resolve()),'--config',str(Path(a.config).resolve()),'--stage',stage],check=True,cwd=cfg['root'])
        return
    import torch,numpy as np,cv2
    torch.set_num_threads(min(8,max(1,int(cfg.get("threads",2)))));torch.set_num_interop_threads(1);torch.manual_seed(1247);np.random.seed(1247);cv2.setNumThreads(2)
    sys.path.insert(0,cfg['root']);os.chdir(cfg['root'])
    # OpenCV's Windows path API cannot reliably open Chinese user/profile names.
    def imread(path,flags=cv2.IMREAD_COLOR):
        try:return cv2.imdecode(np.fromfile(path,dtype=np.uint8),flags)
        except OSError:return None
    def imwrite(path,img,*args):
        ok,data=cv2.imencode(Path(path).suffix,img,*args)
        if ok:data.tofile(path)
        return ok
    cv2.imread=imread;cv2.imwrite=imwrite
    from src.utils.init_path import init_path
    paths=init_path(str(Path(cfg['root'])/'checkpoints'),str(Path(cfg['root'])/'src/config'),512,False,'full')
    state_path=work/'state.json';device='cpu'
    if a.stage=='prepare':
        from src.utils.preprocess import CropAndExtract
        engine=CropAndExtract(paths,device)
        reference=work/'reference.png';image=imread(cfg['source'])
        if image is None:raise RuntimeError('無法讀取照片')
        scale=min(1.,int(cfg.get('max_side',1920))/max(image.shape[:2]))
        if scale<1:image=cv2.resize(image,(max(2,int(image.shape[1]*scale)//2*2),max(2,int(image.shape[0]*scale)//2*2)),interpolation=cv2.INTER_AREA)
        imwrite(str(reference),image);folder=work/'coeff';folder.mkdir(exist_ok=True)
        coeff,crop,info=engine.generate(str(reference),str(folder),'full',source_image_flag=True,pic_size=512)
        if not coeff:raise RuntimeError('SadTalker 沒有找到可用的正面人臉；請換清楚單人照片')
        state_path.write_text(json.dumps({'coeff':coeff,'crop':crop,'crop_info':info}),encoding='utf-8')
    elif a.stage=='audio':
        from src.test_audio2coeff import Audio2Coeff
        import src.generate_batch as batch_module
        # The upstream default flashes a 5-frame blink every 0.4–2.8 sec.
        # Use the same learned eyelid coefficients with a slower sparse driver.
        from quality import blink_amount
        batch_module.generate_blink_seq_randomly=lambda n:np.array([[blink_amount(i/25)] for i in range(n)])
        state=json.loads(state_path.read_text(encoding='utf-8'))
        batch=batch_module.get_data(state['coeff'],cfg['audio'],device,None)
        engine=Audio2Coeff(paths,device);motion=engine.generate(batch,str(work),cfg.get('pose_style',0),None)
        state['motion']=motion;state_path.write_text(json.dumps(state),encoding='utf-8')
    else:
        from src.facerender.animate import AnimateFromCoeff
        from src.facerender.modules.make_animation import keypoint_transformation
        from src.generate_facerender_batch import get_facerender_data
        state=json.loads(state_path.read_text(encoding='utf-8'));engine=AnimateFromCoeff(paths,device)
        data=get_facerender_data(state['motion'],state['crop'],state['coeff'],cfg['audio'],1,expression_scale=cfg.get('expression_scale',1),preprocess='full',size=512)
        source=data['source_image'].float();semantics=data['source_semantics'].float();targets=data['target_semantics_list'].float();total=data['frame_num']
        base=imread(str(work/'reference.png'))
        if base is None:raise RuntimeError('原始照片讀取失敗')
        h,w=base.shape[:2];w-=w%2;h-=h%2;base=base[:h,:w].copy()
        info=state['crop_info'];clx,cly,crx,cry=info[1];lx,ly,rx,ry=map(int,info[2]);x1,y1,x2,y2=clx+lx,cly+ly,clx+rx,cly+ry
        if not (0<=x1<x2<=w and 0<=y1<y2<=h):raise RuntimeError('SadTalker 臉部回貼超出畫面，請換較完整照片')
        temp=Path(cfg['output']).with_name('sad-encoding.mp4');flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
        cmd=[cfg['ffmpeg'],'-v','error','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r','25','-i','pipe:0','-i',cfg['audio'],'-map','0:v:0','-map','1:a:0','-shortest','-c:v','libx264','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart','-threads','2',str(temp)]
        log=open(work/'encoding.log','wb');proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=log,creationflags=flags);started=time.monotonic()
        try:
            with torch.inference_mode():
                canonical=engine.kp_extractor(source);kp_source=keypoint_transformation(canonical,engine.mapping(semantics))
                for i in range(total):
                    driving=keypoint_transformation(canonical,engine.mapping(targets[:,i]))
                    out=engine.generator(source,kp_source=kp_source,kp_driving=driving)
                    image=out['prediction'][0].permute(1,2,0).cpu().numpy()
                    image=np.clip(np.rint(image*255),0,255).astype(np.uint8)[:,:,::-1]
                    patch=cv2.resize(image,(x2-x1,y2-y1),interpolation=cv2.INTER_LANCZOS4)
                    # Same full-image Poisson composition as upstream, streamed.
                    frame=cv2.seamlessClone(patch,base,np.full(patch.shape[:2],255,np.uint8),((x1+x2)//2,(y1+y2)//2),cv2.NORMAL_CLONE)
                    proc.stdin.write(frame.tobytes());del out,image,frame
                    emit(stage='render',stage_name='sad_render',frame=i+1,total=total,eta=round((time.monotonic()-started)/(i+1)*(total-i-1)),actual={'sadtalker':['CPU']},message=f'SadTalker 512：逐格渲染 {i+1}/{total}；不累積整部影片於記憶體')
            proc.stdin.close()
            if proc.wait(timeout=180)!=0:raise RuntimeError('SadTalker 編碼失敗，請查看 encoding.log')
            os.replace(temp,cfg['output'])
        finally:
            if proc.poll() is None:proc.kill();proc.wait()
            log.close()
    report_peak('sad_'+a.stage)

if __name__=='__main__':
    try:main()
    except Exception as e:emit(stage='error',message=str(e));traceback.print_exc();sys.exit(1)
