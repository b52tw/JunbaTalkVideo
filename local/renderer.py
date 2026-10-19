"""Second, isolated worker: optional mouth SR and full-resolution composition.
The large MuseTalk process has exited before this worker starts.
"""
import argparse,json,os,subprocess,sys,time,traceback
from pathlib import Path
# Windows embedded Python ._pth excludes this sibling script directory.
# Use the worker location, never the current working directory or PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules['openvino_telemetry']=None
from runtime_support import process_policy,report_peak
from semantic_mouth import ContourNotReliable

def emit(**obj):print(json.dumps(obj,ensure_ascii=False),flush=True)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    import cv2,numpy as np
    from quality import blend_mouth,photo_motion
    cv2.setNumThreads(2)
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8-sig'));process_policy(cfg)
    threads=2 if cfg.get('low_impact',True) else 4
    target=Path(cfg['output']);meta=json.loads(target.with_name('mouth-cache.json').read_text(encoding='utf-8'))
    cache=target.with_name('mouth-cache');actual=meta['actual'];total=meta['frames'];fps=meta['fps'];w=meta['width'];h=meta['height']
    photo=cfg.get('kind')=='photo';flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    if photo:first=cv2.imdecode(np.fromfile(cfg['source'],dtype=np.uint8),cv2.IMREAD_COLOR)
    else:
        read=subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-i',cfg['source'],'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,creationflags=flags)
        if read.returncode:raise RuntimeError('無法讀取素材影片')
        first=cv2.imdecode(np.frombuffer(read.stdout,dtype=np.uint8),cv2.IMREAD_COLOR)
    if first is None:raise RuntimeError('無法讀取原始素材')
    base=cv2.resize(first,(w,h),interpolation=cv2.INTER_AREA)
    contour_report=None;contour_tracker=None;source_geometry=None
    contour_file=target.with_name('mouth-contour.json')
    if contour_file.is_file():
        candidate=json.loads(contour_file.read_text(encoding='utf-8'))
        if candidate.get('run_id')==cfg.get('run_id','direct'):contour_report=candidate
    if cfg.get('contour_mode',False) and contour_report is None:
        from contour_preflight import prepare_source
        contour_report=prepare_source(base,cfg,target.parent)
    if contour_report is not None:
        from contour_mask import ContourTracker,confirmed_envelope
        from semantic_mouth import dynamic_geometry,union_geometry,ContourNotReliable
        if contour_report['source_size']!=[w,h]:raise ValueError('唇線暫存尺寸不符，請重建遮罩')
        source_geometry=contour_report['mouth_geometry']
        cfg['mouth_box']=contour_report['measured_mouth_box']
        contour_tracker=ContourTracker(source_geometry,confirmed_envelope(source_geometry,cfg['mouth_box']))
        profile=contour_report['profile']
        cfg['real_detail']=profile['real_detail'];cfg['anime_detail']=profile['anime_detail'];cfg['detail_strength']=.35
        emit(stage='compose',message='唇線分層融合：'+profile['style_type']+'；修復強度 35%；眼框及眨眼排程不變')
    enhancer=None
    if cfg.get('anime_detail') or cfg.get('real_detail'):
        try:
            name='gfpgan_1.4' if cfg.get('real_detail') else 'sr_anime'
            try:
                subprocess.run([sys.executable,'-I','-u',str(Path(__file__).with_name('detail_cache.py')),'--root',cfg['models'],'--name',name],check=True)
            except subprocess.CalledProcessError:
                emit(stage='fallback',message='修復模型轉換失敗；停用選用修復，保留已生成嘴型與 CPU 合成')
                cfg['real_detail']=False;cfg['anime_detail']=False
            from detail_cache import path_for
            from inference_backend import NativeModel
            real_detail=cfg.get('real_detail',False);size=512 if real_detail else 256
            enhancer=NativeModel(path_for(cfg['models'],'gfpgan_1.4' if real_detail else 'sr_anime'),'real_mouth_detail' if real_detail else 'anime_mouth_detail',cfg,emit,actual,shapes={'input':[1,3,size,size]})
        except __import__('inference_backend').CPUFallbackRequired:raise
        except Exception as e:
            enhancer=None
            from inference_backend import fault
            fault(cfg,emit,'mouth_detail',e,'選用修復不可用，改 CPU 原圖融合；保留神經嘴型')
    lead_frames=round(float(cfg.get('lead',0))*fps);tail_frames=round(float(cfg.get('tail',0))*fps)
    lead=lead_frames/fps;duration=(lead_frames+total+tail_frames)/fps
    temp=target.with_name('local-render.mp4')
    command=[cfg['ffmpeg'],'-hide_banner','-nostdin','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-i',cfg['audio'],'-map','0:v:0','-map','1:a:0','-af',f'adelay={round(lead*1000)}:all=1,apad','-t',str(duration),'-c:v','libx264','-preset','medium','-crf','16','-pix_fmt','yuv420p','-c:a','aac','-b:a','160k','-movflags','+faststart','-threads','2',str(temp)]
    decoder=None;started=time.monotonic();log=open(target.with_name('local-ffmpeg.log'),'wb');decode_log=open(target.with_name('local-compose-decode.log'),'wb')
    proc=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=log,stderr=log,creationflags=flags)
    if not photo:decoder=subprocess.Popen([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-stream_loop','-1','-i',cfg['source'],'-an','-vf',f'fps={fps},scale={w}:{h}','-f','rawvideo','-pix_fmt','bgr24','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=decode_log,creationflags=flags)
    torso=None
    if photo and cfg.get('body_motion')=='speech':
        from body_motion import TorsoMotion,audio_drive
        torso=TorsoMotion(base,cfg['bbox'],audio_drive(cfg['audio']))
        actual['torso_motion']=['CPU / GrabCut + RMS/Pitch damping' if torso.ready else '未套用：無可靠人物遮罩']
        emit(stage='compose' if torso.ready else 'fallback',message='肩膀：GrabCut 前景＋呼吸／語音阻尼，背景鎖定' if torso.ready else '無可靠人物遮罩，保留原姿勢；不拉伸背景')
    frame=base;last_result=base
    try:
        for i in range(total):
            if decoder:
                raw=decoder.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise RuntimeError('原始影片讀取中斷')
                frame=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3)
            generated=cv2.imdecode(np.fromfile(str(cache/f'{i:05d}.png'),dtype=np.uint8),cv2.IMREAD_COLOR)
            if generated is None:raise RuntimeError('嘴型暫存不完整，請重新生成')
            active_geometry=None
            if contour_tracker is not None:
                try:found=dynamic_geometry(generated,cfg['bbox'],cfg['mouth_box'])
                except ContourNotReliable:found=None
                current_geometry,tracking_status=contour_tracker.update(found)
                active_geometry=current_geometry if tracking_status=='confirmed mouth envelope' else union_geometry(source_geometry,current_geometry)
                if found is None:emit(stage='compose',message='動態唇線不足；沿用上一精確輪廓'+('，此格改用已確認嘴區範圍' if tracking_status=='confirmed mouth envelope' else ''))
            if enhancer is not None:
                if cfg.get('real_detail'):
                    x=cv2.resize(generated,(512,512),interpolation=cv2.INTER_CUBIC)
                    enhanced=enhancer([(x[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/127.5-1)])[0][0]
                    restored=np.clip((enhanced.transpose(1,2,0)+1)*127.5,0,255).astype(np.uint8)[:,:,::-1]
                    # High-frequency residual only: retain the speech model's
                    # low-frequency mouth opening/shape instead of replacing it.
                    detail_base=cv2.resize(generated,(512,512),interpolation=cv2.INTER_CUBIC).astype(np.float32)
                    residual=restored.astype(np.float32)-cv2.GaussianBlur(restored.astype(np.float32),(0,0),2.0)
                    strength=float(cfg.get('detail_strength',.5))
                    generated=np.clip(detail_base+np.clip(residual,-24,24)*strength,0,255).astype(np.uint8)
                else:
                    enhanced=enhancer([generated[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255.])[0][0]
                    restored=np.clip(enhanced.transpose(1,2,0)*255.,0,255).astype(np.uint8)[:,:,::-1]
                    base_detail=cv2.resize(generated,(restored.shape[1],restored.shape[0]),interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
                    residual=restored.astype(np.float32)-cv2.GaussianBlur(restored.astype(np.float32),(0,0),1.6)
                    generated=np.clip(base_detail+np.clip(residual,-18,18)*float(cfg.get('detail_strength',.5)),0,255).astype(np.uint8)
            motion_frame=torso.apply(i/fps) if torso is not None else frame
            blend_args=dict(blend=cfg.get('blend_mode','feather'),scope=cfg.get('mouth_scope','lips'),full=cfg.get('composition_mode')=='full',color_transfer=cfg.get('color_transfer',True),feather_px=contour_report['feather_px'] if contour_report else 4)
            try:result=blend_mouth(motion_frame,generated,cfg['bbox'],cfg.get('mouth_box'),mouth_geometry=active_geometry,**blend_args)
            except ContourNotReliable:
                # v0.8.3b: one bad dynamic contour must not abort the whole video.
                # Fall back to the user-confirmed source lip contour for this frame.
                if active_geometry is None or source_geometry is None:raise
                emit(stage='compose',message=f'第 {i+1} 格動態唇線超出嘴區；此格改用原圖精確唇線')
                result=blend_mouth(motion_frame,generated,cfg['bbox'],cfg.get('mouth_box'),mouth_geometry=source_geometry,**blend_args)
            if photo:result=photo_motion(result,cfg['bbox'],i/fps,cfg.get('eye_boxes',[]) if cfg.get('blink') or cfg.get('gaze') else [],cfg.get('gaze',False),cfg.get('micro_motion',False),cfg.get('blink',False),'' if cfg.get('body_motion')=='speech' else cfg.get('body_motion',''),cfg.get('hand_box',[]),cfg.get('blink_profile','bezier'),cfg.get('brow_motion',False),cfg.get('lid_shadow',True))
            if i==0:
                for k in range(lead_frames):
                    weight=max(0,min(1,(k/fps-(lead-.20))/.20));weight=weight*weight*(3-2*weight)
                    proc.stdin.write(cv2.addWeighted(base,1-weight,result,weight,0).tobytes())
            proc.stdin.write(result.tobytes());last_result=result
            elapsed=time.monotonic()-started
            emit(stage='compose',frame=i+1,total=total,eta=round(elapsed/(i+1)*(total-i-1)),actual=actual,message=f'細節／原圖合成 {i+1}/{total} 格')
        for k in range(tail_frames):
            weight=min(1,(k+1)/max(1,.20*fps));weight=weight*weight*(3-2*weight)
            proc.stdin.write(cv2.addWeighted(last_result,1-weight,frame,weight,0).tobytes())
        proc.stdin.close()
        if proc.wait(timeout=180)!=0:raise RuntimeError('影片編碼失敗，請查看 local-ffmpeg.log')
        os.replace(temp,target)
    finally:
        if proc.poll() is None:proc.kill();proc.wait()
        log.close();decode_log.close()
        if decoder:
            decoder.stdout.close()
            if decoder.poll() is None:decoder.kill()
            decoder.wait()
    # Keep small lossless patches for recovery. No user files are changed.
    note=''
    if contour_tracker is not None and contour_tracker.fallback_frames:
        note=f'；其中 {contour_tracker.fallback_frames} 格動態唇線不足，已用確認嘴區範圍合成，請檢查這些畫面'
    emit(stage='done',actual=actual,message='精緻對嘴完成；原圖細節及片頭片尾已保留'+note)
if __name__=='__main__':
    try:main()
    except __import__('inference_backend').CPUFallbackRequired as e:emit(stage='fallback',message='退出修復後端，等待新 CPU 程序：'+str(e));sys.exit(75)
    except Exception as e:
        emit(stage='error',message=str(e));traceback.print_exc(file=sys.stderr);sys.exit(1)

    finally:report_peak("compose")
