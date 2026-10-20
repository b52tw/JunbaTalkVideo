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
    target=Path(cfg['output']);puppet_mode=cfg.get('engine')=='puppet'
    if not puppet_mode:
        meta=json.loads(target.with_name('mouth-cache.json').read_text(encoding='utf-8'))
        cache=target.with_name('mouth-cache');actual=meta['actual'];total=meta['frames'];fps=meta['fps'];w=meta['width'];h=meta['height']
    photo=cfg.get('kind')=='photo';flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    if photo:first=cv2.imdecode(np.fromfile(cfg['source'],dtype=np.uint8),cv2.IMREAD_COLOR)
    else:
        read=subprocess.run([cfg['ffmpeg'],'-hide_banner','-loglevel','error','-nostdin','-i',cfg['source'],'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','2','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60,creationflags=flags)
        if read.returncode:raise RuntimeError('無法讀取素材影片')
        first=cv2.imdecode(np.frombuffer(read.stdout,dtype=np.uint8),cv2.IMREAD_COLOR)
    if first is None:raise RuntimeError('無法讀取原始素材')
    puppet=None;puppet_track=None
    if puppet_mode:
        # v0.8.3b cartoon lip puppet: no neural model; warp the original mouth.
        if not photo:raise RuntimeError('卡通原圖唇形只支援照片素材')
        h0,w0=first.shape[:2];scale=min(1.,float(cfg.get('max_side',1920))/max(h0,w0))
        w,h=max(2,int(w0*scale)//2*2),max(2,int(h0*scale)//2*2);fps=int(cfg.get('fps',25) or 25)
        cfg['anime_detail']=False;cfg['real_detail']=False;cfg['contour_mode']=False
    base=cv2.resize(first,(w,h),interpolation=cv2.INTER_AREA)
    if puppet_mode:
        from cartoon_puppet import CartoonPuppet,mouth_drive,viseme_track
        from semantic_mouth import locate_mouth
        try:outer=locate_mouth(base,dict(cfg,landmarks=[]))['outer']
        except Exception as e:
            # No measurable lip line: use an ellipse inside the confirmed mouth box.
            x,y,bw_,bh_=cfg['mouth_box'];t=np.linspace(0,2*np.pi,32,endpoint=False)
            outer=np.c_[x+bw_/2+bw_*.48*np.cos(t),y+bh_/2+bh_*.48*np.sin(t)].tolist()
            emit(stage='compose',message='警告：原圖唇線不清楚（'+str(e)[:80]+'）；改用確認嘴框推估上下唇')
        shapes=cfg.get('puppet_style','smooth')=='shapes'
        puppet_track=viseme_track(cfg['audio'],fps) if shapes else [(float(v),0.) for v in mouth_drive(cfg['audio'],fps)]
        total=len(puppet_track);actual={'cartoon_puppet':['CPU / '+('五種口型切換' if shapes else '連續開合')]}
        # v0.8.5 multi-speaker dialogue: one puppet per speaker mouth, each
        # driven only during that speaker's segments.
        from puppet_motion import speaker_masks
        dialogue=[]
        if cfg.get('dialogue') and Path(cfg['dialogue']).is_file():
            dialogue=json.loads(Path(cfg['dialogue']).read_text(encoding='utf-8')).get('lines',[])
        speakers=[s for s in (cfg.get('speakers') or []) if s.get('mouth_box')]
        if not speakers:speakers=[dict(id='A',mouth_box=outer,face_box=cfg['bbox'],_outer=outer)]
        masks=speaker_masks(dialogue,total,fps) if dialogue else {}
        puppets=[]
        for sp in speakers:
            sid=str(sp.get('id','A')).upper()
            if sp.get('_outer'):sp_outer=sp['_outer']
            elif sid=='A' and sp.get('mouth_box')==cfg.get('mouth_box'):sp_outer=outer
            else:
                try:sp_outer=locate_mouth(base,dict(cfg,landmarks=[],mouth_box=sp['mouth_box']))['outer']
                except Exception:
                    x,y,bw_,bh_=sp['mouth_box'];t=np.linspace(0,2*np.pi,32,endpoint=False)
                    sp_outer=np.c_[x+bw_/2+bw_*.48*np.cos(t),y+bh_/2+bh_*.48*np.sin(t)].tolist()
                    emit(stage='compose',message=f'警告：說話者 {sid} 唇線不清楚；改用嘴框推估')
            fb=sp.get('face_box') or (cfg['bbox'] if sid=='A' else None)
            try:rig=CartoonPuppet(base,sp_outer,max_open=float(cfg.get('puppet_open',.55)),face_h=(fb[3]*h if fb else None))
            except Exception as e:
                emit(stage='compose',message=f'警告：說話者 {sid} 嘴部無法變形（{e}）；此人保持原圖');continue
            mask=masks.get(sid) if dialogue else None
            if dialogue and mask is None:mask=np.zeros(total,np.float32)
            puppets.append((sid,rig,mask,sp.get('face_box') or (cfg['bbox'] if sid=='A' else None)))
        if dialogue:emit(stage='compose',message='多人對話：'+'、'.join(p[0] for p in puppets)+f'；共 {len(dialogue)} 段台詞')
        if not puppets:raise RuntimeError('無法建立任何說話者的嘴部變形；請重新框選嘴巴')
        puppet=puppets[0][1]
        emit(stage='compose',actual=actual,message='卡通原圖唇形擺動：'+('B 口型切換（閉／半開／大開／圓唇／咧嘴）' if shapes else 'A 連續開合')+'；不使用 AI 重繪')
    contour_report=None;contour_tracker=None;source_geometry=None
    contour_file=target.with_name('mouth-contour.json')
    if contour_file.is_file() and not puppet_mode:
        candidate=json.loads(contour_file.read_text(encoding='utf-8'))
        if candidate.get('run_id')==cfg.get('run_id','direct'):contour_report=candidate
    user_mouth_box=list(cfg['mouth_box']) if cfg.get('mouth_box') else None
    box_fallback_frames=0
    if cfg.get('contour_mode',False) and contour_report is None:
        from contour_preflight import prepare_source
        try:contour_report=prepare_source(base,cfg,target.parent)
        except Exception as e:
            # v0.8.3b: no reliable source lip line -> feather the confirmed box.
            contour_report=None
            emit(stage='compose',message='警告：原圖唇線分析失敗（'+str(e)[:120]+'）；改用確認嘴框＋邊緣羽化合成')
    # v0.8.3b: cartoon vs real compositing. Cartoons replace mouth + corners +
    # smile creases as one feathered block; real faces use the precise lip line.
    style=cfg.get('compose_style','auto')
    detected=contour_report['profile']['style_type'] if contour_report else 'cartoon'
    cartoon_block=style=='cartoon' or (style=='auto' and detected=='cartoon')
    if contour_report is not None and cartoon_block:
        profile=contour_report['profile']
        cfg['real_detail']=False;cfg['anime_detail']=bool(cfg.get('anime_detail'));cfg['detail_strength']=.35
        if user_mouth_box:cfg['mouth_box']=user_mouth_box
        emit(stage='compose',message='卡通整塊融合：嘴巴、嘴角與笑紋一起替換（大範圍羽化）；眼框及眨眼排程不變')
        contour_report=dict(contour_report,feather_px=4);contour_report['_cartoon']=True
    if contour_report is not None and not cartoon_block:
        from contour_mask import ContourTracker,confirmed_envelope
        from semantic_mouth import dynamic_geometry,union_geometry,ContourNotReliable
        if contour_report['source_size']!=[w,h]:raise ValueError('唇線暫存尺寸不符，請重建遮罩')
        source_geometry=contour_report['mouth_geometry']
        cfg['mouth_box']=contour_report['measured_mouth_box']
        if user_mouth_box:
            # Envelope/safety area = measured lips UNION the user-confirmed box.
            mx,my,mw,mh=cfg['mouth_box'];ux,uy,uw,uh=user_mouth_box
            x0,y0=min(mx,ux),min(my,uy);x1_,y1_=max(mx+mw,ux+uw),max(my+mh,uy+uh)
            user_mouth_box=[x0,y0,x1_-x0,y1_-y0]
        else:user_mouth_box=list(cfg['mouth_box'])
        contour_tracker=ContourTracker(source_geometry,confirmed_envelope(source_geometry,user_mouth_box))
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
        torso=TorsoMotion(base,cfg['bbox'],audio_drive(cfg['audio']),scale={'small':1.0,'medium':1.6,'large':2.3}.get(cfg.get('body_amp','small'),1.0))
        actual['torso_motion']=['CPU / GrabCut + RMS/Pitch damping' if torso.ready else '未套用：無可靠人物遮罩']
        emit(stage='compose' if torso.ready else 'fallback',message='肩膀：GrabCut 前景＋呼吸／語音阻尼，背景鎖定' if torso.ready else '無可靠人物遮罩，保留原姿勢；不拉伸背景')
    # v0.8.5 head / brow motion (photos). Head moves last so blink, mouth and
    # brows travel with it; brows only on the lip-puppet path.
    head_rigs=[];body_rigs=[];brow_rig=None;brow_curve=None
    head_amp=cfg.get('head_motion','off')
    tagged=[]
    if puppet_mode and dialogue:
        from puppet_motion import dialogue_actions
        tagged=dialogue_actions(dialogue)
    if photo and (head_amp!='off' or cfg.get('motion_timeline') or tagged):
        from puppet_motion import HeadRig,BrowRig,BodyRig,motion_curves,parse_timeline,body_sway_curve
        from cartoon_puppet import audio_features
        level,_=audio_features(cfg['audio'],fps);level=np.resize(level,total) if len(level)!=total else level
        timeline=parse_timeline(cfg.get('motion_timeline',''))
        lead_s=float(cfg.get('lead',0))
        timeline=[(who,a_-lead_s,b_-lead_s,act) for who,a_,b_,act in timeline]
        timeline+=tagged
        targets=[(p[0],p[3],p[2]) for p in puppets if p[3]] if puppet_mode else [('A',cfg['bbox'],None)]
        for sid,fbox,mask in targets:
            curves=motion_curves(level,fps,head_amp,timeline,active=mask,speaker=sid,continuous=head_amp!='off')
            head_rigs.append((HeadRig(base.shape,fbox),curves))
            # v0.8.7: body sway removed (2-D warp bent arms/hands next to the body).
            if puppet_mode and sid=='A' and cfg.get('eye_boxes'):
                brow_rig=BrowRig(base.shape,cfg['eye_boxes']);brow_curve=curves[3]
        actual['head_motion']=['CPU / 2D 頭部擺動 '+head_amp+(' + 時間軸' if timeline else '')]
        emit(stage='compose',message=f'頭部動作：{head_amp}；{len(head_rigs)} 人'+('；含眉毛' if brow_rig else '')+(f'；時間軸 {len(timeline)} 項' if timeline else ''))
    frame=base;last_result=base
    try:
        for i in range(total):
            if decoder:
                raw=decoder.stdout.read(w*h*3)
                if len(raw)!=w*h*3:raise RuntimeError('原始影片讀取中斷')
                frame=np.frombuffer(raw,dtype=np.uint8).reshape(h,w,3)
            if puppet is not None:
                motion_frame=torso.apply(i/fps) if torso is not None else frame
                result=brow_rig.apply(motion_frame,float(brow_curve[i])) if brow_rig is not None else motion_frame
                opening,width=puppet_track[i]
                for sid,rig,mask,_fb in puppets:
                    m=1.0 if mask is None else float(mask[i])
                    if m>0:result=rig.frame(opening*m,result,width*m)
            else:
                generated=cv2.imdecode(np.fromfile(str(cache/f'{i:05d}.png'),dtype=np.uint8),cv2.IMREAD_COLOR)
                if generated is None:raise RuntimeError('嘴型暫存不完整，請重新生成')
                active_geometry=None;tracking_status=''
                if contour_tracker is not None:
                    try:found=dynamic_geometry(generated,cfg['bbox'],cfg['mouth_box'])
                    except Exception:found=None  # any detection failure = lost contour, never fatal
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
                blend_args=dict(blend=cfg.get('blend_mode','feather'),scope='cartoon' if cartoon_block else cfg.get('mouth_scope','lips'),full=cfg.get('composition_mode')=='full',color_transfer=cfg.get('color_transfer',True),feather_px=contour_report['feather_px'] if contour_report else 4)
                safety=user_mouth_box if tracking_status=='confirmed mouth envelope' else cfg.get('mouth_box')
                result=None
                # v0.8.3b fallback chain, never aborting the video:
                # dynamic contour -> source contour -> confirmed box with feathering.
                for label,geom,box in (('動態唇線',active_geometry,safety),('原圖精確唇線',source_geometry,cfg.get('mouth_box'))):
                    if geom is None:continue
                    try:result=blend_mouth(motion_frame,generated,cfg['bbox'],box,mouth_geometry=geom,**blend_args);break
                    except (ContourNotReliable,ValueError) as e:emit(stage='compose',message=f'警告：第 {i+1} 格{label}無法使用（{e}）')
                if result is None:
                    if active_geometry is not None:box_fallback_frames+=1
                    result=blend_mouth(motion_frame,generated,cfg['bbox'],user_mouth_box or cfg.get('mouth_box'),mouth_geometry=None,**blend_args)
            if photo:result=photo_motion(result,cfg['bbox'],i/fps,cfg.get('eye_boxes',[]) if cfg.get('blink') or cfg.get('gaze') else [],cfg.get('gaze',False),cfg.get('micro_motion',False),cfg.get('blink',False),'' if cfg.get('body_motion')=='speech' else cfg.get('body_motion',''),cfg.get('hand_box',[]),cfg.get('blink_profile','bezier'),cfg.get('brow_motion',False),cfg.get('lid_shadow',True))
            for rig,(ang,nod_,shk,_b) in head_rigs:result=rig.apply(result,float(ang[i]),float(nod_[i]),float(shk[i]))
            for rig,sw in body_rigs:result=rig.apply(result,float(sw[i]),0.,0.)
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
        note+=f'；警告：{contour_tracker.fallback_frames} 格動態唇線不足，已用確認嘴區範圍合成'
    if box_fallback_frames:
        note+=f'；警告：{box_fallback_frames} 格改用確認嘴框＋邊緣羽化'
    if note:note+='，請目視檢查'
    if puppet_mode:mode_text='卡通原圖唇形擺動（'+('B 口型切換' if cfg.get('puppet_style')=='shapes' else 'A 連續開合')+'）'
    else:mode_text='MuseTalk AI 對嘴（合成：'+('卡通整塊融合' if cartoon_block else '真人精準唇線')+'）'
    emit(stage='done',actual=actual,message='完成；本次使用：'+mode_text+'；原圖細節及片頭片尾已保留'+note)
if __name__=='__main__':
    try:main()
    except __import__('inference_backend').CPUFallbackRequired as e:emit(stage='fallback',message='退出修復後端，等待新 CPU 程序：'+str(e));sys.exit(75)
    except Exception as e:
        emit(stage='error',message=str(e));traceback.print_exc(file=sys.stderr);sys.exit(1)

    finally:report_peak("compose")
