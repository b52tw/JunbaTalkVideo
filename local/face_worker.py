"""Isolated YuNet + OpenVINO 98-point facial localization. Offline after install."""
import argparse,json,sys,subprocess,os,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.modules['openvino_telemetry']=None
from runtime_support import emit,process_policy,report_peak

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));process_policy(cfg)
    import cv2,numpy as np,openvino as ov
    from face_geometry import decode_heatmaps,regions_from_landmarks,reliable_regions,regions_from_five
    from semantic_mouth import heatmap_reliability,from_landmarks,edge_face_anchors,ContourNotReliable
    cv2.setNumThreads(2)
    if cfg.get('kind')=='video':
        r=subprocess.run([cfg['ffmpeg'],'-v','error','-nostdin','-i',cfg['source'],'-frames:v','1','-f','image2pipe','-vcodec','png','-threads','2','pipe:1'],capture_output=True,timeout=45,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        if r.returncode:raise RuntimeError('無法讀取影片第一格')
        image=cv2.imdecode(np.frombuffer(r.stdout,np.uint8),cv2.IMREAD_COLOR)
    else:image=cv2.imdecode(np.fromfile(cfg['source'],dtype=np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise RuntimeError('無法讀取素材')
    h,w=image.shape[:2];scale=min(1,1280/max(w,h));small=cv2.resize(image,(round(w*scale),round(h*scale)))
    det=cv2.FaceDetectorYN.create(str(Path(__file__).with_name('yunet.onnx')),'',(small.shape[1],small.shape[0]),.75,.3,5000)
    _,faces=det.detect(small)
    if faces is None or not len(faces):raise RuntimeError('未找到可靠的真人臉；卡通、遮擋或側臉請改用手動框選。')
    if len(faces)>1:raise RuntimeError('偵測到多張人臉，請先裁切成單人素材再試；不會自動猜選人物。')
    f=faces[0];x,y,bw,bh=f[:4]/scale
    # Landmark model expects a nearly square crop around the face.
    side=max(bw,bh)*1.12;cx=x+bw/2;cy=y+bh/2
    roi=[max(0,round(cx-side/2)),max(0,round(cy-side/2)),min(w,round(cx+side/2)),min(h,round(cy+side/2))]
    x1,y1,x2,y2=roi;crop=image[y1:y2,x1:x2]
    core=ov.Core();requested=cfg.get('device','AUTO');device=next((x for x in core.available_devices if x.startswith('GPU')),'CPU') if requested=='AUTO' else requested
    options={'PERFORMANCE_HINT':'LATENCY'}
    if device=='CPU':options.update(INFERENCE_NUM_THREADS=2,NUM_STREAMS='1')
    model=core.read_model(str(Path(cfg['models'])/'landmarks98.xml'));fallback=''
    try:compiled=core.compile_model(model,device,options)
    except Exception as e:
        if requested!='AUTO':raise
        fallback=str(e)[:200];compiled=core.compile_model(model,'CPU',{'INFERENCE_NUM_THREADS':2,'NUM_STREAMS':'1'})
    tensor=cv2.resize(crop,(64,64)).transpose(2,0,1)[None].astype(np.float32)
    heatmaps=compiled([tensor])[0];points=decode_heatmaps(heatmaps,roi)
    regions=regions_from_landmarks(points,[x,y,x+bw,y+bh],w,h)
    method='landmarks98';warnings=[];reliable,heatmap_diagnostics=heatmap_reliability(heatmaps)
    try:
        if not reliable or not reliable_regions(regions,[x/w,y/h,(x+bw)/w,(y+bh)/h]):raise ContourNotReliable('98 點品質不足')
        mouth_geometry=from_landmarks(regions['landmarks'])
        eye_contours=[regions['landmarks'][s:s+8] for s in (60,68)]
    except ContourNotReliable:
        # Five points are SEARCH SEEDS only. No five-point mouth/eye boxes escape.
        seeds=regions_from_five(f[4:14].reshape(5,2)/scale,[x,y,x+bw,y+bh],w,h)
        found=edge_face_anchors(image,seeds,cfg.get('preserve_eye_boxes'))
        mouth_geometry=found['mouth_geometry'];eye_contours=found['eye_contours']
        polygon=np.asarray(mouth_geometry['outer']);lo=np.maximum(0,polygon.min(0)-[5/w,5/h]);hi=np.minimum(1,polygon.max(0)+[5/w,5/h])
        regions=dict(bbox=seeds['bbox'],mouth_box=[float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])],
                     eye_boxes=found['eye_boxes'],landmarks=mouth_geometry['outer']+sum(eye_contours,[]))
        method=found['method']
        warnings.append('98 點不足；已採用影像輪廓輔助點，插值點不是新增的語義證據。請先檢查遮罩預覽。')
    # Reject grossly inconsistent landmark layouts instead of returning bad boxes.
    eye_y=sum(b[1]+b[3]/2 for b in regions['eye_boxes'])/2;mouth_y=regions['mouth_box'][1]+regions['mouth_box'][3]/2
    if mouth_y<=eye_y or mouth_y>regions['bbox'][1]+regions['bbox'][3]:raise RuntimeError('關鍵點結果不穩定，請改用手動框選。')
    message=('98 點定位完成' if method=='landmarks98' else '輪廓備用定位完成（未採用 5 點粗框）')+'；請先確認唇線遮罩與原眼框。'
    if bw<120:warnings.append('原圖臉部寬度不足 120 像素，嘴部資訊有限；建議更近、更清楚的素材。')
    if cfg.get('kind')=='video':warnings.append('本次只定位第一格，尚未逐格追蹤；大幅移動的影片仍需其他引擎。')
    result=dict(stage='face_detected',**regions,mouth_geometry=mouth_geometry,eye_contours=eye_contours,heatmap_diagnostics=heatmap_diagnostics,method=method,confidence=round(float(f[-1]),4),width=w,height=h,face_pixels=[round(float(bw)),round(float(bh))],actual={'face_detector':['CPU'],'landmarks98':[str(x) for x in compiled.get_property('EXECUTION_DEVICES')]},fallback=fallback,warnings=warnings,message=message)
    Path(cfg['result']).write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8');emit(**result)
if __name__=='__main__':
    try:main()
    except Exception as e:emit(stage='error',message=str(e));traceback.print_exc(file=sys.stderr);sys.exit(1)
    finally:report_peak('face_detection')
