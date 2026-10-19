"""Local static A/B verification. Never compiles or launches a video render."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'local'))

def main():
    parser=argparse.ArgumentParser(description='靜態唇線／遮罩比對；不編譯、不生成影片')
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--config',help='既有 local-config.json 或預覽設定')
    group.add_argument('--project',help='既有專案資料夾')
    group.add_argument('--image',help='新照片；需已有五官模型，獨立子程序定位')
    parser.add_argument('--models',help='已安裝五官模型資料夾；不會下載')
    parser.add_argument('--style',choices=['auto','real','cartoon'],default='auto')
    parser.add_argument('--feather',type=int,choices=[3,4,5],default=4)
    parser.add_argument('--cached-frame',help='選用：既存 mouth-cache 的單格 PNG，不呼叫神經推論')
    parser.add_argument('--out',help='新建報告目錄；預設 verification/日期時間')
    args=parser.parse_args()
    folder=Path(args.out).resolve() if args.out else ROOT/'verification'/time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True,exist_ok=True)
    cached=None
    if args.image:
        if not args.models:parser.error('--image 需要 --models 指向已安裝模型；沒有模型不會猜粗框')
        cfg=dict(source=str(Path(args.image).resolve()),kind='photo',models=str(Path(args.models).resolve()),
                 device='CPU',result=str(folder/'face-result.json'),low_impact=True)
        detection=folder/'face-input.json';detection.write_text(json.dumps(cfg),encoding='utf-8')
        # This child exits before OpenCV and the compositor load in this process.
        with (folder/'face-detection.log').open('w',encoding='utf-8') as log:
            p=subprocess.run([sys.executable,'-I','-u',str(ROOT/'local/face_worker.py'),'--config',str(detection)],
                             stdout=log,stderr=subprocess.STDOUT,timeout=90)
        if p.returncode:raise RuntimeError('定位失敗；請查看 face-detection.log。不會改採 5 點粗框。')
        result=json.loads((folder/'face-result.json').read_text(encoding='utf-8'))
        cfg.update({k:result[k] for k in ('bbox','mouth_box','eye_boxes','landmarks','mouth_geometry')})
    else:
        config=Path(args.config).resolve() if args.config else Path(args.project).resolve()/'local-config.json'
        cfg=json.loads(config.read_text(encoding='utf-8-sig'))
        if cfg.get('kind','photo')!='photo':raise ValueError('此階段只檢查照片，不解碼／生成整段影片')
        project=config.parent
        if (project/'mouth-cache').is_dir():
            choices=sorted((project/'mouth-cache').glob('*.png'))
            if choices:cached=choices[len(choices)//2]
    cfg.update(style_type=args.style,contour_feather_px=args.feather)
    import cv2,numpy as np
    from contour_preflight import prepare_source
    from contour_mask import masks
    from semantic_mouth import dynamic_geometry,union_geometry,ContourNotReliable
    from quality import blend_mouth,rect
    image=cv2.imdecode(np.fromfile(cfg['source'],np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('來源照片無法讀取；原始檔未更動')
    # Feather is measured in requested OUTPUT pixels, matching stage_worker.
    limit=int(cfg.get('max_side',max(image.shape[:2])))
    scale=min(1.,limit/max(image.shape[:2]));h,w=image.shape[:2]
    w,h=max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)
    image=cv2.resize(image,(w,h),interpolation=cv2.INTER_AREA)
    report=prepare_source(image,cfg,folder);geometry=report['mouth_geometry']
    report['mask_stage_neural_weights_loaded']=False
    report['localization_model_used']=bool(args.image)
    report['neural_weights_loaded']=bool(args.image)
    report['lip_sync_model_loaded']=False
    cfg['mouth_box']=report['measured_mouth_box']
    layers=masks(geometry,cfg['bbox'],w,h,args.feather,cfg['mouth_box'])
    outside=cv2.distanceTransform(1-layers['hard'],cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>=args.feather
    report['outer_feather_hard_zero']=bool(np.all(layers['alpha'][:,:,0][outside]==0))
    report['full_frame_render_tested']=False
    report['teeth_shape_preserved']='requires human review; no new neural reconstruction was run'
    cached=Path(args.cached_frame).resolve() if args.cached_frame else cached
    if cached:
        generated=cv2.imdecode(np.fromfile(cached,np.uint8),cv2.IMREAD_COLOR)
        if generated is None:raise ValueError('快取影像無法讀取')
        try:active=union_geometry(geometry,dynamic_geometry(generated,cfg['bbox'],cfg['mouth_box']))
        except ContourNotReliable:
            active=geometry;report['cached_geometry_note']='當格邊緣不足；這張靜態比對只採原圖精確輪廓，未驗證動態追蹤'
        blended=blend_mouth(image,generated,cfg['bbox'],cfg['mouth_box'],mouth_geometry=active,feather_px=args.feather)
        used=masks(active,cfg['bbox'],w,h,args.feather,cfg['mouth_box'])
        x1,y1,x2,y2=used['roi'];full=np.zeros((h,w),bool);full[y1:y2,x1:x2]=used['alpha'][:,:,0]>0
        report['outside_pixels_identical']=bool(np.array_equal(blended[~full],image[~full]))
        report['eye_pixels_identical']=all(np.array_equal(blended[y1:y2,x1:x2],image[y1:y2,x1:x2])
            for x1,y1,x2,y2 in (rect(b,w,h) for b in cfg.get('eye_boxes',[]))) if cfg.get('eye_boxes') else None
        cv2.imencode('.png',blended)[1].tofile(str(folder/'cached-blend.png'))
        report['cached_frame']=str(cached)
    else:
        blended=image;report['cached_frame']=None
        report['comparison_note']='沒有嘴型快取；只驗證定位與遮罩，不能判定生成後牙齒或模糊改善'
    overlay_image=cv2.imdecode(np.fromfile(folder/'mouth-overlay.png',np.uint8),cv2.IMREAD_COLOR)
    # Same crop, same scale: left source; middle red mask; right cached blend or source.
    x1,y1,x2,y2=rect(cfg['bbox'],w,h)
    panels=[im[y1:y2,x1:x2] for im in (image,overlay_image,blended)]
    comparison=np.concatenate(panels,axis=1)
    cv2.imencode('.png',comparison)[1].tofile(str(folder/'comparison.png'))
    (folder/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('完成。左：原圖；中：紅色遮罩／黃色唇線；右：快取融合（沒有快取則顯示原圖）。')
    print('請人工確認唇線、牙齒、嘴周與重繪範圍；報告沒有把靜態檢查當成影片驗收。')
    print(str(folder/'comparison.png'))
    if not report['outer_feather_hard_zero']:raise RuntimeError('羽化範圍驗證失敗')

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('驗證未完成：'+str(error),file=sys.stderr)
        sys.exit(1)
