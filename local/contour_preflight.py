"""One source-frame mask preview, no neural weights or model inference."""
import argparse,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))

def prepare_source(image,cfg,folder):
    import cv2,numpy as np
    from semantic_mouth import enrich_config,locate_mouth,style_profile,measured_safety_box
    from contour_mask import overlay
    cfg=enrich_config(cfg);g=locate_mouth(image,cfg)
    profile=style_profile(cfg.get('style_type','auto'),g,cfg.get('real_detail',False),cfg.get('anime_detail',False))
    feather=int(cfg.get('contour_feather_px',4))
    safety=measured_safety_box(g,image,feather)
    preview,layers=overlay(image,g,cfg['bbox'],feather,safety)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name,data in [('mouth-overlay.png',preview),('mouth-alpha.png',np.rint(layers['alpha'][:,:,0]*255).astype(np.uint8))]:
        ok,encoded=cv2.imencode('.png',data)
        if not ok:raise RuntimeError('遮罩預覽編碼失敗')
        encoded.tofile(str(folder/name))
    report=dict(run_id=cfg.get('run_id','direct'),source_size=[image.shape[1],image.shape[0]],
                face=cfg['bbox'],mouth_geometry=g,profile=profile,feather_px=feather,
                mouth_seed=cfg.get('mouth_box'),measured_mouth_box=safety,
                neural_weights_loaded=False,eyes_modified=False,
                review_required=True,notes=['紅色=重繪與羽化，黃線=測得唇線；須檢查是否定位正確。',
                'Canny 與插值不是訓練出的卡通語義模型；置信不足會停止。',
                '卡通停用寫實皮膚修復；不能保證未知牙齒形狀或所有畫風。'])
    (folder/'mouth-contour.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report

def main():
    import cv2,numpy as np
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'))
    if cfg.get('kind','photo')!='photo':raise ValueError('本階段預覽只接受靜態照片；不解碼整段影片')
    image=cv2.imdecode(np.fromfile(cfg['source'],np.uint8),cv2.IMREAD_COLOR)
    if image is None:raise ValueError('無法讀取照片')
    r=prepare_source(image,cfg,Path(a.out))
    print(json.dumps(dict(ok=True,profile=r['profile'],mouth_method=r['mouth_geometry']['method'],
                         overlay=str(Path(a.out)/'mouth-overlay.png'),neural_weights_loaded=False),ensure_ascii=False))

if __name__=='__main__':main()
