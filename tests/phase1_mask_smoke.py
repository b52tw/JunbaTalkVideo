"""One synthetic frame, no model imports; verify reject/geometry/safety behavior."""
import argparse,ast,hashlib,json,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local'))

def main():
    import cv2,numpy as np
    from semantic_mouth import locate_mouth,edge_face_anchors,heatmap_reliability,style_profile,prepare_face,ContourNotReliable,geometry,union_geometry
    from contour_mask import masks,overlay,ContourTracker
    from quality import blend_mouth
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--baseline-source');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    image=np.full((256,320,3),(150,185,220),np.uint8)
    cv2.ellipse(image,(160,168),(48,18),0,0,360,(20,25,30),-1)
    cv2.rectangle(image,(126,155),(194,164),(235,240,245),-1)
    for center in ((106,94),(214,94)):
        cv2.ellipse(image,center,(18,8),0,0,360,(15,20,25),-1)
    face=[.15,.12,.70,.76];mouth=[.31,.56,.38,.19]
    seeds=dict(mouth_box=mouth,eye_boxes=[[.26,.31,.15,.12],[.59,.31,.15,.12]])
    cfg=dict(bbox=face,mouth_box=mouth,landmarks=[[.1,.1]]*5)
    g=locate_mouth(image,cfg);assert g['method'].startswith('edge') and len(g['outer'])==32
    found=edge_face_anchors(image,seeds,preserved_eyes=seeds['eye_boxes'])
    assert found['eye_boxes']==seeds['eye_boxes']
    rejected=False
    try:locate_mouth(np.full_like(image,200),cfg)
    except ContourNotReliable:rejected=True
    assert rejected,'blank frame must not fall back to a five-point box'
    assert heatmap_reliability(np.zeros((1,98,16,16),np.float32))[0] is False
    for style in ('cartoon','real'):
        profile=style_profile(style,g,real_detail=True,anime_detail=True)
        assert profile['real_detail']==(style=='real')
        assert profile['detail_strength']==.35
        processed=prepare_face(image,profile)
        if style=='real':assert np.array_equal(processed,image)
        else:assert np.max(np.abs(processed.astype(float)-image))<=2
    for feather in (3,4,5):
        layers=masks(g,face,320,256,feather,mouth);x1,y1,x2,y2=layers['roi']
        outside=cv2.distanceTransform(1-layers['hard'],cv2.DIST_L2,cv2.DIST_MASK_PRECISE)>=feather
        assert np.all(layers['alpha'][:,:,0][outside]==0)
        generated=np.full((256,256,3),(180,100,160),np.uint8)
        result=blend_mouth(image,generated,face,mouth,mouth_geometry=g,feather_px=feather)
        fullmask=np.zeros((256,320),bool);fullmask[y1:y2,x1:x2]=layers['alpha'][:,:,0]>0
        assert np.array_equal(result[~fullmask],image[~fullmask])
        assert np.any(result[fullmask]!=image[fullmask])
        assert np.array_equal(result[75:110],image[75:110]),'eye pixels changed'
    tracker=ContourTracker(g)
    crescent=geometry([[.35,.5],[.5,.58],[.65,.5],[.6,.65],[.5,.68],[.4,.65]],[],'test_measured_smile')
    united=masks(union_geometry(crescent,crescent),face,320,256,4)
    x1,y1,_,_=united['roi']
    assert united['alpha'][round(.52*256)-y1,160-x1,0]==0,'smile concavity bridged into skin below nose'
    for _ in range(5):tracker.update(None)
    try:tracker.update(None)
    except ContourNotReliable:pass
    else:raise AssertionError('unbounded lost-contour fallback')
    blink_unchanged=None
    if a.baseline_source:
        with zipfile.ZipFile(a.baseline_source) as z:old=ast.parse(z.read('local/quality.py').decode())
        new=ast.parse((Path(__file__).resolve().parents[1]/'local/quality.py').read_text(encoding='utf-8'))
        def funcs(tree):return {n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,ast.FunctionDef)}
        old,new=funcs(old),funcs(new)
        for name in ('blink_starts','blink_amount','animate_eye','photo_motion'):assert old[name]==new[name],name+' changed'
        blink_unchanged=True
    preview,layers=overlay(image,g,face,4,mouth)
    cv2.imencode('.png',preview)[1].tofile(str(out/'synthetic-mouth-overlay.png'))
    report=dict(ok=True,synthetic_frames=1,neural_weights_loaded=False,no_five_point_output=True,
                feather_outside_zero=[3,4,5],outside_pixels_identical=True,eyes_identical=True,
                blink_functions_unchanged=blink_unchanged,real_detail_strength=.35,cartoon_real_detail=False)
    (out/'smoke.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__':main()
