"""Phase 1 geometry only. No weights, identity lookup, or invented semantic confidence.

Five detector points may SEED an edge search; they are never returned as a mouth
contour. Interpolated vertices are labelled geometric, not learned landmarks.
"""
import cv2
import numpy as np

class ContourNotReliable(RuntimeError):
    pass

def box_pixels(box, width, height):
    x,y,w,h=map(float,box)
    return (max(0,int(x*width)),max(0,int(y*height)),
            min(width,int(np.ceil((x+w)*width))),min(height,int(np.ceil((y+h)*height))))

def resample_closed(points, count=32):
    p=np.asarray(points,np.float32).reshape(-1,2)
    p=np.concatenate([p,p[:1]])
    distances=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))]
    good=np.r_[True,np.diff(distances)>1e-5]
    p,distances=p[good],distances[good]
    if len(p)<4 or distances[-1]<1e-5:raise ContourNotReliable('唇線長度不足')
    t=np.linspace(0,distances[-1],count,endpoint=False)
    return np.stack([np.interp(t,distances,p[:,0]),np.interp(t,distances,p[:,1])],axis=1).astype(np.float32)

def valid_polygon(points):
    p=np.asarray(points,np.float32)
    return p.ndim==2 and p.shape[1]==2 and len(p)>=6 and np.isfinite(p).all() and abs(cv2.contourArea(p))>1e-7

def geometry(outer, inner, method, diagnostics=None):
    outer=np.asarray(outer,np.float32)
    if not valid_polygon(outer):raise ContourNotReliable('未取得可靠的閉合外唇輪廓；不採用 5 點粗框')
    if np.any(outer<0) or np.any(outer>1):raise ContourNotReliable('唇線超出圖片')
    inner=np.asarray(inner,np.float32).reshape(-1,2)
    anchors={name:outer[int(np.argmin(outer[:,axis])) if minimum else int(np.argmax(outer[:,axis]))].tolist()
             for name,axis,minimum in [('left_corner',0,True),('right_corner',0,False),('upper_peak',1,True),('lower_peak',1,False)]}
    return dict(outer=resample_closed(outer).tolist(),inner=resample_closed(inner,24).tolist() if valid_polygon(inner) else [],
                anchors=anchors,method=method,vertex_kind='measured contour / geometric interpolation',
                inner_visible=bool(valid_polygon(inner)),diagnostics=diagnostics or {})

def heatmap_reliability(heatmaps):
    maps=np.asarray(heatmaps,np.float32)[0]
    if maps.ndim!=3 or len(maps)!=98 or not np.isfinite(maps).all():return False,{'reason':'invalid_heatmaps'}
    scores=[]
    for hm in maps[60:96]:
        spread=float(hm.max()-np.median(hm));noise=float(np.mean(np.abs(hm-np.median(hm))))
        scores.append(spread/max(1e-6,noise))
    # A quality heuristic, NOT a calibrated landmark probability.
    accepted=np.mean(np.asarray(scores)>=4.0)>=.8
    return bool(accepted),dict(quality_kind='peak/mean-deviation heuristic, not probability',
                                strong_fraction=float(np.mean(np.asarray(scores)>=4.0)))

def from_landmarks(points):
    p=np.asarray(points,np.float32)
    if p.shape!=(98,2) or not np.isfinite(p).all():raise ContourNotReliable('98 點資料無效')
    mouth=p[76:88];eyes=p[60:76];mw=np.ptp(mouth[:,0]);mh=np.ptp(mouth[:,1])
    if mw<.01 or mh<.001 or mh>mw or np.mean(mouth[:,1])<=np.mean(eyes[:,1]):
        raise ContourNotReliable('98 點嘴部幾何不合理')
    return geometry(mouth,p[88:96],'landmarks98_contour')

def _edge_contour(image, seed, kind='mouth'):
    h,w=image.shape[:2];x1,y1,x2,y2=box_pixels(seed,w,h)
    # A rough detector seed may clip upturned smile corners. Expand SEARCH only;
    # the returned mask still follows measured contours, never this rectangle.
    # v0.8.3b: cartoon smiles extend well past the five-point mouth corners and
    # below the detector's flat mouth seed; widen the SEARCH area for the mouth.
    if kind=='mouth':padx=max(3,round((x2-x1)*.15));pady=max(3,round((y2-y1)*.6))
    else:padx=max(3,round((x2-x1)*.06));pady=max(3,round((y2-y1)*.24))
    x1,y1,x2,y2=max(0,x1-padx),max(0,y1-pady),min(w,x2+padx),min(h,y2+pady)
    roi=image[y1:y2,x1:x2]
    if min(roi.shape[:2])<8:raise ContourNotReliable('輪廓搜尋範圍太小')
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(3,3),.6)
    edges=cv2.Canny(gray,25,75)
    edges=cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    contours,hierarchy=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    rh,rw=gray.shape;choices=[];color_start=len(contours)
    neutral=np.zeros_like(gray,bool);dark_cut=60
    if kind=='mouth':
        lab=cv2.cvtColor(roi,cv2.COLOR_BGR2LAB)
        border_a=np.r_[lab[:3,:,1].ravel(),lab[-3:,:,1].ravel(),lab[:,:3,1].ravel(),lab[:,-3:,1].ravel()]
        border_gray=np.r_[gray[:3].ravel(),gray[-3:].ravel(),gray[:,:3].ravel(),gray[:,-3:].ravel()]
        neutral=(lab[:,:,1].astype(float)<float(np.median(border_a))-6)&(gray>max(90,float(np.median(border_gray))*.6))
        dark_cut=float(np.clip(np.median(border_gray)-40,30,180))
        if neutral.sum()>rw*rh*.015:
            combined=((gray<dark_cut)|neutral).astype(np.uint8)*255
            combined=cv2.morphologyEx(combined,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
            # Remove thin dimple/crease branches, not speech pixels. This operation
            # only estimates geometry; the source image is never smoothed here.
            size=max(3,min(25,int(rw*.06)))|1
            combined=cv2.morphologyEx(combined,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
            extra,_=cv2.findContours(combined,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            # v0.8.3b: also offer the un-opened lip/teeth envelope. Opening can cut
            # a thin cartoon smile into pieces and keep only the teeth; its
            # external contour encloses teeth, cavity and both corners.
            whole=cv2.morphologyEx(((gray<dark_cut)|neutral).astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
            envelope,_=cv2.findContours(whole,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            contours=list(contours)+list(extra)+list(envelope)
    for i,c in enumerate(contours):
        if len(c)<12:continue
        x,y,bw,bh=cv2.boundingRect(c);area=abs(cv2.contourArea(c))
        if not (.30*rw<=bw<=.98*rw and .10*rh<=bh<=.96*rh and area>max(12,rw*rh*.025)):continue
        # Reject contours touching the search border, where cheeks/nose leak in.
        if x<1 or y<1 or x+bw>=rw-1 or y+bh>=rh-1:continue
        center_error=((x+bw/2-rw/2)/rw)**2+((y+bh/2-rh/2)/rh)**2
        if center_error>.15:continue
        if kind=='eye' and bw/max(1,bh)<.75:continue
        score=area/(rw*rh)-center_error*.8
        if i>=color_start:score+=.25
        choices.append((score,i,c))
    if not choices:raise ContourNotReliable('邊緣不足或輪廓碰到邊界；請重新確認嘴／眼範圍，不退回 5 點粗框')
    choices.sort(key=lambda item:item[0],reverse=True);_,index,outer=choices[0]
    inner=[];color_candidate=index>=color_start
    if color_candidate:
        dark=(gray<dark_cut).astype(np.uint8)*255
        cavities,_=cv2.findContours(dark,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        inside=np.zeros_like(gray);cv2.fillPoly(inside,[outer],1)
        valid=[]
        for c in cavities:
            m=cv2.moments(c)
            if m['m00']>12:
                cx,cy=int(m['m10']/m['m00']),int(m['m01']/m['m00'])
                if 0<=cy<rh and 0<=cx<rw and inside[cy,cx]:valid.append(c)
        if valid:inner=max(valid,key=cv2.contourArea)
    elif hierarchy is not None:
        for j,c in enumerate(contours):
            if j>=color_start:break
            area=abs(cv2.contourArea(c));oa=abs(cv2.contourArea(outer))
            if hierarchy[0,j,3]==index and .04*oa<area<.75*oa and len(c)>=8:
                if len(inner)==0 or area>abs(cv2.contourArea(inner)):inner=c
    scale=np.array([w,h],np.float32);offset=np.array([x1,y1],np.float32)
    to_global=lambda c:((np.asarray(c,np.float32).reshape(-1,2)+offset)/scale).tolist()
    result=geometry(to_global(outer),to_global(inner) if len(inner) else [],'edge_color_contour' if color_candidate else 'edge_contour',
                    dict(candidate_count=len(choices),search_kind=kind,learned_semantics=False))
    return result

def measured_safety_box(mouth_geometry,image,feather_px):
    h,w=image.shape[:2];p=np.asarray(mouth_geometry['outer'],np.float32)
    lo=np.maximum(0,p.min(0)-[feather_px/w,feather_px/h]);hi=np.minimum(1,p.max(0)+[feather_px/w,feather_px/h])
    return [float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])]

def refine_eye_seed(image, seed, floor=None):
    """v0.8.3b: move an eye seed off a thick cartoon eyebrow onto the real eye.

    Real-face detectors often place cartoon eye points on the bold eyebrow.
    Search slightly above and well below the seed for the eye's white sclera
    (low saturation, bright) joined with a dark pupil; eyebrows have no sclera.
    Returns a measured box, or None so the caller keeps the original seed.
    """
    h,w=image.shape[:2];x1,y1,x2,y2=box_pixels(seed,w,h);sw,sh=max(4,x2-x1),max(4,y2-y1)
    sx1,sx2=max(0,x1-round(sw*.35)),min(w,x2+round(sw*.35))
    sy1,sy2=max(0,y1-round(sh*.5)),min(h,y2+round(sh*3.0))
    if floor is not None:sy2=min(sy2,int(floor*h))
    if sx2-sx1<8 or sy2-sy1<8:return None
    roi=image[sy1:sy2,sx1:sx2];hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    sat,val=hsv[:,:,1].astype(np.int16),hsv[:,:,2].astype(np.int16)
    # Absolute white OR clearly less saturated/brighter than this face's skin
    # (shaded or warm-lit cartoon sclera is not pure white).
    ms,mv=float(np.median(sat)),float(np.median(val))
    sclera=(((sat<70)&(val>165))|((sat<ms*.75)&(val>mv*.8))).astype(np.uint8)
    sclera=cv2.morphologyEx(sclera,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    dark=(val<95).astype(np.uint8)
    n,labels,stats,cent=cv2.connectedComponentsWithStats(sclera)
    rh,rw=sclera.shape;best=None;seed_cx=(x1+x2)/2-sx1
    for i in range(1,n):
        x,y,bw,bh,area=stats[i]
        if area<max(12,rw*rh*.004) or bw>rw*.95 or bh>rh*.8:continue
        # The sclera must touch a dark pupil/iris; skin highlights do not.
        ring=cv2.dilate((labels==i).astype(np.uint8),np.ones((7,7),np.uint8))
        if not np.any(ring&dark):continue
        score=area-abs(cent[i][0]-seed_cx)*2
        if best is None or score>best[0]:best=(score,i)
    if best is None:return None
    eye=cv2.dilate((labels==best[1]).astype(np.uint8),np.ones((7,7),np.uint8))
    for k in range(1,n):
        if k!=best[1] and stats[k][4]>=12 and np.any((labels==k)&(cv2.dilate(eye,np.ones((15,15),np.uint8))>0)):eye|=(labels==k).astype(np.uint8)
    n2,l2,st2,_=cv2.connectedComponentsWithStats(dark)
    for j in range(1,n2):
        x,y,bw,bh,area=st2[j]
        # Join the pupil/iris (roundish), never the long eyebrow arc.
        if np.any((l2==j)&(eye>0)) and bw<=3*max(1,bh) and bw<rw*.6:eye|=(l2==j).astype(np.uint8)
    ys,xs=np.nonzero(eye)
    ex1,ex2,ey1,ey2=xs.min(),xs.max()+1,ys.min(),ys.max()+1;padx,pady=(ex2-ex1)*.12,(ey2-ey1)*.25
    gx1,gy1=max(0,sx1+ex1-padx),max(0,sy1+ey1-pady);gx2,gy2=min(w,sx1+ex2+padx),min(h,sy1+ey2+pady)
    return [float(gx1/w),float(gy1/h),float((gx2-gx1)/w),float((gy2-gy1)/h)]

def _box_ellipse(box,count=32):
    x,y,bw,bh=box;t=np.linspace(0,2*np.pi,count,endpoint=False)
    return np.c_[x+bw/2+bw/2*np.cos(t),y+bh/2+bh/2*np.sin(t)].tolist()

def edge_face_anchors(image, seeds, preserved_eyes=None):
    mouth=_edge_contour(image,seeds['mouth_box'])
    eye_contours=[];eye_boxes=[]
    mouth_top=float(np.asarray(mouth['outer'])[:,1].min())
    for seed in seeds['eye_boxes']:
        refined=None if preserved_eyes else refine_eye_seed(image,seed,floor=mouth_top)
        if refined is not None:
            try:eye=_edge_contour(image,refined,'eye')
            except ContourNotReliable:eye=dict(outer=_box_ellipse(refined))
            p=np.asarray(eye['outer']);lo=p.min(0);hi=p.max(0)
            # Keep the edge contour only if it stays on the measured eye.
            if hi[1]<refined[1] or lo[1]>refined[1]+refined[3]:eye=dict(outer=_box_ellipse(refined))
            eye_contours.append(eye['outer']);eye_boxes.append(refined);continue
        try:eye=_edge_contour(image,seed,'eye')
        except ContourNotReliable:
            if preserved_eyes:continue
            raise
        p=np.asarray(eye['outer']);lo=p.min(0);hi=p.max(0)
        eye_contours.append(eye['outer']);eye_boxes.append([float(lo[0]),float(lo[1]),float(hi[0]-lo[0]),float(hi[1]-lo[1])])
    return dict(mouth_geometry=mouth,eye_contours=eye_contours,
                eye_boxes=preserved_eyes if preserved_eyes else eye_boxes,
                method='edge_contours_not_five_point_boxes')

def locate_mouth(image,cfg):
    if cfg.get('mouth_geometry'):
        g=cfg['mouth_geometry'];return geometry(g['outer'],g.get('inner',[]),g.get('method','supplied_contour'))
    points=cfg.get('landmarks',[])
    if len(points)==98:
        try:
            found=from_landmarks(points)
            if cfg.get('mouth_box'):
                x,y,w,h=cfg['mouth_box'];p=np.asarray(found['outer'])
                if np.any(p<[x,y]) or np.any(p>[x+w,y+h]):raise ContourNotReliable('98 點唇線不在確認的嘴區內')
            return found
        except ContourNotReliable:pass
    if not cfg.get('mouth_box'):raise ContourNotReliable('缺少嘴部搜尋範圍；不會猜測成品嘴框')
    return _edge_contour(image,cfg['mouth_box'])

def style_profile(requested, mouth_geometry, real_detail=False, anime_detail=False):
    if requested not in ('auto','real','cartoon'):raise ValueError('style_type 只能為 auto / real / cartoon')
    # Edge fallback suggests stylization; it is deliberately labelled uncertain.
    selected=requested if requested!='auto' else ('cartoon' if mouth_geometry['method'].startswith('edge') else 'real')
    return dict(style_type=selected,selection='manual' if requested!='auto' else 'geometry heuristic; uncertain',
                real_detail=bool(real_detail and selected=='real'),anime_detail=bool(anime_detail and selected=='cartoon'),
                detail_strength=.35,edge_sharpen=selected=='cartoon',contour_weight=1.0,
                blink_policy='unchanged',teeth_policy='preserve style; cannot guarantee unseen teeth')

def prepare_face(face,profile):
    if not profile['edge_sharpen']:return face
    # Edge-aware bounded sharpening, not neural super-resolution. No skin smoothing.
    gray=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,30,90)
    support=cv2.dilate(edges,np.ones((3,3),np.uint8)).astype(np.float32)/255
    base=face.astype(np.float32);residual=np.clip(base-cv2.GaussianBlur(base,(0,0),.7),-8,8)
    return np.rint(np.clip(base+residual*.25*support[:,:,None],0,255)).astype(np.uint8)

def enrich_config(cfg):
    """Compatibility bridge for existing Go projects, until UI wiring is reviewed."""
    from pathlib import Path
    import json
    cfg=dict(cfg)
    if not cfg.get('landmarks') and cfg.get('output'):
        path=Path(cfg['output']).parent/'project.json'
        if path.is_file() and path.stat().st_size<1024*1024:
            settings=json.loads(path.read_text(encoding='utf-8-sig')).get('local_settings') or {}
            cfg['landmarks']=settings.get('landmarks',[])
    return cfg

def dynamic_geometry(generated,face,mouth):
    fx,fy,fw,fh=face;mx,my,mw,mh=mouth
    local=[(mx-fx)/fw,(my-fy)/fh,mw/fw,mh/fh]
    found=_edge_contour(generated,local)
    convert=lambda points:[[fx+x*fw,fy+y*fh] for x,y in points]
    return geometry(convert(found['outer']),convert(found['inner']),'generated_edge_contour')

def union_geometry(source,current):
    # Cover both old and new lip interiors; never leave the original teeth outside
    # a smaller closing-mouth mask. Feather is still separately bounded to 3–5 px.
    result=geometry(current['outer'],current.get('inner',[]),'source_and_current_lip_envelope')
    # A convex hull would bridge the concave upper edge of a smile and repaint
    # skin below the nose. Preserve both polygons; rasterize their exact union.
    result['support_polygons']=[source['outer'],current['outer']]
    return result
