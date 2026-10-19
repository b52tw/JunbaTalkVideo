"""Spatial face landmarks only: no identity matching or biometrics database."""
import cv2
import numpy as np

def normalized_box(points,w,h,padx=.15,pady=.5,min_height=0):
    p=np.asarray(points,dtype=np.float32);lo=p.min(axis=0);hi=p.max(axis=0);bw=max(hi[0]-lo[0],4);bh=max(hi[1]-lo[1],min_height,4)
    center=(hi+lo)/2;lo=center-[bw*(.5+padx),bh*(.5+pady)];hi=center+[bw*(.5+padx),bh*(.5+pady)]
    lo=np.clip(lo,[0,0],[w-2,h-2]);hi=np.clip(hi,lo+1,[w,h])
    return [round(float(lo[0]/w),6),round(float(lo[1]/h),6),round(float((hi[0]-lo[0])/w),6),round(float((hi[1]-lo[1])/h),6)]

def decode_heatmaps(heatmaps,roi):
    maps=np.asarray(heatmaps)[0];h,w=maps.shape[-2:];points=[]
    x1,y1,x2,y2=roi
    for hm in maps:
        y,x=np.unravel_index(np.argmax(hm),hm.shape);px=float(x);py=float(y)
        if 0<x<w-1:px+=.25*np.sign(hm[y,x+1]-hm[y,x-1])
        if 0<y<h-1:py+=.25*np.sign(hm[y+1,x]-hm[y-1,x])
        points.append([x1+px/w*(x2-x1),y1+py/h*(y2-y1)])
    return np.asarray(points,dtype=np.float32)

def regions_from_landmarks(points,face,w,h):
    p=np.asarray(points,dtype=np.float32)
    if p.shape!=(98,2) or not np.isfinite(p).all():raise ValueError('臉部關鍵點資料無效')
    # WFLW-98 topology: eye contours 60:68/68:76; outer+inner lips 76:96.
    mouth=normalized_box(p[76:96],w,h,.14,.55,min_height=max(6,(face[3]-face[1])*.055))
    eyes=[normalized_box(p[s:s+8],w,h,.12,.5) for s in (60,68)]
    eyes.sort(key=lambda b:b[0])
    face_box=normalized_box(np.array([[face[0],face[1]],[face[2],face[3]]]),w,h,.07,.08)
    return dict(bbox=face_box,mouth_box=mouth,eye_boxes=eyes,landmarks=[[round(float(x/w),6),round(float(y/h),6)] for x,y in p])


def reliable_regions(regions,face):
    fw=face[2]-face[0];eyes=regions['eye_boxes'];centers=[b[0]+b[2]/2 for b in eyes]
    return all(b[2]<fw*.48 for b in eyes) and centers[1]-centers[0]>fw*.15

def regions_from_five(points,face,w,h):
    p=np.asarray(points,dtype=np.float32);eyes=sorted(p[:2].tolist(),key=lambda q:q[0]);dist=np.linalg.norm(np.array(eyes[1])-eyes[0])
    eye_boxes=[]
    for x,y in eyes:
        eye_boxes.append(normalized_box([[x-dist*.22,y-dist*.08],[x+dist*.22,y+dist*.08]],w,h,.08,.6))
    mouth=normalized_box(p[3:5],w,h,.22,.8,min_height=dist*.22)
    face_box=normalized_box([[face[0],face[1]],[face[2],face[3]]],w,h,.07,.08)
    return dict(bbox=face_box,mouth_box=mouth,eye_boxes=eye_boxes,landmarks=[[round(float(x/w),6),round(float(y/h),6)] for x,y in p])
