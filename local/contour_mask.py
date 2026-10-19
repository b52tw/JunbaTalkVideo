"""Lip polygon and bounded 3–5 output-pixel feather, independent of eye animation."""
import cv2
import numpy as np
from semantic_mouth import ContourNotReliable,valid_polygon,box_pixels

def masks(geometry,face,width,height,feather_px=4,safety_box=None):
    if feather_px not in (3,4,5):raise ValueError('輪廓外羽化只允許 3、4、5 px')
    fx,fy,fw,fh=face
    x1,y1,x2,y2=max(0,round(fx*width)),max(0,round(fy*height)),min(width,round((fx+fw)*width)),min(height,round((fy+fh)*height))
    shape=(y2-y1,x2-x1)
    if min(shape)<8:raise ContourNotReliable('臉部範圍太小')
    outer=np.asarray(geometry['outer'],np.float32)
    if not valid_polygon(outer):raise ContourNotReliable('缺少可靠唇線，不使用矩形替代')
    points=np.rint(outer*[width,height]-[x1,y1]).astype(np.int32)
    if np.any(points<0) or np.any(points[:,0]>=shape[1]) or np.any(points[:,1]>=shape[0]):
        raise ContourNotReliable('唇線不在完整臉框內')
    hard=np.zeros(shape,np.uint8)
    for polygon in geometry.get('support_polygons',[geometry['outer']]):
        vertices=np.rint(np.asarray(polygon)*[width,height]-[x1,y1]).astype(np.int32)
        if np.any(vertices<0) or np.any(vertices[:,0]>=shape[1]) or np.any(vertices[:,1]>=shape[0]):
            raise ContourNotReliable('動態唇線超出完整臉框')
        cv2.fillPoly(hard,[vertices],1)
    outside=cv2.distanceTransform(1-hard,cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
    # Hard zero beyond N output pixels: no Gaussian tail into cheeks/chin.
    t=np.clip(1-outside/feather_px,0,1);alpha=t*t*(3-2*t)
    allowed=outside<feather_px
    if safety_box is not None:
        bx1,by1,bx2,by2=box_pixels(safety_box,width,height)
        safe=np.zeros(shape,bool)
        safe[max(0,by1-y1):max(0,min(shape[0],by2-y1)),max(0,bx1-x1):max(0,min(shape[1],bx2-x1))]=True
        if np.any((hard>0)&~safe):raise ContourNotReliable('唇線越過已確認嘴區；不擴張到鼻下或下巴')
        alpha*=safe;allowed&=safe
    inner=np.zeros(shape,np.uint8)
    if geometry.get('inner'):
        ip=np.rint(np.asarray(geometry['inner'])*[width,height]-[x1,y1]).astype(np.int32)
        cv2.fillPoly(inner,[ip],1);inner*=hard
    return dict(alpha=alpha[:,:,None],hard=hard,inner=inner,
                lips=hard-inner,allowed=allowed,roi=(x1,y1,x2,y2))

class ContourTracker:
    """At most one prior geometry; never smooth speech pixels or invent lost points."""
    def __init__(self,initial):self.geometry=initial;self.misses=0
    def update(self,candidate):
        if candidate is None:
            self.misses+=1
            if self.misses>5:raise ContourNotReliable('連續多格無可靠動態唇線，停止輸出以免貼錯')
            return self.geometry,'previous precise contour'
        self.geometry=candidate;self.misses=0
        return candidate,'current contour'

def overlay(image,geometry,face,feather_px=4,safety_box=None):
    result=masks(geometry,face,image.shape[1],image.shape[0],feather_px,safety_box)
    x1,y1,x2,y2=result['roi'];out=image.copy();roi=out[y1:y2,x1:x2]
    alpha=result['alpha']*.42;red=np.zeros_like(roi);red[:,:,2]=255
    out[y1:y2,x1:x2]=np.rint(roi*(1-alpha)+red*alpha).astype(np.uint8)
    p=np.rint(np.asarray(geometry['outer'])*[image.shape[1],image.shape[0]]).astype(np.int32)
    cv2.polylines(out,[p],True,(0,255,255),1,cv2.LINE_AA)
    return out,result
