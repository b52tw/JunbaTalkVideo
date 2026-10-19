"""Original-preserving composition and explicitly optional 2-D photo motion.
No learned eye/pose generation: eye boxes must be selected by the user.
"""
import math,random
import cv2
import numpy as np
from visual_polish import lab_transfer,lid_shadow


def rect(box, w, h):
    x,y,bw,bh=box
    return max(0,round(x*w)),max(0,round(y*h)),min(w,round((x+bw)*w)),min(h,round((y+bh)*h))


def mouth_mask(shape, face, mouth, scope='lips'):
    h,w=shape[:2]
    fx,fy,fw,fh=face
    if not mouth:
        mouth=[fx+fw*.23,fy+fh*.54,fw*.54,fh*.25]
    x,y,bw,bh=mouth
    if scope=='jaw':y+=bh*.10;bh*=1.75
    # Cover the original lip corners too; keep nose, cheeks and eyes outside.
    yy,xx=np.mgrid[:h,:w].astype(np.float32)
    cx=(x+bw/2-fx)/fw*w; cy=(y+bh/2-fy)/fh*h
    # Rounded rectangle includes both lip corners but stops below the nose.
    # An ellipse tall enough for wide smiles would repaint the nose tip.
    hw=max(2,bw/fw*w*.57); hh=max(2,bh/fh*h*.56)
    corner=max(1,min(hw,hh)*.38)
    qx=np.abs(xx-cx)-(hw-corner); qy=np.abs(yy-cy)-(hh-corner)
    distance=np.sqrt(np.maximum(qx,0)**2+np.maximum(qy,0)**2)+np.minimum(np.maximum(qx,qy),0)-corner
    feather=max(1,bh/fh*h*.16)
    t=np.clip(-distance/feather,0,1)
    return (t*t*(3-2*t))[:,:,None]


def blend_mouth(source, generated, face, mouth, detail=None, blend='feather', scope='lips', full=False, color_transfer=True, mouth_geometry=None, feather_px=4):
    h,w=source.shape[:2]
    x1,y1,x2,y2=rect(face,w,h)
    original=source[y1:y2,x1:x2]
    patch=cv2.resize(generated,(x2-x1,y2-y1),interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    contour_layers=None
    if mouth_geometry is not None:
        if full:raise ValueError('唇線模式不允許整臉重繪')
        from contour_mask import masks
        contour_layers=masks(mouth_geometry,face,w,h,feather_px,mouth);mask=contour_layers['alpha']
    elif full:
        ph,pw=original.shape[:2];ey,ex=np.mgrid[:ph,:pw].astype(np.float32);edge=np.minimum.reduce([ex,pw-1-ex,ey,ph-1-ey]);edge=np.clip(edge/max(3,min(ph,pw)*.035),0,1);mask=(edge*edge*(3-2*edge))[:,:,None]
    else:mask=mouth_mask(original.shape,face,mouth,scope)
    if color_transfer and not full:
        corrected=lab_transfer(original,patch,mask)
        # Lip/inner-mouth pixels keep their generated colors. Only the narrow
        # surrounding skin ring receives color transfer; teeth are not recolored.
        patch=np.where(contour_layers['hard'][:,:,None]>0,patch,corrected) if contour_layers else corrected
    ring=(mask[:,:,0]>.03)&(mask[:,:,0]<.35)
    if ring.any() and not color_transfer:
        bias=np.median(original.astype(np.float32)[ring]-patch[ring],axis=0)
        patch=np.clip(patch+np.clip(bias,-12,12),0,255)
    out=source.copy()
    if blend=='poisson':
        hard=np.rint(mask[:,:,0]*255).astype(np.uint8);x,y,bw,bh=cv2.boundingRect(hard)
        if bw>3 and bh>3:patch=cv2.seamlessClone(patch.astype(np.uint8),original,hard,(x+bw//2,y+bh//2),cv2.NORMAL_CLONE).astype(np.float32)
    out[y1:y2,x1:x2]=np.rint(patch*mask+original*(1-mask)).clip(0,255).astype(np.uint8)
    return out


def cubic_bezier(x,x1,y1,x2,y2):
    lo,hi=0.,1.
    for _ in range(18):
        u=(lo+hi)/2;v=1-u;tx=3*v*v*u*x1+3*v*u*u*x2+u*u*u
        if tx<x:lo=u
        else:hi=u
    u=(lo+hi)/2;v=1-u
    return 3*v*v*u*y1+3*v*u*u*y2+u*u*u

def blink_starts(seed=1247,duration=600):
    rng=random.Random(seed);t=rng.uniform(3.5,6.);out=[]
    while t<duration:out.append(t);t+=.35+rng.uniform(3.5,6.)
    return out
_BLINK_STARTS=blink_starts()
def blink_amount(t,profile='bezier'):
    starts=_BLINK_STARTS if profile=='bezier' else (3.8,10.6,18.7,24.9,33.2,40.5,48.8,56.3)
    close_time,open_time=(.10,.25) if profile=='bezier' else (.16,.34)
    for start in starts:
        u=t-start
        if 0<=u<close_time:return .78*cubic_bezier(u/close_time,.35,0.,.65,1.)
        if close_time<=u<close_time+open_time:return .78*(1-cubic_bezier((u-close_time)/open_time,.15,.05,.25,1.))
    return 0.


def animate_eye(frame, box, t, gaze, blink=True, profile='bezier', brow=False, shadow=True):
    h,w=frame.shape[:2]; x1,y1,x2,y2=rect(box,w,h)
    ew,eh=x2-x1,y2-y1
    if ew<6 or eh<3:
        return frame
    # Stretch only nearby lid skin; compress the selected eye to a closed line.
    pad=max(3,round(eh*.85)); top=max(0,y1-pad); bottom=min(h,y2+pad)
    left=max(0,x1-round(ew*.25)); right=min(w,x2+round(ew*.25))
    patch=frame[top:bottom,left:right]; ph,pw=patch.shape[:2]
    yy,xx=np.mgrid[:ph,:pw].astype(np.float32)
    a=y1-top; b=y2-top; mid=(a+b)/2
    close=blink_amount(t,profile) if blink else 0; half=max(1.2,(b-a)/2*(1-.78*close))
    fx=np.clip((xx-(x1-left))/max(1,ew),0,1)
    curve=eh*.10*close*np.cos(2*np.pi*fx)
    lo,hi=mid+curve-half,mid+curve+half
    mapped=np.where(yy<lo, yy*a/np.maximum(lo,.001),np.where(yy>hi,b+(yy-hi)*(ph-1-b)/np.maximum(ph-1-hi,.001), a+(yy-lo)*(b-a)/np.maximum(hi-lo,.001)))
    feather=max(1,ew*.22)
    side=np.clip(np.minimum(xx-(x1-left)+feather,(x2-left)+feather-xx)/feather,0,1)
    side=side*side*(3-2*side)
    vertical=np.exp(-((yy-mid)/max(1,eh*.70))**4)
    mapy=yy+(mapped-yy)*side
    move=(.020*ew*math.sin(t*.8)+.010*ew*math.sin(t*1.7)) if gaze else 0
    mapx=xx-move*side*vertical*(1-close)
    warped=cv2.remap(patch,mapx.astype(np.float32),mapy.astype(np.float32),cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    out=frame.copy(); out[top:bottom,left:right]=warped
    if brow and close:
        y0=max(0,y1-round(eh*2.0));yend=max(y0+1,y1-round(eh*.65));region=out[y0:yend,left:right].copy()
        by,bx=np.mgrid[:region.shape[0],:region.shape[1]].astype(np.float32)
        weight=np.sin(np.pi*(by+.5)/region.shape[0])**2*np.sin(np.pi*(bx+.5)/region.shape[1])**2
        out[y0:yend,left:right]=cv2.remap(region,bx,by-1.5*close*weight,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    if shadow and close:out=lid_shadow(out,box,close/.78)
    return out


def photo_motion(frame, face, t, eyes=(), gaze=False, micro=False, blink=True, motion="", hand=(), blink_profile="bezier", brow=False, shadow=True):
    out=frame
    for box in eyes:
        out=animate_eye(out,box,t,gaze,blink,blink_profile,brow,shadow)
    if not motion and micro:
        motion="nod"
    if motion:
        h,w=out.shape[:2]; x,y,bw,bh=face
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        cx=(x+bw/2)*w; cy=(y+bh*.75)*h
        if motion=="breathe":
            cy=min(.70*h,(y+bh*1.40)*h)
            field=np.exp(-(((xx-cx)/(w*.25))**2+((yy-cy)/(h*.19))**2)*2)
            phase=math.sin(t*2*math.pi/3.6)
            dx=(xx-cx)*.006*phase*field
            dy=-.0018*h*phase*field
        elif motion=="sway":
            field=np.exp(-(((xx-cx)/(w*.38))**2+((yy-h*.50)/(h*.36))**2)*2)
            dx=.004*w*math.sin(t*1.7)*field
            dy=.0008*h*math.sin(t*3.4)*field
        elif motion=="hand" and hand:
            hx,hy,hw,hh=hand;cx=(hx+hw/2)*w;cy=(hy+hh/2)*h
            field=np.exp(-(((xx-cx)/(hw*w*.65))**2+((yy-cy)/(hh*h*.65))**2)*2)
            dx=.035*hw*w*math.sin(t*3.5)*field
            dy=.060*hh*h*math.sin(t*7.0)*field
        else:
            field=np.exp(-(((xx-cx)/(bw*w*.9))**2+((yy-cy)/(bh*h*1.5))**2)*2)
            dx=.002*w*math.sin(t*1.25)*field
            dy=.002*h*(math.sin(t*1.7)+.25*math.sin(t*.61))*field
        out=cv2.remap(out,xx-dx,yy-dy,cv2.INTER_CUBIC,borderMode=cv2.BORDER_REFLECT_101)
    return out
