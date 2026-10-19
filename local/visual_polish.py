"""Conservative appearance-only corrections; no invented teeth/eyelid texture."""
import cv2
import numpy as np

def lab_transfer(original,patch,mask,strength=.65):
    """Align Lab mean/std on a lower-face skin ring, with bounded gain and shift."""
    m=mask[:,:,0];h,w=m.shape
    k=max(3,int(min(h,w)*.07)|1)
    near=cv2.dilate((m>.01).astype(np.uint8),np.ones((k,k),np.uint8))>0
    yy=np.arange(h)[:,None];region=near&(m<=.01)&(yy>=h*.40)
    if np.count_nonzero(region)<24:region=(m>.01)&(m<.3)
    if np.count_nonzero(region)<12:return patch
    src=cv2.cvtColor(np.clip(patch/255.,0,1).astype(np.float32),cv2.COLOR_BGR2LAB)
    dst=cv2.cvtColor(original.astype(np.float32)/255.,cv2.COLOR_BGR2LAB)
    def stats(values):
        lo,hi=np.percentile(values,[3,97],axis=0);v=np.clip(values,lo,hi)
        return v.mean(0),v.std(0)
    sm,ss=stats(src[region]);tm,ts=stats(dst[region])
    gain=np.clip(ts/np.maximum(ss,1.5),.75,1.33)
    # Flat skin must not amplify quantization noise.
    gain=np.where(ss<1.5,1.,gain)
    adjusted=(src-sm)*gain+sm+np.clip(tm-sm,[-10,-7,-7],[10,7,7])
    # Protect very bright teeth and saturated lips from strong global recoloring.
    protect=np.where((src[:,:,0]>85)|(np.linalg.norm(src[:,:,1:],axis=2)>35),.35,1.)[:,:,None]
    out=src+(adjusted-src)*float(np.clip(strength,0,1))*protect
    out[:,:,0]=np.clip(out[:,:,0],0,100);out[:,:,1:]=np.clip(out[:,:,1:],-127,127)
    return np.clip(cv2.cvtColor(out.astype(np.float32),cv2.COLOR_LAB2BGR)*255.,0,255)

def lid_shadow(frame,box,closure):
    c=float(np.clip(closure,0,1))
    if c<=0:return frame
    h,w=frame.shape[:2];x,y,bw,bh=box;x1=max(0,int(x*w));x2=min(w,int((x+bw)*w));y1=max(0,int((y-bh*.25)*h));y2=min(h,int((y+bh)*h))
    if x2-x1<6 or y2-y1<3:return frame
    yy,xx=np.mgrid[y1:y2,x1:x2].astype(np.float32);cx=(x+bw*.5)*w;crease=(y+bh*(.22+.25*c))*h
    gaussian=np.exp(-.5*((xx-cx)/max(1,bw*w*.30))**2-.5*((yy-crease)/max(.8,bh*h*.10))**2)
    alpha=(.05*c+.10*c*c)*gaussian
    out=frame.copy();patch=frame[y1:y2,x1:x2].astype(np.float32)
    out[y1:y2,x1:x2]=np.rint(patch*(1-alpha[:,:,None])).clip(0,255).astype(np.uint8)
    return out
