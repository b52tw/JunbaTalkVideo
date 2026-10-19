"""Small 2-D torso animation with an audio-driven damped spring and locked background.
Not a 3-D body/gesture generator. All displacements are bounded to five pixels.
"""
import cv2
import numpy as np

def audio_drive(audio_path):
    from scipy.io import wavfile
    from scipy.signal import resample_poly
    from math import gcd
    sr,a=wavfile.read(audio_path)
    if np.issubdtype(a.dtype,np.integer):a=a.astype(np.float32)/max(abs(np.iinfo(a.dtype).min),np.iinfo(a.dtype).max)
    else:a=a.astype(np.float32)
    if a.ndim>1:a=a.mean(1)
    if sr!=16000:g=gcd(sr,16000);a=resample_poly(a,16000//g,sr//g)
    sr=16000;hop=320;size=640;n=max(1,int(np.ceil(len(a)/hop)));rms=[];pitch=[];window=np.hanning(size)
    for i in range(n):
        segment=a[i*hop:i*hop+size];segment=np.pad(segment,(0,max(0,size-len(segment))));energy=float(np.sqrt(np.mean(segment**2)));rms.append(energy)
        if energy<.002:pitch.append(0.);continue
        f=np.fft.rfft((segment-segment.mean())*window,2048);corr=np.fft.irfft(f*f.conj(),2048).real[:size]
        lo,hi=40,201;lag=lo+int(np.argmax(corr[lo:hi]));pitch.append(sr/lag if corr[lag]/max(corr[0],1e-9)>.35 else 0.)
    rms=np.asarray(rms);pitch=np.asarray(pitch);log=np.log(np.maximum(rms,1e-6));median=np.median(log);spread=max(.3,np.percentile(log,90)-median)
    e=np.clip((log-median)/spread,0,1);voiced=pitch[pitch>0];p=np.zeros_like(e)
    if len(voiced):p[pitch>0]=np.clip(np.abs(np.log2(pitch[pitch>0]/np.median(voiced)))/.5,0,1)
    target=4.*np.clip(.75*e+.25*p,0,1)*(rms>.002);position=0.;velocity=0.;result=[]
    for value in target:
        velocity+=(49*(value-position)-2*.85*7*velocity)*.02;position+=velocity*.02;result.append(np.clip(position,0,5))
    return np.asarray(result,np.float32)

class TorsoMotion:
    def __init__(self,source,face,drive=None,alpha=None,scale=1.0):
        from scipy.ndimage import distance_transform_edt
        self.scale=float(scale);self.base=source;self.face=face;self.drive=drive if drive is not None else np.zeros(1);self.ready=False;h,w=source.shape[:2]
        if alpha is None:
            scale=min(1.,512/max(h,w));small=cv2.resize(source,(max(2,round(w*scale)),max(2,round(h*scale))))
            sh,sw=small.shape[:2];x,y,bw,bh=face;x1=max(1,int((x-.8*bw)*sw));y1=max(1,int((y-.2*bh)*sh));x2=min(sw-1,int((x+1.8*bw)*sw));y2=sh-1
            if x2-x1<8 or y2-y1<8:return
            mask=np.zeros((sh,sw),np.uint8)
            try:cv2.grabCut(small,mask,(x1,y1,x2-x1,y2-y1),np.zeros((1,65)),np.zeros((1,65)),2,cv2.GC_INIT_WITH_RECT)
            except cv2.error:return
            hard=((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD)).astype(np.uint8)
            ratio=hard.mean();cx=int((x+bw*.5)*sw);cy=int((y+bh*.5)*sh)
            if not (.01<ratio<.85) or not hard[np.clip(cy,0,sh-1),np.clip(cx,0,sw-1)]:return
            _,labels=cv2.connectedComponents(hard);label=labels[np.clip(cy,0,sh-1),np.clip(cx,0,sw-1)];hard=(labels==label).astype(np.uint8)
            alpha=cv2.resize(hard.astype(np.float32),(w,h),interpolation=cv2.INTER_LINEAR)
        self.alpha=cv2.GaussianBlur(np.clip(alpha.astype(np.float32),0,1),(0,0),1.4)
        small_scale=min(1.,512/max(h,w));size=(max(2,round(w*small_scale)),max(2,round(h*small_scale)));small=cv2.resize(source,size);m=cv2.resize((self.alpha>.05).astype(np.uint8),size,interpolation=cv2.INTER_NEAREST)
        if not np.any(m==0):return
        indices=distance_transform_edt(m,return_distances=False,return_indices=True);fill=small[indices[0],indices[1]];fill=cv2.resize(fill,(w,h),interpolation=cv2.INTER_LINEAR)
        self.background=np.where((self.alpha>.001)[:,:,None],fill,source).astype(np.float32)
        self.yy,self.xx=np.mgrid[:h,:w].astype(np.float32);neck=(face[1]+face[3]*.92)*h
        self.neck=np.clip((self.yy-neck)/max(8,face[3]*h*.28),0,1)
        self.safe=np.clip(np.minimum.reduce([self.xx,w-1-self.xx,self.yy,h-1-self.yy])/8.,0,1);self.ready=True
    def apply(self,t):
        if not self.ready:return self.base
        h,w=self.base.shape[:2];accent=float(self.drive[min(len(self.drive)-1,max(0,int(t*50)))])
        breath=4.*np.sin(2*np.pi*t/3.6);field=self.neck*self.safe
        k=self.scale;dy=np.clip((breath+.65*accent)*k,-5*k,5*k)*field;dx=np.clip(((self.xx-w*.5)*.0007*np.sin(2*np.pi*t/3.6)+.30*accent)*k,-3*k,3*k)*field
        if np.max(np.abs(dy)+np.abs(dx))<1e-5:return self.base
        mx=(self.xx-dx).astype(np.float32);my=(self.yy-dy).astype(np.float32);a=cv2.remap(self.alpha,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        prem=cv2.remap(self.base.astype(np.float32)*self.alpha[:,:,None],mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        composed=prem+self.background*(1-a[:,:,None]);effect=np.clip(np.abs(dy)+np.abs(dx),0,1)
        union=(self.alpha>.001)|(a>.001);effect*=union
        return np.rint(composed*effect[:,:,None]+self.base*(1-effect[:,:,None])).clip(0,255).astype(np.uint8)
