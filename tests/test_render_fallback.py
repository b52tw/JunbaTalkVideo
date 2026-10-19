"""v0.8.3b: unreadable generated mouths must never abort the render."""
import json,shutil,subprocess,sys,tempfile,unittest,wave
from pathlib import Path
import cv2,numpy as np

ROOT=Path(__file__).resolve().parents[1]

class RenderFallback(unittest.TestCase):
    def test_lost_contours_finish_with_warning(self):
        ffmpeg=shutil.which('ffmpeg')
        if not ffmpeg:self.skipTest('FFmpeg required')
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);cache=d/'mouth-cache';cache.mkdir()
            img=np.full((480,360,3),(150,185,235),np.uint8)
            cv2.ellipse(img,(180,300),(60,22),0,0,180,(40,30,120),-1)        # open smile
            cv2.rectangle(img,(130,300),(230,312),(245,245,245),-1)          # teeth
            cv2.imwrite(str(d/'src.png'),img)
            face=[.2,.3,.6,.5];x1,y1,x2,y2=72,144,288,384
            crop=cv2.resize(img[y1:y2,x1:x2],(256,256))
            for i in range(16):
                g=crop.copy()
                if i>=2:g[100:240,30:226]=cv2.GaussianBlur(g[100:240,30:226],(0,0),14)
                cv2.imwrite(str(cache/f'{i:05d}.png'),g)
            (d/'mouth-cache.json').write_text(json.dumps({'frames':16,'fps':8,'width':360,'height':480,'actual':{}}),encoding='utf-8')
            s=(np.sin(np.arange(32000)*2*np.pi*220/16000)*6000).astype('<i2')
            with wave.open(str(d/'a.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(s.tobytes())
            cfg=dict(source=str(d/'src.png'),audio=str(d/'a.wav'),output=str(d/'out.mp4'),ffmpeg=ffmpeg,kind='photo',device='CPU',
                     bbox=face,mouth_box=[.3,.58,.4,.12],eye_boxes=[],contour_mode=True,anime_detail=False,real_detail=False,lead=.2,tail=.2,run_id='t')
            (d/'cfg.json').write_text(json.dumps(cfg),encoding='utf-8')
            p=subprocess.run([sys.executable,'-I',str(ROOT/'local/renderer.py'),'--config',str(d/'cfg.json')],
                             capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
            self.assertEqual(p.returncode,0,p.stdout[-2000:]+p.stderr[-2000:])
            done=[json.loads(l) for l in p.stdout.splitlines() if l.startswith('{') and '"done"' in l]
            self.assertTrue(done and '警告' in done[-1]['message'],p.stdout[-1500:])
            self.assertTrue((d/'out.mp4').stat().st_size>1000)

if __name__=='__main__':unittest.main()
