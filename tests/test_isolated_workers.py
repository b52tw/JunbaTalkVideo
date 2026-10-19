"""Regression for Windows embedded Python excluding the sibling worker folder.
Run with the app's dependencies: python tests/test_isolated_workers.py
"""
import json,shutil,subprocess,sys,tempfile,unittest,wave
from pathlib import Path
import cv2
import numpy as np

BUNDLE=Path(__file__).resolve().parents[1]/'local'
class IsolatedWorkers(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='junba-isolation-')
        self.root=Path(self.tmp.name);self.scripts=self.root/'可攜 版本'/'工作模組';self.scripts.mkdir(parents=True)
        for name in ('visual_polish.py','body_motion.py','runner.py','renderer.py','quality.py','runtime_support.py','stage_worker.py','face_worker.py','face_geometry.py','yunet.onnx'):
            shutil.copy2(BUNDLE/name,self.scripts/name)
        self.cwd=self.root/'另一個工作目錄';self.cwd.mkdir()
    def tearDown(self):self.tmp.cleanup()
    def test_runner_checks_sibling_import_before_model_loading(self):
        p=subprocess.run([sys.executable,'-I',str(self.scripts/'runner.py'),'--check-imports'],cwd=self.cwd,capture_output=True,text=True,timeout=30)
        self.assertEqual(p.returncode,0,p.stdout+p.stderr)
        result=next(json.loads(line) for line in p.stdout.splitlines() if json.loads(line).get('stage')=='checked')
        self.assertTrue(result['isolated'])
        self.assertEqual(Path(result['path']),self.scripts/'quality.py')
    def test_face_worker_rejects_blank_image_without_guessing(self):
        out=self.root/'五官偵測';out.mkdir()
        ok,b=cv2.imencode('.png',np.full((320,320,3),127,np.uint8));(out/'空白.png').write_bytes(b.tobytes())
        config=out/'設定.json';config.write_text(json.dumps({'source':str(out/'空白.png'),'kind':'photo','models':str(out),'result':str(out/'result.json'),'device':'CPU'}),encoding='utf-8')
        p=subprocess.run([sys.executable,'-I',str(self.scripts/'face_worker.py'),'--config',str(config)],cwd=self.cwd,capture_output=True,text=True,timeout=30)
        self.assertNotEqual(p.returncode,0);self.assertIn('未找到可靠的真人臉',p.stdout);self.assertFalse((out/'result.json').exists())
    def test_all_body_modes_render_in_isolated_runtime(self):
        ffmpeg=shutil.which('ffmpeg');ffprobe=shutil.which('ffprobe')
        if not ffmpeg or not ffprobe:self.skipTest('FFmpeg tools required')
        output=self.root/'專案';output.mkdir();cache=output/'mouth-cache';cache.mkdir()
        base=np.full((384,256,3),130,np.uint8);base[:,:,1]=180
        ok,b=cv2.imencode('.png',base);self.assertTrue(ok);(output/'原始素材.png').write_bytes(b.tobytes())
        patch=np.full((256,256,3),110,np.uint8)
        for i in range(8):
            ok,b=cv2.imencode('.png',patch);(cache/f'{i:05d}.png').write_bytes(b.tobytes())
        (output/'mouth-cache.json').write_text(json.dumps({'frames':8,'fps':6,'width':128,'height':192,'actual':{}}),encoding='utf-8')
        samples=(np.sin(np.arange(21333)*2*np.pi*440/16000)*8000).astype('<i2')
        with wave.open(str(output/'語音.wav'),'wb') as f:f.setnchannels(1);f.setsampwidth(2);f.setframerate(16000);f.writeframes(samples.tobytes())
        for mode in ('','breathe','sway','nod','hand'):
            with self.subTest(mode=mode or 'none'):
                target=output/((mode or 'none')+'.mp4')
                cfg={'source':str(output/'原始素材.png'),'audio':str(output/'語音.wav'),'output':str(target),'ffmpeg':ffmpeg,'kind':'photo','device':'CPU','bbox':[.2,.15,.6,.5],'mouth_box':[.35,.43,.3,.1],'eye_boxes':[[.3,.28,.10,.04],[.57,.28,.10,.04]],'blink':True,'gaze':True,'body_motion':mode,'hand_box':[.25,.6,.23,.1],'lead':.5,'tail':.5}
                config=output/'設定.json';config.write_text(json.dumps(cfg,ensure_ascii=False),encoding='utf-8')
                p=subprocess.run([sys.executable,'-I',str(self.scripts/'renderer.py'),'--config',str(config)],cwd=self.cwd,capture_output=True,text=True,timeout=30)
                self.assertEqual(p.returncode,0,p.stdout+p.stderr)
                self.assertTrue(any(json.loads(line).get('stage')=='done' for line in p.stdout.splitlines()))
                self.assertTrue(any(json.loads(line).get('stage')=='memory' for line in p.stdout.splitlines()))
                data=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-of','json',str(target)]))
                video=next(s for s in data['streams'] if s['codec_type']=='video')
                self.assertEqual((video['width'],video['height'],int(video['nb_frames'])),(128,192,14))
                self.assertTrue(any(s['codec_type']=='audio' for s in data['streams']))
if __name__=='__main__':unittest.main()
