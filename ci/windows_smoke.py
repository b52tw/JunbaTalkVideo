"""Native Windows DLL and small-model numerical tests; no GPU/NPU claims."""
import hashlib,json,shutil,subprocess,sys,urllib.request,zipfile
from pathlib import Path
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT=Path(__file__).resolve().parents[1];WORK=ROOT/'.build/Windows smoke 峻爸';WORK.mkdir(parents=True,exist_ok=True);report={'full_models_tested':False,'gpu_npu_tested':False,'checks':[]}
def run(args):
 p=subprocess.run([str(a) for a in args],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace');print(p.stdout,flush=True)
 if p.returncode:raise RuntimeError('command failed: '+str(args[0]))
 return p.stdout
def wheels(py,name):
 folder=WORK/'wheels';folder.mkdir(exist_ok=True);paths=[]
 for r in json.loads((ROOT/'local'/name).read_text()):
  p=folder/r['name']
  if not p.exists():
   if r['url'].startswith('bundled:'):shutil.copyfile(ROOT/'local'/r['name'],p)
   else:
    with urllib.request.urlopen(r['url'],timeout=180) as src,p.open('wb') as dst:shutil.copyfileobj(src,dst)
  assert p.stat().st_size==r['size'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'];paths.append(p)
 run([py,'-m','pip','install','--no-index','--no-deps','--no-compile',*paths])
def env(name):
 folder=WORK/name;py=folder/'python';py.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(ROOT/'local/python-3.11.9-embed-amd64.zip') as z:z.extractall(py)
 (py/'python311._pth').write_text('python311.zip\n.\nLib\\site-packages\nimport site\n')
 with zipfile.ZipFile(ROOT/'local/pip-25.0.1-py3-none-any.whl') as z:z.extractall(py/'Lib/site-packages')
 scripts=folder/'workers';scripts.mkdir(exist_ok=True)
 for f in ('backend_probe.py','inference_backend.py','probe_matmul.onnx','runtime_support.py'):shutil.copyfile(ROOT/'local'/f,scripts/f)
 return folder,py/'python.exe',scripts
def probe(py,scripts,folder,runtime):
 cfg=folder/(runtime+'.json');cfg.write_text(json.dumps(dict(runtime=runtime,device='CPU',output=str(folder/'test.mp4'))));out=run([py,'-I','-u',scripts/'backend_probe.py','--config',cfg]);r=next(json.loads(s) for s in out.splitlines() if s.startswith('{') and '"probe_ok": true' in s);report['checks'].append(dict(environment=folder.name,**r))
try:
 assert sys.platform=='win32'
 folder,py,scripts=env('MuseTalk');run([py,'-m','pip','install','--only-binary=:all:','--require-hashes','-r',ROOT/'local/requirements.txt']);probe(py,scripts,folder,'openvino');wheels(py,'directml_wheels.json');probe(py,scripts,folder,'directml');run([py,'-m','pip','check'])
 folder,py,scripts=env('SadTalker');wheels(py,'sadtalker_wheels.json');run([py,'-I','-c','import torch,cv2,librosa,safetensors,onnx,openvino;print(torch.__version__,openvino.__version__);print(torch.mm(torch.ones(2,2),torch.eye(2)))']);probe(py,scripts,folder,'openvino');wheels(py,'directml_wheels.json');probe(py,scripts,folder,'directml');run([py,'-m','pip','check']);report['ok']=True
except Exception as e:report.update(ok=False,error=str(e));raise
finally:
 (ROOT/'dist').mkdir(exist_ok=True);(ROOT/'dist/Windows_runtime_smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
