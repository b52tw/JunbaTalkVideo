"""Build the Windows portable Embedded Python + OpenVINO + DirectML submodule.
No real model weights are downloaded or opened. Only a micro graph is executed.
"""
import hashlib,json,shutil,subprocess,sys,urllib.request,zipfile
from pathlib import Path
if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8',errors='replace')
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'.build/portable-runtime/base'
def run(args):
    result=subprocess.run([str(x) for x in args],text=True,encoding='utf-8',errors='replace',stdout=subprocess.PIPE,stderr=subprocess.STDOUT);print(result.stdout,flush=True)
    if result.returncode:raise RuntimeError('Runtime preparation failed: '+str(args[0]))
    return result.stdout
assert sys.platform=='win32','Build portable SDK on Windows only'
py=BASE/'python';py.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(ROOT/'local/python-3.11.9-embed-amd64.zip') as z:z.extractall(py)
(py/'python311._pth').write_text('python311.zip\n.\nLib\\site-packages\nimport site\n')
with zipfile.ZipFile(ROOT/'local/pip-25.0.1-py3-none-any.whl') as z:z.extractall(py/'Lib/site-packages')
python=py/'python.exe'
run([python,'-m','pip','install','--no-compile','--disable-pip-version-check','--only-binary=:all:','--require-hashes','-r',ROOT/'local/requirements.txt'])
folder=ROOT/'.build/dml-wheels';folder.mkdir(exist_ok=True);paths=[]
for row in json.loads((ROOT/'local/directml_wheels.json').read_text()):
    path=folder/row['name']
    if not path.exists():
        with urllib.request.urlopen(row['url'],timeout=180) as src,path.open('wb') as dst:shutil.copyfileobj(src,dst)
    assert path.stat().st_size==row['size']
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    assert h.hexdigest()==row['sha256'];paths.append(path)
run([python,'-m','pip','install','--no-index','--no-deps','--no-compile',*paths]);run([python,'-m','pip','check'])
workers=BASE/'workers';workers.mkdir(exist_ok=True)
for name in ('hardware_check.py','micro_graph.py','inference_backend.py','runtime_support.py'):shutil.copyfile(ROOT/'local'/name,workers/name)
probe=ROOT/'.build/sdk-probe';probe.mkdir(exist_ok=True);cfg=probe/'config.json';cfg.write_text(json.dumps(dict(mode='full',device='CPU',report=str(probe/'report.json'))))
run([python,'-I','-u',workers/'hardware_check.py','--config',cfg]);report=json.loads((probe/'report.json').read_text());assert report['cpu_verified'];assert report['real_weights_loaded'] is False
(BASE/'bundle.json').write_text(json.dumps(dict(id='junba-sdk-v081',python='3.11.9',openvino='2026.4.1',ort_directml='1.24.4',real_models_included=False,platform='windows-x64'),indent=2));(BASE/'directml-installed.txt').write_text('ort-directml-1.24.4-v08')
(ROOT/'dist').mkdir(exist_ok=True);shutil.copyfile(probe/'report.json',ROOT/'dist/Windows_runtime_smoke.json')
print('Portable Embedded Python SDK prepared; no full model inference was performed.')
