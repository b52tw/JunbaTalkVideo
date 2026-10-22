import hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];DIST=ROOT/'dist';DIST.mkdir(exist_ok=True);VERSION=(ROOT/'VERSION').read_text(encoding='utf-8').strip()
exe=DIST/f'JunbaTalkVideo_v{VERSION}.exe';shutil.copyfile(ROOT/'JunbaTalkVideo.exe',exe)
def docs(folder):
    folder.mkdir(exist_ok=True,parents=True);shutil.copyfile(exe,folder/'JunbaTalkVideo.exe')
    for name in ('README.md','GITHUB_BUILD.md','ENGINE_GUIDE.md','DEVELOPMENT_RULES.md','MOTION_ROADMAP.md','SOURCES.md','VALIDATION_v0.8.json','VALIDATION_v0.9.0.json','LICENSE-MIT.txt'):
        if (ROOT/name).exists():shutil.copyfile(ROOT/name,folder/name)
    shutil.copytree(ROOT/'third_party',folder/'third_party',dirs_exist_ok=True)
def pack(folder,name):
    target=DIST/name
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=False) as z:
        for p in folder.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(folder).as_posix())
    return target
small=ROOT/'.build/portable-small';docs(small);outputs=[exe,pack(small,f'JunbaTalkVideo_v{VERSION}_Windows64.zip')]
sdk=ROOT/'.build/portable-runtime'
if (sdk/'base/bundle.json').exists():
    full=ROOT/'.build/portable-full';docs(full);shutil.copytree(sdk,full/'runtime',dirs_exist_ok=True)
    outputs.append(pack(full,f'JunbaTalkVideo_v{VERSION}_EmbeddedPortable.zip'))
rows=[]
for p in outputs:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    rows.append(dict(name=p.name,size=p.stat().st_size,sha256=h.hexdigest()))
(DIST/'SHA256.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows,indent=2))
