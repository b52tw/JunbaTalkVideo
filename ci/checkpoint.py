"""Source first: create both archives BEFORE running any validation/build.
No model inference, downloads or test invocation belongs in this script.
"""
import argparse,hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--out',default=str(ROOT/'dist-source'));a=p.parse_args();out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
files=[]
for path in ROOT.rglob('*'):
    if not path.is_file():continue
    rel=path.relative_to(ROOT)
    if any(x in ('.git','.build','dist','dist-native','dist-source','build','__pycache__','node_modules') for x in rel.parts):continue
    if path.suffix in ('.exe','.pyc','.gz') or out in path.parents:continue
    files.append((path,rel.as_posix()))
rows=[]
for name in ('JunbaTalkVideo_v0.8_Source.zip','JunbaTalkVideo_v0.8_GitHub_Builder.zip'):
    archive=out/name
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=False) as z:
        for path,n in files:z.write(path,n)
    rows.append(dict(name=name,size=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(files)))
(out/'Source_SHA256.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8');print(json.dumps(rows,indent=2))
