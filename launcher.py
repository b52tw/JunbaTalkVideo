import hashlib,os,subprocess,sys,tempfile
from pathlib import Path
source=Path(getattr(sys,'_MEIPASS',Path(__file__).parent))/'native/JunbaTalkVideo.exe';data=source.read_bytes();digest=hashlib.sha256(data).hexdigest();root=Path(os.environ.get('LOCALAPPDATA',tempfile.gettempdir()))/'JunbaTalkVideo/launcher'/digest;root.mkdir(parents=True,exist_ok=True);target=root/'JunbaTalkVideo.exe'
if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
 fd,tmp=tempfile.mkstemp(dir=root,suffix='.tmp')
 try:
  with os.fdopen(fd,'wb') as f:f.write(data)
  os.replace(tmp,target)
 finally:
  if Path(tmp).exists():Path(tmp).unlink()
p=subprocess.Popen([str(target),*sys.argv[1:]],cwd=root)
if '--self-test-report' in sys.argv:raise SystemExit(p.wait())
