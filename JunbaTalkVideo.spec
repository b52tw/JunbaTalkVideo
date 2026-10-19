from pathlib import Path
root=Path(SPECPATH)
a=Analysis([str(root/'launcher.py')],pathex=[str(root)],binaries=[],datas=[(str(root/'dist-native/JunbaTalkVideo.exe'),'native')],hiddenimports=[],hookspath=[],hooksconfig={},runtime_hooks=[],excludes=[],noarchive=False,optimize=1)
pyz=PYZ(a.pure)
exe=EXE(pyz,a.scripts,a.binaries,a.datas,[],name='JunbaTalkVideo_PyInstaller',debug=False,bootloader_ignore_signals=False,strip=False,upx=False,console=False)
