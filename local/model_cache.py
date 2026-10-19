"""Disk-backed OpenVINO IR weights; never keep ONNX parsing and inference together."""
import gc,json,os
from pathlib import Path

VERSION='ir-v07-fp16'
NAMES=('whisper_encoder','vae_encoder','positional_encoding','unet','vae_decoder')

def cached_path(root,name):
    root=Path(root);source=root/(name+'_fp16.onnx');folder=root/VERSION
    xml=folder/(name+'.xml');meta=folder/(name+'.json');binary=xml.with_suffix('.bin')
    stat=source.stat();key={'version':VERSION,'size':stat.st_size,'mtime':stat.st_mtime_ns}
    try:
        saved=json.loads(meta.read_text(encoding='utf-8'))
        if saved['source']==key and xml.stat().st_size==saved['xml'] and binary.stat().st_size==saved['bin']:return xml
    except (OSError,ValueError,KeyError):pass
    return None

def prepare(root,emit):
    import openvino as ov
    root=Path(root);folder=root/VERSION;folder.mkdir(exist_ok=True)
    for name in NAMES:
        if cached_path(root,name):continue
        source=root/(name+'_fp16.onnx');stat=source.stat()
        emit(stage='compile',message='首次建立磁碟映射模型：'+name+'；完成後釋放，之後直接讀取快取')
        core=ov.Core();core.set_property({'ENABLE_MMAP':True});model=core.read_model(str(source));temp=folder/(name+'.pending.xml')
        ov.save_model(model,str(temp),compress_to_fp16=True)
        del model,core;gc.collect()
        final=folder/(name+'.xml');os.replace(temp.with_suffix('.bin'),final.with_suffix('.bin'))
        # OpenVINO XML does not contain the external BIN filename.
        os.replace(temp,final)
        key={'source':{'version':VERSION,'size':stat.st_size,'mtime':stat.st_mtime_ns},'xml':final.stat().st_size,'bin':final.with_suffix('.bin').stat().st_size}
        meta=folder/(name+'.json');tmp=meta.with_suffix('.tmp');tmp.write_text(json.dumps(key),encoding='utf-8');os.replace(tmp,meta)
    emit(stage='completed',stage_name='cache',message='磁碟映射模型已備妥；釋放轉換程序')
