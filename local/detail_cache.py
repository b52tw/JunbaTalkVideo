"""Short-lived FP16 IR conversion, exited before enhancement compilation."""
import argparse,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.modules['openvino_telemetry']=None
from runtime_support import emit,report_peak
VERSION='detail-fp16-v081'
def path_for(root,name):return Path(root)/'ir-detail-fp16'/(name+'.xml')
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--name',choices=['gfpgan_1.4','sr_anime'],required=True);a=p.parse_args();root=Path(a.root);source=root/(a.name+'.onnx');target=path_for(root,a.name);target.parent.mkdir(exist_ok=True);marker=target.with_suffix('.json');stat=source.stat();key={'version':VERSION,'size':stat.st_size,'mtime':stat.st_mtime_ns}
    try:
        if json.loads(marker.read_text())==key and target.is_file() and target.with_suffix('.bin').is_file():return
    except (OSError,ValueError):pass
    import openvino as ov
    core=ov.Core();core.set_property({'ENABLE_MMAP':True});model=core.read_model(str(source));pending=target.with_name(a.name+'.pending.xml');ov.save_model(model,str(pending),compress_to_fp16=True);os.replace(pending.with_suffix('.bin'),target.with_suffix('.bin'));os.replace(pending,target);marker.write_text(json.dumps(key));emit(stage='completed',message='局部修復 FP16 權重快取完成；轉換程序退出後才載入')
if __name__=='__main__':
    try:main()
    finally:report_peak('detail_cache')
