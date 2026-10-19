"""Micro Conv+ReLU only. Never open an installed neural-model weight."""
import argparse,json,sys,tempfile,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.modules['openvino_telemetry']=None
from runtime_support import emit,report_peak
try:
    import numpy as np
    from inference_backend import NativeModel,OrtModel
    from micro_graph import create
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));actual={}
    with tempfile.TemporaryDirectory(prefix='junba-micro-') as folder:
        path=create(Path(folder)/'micro.onnx');m=(OrtModel if cfg.get('runtime')=='directml' else NativeModel)(path,'micro_conv_relu',cfg,emit,actual)
        x=np.linspace(-1,1,48,dtype=np.float32).reshape(1,3,4,4);out=m([x])[0]
        if not np.allclose(out,np.maximum(x,0),rtol=.001,atol=.005):raise RuntimeError('Micro graph 數值不符')
        emit(probe_ok=True,actual=actual,graph_bytes=path.stat().st_size,message='微型 Conv＋ReLU 推論通過；真實模型未載入，完整模型仍需本機測試')
except Exception as e:emit(stage='error',message=str(e));traceback.print_exc();sys.exit(1)
finally:report_peak('micro_probe')
