import argparse,gc,json,os,sys,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));sys.modules['openvino_telemetry']=None
from runtime_support import emit,report_peak,process_policy
VERSION='sad512-grid3d-v08-2'
IR_VERSION='sad512-ir-fp16-v081'
def paths_for(root):
    folder=Path(root)/'accelerated-v08';return folder,folder/'generator512.onnx',folder/'generator512.xml',folder/'ready.json'
def valid(root,ir=False):
    _,onnx,xml,marker=paths_for(root)
    try:
        m=json.loads(marker.read_text(encoding='utf-8'))
        if m['version']!=VERSION or m['weight_size']!=(Path(root)/'checkpoints/SadTalker_V0.0.2_512.safetensors').stat().st_size:return False
        return m.get('ir_version')==IR_VERSION and xml.is_file() and xml.with_suffix('.bin').is_file() if ir else onnx.is_file()
    except (OSError,ValueError,KeyError):return False
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--stage',choices=['export','cache'],required=True);a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8-sig'));root=Path(cfg['root']);folder,onnx,xml,marker=paths_for(root);folder.mkdir(exist_ok=True);process_policy({'low_impact':True})
    if valid(root,a.stage=='cache'):emit(stage='completed',message='SadTalker 加速快取已就緒');return
    if a.stage=='export':
        import torch,yaml,safetensors.torch
        sys.path.insert(0,str(root));torch.set_num_threads(min(8,int(cfg.get('threads',2))))
        from src.facerender.modules.generator import OcclusionAwareSPADEGenerator
        from torch_grid3d import install_export_sampler
        config=yaml.safe_load((root/'src/config/facerender_still.yaml').read_text(encoding='utf-8'))
        model=OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],**config['model_params']['common_params']).eval().requires_grad_(False)
        weight=root/'checkpoints/SadTalker_V0.0.2_512.safetensors'
        with safetensors.safe_open(str(weight),framework='pt',device='cpu') as f:state={k[len('generator.'):]:f.get_tensor(k) for k in f.keys() if k.startswith('generator.')}
        model.load_state_dict(state);del state;gc.collect()
        class Export(torch.nn.Module):
            def __init__(self):super().__init__();self.generator=model
            def forward(self,image,source,driving):return self.generator(image,kp_source={'value':source},kp_driving={'value':driving})['prediction']
        install_export_sampler();image=torch.full((1,3,512,512),.5);points=torch.linspace(-.5,.5,45).reshape(1,15,3);temp=folder/'generator512.pending.onnx'
        emit(stage='compile',message='首次轉換 SadTalker 3D 變形；完成後重用，請稍候')
        torch.onnx.export(Export(),(image,points,points.clone()),str(temp),opset_version=17,input_names=['image','source','driving'],output_names=['prediction'],do_constant_folding=True)
        os.replace(temp,onnx);marker.write_text(json.dumps({'version':VERSION,'weight_size':weight.stat().st_size}),encoding='utf-8')
    else:
        import openvino as ov
        core=ov.Core();core.set_property({'ENABLE_MMAP':True});model=core.read_model(str(onnx));temp=folder/'generator512.pending.xml';ov.save_model(model,str(temp),compress_to_fp16=True);del model,core;gc.collect();os.replace(temp.with_suffix('.bin'),xml.with_suffix('.bin'));os.replace(temp,xml)
        m=json.loads(marker.read_text(encoding='utf-8'));m['ir_version']=IR_VERSION;marker.write_text(json.dumps(m),encoding='utf-8')
    emit(stage='completed',message='SadTalker '+a.stage+' 完成')
if __name__=='__main__':
    try:main()
    except Exception as e:emit(stage='error',message=str(e));traceback.print_exc();sys.exit(1)
    finally:report_peak('sad_export')
