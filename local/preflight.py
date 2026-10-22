"""Startup preflight without loading neural model weights."""
import argparse
import importlib
import json
import shutil
import sys
import traceback
from pathlib import Path

def emit(**event):
    print(json.dumps(event, ensure_ascii=False), flush=True)

def require_file(path, label):
    p = Path(path) if path else None
    if p is None or not p.is_file() or p.stat().st_size <= 0:
        raise RuntimeError(f"啟動檢查失敗：{label}不存在或是空檔：{path or '(未設定)'}")
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    emit(stage='preflight', stage_name='preflight',
         message='啟動前檢查素材、套件、後端、權重與磁碟；不載入神經模型權重')
    try:
        cfg_path = require_file(args.config, '工作設定檔')
        cfg = json.loads(cfg_path.read_text(encoding='utf-8-sig'))
        require_file(cfg.get('source'), '原始素材')
        require_file(cfg.get('audio'), '語音 WAV')
        require_file(cfg.get('ffmpeg'), 'FFmpeg')
        if sys.version_info[:2] != (3, 11):
            raise RuntimeError(f"Python 版本不符：目前 {sys.version.split()[0]}；本機 SDK 預期 Python 3.11.x")
        for module in ('numpy', 'cv2', 'scipy'):
            try:
                importlib.import_module(module)
            except Exception as exc:
                raise RuntimeError(f"無法載入必要套件 {module}：{type(exc).__name__}: {exc}") from exc
        engine = cfg.get('engine', 'musetalk')
        if engine not in ('puppet', 'musetalk'):
            raise RuntimeError(f'本機預檢不支援此引擎：{engine}')
        devices = []
        if engine == 'musetalk':
            runtime = cfg.get('runtime', 'openvino')
            if runtime == 'directml':
                try:
                    ort = importlib.import_module('onnxruntime')
                    devices = list(ort.get_available_providers())
                    if 'DmlExecutionProvider' not in devices:
                        raise RuntimeError('ONNX Runtime 未列出 DmlExecutionProvider')
                except Exception as exc:
                    raise RuntimeError(f'DirectML 啟動檢查失敗：{type(exc).__name__}: {exc}') from exc
            elif runtime in ('openvino', 'openvino_onnx_cpu'):
                try:
                    ov = importlib.import_module('openvino')
                    devices = list(ov.Core().available_devices)
                    if not devices:
                        raise RuntimeError('OpenVINO 沒有列出可用裝置')
                except Exception as exc:
                    raise RuntimeError(f'OpenVINO 啟動檢查失敗：{type(exc).__name__}: {exc}') from exc
            else:
                raise RuntimeError(f'不支援的 MuseTalk 執行後端：{runtime}')
            models = Path(cfg.get('models', ''))
            if not models.is_dir():
                raise RuntimeError(f'找不到模型資料夾：{models}')
            required = ('whisper_encoder_fp16.onnx', 'vae_encoder_fp16.onnx',
                        'positional_encoding_fp16.onnx', 'unet_fp16.onnx', 'vae_decoder_fp16.onnx')
            missing = [name for name in required
                       if not (models / name).is_file() or (models / name).stat().st_size <= 0]
            if missing:
                raise RuntimeError('模型權重缺漏：' + '、'.join(missing) + '；請按「安裝／修復完整 AI 模型」')
        out = Path(cfg.get('output', ''))
        out.parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(out.parent).free
        if free < 256 * 1024 * 1024:
            raise RuntimeError(f'輸出磁碟可用空間不足 256 MiB（目前 {free // (1024*1024)} MiB）；請先清出空間')
        emit(stage='preflight_ok', stage_name='preflight',
             message='啟動檢查通過；尚未載入模型權重' +
             (('；可用後端：' + '、'.join(map(str, devices))) if devices else '；卡通原圖模式不載入 AI 權重'),
             actual={'startup': devices} if devices else {})
        return 0
    except Exception as exc:
        emit(stage='error', stage_name='preflight', message=str(exc))
        traceback.print_exc(file=sys.stderr)
        return 1

if __name__ == '__main__':
    raise SystemExit(main())
