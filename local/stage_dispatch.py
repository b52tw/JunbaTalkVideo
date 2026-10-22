"""Stdlib supervisor: reap failed native workers BEFORE starting CPU fallback."""
import argparse, json, subprocess, sys, threading, time
from pathlib import Path

ALLOWED = {'preflight.py', 'stage_worker.py', 'sad_worker.py', 'sad_export.py', 'renderer.py', 'detail_cache.py'}
MEMORY_ERRORS = ('bad_alloc', 'out of memory', 'not enough memory', 'failed to allocate',
                 'cannot allocate', 'memoryerror', '0x8007000e', '記憶體不足')

def emit(**event):
    print(json.dumps(event, ensure_ascii=False), flush=True)

def save_override(folder, cfg, **values):
    override = dict(run_id=cfg.get('run_id', 'direct'), **values)
    pending = folder / 'backend-fallback.pending.json'
    pending.write_text(json.dumps(override), encoding='utf-8')
    pending.replace(folder / 'backend-fallback.json')
    return override

def run_stage(script, config, extra):
    cfg = json.loads(config.read_text(encoding='utf-8-sig'))
    folder = Path(cfg['output']).parent
    folder.mkdir(parents=True, exist_ok=True)
    phase = extra[extra.index('--stage') + 1] if '--stage' in extra else script.removesuffix('.py')
    fallback = folder / 'backend-fallback.json'
    if phase != 'cache' and fallback.is_file():
        override = json.loads(fallback.read_text(encoding='utf-8'))
        if override.get('run_id') == cfg.get('run_id', 'direct'):
            cfg.update(override)
    max_attempts = 1 if script == 'preflight.py' else 4
    for attempt in range(max_attempts):
        attempt_config = folder / ('attempt-' + phase + '-' + str(attempt + 1) + '.json')
        attempt_config.write_text(json.dumps(cfg), encoding='utf-8')
        event = dict(time=time.time(), run_id=cfg.get('run_id', 'direct'), script=script,
                     phase=phase, attempt=attempt + 1, runtime=cfg.get('runtime', 'openvino'),
                     device=cfg.get('device', 'AUTO'), peak_budget_mb=cfg.get('peak_budget_mb', 4096))
        emit(stage='worker_start', stage_name=phase,
             message=f'{phase}：第 {attempt + 1} 次，{event["runtime"]} / {event["device"]}；私有提交預算 {event["peak_budget_mb"]} MiB')
        tail = bytearray()
        def stderr_copy(pipe):
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                tail.extend(chunk)
                del tail[:-4096]
                sys.stderr.buffer.write(chunk)
                sys.stderr.buffer.flush()
        child = subprocess.Popen([sys.executable, '-I', '-X', 'faulthandler', '-u',
                                  str(Path(__file__).with_name(script)), '--config', str(attempt_config), *extra],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        thread = threading.Thread(target=stderr_copy, args=(child.stderr,), daemon=True)
        thread.start()
        last_error = ''
        for line in child.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
            try:
                item = json.loads(line)
                if item.get('stage') == 'error':
                    last_error = str(item.get('message', ''))
            except (ValueError, AttributeError):
                pass
        rc = child.wait()  # Native allocator is gone before a new child exists.
        thread.join()
        child.stdout.close()
        child.stderr.close()
        event.update(pid=child.pid, finished=time.time(), exit_code=rc,
                     exit_hex=f'0x{rc & 0xffffffff:08x}', error=last_error,
                     stderr_tail=tail.decode('utf-8', errors='replace'))
        with (folder / 'process-exits.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(event, ensure_ascii=False) + '\n')
        if rc == 0:
            return 0
        if rc == 75 and fallback.is_file():
            override = json.loads(fallback.read_text(encoding='utf-8'))
            if override.get('run_id') != cfg.get('run_id', 'direct'):
                break
        else:
            memory_error = any(x in (last_error + event['stderr_tail']).lower() for x in MEMORY_ERRORS)
            native_failure = rc < 0 or rc > 0x7fffffff or (rc != 75 and not last_error)
            accelerated = cfg.get('device', 'AUTO') != 'CPU' or cfg.get('npu_assist', False)
            if (native_failure or memory_error) and accelerated and script != 'sad_export.py':
                override = save_override(folder, cfg, device='CPU', npu_assist=False,
                                         runtime=cfg.get('runtime', 'openvino'),
                                         fallback_attempt=int(cfg.get('fallback_attempt', 0)) + 1)
                if phase == 'cache':
                    save_override(folder, cfg, runtime='openvino_onnx_cpu', device='CPU', npu_assist=False)
                    emit(stage='fallback', message='轉換程序已完全退出；後續用新 CPU 程序直接讀 ONNX，保留既有快取。')
                    time.sleep(1.0)
                    return 0
            else:
                emit(stage='error', stage_name=phase,
                     message=f'{phase} 子程序結束（{event["exit_hex"]}）：' +
                     (last_error or '未回報 Python 例外；可能是原生 SDK／驅動或記憶體配置失敗。') +
                     f' 私有提交預算 {event["peak_budget_mb"]} MiB；請下載錯誤報告。')
                return 1
        cfg.update(override)
        emit(stage='fallback', stage_name=phase,
             message=f'{phase} 前一程序已退出（{event["exit_hex"]}）；另開 CPU 程序重試。')
        # Popen.wait uses WaitForSingleObject on Windows. Add a handoff grace period.
        time.sleep(1.0)
    emit(stage='error', stage_name=phase, message='CPU／替代後端重試已達上限；素材與日誌已保留，請下載錯誤報告。')
    return 1

def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    parser = argparse.ArgumentParser()
    parser.add_argument('--script', required=True, choices=sorted(ALLOWED))
    parser.add_argument('--config', required=True)
    parser.add_argument('--start-gate')
    args, extra = parser.parse_known_args()
    # Do not spawn SDK children before Go has attached the Windows Job limit.
    if args.start_gate:
        deadline = time.monotonic() + 30
        while not Path(args.start_gate).is_file():
            if time.monotonic() >= deadline:
                emit(stage='error', message='等待記憶體預算啟用逾時；未啟動模型。')
                return 1
            time.sleep(.02)
    return run_stage(args.script, Path(args.config).resolve(), extra)

if __name__ == '__main__':
    raise SystemExit(main())
