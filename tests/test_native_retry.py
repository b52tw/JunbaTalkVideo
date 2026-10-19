"""No neural imports: exercise real subprocess crashes and isolated handoff."""
import json, shutil, subprocess, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE = '''import argparse,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--stage');a=p.parse_args()
c=json.loads(Path(a.config).read_text(encoding='utf-8'));folder=Path(c['output']).parent
if c.get('scenario')=='bad_input':
 print(json.dumps(dict(stage='error',message='source missing')),flush=True);sys.exit(1)
if c.get('device')!='CPU':
 if c.get('scenario')=='requested':
  (folder/'backend-fallback.json').write_text(json.dumps(dict(run_id=c['run_id'],device='CPU',runtime='openvino',npu_assist=False)),encoding='utf-8')
  sys.exit(75)
 # Windows ExitProcess(-1) is 0xffffffff; POSIX truncates it to 255.
 sys.stderr.write('simulated native SDK termination\\n');sys.stderr.flush()
 sys.exit(-1)
assert c.get('npu_assist') is False
events=[json.loads(x) for x in (folder/'process-exits.jsonl').read_text(encoding='utf-8').splitlines()]
assert time.time()-events[-1]['finished']>=.95, 'CPU started before handoff grace period'
if c.get('scenario')=='cpu_crash':sys.exit(-1)
(folder/'cpu-success.json').write_text(json.dumps(dict(pid=os.getpid(),budget=c['peak_budget_mb'])),encoding='utf-8')
print(json.dumps(dict(stage='completed',actual=dict(unet=['CPU']))),flush=True)
'''

class RetryTests(unittest.TestCase):
    def exercise(self, scenario):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            shutil.copyfile(ROOT/'local/stage_dispatch.py', folder/'stage_dispatch.py')
            (folder/'stage_worker.py').write_text(FAKE,encoding='utf-8')
            cfg = folder/'input.json'
            cfg.write_text(json.dumps(dict(output=str(folder/'video.mp4'), run_id='test',
                device='GPU.0', runtime='openvino', npu_assist=True, peak_budget_mb=6144, scenario=scenario)),encoding='utf-8')
            result = subprocess.run([sys.executable,'-I',str(folder/'stage_dispatch.py'),
                '--script','stage_worker.py','--config',str(cfg),'--stage','infer'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=10)
            events=[json.loads(x) for x in (folder/'process-exits.jsonl').read_text(encoding='utf-8').splitlines()]
            success=json.loads((folder/'cpu-success.json').read_text(encoding='utf-8')) if (folder/'cpu-success.json').exists() else None
            return result, events, success

    def test_native_crash_reaps_then_cpu(self):
        result, events, success=self.exercise('native')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(len(events),2)
        self.assertNotEqual(events[0]['pid'],events[1]['pid'])
        self.assertEqual(events[1]['device'],'CPU')
        self.assertEqual(success['budget'],6144)
        self.assertIn('simulated native',events[0]['stderr_tail'])

    def test_existing_fallback_75(self):
        result, events, success=self.exercise('requested')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(events[0]['exit_code'],75)
        self.assertIsNotNone(success)

    def test_cpu_crash_is_bounded(self):
        result, events, success=self.exercise('cpu_crash')
        self.assertEqual(result.returncode,1)
        self.assertEqual(len(events),2)
        self.assertIsNone(success)
        self.assertIn('6144 MiB',result.stdout)

    def test_invalid_input_not_retried(self):
        result, events, success=self.exercise('bad_input')
        self.assertEqual(result.returncode,1)
        self.assertEqual(len(events),1)
        self.assertIsNone(success)

if __name__=='__main__':unittest.main()
