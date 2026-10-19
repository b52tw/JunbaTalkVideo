"""Small offline worker utilities: explicit memory measurements and CPU policy."""
import json,os,sys
# -I ignores PYTHONUTF8; configure redirected Windows pipes explicitly.
for stream in (sys.stdout,sys.stderr):
    if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8',errors='replace')

def emit(**obj):print(json.dumps(obj,ensure_ascii=False),flush=True)
def process_policy(cfg):
    if not cfg.get('low_impact',True):return
    try:
        if os.name=='nt':
            import ctypes
            # CPU scheduling only; this does not impose a GPU-memory quota.
            kernel=ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype=ctypes.c_void_p
            kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
            kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)
        else:os.nice(5)
    except (OSError,AttributeError):pass

def peak_mb():
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage')]
        c=Counters();c.cb=ctypes.sizeof(c)
        kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=wintypes.HANDLE
        memory=ctypes.windll.psapi.GetProcessMemoryInfo;memory.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];memory.restype=wintypes.BOOL
        if memory(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):return round(c.PeakWorkingSetSize/(1024*1024),1)
        return 0
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(value/(1024*1024 if sys.platform=='darwin' else 1024),1)

def report_peak(stage_name):
    try:emit(stage='memory',stage_name=stage_name,memory_peak_mb=peak_mb(),worker_pid=os.getpid())
    except Exception:pass
