"""Generate a 1x1 Conv + ReLU ONNX probe using only Python stdlib (<1 KiB).
No neural model files, ONNX package or external weights are read.
"""
import struct
from pathlib import Path

def _v(n):
    out=bytearray()
    while n>127:out.append((n&127)|128);n>>=7
    out.append(n);return bytes(out)
def _i(field,n):return _v(field<<3)+_v(n)
def _b(field,value):
    if isinstance(value,str):value=value.encode()
    return _v((field<<3)|2)+_v(len(value))+value
def _value(name):
    shape=b''.join(_b(1,_i(1,d)) for d in (1,3,4,4))
    return _b(1,name)+_b(2,_b(1,_i(1,1)+_b(2,shape)))
def create(path):
    weight=b''.join(_i(1,n) for n in (3,3,1,1))+_i(2,1)+_b(8,'w')+_b(9,struct.pack('<9f',1,0,0,0,1,0,0,0,1))
    conv=_b(1,'input')+_b(1,'w')+_b(2,'conv')+_b(4,'Conv')
    relu=_b(1,'conv')+_b(2,'output')+_b(4,'Relu')
    graph=_b(1,conv)+_b(1,relu)+_b(2,'Junba_Micro_Conv_Relu')+_b(5,weight)+_b(11,_value('input'))+_b(12,_value('output'))
    model=_i(1,8)+_b(2,'JunbaTalkVideo')+_b(7,graph)+_b(8,_i(2,17))
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(model)
    if len(model)>=1048576:raise RuntimeError('Micro graph exceeds audit limit')
    return path
