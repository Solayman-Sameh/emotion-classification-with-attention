"""Shared, versioned emotion preprocessing. No learned state outside train."""
import hashlib, json, os, re, unicodedata, pathlib, random, time

LABELS = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
MODELS = ['distilbert', 'lstm', 'gru', 'bilstm_attention']
PREPROCESS_VERSION = 'emotion-nfc-whitespace-v1-word-contraction-punct-v1'
# Preserve contractions, punctuation, redaction markers and each emoji code point.
TOKEN_RE = re.compile(r"\[[A-Za-z_]+\]|\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
def clean(text):
    if not isinstance(text, str): return ''
    return ' '.join(unicodedata.normalize('NFC', text).split())
def tokens(text): return TOKEN_RE.findall(clean(text).lower())
def identity(text): return clean(text).casefold()
def digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(path): return json.loads(pathlib.Path(path).read_text(encoding='utf-8'))
def write(path,obj):
    p=pathlib.Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8'); tmp.replace(p)
def rows(path):
    with open(path,encoding='utf-8') as f: return [json.loads(x) for x in f if x.strip()]
def write_rows(path,records):
    with open(path,'w',encoding='utf-8') as f:
        for r in records: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def seed_all(seed):
    random.seed(seed)
    import numpy as np, torch
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def encode(text,vocab,cap):
    ts=tokens(text)
    return [vocab.get(t,1) for t in ts[:cap]],ts[:cap],len(ts)>cap
def validate_mapping(mapping,names):
    if set(mapping)!=set(LABELS): raise ValueError('Mapping targets changed')
    flattened=[x for v in mapping.values() for x in v]
    if len(flattened)!=len(set(flattened)) or set(flattened)!=(set(names)-{'neutral'}):
        raise ValueError('Unknown or multiply mapped fine-grained labels')
    return {x:k for k,v in mapping.items() for x in v}
def classify(raw_labels,names,mapping,text):
    if not isinstance(raw_labels,list) or any(type(i) is not int or i<0 or i>=len(names) for i in raw_labels):
        raise ValueError('Unexpected label schema/ID')
    source=[names[i] for i in raw_labels]
    if not clean(text) or not tokens(text): return None,'unusable_text',source
    if not source: return None,'empty_labels',source
    if 'neutral' in source: return None,'neutral',source
    mapped={mapping[n] for n in source}
    if len(mapped)!=1: return None,'cross_emotion',source
    return LABELS.index(next(iter(mapped))),'candidate',source
def dependencies():
    import importlib.metadata as m, platform
    versions={p:m.version(p) for p in ['torch','numpy','pandas','pyarrow','scikit-learn','transformers','accelerate','huggingface-hub','tokenizers','safetensors','matplotlib','gradio']}
    versions['python']=platform.python_version(); return versions
def file_manifest(root):
    root=pathlib.Path(root)
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in str(p) and p.name!='files_manifest.json'}
def verify_manifest(root):
    root=pathlib.Path(root); manifest=read(root/'files_manifest.json')
    for name,expected in manifest.items():
        if not (root/name).is_file() or sha(root/name)!=expected: raise RuntimeError('Artifact identity mismatch: '+name)
