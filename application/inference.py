"""Inference with verified local artifacts first and Hugging Face cache fallback."""
import html, pathlib, time
import numpy as np, torch
from common import LABELS,clean,tokens,encode,read,verify_manifest
from models import load_recurrent
from download_models import resolve_model_root

def load_model(kind,path,tokenizer_path,device='cpu',repo_id=None,revision=None,cache_dir=None):
    path=pathlib.Path(path)
    tokenizer_path=pathlib.Path(tokenizer_path)
    needed=[path/'config.json',path/'model.safetensors',tokenizer_path/'tokenizer.json',tokenizer_path/'tokenizer_config.json'] if kind=='distilbert' else [path/'best.pt']
    if not all(item.is_file() for item in needed):
        root=resolve_model_root(path.parent.parent,repo_id=repo_id,revision=revision,cache_dir=cache_dir)
        path=root/'models'/kind;tokenizer_path=root/'tokenizer'
    if kind=='distilbert':
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True,trust_remote_code=False)
        model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,trust_remote_code=False).to(device).eval()
        return model,tok
    return load_recurrent(path/'best.pt',device)[0],None

def predict(model,kind,texts,vocab,config,tokenizer=None,device='cpu',batch=32,attention=False):
    probs=[];attentions=[]
    with torch.inference_mode():
        for start in range(0,len(texts),batch):
            chunk=texts[start:start+batch]
            if any(not tokens(t) for t in chunk): raise ValueError('Enter nonblank usable text')
            if kind=='distilbert':
                inputs=tokenizer([clean(t) for t in chunk],padding=True,truncation=True,max_length=config['subword_cap'],return_tensors='pt')
                logits=model(**{k:v.to(device) for k,v in inputs.items()}).logits;weights=None
            else:
                seqs=[encode(t,vocab,config['word_cap'])[0] for t in chunk]
                ids=torch.zeros((len(seqs),max(map(len,seqs))),dtype=torch.long,device=device)
                for i,seq in enumerate(seqs):ids[i,:len(seq)]=torch.tensor(seq,device=device)
                logits,weights=model(ids)
                if attention:
                    if weights is None: raise ValueError('This architecture has no additive attention')
                    attentions.extend([weights[i,:len(seq)].cpu().numpy() for i,seq in enumerate(seqs)])
            probs.append(torch.softmax(logits.float(),dim=-1).cpu().numpy())
    p=np.concatenate(probs)
    if not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-6): raise RuntimeError('Invalid probabilities')
    return p,attentions

def attention_html(text,weights,cap):
    all_tokens=tokens(text);used=all_tokens[:cap]
    if len(used)!=len(weights) or not used:raise ValueError('Token/attention alignment mismatch')
    if not np.isfinite(weights).all() or not np.isclose(sum(weights),1,atol=1e-5):raise ValueError('Invalid attention weights')
    FULL=0.25;STRIP=22  # fixed scale for every input: a bar this tall means weight >= 25%. Never rescaled per example.
    def pct(w):return f'{w*100:.0f}%' if w>=0.1 else f'{w*100:.1f}%'
    spans=[]
    for i,(token,w) in enumerate(zip(used,weights)):
        w=float(w);ratio=min(w/FULL,1.0);h=ratio*STRIP
        tint=int(round(ratio*18));border=int(round(12+ratio*45))
        spans.append(
            f'<span class="e-tok" data-index="{i}" data-attention="{w:.10f}" style="--tok-tint:{tint}%;--tok-border:{border}%" title="Word #{i+1}: {w*100:.1f}% focus ({w:.8f})">'
            f'<span class="e-t">{html.escape(token)}</span>'
            f'<span class="e-bw"><span class="e-b" style="height:{h:.2f}px"></span></span>'
            f'<span class="e-weight">{pct(w)}</span></span>'
        )
    notes=[]
    if len(all_tokens)>cap:notes.insert(0,f'Showing the first {cap} of {len(all_tokens)} tokens (words, punctuation or emoji); the rest were cut off before this reader saw them.')
    return (f'<div class="e-tokens">{"".join(spans)}</div>'
            + (f'<p class="e-note">{html.escape(" ".join(notes))}</p>' if notes else ''))

class Application:
    def __init__(self,root,repo_id=None,revision=None,cache_dir=None):
        self.root=resolve_model_root(root,repo_id=repo_id,revision=revision,cache_dir=cache_dir);verify_manifest(self.root)
        self.manifest=read(self.root/'application.json');self.config=read(self.root/'config.json');self.vocab=read(self.root/'vocab.json')
        self.loaded={}
        for kind in set([self.manifest['selected'],'bilstm_attention']):
            self.loaded[kind]=load_model(kind,self.root/'models'/kind,self.root/'tokenizer')
    def run(self,text):
        if not isinstance(text,str) or not tokens(text):raise ValueError('Please enter nonblank text.')
        if len(text)>1_000_000:raise ValueError('Input limit is 1,000,000 characters; shorten the text.')
        main=self.manifest['selected'];model,tok=self.loaded[main]
        p,_=predict(model,main,[text],self.vocab,self.config,tok)
        am,at=self.loaded['bilstm_attention']
        ap,aw=predict(am,'bilstm_attention',[text],self.vocab,self.config,attention=True)
        winner=LABELS[int(p[0].argmax())];attention_winner=LABELS[int(ap[0].argmax())]
        status=f'BiLSTM + Attention predicts {attention_winner}. '+('Models agree.' if winner==attention_winner else 'Models disagree; this heatmap belongs to BiLSTM + Attention.')
        if main=='distilbert':
            n=len(tok(clean(text),add_special_tokens=True)['input_ids']);status+=f' Main classifier uses {min(n,self.config["subword_cap"])} of {n} subwords.'
        return {'selected_model':main,'prediction':winner,'probabilities':dict(zip(LABELS,map(float,p[0]))),'attention_prediction':attention_winner,
                'attention_status':status,'attention':list(map(float,aw[0])),'attention_html':attention_html(text,aw[0],self.config['word_cap'])}