"""Exercise the real local Gradio HTTP API using the pinned, already cached models.

This is an API check, not a visual browser check or a fresh dependency install.
"""
import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import urllib.request


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout-root',type=pathlib.Path,default=pathlib.Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.checkout_root.resolve()
    scratch=root/'.release-checks';scratch.mkdir(exist_ok=True)
    os.environ.update(NO_PROXY='127.0.0.1,localhost',GRADIO_ANALYTICS_ENABLED='False',HF_HUB_DISABLE_TELEMETRY='1')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',USE_TF='0',USE_FLAX='0',TOKENIZERS_PARALLELISM='false',GRADIO_SERVER_PORT='7867',GRADIO_SERVER_NAME='127.0.0.1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    log=open(scratch/'api-server.log','w',encoding='utf-8')
    process=subprocess.Popen([sys.executable,'-u','app.py'],cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    result={'mode':'real HTTP API with cached models; no browser or fresh download','samples':[]}
    base='http://127.0.0.1:7867';opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        for _ in range(120):
            if process.poll() is not None:raise RuntimeError('App exited: inspect api-server.log')
            try:
                with opener.open(base,timeout=1) as response:
                    if response.status==200:break
            except Exception:time.sleep(.5)
        else:raise TimeoutError('App startup timeout')
        from gradio_client import Client
        client=Client(base,verbose=False)
        samples=['I am so happy you made it!','I am not happy about this.','happy happy sad happy','\U0001f622\U0001f622','x','happy '*160,'<script>alert(1)</script>']
        for sample in samples:
            prediction,heatmap=client.predict(sample,api_name='/classify')
            bars=re.findall(r'data-emotion="([a-z]+)" data-probability="([0-9.]+)"',prediction)
            assert len(bars)==6 and {label for label,_ in bars}=={'joy','sadness','anger','fear','surprise','disgust'}
            probabilities=[float(value) for _,value in bars]
            assert abs(sum(probabilities)-1)<1e-5
            assert probabilities==sorted(probabilities,reverse=True)
            assert 'data-model="distilbert"' in prediction and 'BiLSTM + Attention' in heatmap and 'Words that stood out' in heatmap
            assert '<script>' not in heatmap
            weights=[float(x) for x in re.findall(r'data-attention="([0-9.]+)"',heatmap)]
            assert weights and abs(sum(weights)-1)<1e-5
            if sample=='happy '*160:assert len(weights)==128 and 'Showing the first 128 of 160 tokens' in heatmap
            if sample=='happy happy sad happy':assert len(weights)==4 and heatmap.count('>happy</span>')==3
            result['samples'].append({'input':sample[:65],'probability_order':[label for label,_ in bars],'attention_tokens':len(weights)})
        try:client.predict('',api_name='/classify')
        except Exception as error:
            assert 'Please enter nonblank text.' in str(error),str(error)
            result['blank_rejected']=True
        else:raise AssertionError('Blank input accepted')
        result['status']='passed'
    finally:
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close();(scratch/'api_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


if __name__=='__main__':main()
