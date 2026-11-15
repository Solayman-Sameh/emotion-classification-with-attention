import os
os.environ.setdefault('MPLBACKEND','Agg')
os.environ.setdefault('USE_TF','0');os.environ.setdefault('USE_FLAX','0');os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
import html,inspect,pathlib,re,torch
torch.set_num_threads(2)
import gradio as gr
from common import LABELS
from inference import Application

EMOJI={'joy':'😊','sadness':'😢','anger':'😠','fear':'😨','surprise':'😲','disgust':'🤢'}
MODEL_NAMES={'distilbert':'Main Reader','bilstm_attention':'Word-Focus Reader'}
EXAMPLES=['I am so happy you made it!','I am not happy about this.','That is disgusting.','Wait, what just happened?!','😢😢','happy happy sad happy']

THEME=gr.themes.Base(primary_hue='indigo',neutral_hue='slate',radius_size='lg',
    font=[gr.themes.GoogleFont('Instrument Sans'),'ui-sans-serif','system-ui','sans-serif']
).set(button_primary_background_fill='var(--ea)',button_primary_background_fill_hover='var(--ea-hover)',
      button_primary_text_color='white',button_primary_border_color='transparent')

CSS='''
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Instrument+Sans:wght@400;500;600;700&display=swap');
:root{
  --ea:#4f46e5;--ea-hover:#4338ca;--ea-soft:rgba(79,70,229,.10);
  --surface:#ffffff;--surface-subtle:#f8fafc;--border:#e2e8f0;
  --text-main:#0f172a;--text-muted:#64748b;
  --emo-joy:#d97706;--emo-joy-bg:rgba(245,158,11,.13);
  --emo-sadness:#2563eb;--emo-sadness-bg:rgba(59,130,246,.13);
  --emo-anger:#dc2626;--emo-anger-bg:rgba(239,68,68,.13);
  --emo-fear:#7c3aed;--emo-fear-bg:rgba(139,92,246,.13);
  --emo-surprise:#0891b2;--emo-surprise-bg:rgba(6,182,212,.13);
  --emo-disgust:#059669;--emo-disgust-bg:rgba(16,185,129,.13);
}
.dark{
  --ea:#818cf8;--ea-hover:#a5b4fc;--ea-soft:rgba(129,140,248,.16);
  --surface:#0f172a;--surface-subtle:#1e293b;--border:#334155;
  --text-main:#f8fafc;--text-muted:#94a3b8;
  --emo-joy:#fbbf24;--emo-joy-bg:rgba(251,191,36,.16);
  --emo-sadness:#60a5fa;--emo-sadness-bg:rgba(96,165,250,.16);
  --emo-anger:#f87171;--emo-anger-bg:rgba(248,113,113,.16);
  --emo-fear:#c084fc;--emo-fear-bg:rgba(192,132,252,.16);
  --emo-surprise:#22d3ee;--emo-surprise-bg:rgba(34,211,238,.16);
  --emo-disgust:#34d399;--emo-disgust-bg:rgba(52,211,153,.16);
}
.gradio-container{max-width:1040px!important;width:100%!important;margin:0 auto!important;padding:44px 24px 64px!important;box-sizing:border-box!important}
.gradio-container .main,.gradio-container .wrap,.gradio-container .contain{min-width:0!important}
.html-container{padding:0!important}
.gradio-container .column{gap:16px!important}

/* Header */
.e-head{padding:0 0 8px!important;border:0!important;background:none!important;box-shadow:none!important}
.e-eyebrow{display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:999px;background:var(--ea-soft);color:var(--ea);font-size:12px;font-weight:600;letter-spacing:.03em;text-transform:uppercase;margin-bottom:12px}
.e-head h1{font-family:'Bricolage Grotesque','Instrument Sans',system-ui,sans-serif;font-weight:700;font-size:40px;line-height:1.08;letter-spacing:-.025em;margin:0 0 10px;color:var(--body-text-color)}
.e-head p{max-width:60ch;font-size:16.5px;line-height:1.55;margin:0;color:var(--body-text-color-subdued)}

/* Unified Input Composer */
.e-composer{border:1px solid var(--border-color-primary);border-radius:18px;padding:18px 20px 16px;background:var(--background-fill-primary);box-shadow:0 10px 30px -10px rgba(15,23,42,.06);transition:border-color .2s,box-shadow .2s}
.e-composer:focus-within{border-color:var(--ea);box-shadow:0 0 0 3px var(--ea-soft),0 12px 32px -10px rgba(79,70,229,.12)}
.e-composer .e-composer{padding:0!important;border:0!important;border-radius:0;box-shadow:none!important;background:transparent!important}
.e-composer .styler,.e-composer .form{background:transparent!important;border:0!important;box-shadow:none!important}
.e-input{border:0!important;background:transparent!important;box-shadow:none!important;padding:0!important}
.e-input textarea{font-size:18px!important;line-height:1.55!important;background:transparent!important;box-shadow:none!important;border:0!important;padding:4px 2px 12px!important}
.e-input textarea:focus{box-shadow:none!important;outline:none!important}
.e-toolbar{display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px;padding-top:12px;border-top:1px solid var(--border-color-primary)}
.e-actions{gap:10px!important;align-items:center}
.e-actions button{flex:0 0 auto!important;width:auto!important;min-width:0!important;border-radius:10px!important;font-weight:600!important;padding:8px 18px!important}

/* Sample Prompt Chips */
.e-try{margin:8px 0 6px;font-size:13px;font-weight:600;letter-spacing:.01em;color:var(--body-text-color-subdued)}
.e-chips{gap:8px!important;flex-wrap:wrap!important}
.e-chips button{flex:0 0 auto!important;width:auto!important;min-width:0!important;background:var(--background-fill-primary)!important;border:1px solid var(--border-color-primary)!important;color:var(--body-text-color)!important;font-weight:500!important;font-size:13.5px!important;padding:6px 14px!important;border-radius:999px!important;transition:all .15s ease!important}
.e-chips button:hover{border-color:var(--ea)!important;background:var(--ea-soft)!important;color:var(--ea)!important;transform:translateY(-1px)}

/* Cards & Grid */
.e-stretch,.e-stretch .prose,.e-stretch .html-container{height:100%;background:transparent!important;border:0!important;box-shadow:none!important;overflow:visible!important}
.e-results>.column{min-width:min(340px,100%)!important}
.e-card{border:1px solid var(--border-color-primary);border-radius:18px;padding:24px 26px;background:var(--background-fill-primary);height:100%;box-sizing:border-box;min-width:0;overflow-wrap:anywhere;display:flex;flex-direction:column;justify-content:space-between;box-shadow:0 4px 20px -6px rgba(15,23,42,.04)}
.e-card h2{font-family:'Bricolage Grotesque','Instrument Sans',system-ui,sans-serif;font-size:20px;font-weight:600;letter-spacing:-.015em;margin:0 0 4px;color:var(--body-text-color)}
.e-sub{margin:0 0 18px;font-size:14px;color:var(--body-text-color-subdued);line-height:1.5}
.e-empty{color:var(--body-text-color-subdued);font-size:15px;line-height:1.6;margin:12px 0;padding:28px 20px;border-radius:12px;background:color-mix(in srgb,var(--body-text-color) 3%,transparent);text-align:center}

/* Prediction Hero Banner & Rows */
.e-result{display:flex;align-items:center;gap:14px;margin:4px 0 22px;padding:16px 18px;border-radius:14px;background:var(--emo-bg,var(--ea-soft));border:1px solid color-mix(in srgb,var(--emo-color,var(--ea)) 22%,transparent)}
.e-emoji{font-size:38px;line-height:1;flex-shrink:0}
.e-hero-meta{display:flex;flex-direction:column;gap:2px;min-width:0;flex:1}
.e-word{font-family:'Bricolage Grotesque','Instrument Sans',system-ui,sans-serif;font-size:34px;font-weight:700;letter-spacing:-.02em;line-height:1.05;color:var(--emo-color,var(--ea))}
.e-conf-wrap{display:flex;align-items:center;gap:8px;font-size:13.5px;color:var(--body-text-color-subdued)}
.e-conf{font-size:15px;font-weight:700;color:var(--body-text-color);font-variant-numeric:tabular-nums}
.e-row{display:grid;grid-template-columns:92px minmax(0,1fr) 46px;align-items:center;gap:12px;margin:10px 0;font-size:14.5px;color:var(--body-text-color-subdued);text-transform:capitalize}
.e-row.win{color:var(--body-text-color);font-weight:600}
.e-track{height:9px;border-radius:999px;background:color-mix(in srgb,var(--body-text-color) 8%,transparent);overflow:hidden}
.e-fill{height:100%;border-radius:999px;background:color-mix(in srgb,var(--body-text-color) 26%,transparent);transition:width .35s cubic-bezier(.22,1,.36,1)}
.e-row.win .e-fill{background:var(--emo-color,var(--ea))}
.e-pct{text-align:right;font-variant-numeric:tabular-nums;font-size:13.5px}

/* Word Focus & Token Pills */
.e-badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;font-size:12.5px;font-weight:600;margin-left:6px;background:rgba(16,185,129,.12);color:#059669}
.dark .e-badge{color:#34d399}
.e-badge.diff{background:rgba(245,158,11,.15);color:#b45309}
.dark .e-badge.diff{color:#fbbf24}
.e-tokens{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0 14px;padding:18px;border-radius:14px;background:color-mix(in srgb,var(--body-text-color) 3%,transparent);border:1px solid color-mix(in srgb,var(--border-color-primary) 70%,transparent)}
.e-tok{display:inline-flex;flex-direction:column;align-items:center;min-width:48px;padding:8px 12px 7px;border-radius:11px;background:color-mix(in srgb,var(--ea) var(--tok-tint,0%),var(--background-fill-primary));border:1px solid color-mix(in srgb,var(--ea) var(--tok-border,12%),var(--border-color-primary));cursor:default;transition:transform .15s ease,box-shadow .15s ease;max-width:100%}
.e-tok:hover{transform:translateY(-2px);box-shadow:0 6px 16px -4px rgba(79,70,229,.18)}
.e-t{font-size:17px;font-weight:600;line-height:1.25;color:var(--body-text-color);overflow-wrap:anywhere;min-width:0;margin-bottom:6px}
.e-bw{display:flex;align-items:flex-end;justify-content:center;width:100%;height:22px;padding:0 4px;border-bottom:1px solid color-mix(in srgb,var(--border-color-primary) 80%,transparent);box-sizing:border-box}
.e-b{display:block;width:14px;max-width:100%;background:var(--ea);border-radius:3px 3px 0 0;transition:height .3s ease}
.e-weight{margin-top:5px;text-align:center;font-size:11.5px;line-height:1.2;font-weight:600;font-variant-numeric:tabular-nums;color:var(--body-text-color-subdued)}
.e-note{margin:10px 0 0;font-size:13px;line-height:1.45;color:var(--body-text-color-subdued)}
.e-takeaway{margin-top:14px;padding:12px 14px;border-radius:11px;background:color-mix(in srgb,var(--ea) 6%,transparent);font-size:13.5px;line-height:1.45;color:var(--body-text-color-subdued)}
.e-takeaway b{color:var(--body-text-color)}

/* Info Popover */
.e-card-heading{display:flex;align-items:center;justify-content:space-between;gap:8px;position:relative;margin-bottom:4px}
.e-card-heading h2{margin:0}
.e-info summary{list-style:none;cursor:pointer;display:inline-flex;align-items:center;gap:5px;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;color:var(--body-text-color-subdued);border:1px solid var(--border-color-primary);transition:all .15s}
.e-info summary::-webkit-details-marker{display:none}
.e-info summary span{display:inline-flex;align-items:center;justify-content:center;width:15px;height:15px;border-radius:50%;background:var(--ea-soft);color:var(--ea);font:700 11px/1 system-ui}
.e-info summary:hover,.e-info[open] summary{color:var(--ea);border-color:var(--ea);background:var(--ea-soft)}
.e-info-panel{position:absolute;top:calc(100% + 8px);right:0;width:min(380px,92vw);box-sizing:border-box;padding:16px 18px;border:1px solid var(--border-color-primary);border-radius:14px;background:var(--background-fill-primary);color:var(--body-text-color);box-shadow:0 14px 34px rgba(15,23,42,.16);z-index:30;font-size:13.5px;line-height:1.55;font-weight:400}
.e-info-panel p{margin:0 0 10px}.e-info-panel p:last-child{margin-bottom:0}

/* Friendly Footer Guide */
.e-about{margin-top:16px!important;border:1px solid var(--border-color-primary)!important;border-radius:16px!important;padding:4px 18px!important;background:var(--background-fill-primary)!important}
.e-about .block{overflow:visible!important}
.e-guide-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;padding:8px 0 12px}
.e-guide-item{padding:14px 16px;border-radius:12px;background:color-mix(in srgb,var(--body-text-color) 3%,transparent)}
.e-guide-item h4{margin:0 0 6px;font-size:14px;font-weight:600;color:var(--body-text-color)}
.e-guide-item p{margin:0;font-size:13.5px;line-height:1.5;color:var(--body-text-color-subdued)}

.e-tok{box-sizing:border-box}
.e-card-heading{flex-wrap:wrap}
.e-info summary:focus-visible{outline:2px solid var(--ea);outline-offset:3px}
.e-info-panel{width:min(380px,100%);max-height:min(65vh,480px);overflow-y:auto}
@media(max-width:680px){
  .gradio-container .main{padding:0!important}
  .e-composer{padding:14px 12px}
  .e-row{grid-template-columns:64px minmax(0,1fr) 38px;gap:8px}
  .e-chips button{max-width:100%!important;white-space:normal!important}
  .e-result{padding:12px;gap:10px;flex-wrap:wrap}
  .e-word{font-size:clamp(26px,8vw,34px)}
  .e-tokens{padding:10px;gap:8px}

  .gradio-container{padding:24px 14px 36px!important}
  .e-head h1{font-size:32px}
  .e-results{flex-direction:column!important}
  .e-results>.column{width:100%!important;min-width:0!important}
  .e-card{padding:20px 18px}
  .e-guide-grid{grid-template-columns:1fr}
}
'''

def pct(p):
    if p<0.005:return '<1%'
    if p>0.995:return '>99%'
    return f'{p*100:.0f}%'

def attention_heading():
    return ('<div class="e-card-heading"><h2>Words that stood out</h2>'
            '<details class="e-info"><summary aria-label="About word focus"><span aria-hidden="true">?</span>How this works</summary>'
            '<div class="e-info-panel"><p><b>What do these highlights mean?</b><br>A second model, BiLSTM + Attention, gives each word position a share of its focus. Higher percentages mean more attention, not an exact contribution to the prediction. Word representations include surrounding context; punctuation and emoji can receive focus too.</p>'
            '<p><b>Why can the two cards disagree?</b><br>The cards use two different models, each reading the sentence in context. This card shows the second model’s own prediction and focus; it does not explain the main reader’s decision.</p>'
            '<p><b>Reading the scores:</b><br>Percentages add up to about 100%, allowing for rounding. Any word position at 25% or higher reaches the top of its meter. Hover a tile for the precise weight.</p></div></details></div>')

def empty_card(title,hint):
    heading=attention_heading() if title in ('Words in focus','Words that stood out') else f'<h2>{title}</h2>'
    return f'<div class="e-card"><div>{heading}<p class="e-empty">{hint}</p></div></div>'

def prediction_card(r):
    probs=sorted(r['probabilities'].items(),key=lambda kv:-kv[1])
    top,topp=probs[0]
    rows=''.join(
        f'<div class="e-row{" win" if i==0 else ""}" style="--emo-color:var(--emo-{html.escape(k)},var(--ea))" data-emotion="{html.escape(k)}" data-probability="{p:.10f}">'
        f'<span>{EMOJI.get(k,"")} {html.escape(k)}</span>'
        f'<span class="e-track"><span class="e-fill" style="display:block;width:{p*100:.2f}%"></span></span>'
        f'<span class="e-pct">{pct(p)}</span></div>'
        for i,(k,p) in enumerate(probs)
    )
    return (
        f'<div class="e-card" data-model="{html.escape(r["selected_model"])}" style="--emo-color:var(--emo-{html.escape(top)},var(--ea));--emo-bg:var(--emo-{html.escape(top)}-bg,var(--ea-soft))">'
        f'<div><h2>Detected Mood</h2><p class="e-sub">The main reader’s scores across the six emotions.</p>'
        f'<div class="e-result"><span class="e-emoji">{EMOJI.get(top,"")}</span>'
        f'<div class="e-hero-meta"><span class="e-word">{html.escape(top.capitalize())}</span>'
        f'<div class="e-conf-wrap"><span class="e-conf">{pct(topp)}</span><span>model score</span></div></div></div>'
        f'{rows}</div></div>'
    )

def attention_card(r):
    agree=r['attention_prediction']==r['prediction']
    badge=f'<span class="e-badge{"" if agree else " diff"}">{"✓ Matches overall mood" if agree else "⚡ Second opinion"}</span>'
    m=re.search(r'uses (\d+) of (\d+) subwords',r['attention_status'])
    extra=f' The main reader used only the first {m.group(1)} of {m.group(2)} word parts because of length.' if m and m.group(1)!=m.group(2) else ''
    sub=f'The word-focus reader suggests <b>{html.escape(r["attention_prediction"])}</b>.{badge}{extra}'
    return (
        f'<div class="e-card"><div>{attention_heading()}<p class="e-sub">{sub}</p>{r["attention_html"]}</div>'
        f'<div class="e-takeaway"><b>Tip:</b> Taller bars and brighter tiles show where this reader focused. They reach their maximum at 25%; the percentage shows the actual weight. Hover a tile for the precise value.</div></div>'
    )

ABOUT='''<div class="e-guide-grid">
<div class="e-guide-item"><h4>1. Why two cards?</h4><p>The main reader predicts an emotion for your text. A separate reader offers its own prediction and highlights where it focused. The two can disagree. These scores are not calibrated guarantees of accuracy.</p></div>
<div class="e-guide-item"><h4>2. How word scores work</h4><p>Each word position receives a share of the total focus. Percentages add up to about 100%, allowing for rounding. Every tile has the same meter width and reaches full height at 25%; the number shows its actual weight.</p></div>
<div class="e-guide-item"><h4>3. Always picks one of six</h4><p>Even everyday or neutral sentences are matched to the closest of the six emotions (Joy, Surprise, Anger, Sadness, Disgust, or Fear). It reflects the phrasing, not how you personally feel.</p></div>
</div>'''

def build(root=None):
    application=Application(root or pathlib.Path(__file__).parent)
    def classify(text):
        try:r=application.run(text)
        except ValueError as e:raise gr.Error(str(e))
        return prediction_card(r),attention_card(r)
    blocks_options = {'theme': THEME, 'css': CSS} if 'theme' in inspect.signature(gr.Blocks).parameters else {}
    html_options = {'padding': False} if 'padding' in inspect.signature(gr.HTML).parameters else {}
    with gr.Blocks(title='Emotion & Word Focus Explorer',fill_width=False,**blocks_options) as demo:
        gr.HTML('<div class="e-eyebrow">✨ Tone & Word Focus Explorer</div><h1>What emotion does your text express?</h1><p>Type or paste a sentence below. Compare an emotion prediction with a second reader’s word-by-word focus.</p>',elem_classes='e-head',**html_options)
        if application.manifest['mode']!='full':gr.Markdown('**Smoke checkpoints: functional checks only. These results do not establish model quality.**')
        with gr.Group(elem_classes='e-composer'):
            text=gr.Textbox(show_label=False,lines=3,placeholder='Write or paste an English sentence here (e.g., "I am so happy you made it!")...',elem_classes='e-input')
            with gr.Row(elem_classes='e-toolbar'):
                with gr.Row(elem_classes='e-actions'):
                    button=gr.Button('Analyze Tone',variant='primary',size='md')
                    clear=gr.Button('Reset',variant='secondary',size='md')
        gr.HTML('<p class="e-try">Or test with a quick example:</p>',**html_options)
        with gr.Row(elem_classes='e-chips'):
            chips=[gr.Button(ex,size='sm',variant='secondary') for ex in EXAMPLES]
        with gr.Row(equal_height=True,elem_classes='e-results'):
            with gr.Column(scale=5,min_width=340):pred=gr.HTML(empty_card('Detected Mood','Type a sentence above to see its emotional breakdown.'),**html_options,elem_classes='e-stretch')
            with gr.Column(scale=6,min_width=340):attn=gr.HTML(empty_card('Words that stood out','See where the second reader focused as it read your text.'),**html_options,elem_classes='e-stretch')
        with gr.Accordion('How to read these results (Plain-English Guide)',open=False,elem_classes='e-about'):gr.HTML(ABOUT,**html_options)
        outs=[pred,attn]
        button.click(classify,text,outs);text.submit(classify,text,outs)
        for chip,ex in zip(chips,EXAMPLES):chip.click(lambda ex=ex:ex,None,text,queue=False)
        clear.click(lambda:('',empty_card('Detected Mood','Type a sentence above to see its emotional breakdown.'),empty_card('Words that stood out','See where the second reader focused as it read your text.')),None,[text,pred,attn])
    return demo

def launch(root=None):
    demo = build(root)
    parameters = inspect.signature(demo.launch).parameters
    options = {'theme': THEME, 'css': CSS} if 'theme' in parameters else {}
    if 'footer_links' in parameters:
        options['footer_links'] = []
    return demo.launch(**options)

if __name__=='__main__':launch()
