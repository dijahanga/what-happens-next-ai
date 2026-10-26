import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='What Happens Next? | Process Intelligence', page_icon='', layout='wide')
st.markdown('''<style>
.stApp {background:linear-gradient(125deg,#0b1224 0%,#111c33 60%,#0d1830 100%);color:#eaf2ff}
[data-testid="stSidebar"] {background:#111b30}
h1,h2,h3 {color:#f1f7ff!important}
[data-testid="stMetric"] {background:#1a2942;padding:14px;border-radius:13px;border:1px solid #2b405b}
[data-testid="stMetricLabel"] {color:#bdcbe0}
.stButton>button[kind="primary"] {background:#32c6b1;color:#071525;border:0;font-weight:700}
</style>''', unsafe_allow_html=True)

class NextEventLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)
    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(hidden[-1])

@st.cache_resource
def load_model():
    metadata = json.loads((ROOT/'metadata.json').read_text())
    model = NextEventLSTM(len(metadata['tokens']), metadata['embedding_dim'], metadata['hidden_dim'])
    model.load_state_dict(torch.load(ROOT/'model.pt', map_location='cpu', weights_only=True))
    model.eval()
    return model, metadata

@st.cache_data
def load_cases():
    df = pd.read_csv(ROOT/'demo_cases.csv.gz')
    return {str(case): group['event'].tolist() for case,group in df.groupby('case', sort=False)}

model, metadata = load_model()
tokens = metadata['tokens']
reverse = {v:k for k,v in tokens.items()}
cases = load_cases()
SPECIAL = {'<PAD>','<START>'}

# Display translations only. Original BPIC2012 activity IDs remain unchanged for inference.
WORK_LABELS = {
    'W_Afhandelen leads': 'Follow up on enquiry',
    'W_Beoordelen fraude': 'Review fraud risk',
    'W_Completeren aanvraag': 'Complete application details',
    'W_Nabellen incomplete dossiers': 'Follow up on missing documents',
    'W_Nabellen offertes': 'Follow up on offer',
    'W_Valideren aanvraag': 'Validate application',
}
APPLICATION_LABELS = {
    'A_SUBMITTED':'Application submitted', 'A_PARTLYSUBMITTED':'Application partly submitted',
    'A_PREACCEPTED':'Initial acceptance', 'A_ACCEPTED':'Application accepted',
    'A_DECLINED':'Application declined', 'A_CANCELLED':'Application cancelled',
    'A_FINALIZED':'Application finalised', 'A_APPROVED':'Application approved',
    'A_REGISTERED':'Application registered', 'A_ACTIVATED':'Application activated',
}
OFFER_LABELS = {
    'O_CREATED':'Offer created', 'O_SELECTED':'Offer selected',
    'O_SENT':'Offer sent', 'O_SENT_BACK':'Signed offer returned',
    'O_ACCEPTED':'Offer accepted', 'O_DECLINED':'Offer declined',
    'O_CANCELLED':'Offer cancelled',
}
def friendly(event):
    if event == '<END>': return 'Process completed'
    return WORK_LABELS.get(event) or APPLICATION_LABELS.get(event) or OFFER_LABELS.get(event) or event

def category(event):
    if event.startswith('A_'): return 'Application'
    if event.startswith('O_'): return 'Offer'
    if event.startswith('W_'): return 'Work task'
    return 'Completion'

def predict(prefix):
    seq = [tokens['<START>']] + [tokens[e] for e in prefix]
    seq = seq[-metadata['max_len']:]
    padded = [0]*(metadata['max_len']-len(seq))+seq
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor([padded],dtype=torch.long))[0],dim=-1).numpy()
    # Do not hide invalid output probability: display a warning if meaningful.
    invalid = sum(float(probabilities[tokens[t]]) for t in SPECIAL)
    candidates = [(reverse[i],float(p)) for i,p in enumerate(probabilities) if reverse[i] not in SPECIAL]
    candidates.sort(key=lambda pair:pair[1], reverse=True)
    return candidates,invalid

def process_chart(prefix, options, revealed=False, actual=None):
    # Observed trace and predicted branches; NOT a causal explanation.
    history = prefix[-5:]
    fig = go.Figure()
    if not history:
        history = ['<START>']
    xs = list(range(len(history)))
    last = xs[-1]
    for i in range(len(xs)-1):
        fig.add_trace(go.Scatter(x=[xs[i], xs[i+1]], y=[0,0], mode='lines',
            line=dict(color='#54D9C4', width=4), hoverinfo='skip', showlegend=False))
    fig.add_trace(go.Scatter(x=xs,y=[0]*len(xs),mode='markers+text',
        marker=dict(size=[36]*len(xs),color=['#235d67']*(len(xs)-1)+['#39bba9'],
                    line=dict(color='#b1fff0',width=2)),
        text=[str(i+1) for i in xs],textposition='middle center',
        textfont=dict(color='white',size=12),customdata=[friendly(e) for e in history],
        hovertemplate='%{customdata}<extra>Observed</extra>',showlegend=False))
    branch_y = [1.05,0,-1.05]
    colors=['#53DFC7','#A99AFB','#F4BD75']
    for i,(event,prob) in enumerate(options[:3]):
        y=branch_y[i]
        fig.add_trace(go.Scatter(x=[last,last+1.3],y=[0,y],mode='lines',
            line=dict(color=colors[i],width=2.5,dash='dot'),hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter(x=[last+1.3],y=[y],mode='markers+text',
            marker=dict(size=44,color=colors[i],line=dict(color='#f5f7ff',width=1)),
            text=[f'{prob:.0%}'],textposition='middle center',textfont=dict(color='#10192b',size=11),
            customdata=[friendly(event)],hovertemplate='%{customdata}: %{text}<extra>Predicted</extra>',showlegend=False))
        fig.add_annotation(x=last+1.3,y=y+(.29 if i!=2 else -.29),text=friendly(event),
            showarrow=False,font=dict(color='#eef4ff',size=11),xanchor='center')
    if revealed and actual is not None:
        fig.add_trace(go.Scatter(x=[last,last+1.3],y=[0,-1.95],mode='lines',
            line=dict(color='#59B9FF',width=3),hoverinfo='skip',showlegend=False))
        fig.add_trace(go.Scatter(x=[last+1.3],y=[-1.95],mode='markers',
            marker=dict(size=22,color='#59B9FF',symbol='diamond'),
            customdata=[friendly(actual)],hovertemplate='Recorded: %{customdata}<extra></extra>',showlegend=False))
        fig.add_annotation(x=last+1.3,y=-2.3,text='Recorded: '+friendly(actual),
            showarrow=False,font=dict(color='#85CBFF',size=12))
    fig.update_layout(height=400 if revealed else 340,margin=dict(l=15,r=65,t=48,b=45),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#eaf2ff'),xaxis=dict(visible=False,range=[-.4,last+2.25]),
        yaxis=dict(visible=False,range=[-2.65 if revealed else -1.65,1.75]),
        showlegend=False)
    return fig

st.markdown('''<style>
.hero {padding:16px 0 8px 0}
.hero-eyebrow {letter-spacing:.16em;color:#69e0cb;font-size:.85rem;font-weight:750}
.hero-title {font-size:clamp(2.0rem,4vw,3.45rem);line-height:1.12;font-weight:800;color:#f6f9ff;margin:12px 0}
.hero-sub {color:#c1cfe2;font-size:1.06rem;max-width:820px}
.explainer {padding:14px 17px;background:#1a2942;border:1px solid #38516d;border-radius:14px;color:#e8f2ff}
</style>''',unsafe_allow_html=True)

with st.sidebar:
    st.header('Explore an application')
    st.caption('Actual event sequences from the public BPIC2012 loan-application log.')
    case_id = st.selectbox('Choose an example case',list(cases),format_func=lambda c:f'Application #{c}')
    trace = cases[case_id]
    max_prefix = min(len(trace),metadata['max_len']-1)
    n = st.slider('Events observed so far',0,max_prefix,min(5,max_prefix),key=f'prefix_{case_id}')
    st.caption('Move the slider to explore a real recorded journey.')
    presentation = st.toggle('🎤 Presentation mode',value=False,
        help='Shows audience instructions and the guessing experiment.')
    st.divider()
    st.markdown('**Model:** PyTorch LSTM')
    st.markdown('**Source:** BPIC2012')
    st.markdown('**Prediction:** Next recorded activity')

prefix=trace[:n]
options,invalid=predict(prefix)
next_event=trace[n] if n<len(trace) else '<END>'
reveal_key=f'reveal_{case_id}_{n}'
revealed=st.session_state.get(reveal_key,False)
metrics=metadata['results']

st.markdown('''<div class="hero"><div class="hero-eyebrow">✦ AI · PROCESS INTELLIGENCE · RESEARCH IN ACTION</div>
<div class="hero-title">What Happens Next?</div>
<div class="hero-sub">Follow a real loan application. Ask AI to predict its next step. Then reveal what actually happened.</div></div>''',unsafe_allow_html=True)
st.caption('A live interactive demonstration built from doctoral research in predictive process mining.')

if presentation:
    st.info(' Pick a case, look at its history, what could be the next event 🤔.')

c1,c2,c3=st.columns(3)
c1.metric('Held-out top-1 accuracy',f"{metrics['test']['accuracy']:.1%}")
c2.metric('Cases in source log',f"{metrics['cases']:,}")
c3.metric('Activity types',len(tokens)-3)
st.caption('Accuracy is a reported test-set metric, not a guarantee for the selected case. Probabilities are uncalibrated model outputs.')

st.subheader('01 · The application journey')
st.markdown('''<div class="explainer">Imagine someone applying for a loan. Behind the scenes, an application can move through checks, staff follow-ups, offers and decisions. Each step is an event. Our model studies past sequences to forecast the next one.</div>''',unsafe_allow_html=True)
if prefix:
    st.markdown('**Observed so far:** '+'  →  '.join(f'`{friendly(x)}`' for x in prefix[-6:]))
else:
    st.markdown('**Observed so far:** No activities yet')

st.plotly_chart(process_chart(prefix,options,revealed,next_event),use_container_width=True)
st.caption('Solid teal path = observed events · Dotted branches = model predictions · Blue diamond = recorded outcome (after reveal). This is a visualisation of predictions, not a causal explanation of the neural network.')

st.subheader('02 · AI next-event forecast')
left,right=st.columns([1,1.3],gap='large')
with left:
    st.markdown('**Most likely next event**')
    st.markdown(f'## {friendly(options[0][0])}')
    st.metric('Model probability',f'{options[0][1]:.1%}')
    st.caption(f'Original dataset label: `{options[0][0]}`')
    st.caption('The model considers several possible next events, not just one.')
with right:
    top=options[:5]
    bar=go.Figure(go.Bar(x=[p for _,p in top][::-1],
        y=[friendly(e) for e,_ in top][::-1],orientation='h',
        marker_color=['#6d80c8','#6d80c8','#6d80c8','#8c79dc','#43d2ba'][::-1],
        text=[f'{p:.1%}' for _,p in top][::-1],textposition='outside'))
    bar.update_layout(height=300,margin=dict(l=0,r=55,t=10,b=20),
        paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e4edff'),xaxis=dict(range=[0,max(.15,top[0][1]*1.27)],tickformat='.0%',gridcolor='#2b3c58'),
        yaxis=dict(automargin=True))
    st.plotly_chart(bar,use_container_width=True)
if invalid>.01:
    st.warning(f'{invalid:.1%} probability was assigned to reserved input tokens; interpret the predictions cautiously.')

st.subheader('03 · You versus the AI')
if presentation:
    choices=[e for e,_ in options[:5]]
    if next_event not in choices: choices.append(next_event)
    guess=st.radio('What do you think happens next?',choices,
        format_func=friendly,horizontal=False,key=f'guess_{case_id}_{n}',
        disabled=revealed)
    st.caption('Let's see the AI forecast.')
if st.button('✨ what actually happened',type='primary',disabled=revealed):
    st.session_state[reveal_key]=True
    st.rerun()
if revealed:
    correct=options[0][0]==next_event
    if correct:
        st.success(f'**AI top prediction was correct!** Recorded next event: **{friendly(next_event)}**.')
    else:
        st.info(f'**The recorded next event was {friendly(next_event)}.** The model favoured **{friendly(options[0][0])}**. A prediction is not a promise.')
    actual_p=next((p for e,p in options if e==next_event),0.)
    st.caption(f'Model probability assigned to the recorded outcome: {actual_p:.1%}.')
    if presentation:
        st.write('**Audience guess:** '+('Correct! 🎉' if guess==next_event else f'Not this time — the answer was {friendly(next_event)}.'))
    st.caption('Try another application or move the sidebar slider to test a different moment.')

with st.expander('Research, data provenance and important limitations'):
    st.write('**Research foundation:** A Deep Learning Approach to Business Process Mining. This showcase uses a separately retrained PyTorch LSTM for next-activity prediction, inspired by the thesis.')
    st.write('**Dataset:** BPIC2012, a public loan-application event log. Friendly English labels are presentation translations; the model uses the original activity IDs.')
    st.write('**Evaluation:** Application cases were separated into training, validation and testing. The displayed 52.54% is the reported top-1 held-out-prefix test accuracy. Model probabilities are not calibrated.')
    st.write('**What this demo does not do:** It does not predict waiting times, explain the causal internals of the LSTM, detect drift, or test process interventions. The source CSV lacks reliable complete timestamps, so event order follows the source rows.')
    st.write('**Recorded outcomes:** Demo traces come from sampled real cases in the event log; predictions may be incorrect, which is important to show.')
