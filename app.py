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
st.set_page_config(page_title='What Happens Next? | Process Intelligence', page_icon='🔮', layout='wide')
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

def friendly(event):
    if event == '<END>': return 'Process completed'
    if event.startswith('A_'): return 'Application · '+event[2:].replace('_',' ').title()
    if event.startswith('O_'): return 'Offer · '+event[2:].replace('_',' ').title()
    if event.startswith('W_'): return 'Work · '+event[2:].replace('_',' ').title()
    return event

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

def process_chart(prefix, options):
    history = prefix[-5:]
    labels = ([friendly(e).replace('Application · ','').replace('Offer · ','').replace('Work · ','') for e in history] or ['Start'])
    xvals = list(range(len(labels)))
    fig = go.Figure()
    for i in range(len(labels)-1):
        fig.add_trace(go.Scatter(x=[i,i+1],y=[0,0],mode='lines',line=dict(color='#5ce1cc',width=3),hoverinfo='skip',showlegend=False))
    fig.add_trace(go.Scatter(x=xvals,y=[0]*len(labels),mode='markers+text',marker=dict(size=28,color='#235f69',line=dict(width=2,color='#65ead5')),text=[str(i+1) for i in xvals],textposition='middle center',textfont=dict(color='white',size=11),customdata=labels,hovertemplate='%{customdata}<extra></extra>',showlegend=False))
    final_x=len(labels)-1
    for i,(event,p) in enumerate(options[:3]):
        y = [0.8,0,-0.8][i]
        fig.add_trace(go.Scatter(x=[final_x,final_x+1],y=[0,y],mode='lines',line=dict(color=['#57e0c4','#998cf4','#e8b36c'][i],width=max(1,6*p)),showlegend=False,hoverinfo='skip'))
        fig.add_trace(go.Scatter(x=[final_x+1],y=[y],mode='markers+text',marker=dict(size=25,color=['#1e817d','#6654ad','#9b6b39'][i]),text=[f'{p:.0%}'],textposition='middle right',textfont=dict(color='white',size=11),customdata=[friendly(event)],hovertemplate='%{customdata}<extra></extra>',showlegend=False))
    fig.update_layout(height=290,margin=dict(l=15,r=55,t=20,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',xaxis=dict(visible=False,range=[-.3,len(labels)+.65]),yaxis=dict(visible=False,range=[-1.25,1.25]))
    return fig

st.markdown('### ✦ WHAT HAPPENS NEXT?')
st.title('From yesterday’s events to tomorrow’s possibilities')
st.caption('A live AI process-intelligence demonstration inspired by doctoral research in predictive process mining.')

with st.sidebar:
    st.header('Explore an application')
    st.caption('Real activity sequences from the public BPIC2012 event log. Demo cases are sampled, not invented.')
    case_id = st.selectbox('Choose an example case',list(cases),format_func=lambda c:f'Application #{c}')
    trace = cases[case_id]
    max_prefix = min(len(trace),metadata['max_len']-1)
    n = st.slider('Events observed so far',0,max_prefix,min(5,max_prefix),key=f'prefix_{case_id}')
    st.caption('Adjust the slider to move through an actual recorded case.')
    st.divider()
    st.markdown('**Model** · PyTorch LSTM')
    st.markdown('**Source** · BPIC2012 event log')
    st.markdown('**Task** · Next-activity prediction')

prefix=trace[:n]
options,invalid=predict(prefix)
next_event=trace[n] if n<len(trace) else '<END>'
metrics=metadata['results']
a,b,c=st.columns(3)
a.metric('Held-out test accuracy',f"{metrics['test']['accuracy']:.1%}")
b.metric('Recorded application cases',f"{metrics['cases']:,}")
c.metric('Activity types',len(tokens)-3)

left,right=st.columns([1.45,1],gap='large')
with left:
    st.subheader('Application journey')
    if prefix:
        st.write(' → '.join(friendly(x).split(' · ')[-1] for x in prefix[-7:]))
    else: st.info('The case has not started. Predict its first event.')
    st.plotly_chart(process_chart(prefix,options),use_container_width=True)
    st.caption('Branches show the three highest-probability next events; earlier nodes show the observed history. This is a prediction view, not a causal explanation of the LSTM.')
with right:
    st.subheader('🔮 AI next-event forecast')
    st.markdown(f'### {friendly(options[0][0])}')
    st.metric('Top prediction probability',f'{options[0][1]:.1%}')
    top = options[:5]
    chart=go.Figure(go.Bar(x=[p for _,p in top][::-1],y=[friendly(e) for e,_ in top][::-1],orientation='h',marker_color=['#6d80c8','#6d80c8','#6d80c8','#8c79dc','#43d2ba'][::-1],text=[f'{p:.1%}' for _,p in top][::-1],textposition='outside'))
    chart.update_layout(height=295,margin=dict(l=0,r=40,t=0,b=15),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#dbe6fa'),xaxis=dict(range=[0,max(.15,top[0][1]*1.23)],tickformat='.0%',gridcolor='#293950'),yaxis=dict(automargin=True))
    st.plotly_chart(chart,use_container_width=True)
    if invalid>.01: st.warning(f'{invalid:.1%} probability assigned to reserved input tokens. Interpret predictions with care.')

st.divider()
st.subheader('Test the prediction against the recorded next event')
if st.button('Reveal what actually happened',type='primary'):
    st.session_state[f'reveal_{case_id}_{n}']=True
if st.session_state.get(f'reveal_{case_id}_{n}',False):
    correct=options[0][0]==next_event
    if correct: st.success(f'Correct prediction! The recorded next event was: **{friendly(next_event)}**.')
    else: st.info(f'The recorded next event was **{friendly(next_event)}**, while the model favoured **{friendly(options[0][0])}**. Models make mistakes — that is part of the story.')
    actual_p=next((p for e,p in options if e==next_event),0.)
    st.caption(f'Model probability assigned to the recorded outcome: {actual_p:.1%}.')

with st.expander('How it works · Research and limitations'):
    st.write('The LSTM was trained on event sequences from the BPIC2012 loan-application event log. It takes a sequence of observed activity tokens and produces a probability distribution over possible next activities, including process completion.')
    st.write('Training used separate application cases for training, validation and testing. The displayed 52.54% accuracy is top-1 next-event accuracy on held-out prefixes, not a guarantee for any individual case. Probabilities are model scores and have not been calibrated.')
    st.write('The source CSV does not retain reliable full timestamps, so this demo preserves the recorded row order within each case. It does not estimate waiting times or claim causal effects. Case histories in this dashboard are sampled from the original event log.')
    st.write('Research basis: A Deep Learning Approach to Business Process Mining. The showcase interface is a new demonstration of the trained sequence model.')
