import io
import math
import re
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="What Happens Next?",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,520;9..144,620&family=Outfit:wght@380;500;620;700&display=swap');
html, body, [class*="css"] { font-family: Outfit, "Segoe UI", sans-serif; }
.stApp {
  color: #eef4ff;
  background-color: #071018 !important;
  background-image:
    radial-gradient(720px 480px at 16% 6%, rgba(62, 224, 197, 0.55), transparent 64%),
    radial-gradient(680px 460px at 88% 10%, rgba(186, 150, 255, 0.52), transparent 62%),
    radial-gradient(640px 440px at 62% 86%, rgba(240, 177, 90, 0.32), transparent 64%),
    radial-gradient(480px 360px at 40% 46%, rgba(110, 176, 255, 0.22), transparent 70%) !important;
  background-attachment: fixed !important;
}
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"],
.block-container { background: transparent !important; }
[data-testid="stSidebar"] {
  background: rgba(8, 16, 30, 0.46) !important;
  border-right: 1px solid rgba(255, 255, 255, 0.14);
  box-shadow: inset -1px 0 0 rgba(255, 255, 255, 0.05);
  backdrop-filter: blur(22px) saturate(1.45);
  -webkit-backdrop-filter: blur(22px) saturate(1.45);
}
[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stToolbarActions"],
[data-testid="stMainMenu"],
[data-testid="stToolbarActionButton"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }
h1 a, h2 a, h3 a { display: none !important; }
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] button {
  visibility: visible !important;
  color: #f4f8ff !important;
}
[data-testid="stExpandSidebarButton"] {
  background: rgba(255, 255, 255, 0.08);
  border: 1px solid rgba(255, 255, 255, 0.22);
  border-radius: 12px;
  backdrop-filter: blur(12px);
}
[data-testid="stExpandSidebarButton"] *,
[data-testid="stSidebarCollapseButton"] * { color: #f4f8ff !important; fill: #f4f8ff !important; }
h1, h2, h3 { font-family: Fraunces, Georgia, serif !important; font-weight: 520 !important; letter-spacing: -0.02em; color: #f7fbff !important; }
.block-container { padding-top: 1.15rem; padding-bottom: 3.2rem; max-width: 1180px; }
[data-testid="stMetric"],
[data-testid="stExpander"] details,
[data-testid="stAlert"],
[data-testid="stPlotlyChart"] {
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.14), rgba(255, 255, 255, 0.04)) !important;
  border: 1px solid rgba(255, 255, 255, 0.26) !important;
  border-radius: 20px !important;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.22);
  backdrop-filter: blur(16px) saturate(1.35);
  -webkit-backdrop-filter: blur(16px) saturate(1.35);
}
[data-testid="stMetric"] { padding: 12px 14px; }
[data-testid="stPlotlyChart"] { padding: 6px 8px 2px; }
[data-testid="stAlert"] { color: #eef4ff; }
div[data-baseweb="select"] > div {
  background: rgba(255, 255, 255, 0.06) !important;
  border-color: rgba(255, 255, 255, 0.18) !important;
  border-radius: 14px !important;
  backdrop-filter: blur(12px);
}
[data-testid="stFileUploader"] section {
  background: rgba(255, 255, 255, 0.04);
  border: 1px dashed rgba(255, 255, 255, 0.28);
  border-radius: 18px;
}
.stButton > button { border-radius: 999px; padding: 0.65rem 1.2rem; font-weight: 650; }
.stButton > button[kind="primary"] {
  background: linear-gradient(180deg, #7af6e2, #2ed3b6);
  color: #042028;
  border: 0;
  box-shadow: 0 10px 28px rgba(62, 224, 197, 0.34);
}
.stButton > button[kind="secondary"] {
  background: rgba(255, 255, 255, 0.07);
  color: #f4f8ff;
  border: 1px solid rgba(255, 255, 255, 0.22);
  backdrop-filter: blur(12px);
}
[data-testid="stMain"] div[data-testid="stRadio"] > div[role="radiogroup"] {
  gap: 0.15rem;
  width: max-content;
  max-width: 100%;
  box-sizing: border-box;
  padding: 12px 18px 14px 10px;
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.06);
  border: 1px solid rgba(255, 255, 255, 0.16);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.2);
  backdrop-filter: blur(16px);
}
[data-testid="stMain"] div[data-testid="stRadio"] [role="radiogroup"] label {
  border-radius: 12px;
  padding: 0.2rem 0.45rem;
  margin: 0;
}
.hero {
  padding: 22px 26px 18px;
  border-radius: 28px;
  background: linear-gradient(155deg, rgba(255, 255, 255, 0.22), rgba(62, 224, 197, 0.08) 42%, rgba(176, 146, 255, 0.12)) !important;
  border: 1px solid rgba(255, 255, 255, 0.34);
  box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.55);
  backdrop-filter: blur(18px) saturate(1.6);
  -webkit-backdrop-filter: blur(18px) saturate(1.6);
}
.eyebrow { color: #b8fff2; letter-spacing: .16em; text-transform: uppercase; font-size: .76rem; font-weight: 700; }
.title { font-family: Fraunces, Georgia, serif; font-size: clamp(2.15rem, 4.4vw, 3.35rem); line-height: 1.02; color: #f7fbff; margin: 8px 0 8px; font-weight: 520; letter-spacing: -0.03em; }
.sub { color: #d5e2f4; font-size: 1.08rem; max-width: 760px; line-height: 1.45; }
.fine { color: #b7c6dc; font-size: .92rem; line-height: 1.45; margin: 12px 2px 0; }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 16px 0 8px; }
.stat, .panel, .card {
  position: relative;
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.16), rgba(255, 255, 255, 0.05));
  border: 1px solid rgba(255, 255, 255, 0.28);
  box-shadow: 0 18px 40px rgba(0, 0, 0, 0.24), inset 0 1px 0 rgba(255, 255, 255, 0.42);
  backdrop-filter: blur(18px) saturate(1.5);
  -webkit-backdrop-filter: blur(18px) saturate(1.5);
}
.stat::before, .panel::before, .card::before {
  content: "";
  position: absolute;
  left: 12%;
  right: 12%;
  top: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.75), transparent);
  pointer-events: none;
}
.stat { border-radius: 20px; padding: 14px 16px 13px; }
.stat b { display: block; font-family: Fraunces, Georgia, serif; font-size: 1.72rem; font-weight: 520; color: #f7fbff; letter-spacing: -0.03em; line-height: 1.05; }
.stat span { color: #c5d3e6; font-size: .9rem; line-height: 1.35; display: block; margin-top: 4px; }
.panel { border-radius: 20px; padding: 14px 16px; color: #e4eefb; line-height: 1.5; }
.legend { display: flex; gap: 16px; flex-wrap: wrap; color: #d5e2f4; font-size: .92rem; margin: 2px 0 8px; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 99px; margin-right: 6px; vertical-align: middle; }
.rail-wrap { padding: 4px 0 8px; }
.rail { display: flex; flex-wrap: wrap; align-items: stretch; gap: 10px; }
.forecast-row { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin: 2px 0 8px; }
.card {
  border-radius: 18px;
  padding: 12px 12px 11px;
  min-width: 132px;
  flex: 1 1 132px;
  max-width: 190px;
  display: flex;
  flex-direction: column;
  gap: 4px;
  transition: transform .18s ease, border-color .18s ease;
}
.card:hover { transform: translateY(-2px); border-color: rgba(255, 255, 255, 0.34); }
.card strong { font-size: 15.5px; line-height: 1.25; color: #f7fbff; font-weight: 650; }
.card .tag { font-size: 10.5px; letter-spacing: .08em; text-transform: uppercase; color: #b7c8de; font-weight: 700; }
.card .meta { font-size: 12.5px; color: #d0dced; }
.card.app { border-top: 2px solid #3ee0c5; }
.card.work { border-top: 2px solid #f0b15a; }
.card.offer { border-top: 2px solid #b6a4ff; }
.card.end { border-top: 2px solid #79b8ff; }
.card.other { border-top: 2px solid #9eb0c8; }
.card.now { box-shadow: 0 0 0 1px rgba(255, 255, 255, 0.45), 0 16px 40px rgba(62, 224, 197, 0.18), inset 0 1px 0 rgba(255, 255, 255, 0.4); }
.card.forecast { min-width: 0; max-width: none; flex: none; background: rgba(16, 36, 48, 0.46); }
.card.forecast.hit { border-color: rgba(121, 184, 255, 0.8); }
.card.forecast.miss { opacity: .5; }
.card.earlier { min-width: 118px; max-width: 128px; border-style: dashed; align-items: center; justify-content: center; text-align: center; color: #d0dced; }
.meter { height: 6px; background: rgba(255, 255, 255, 0.12); border-radius: 99px; overflow: hidden; margin-top: 6px; }
.meter > span { display: block; height: 100%; border-radius: 99px; box-shadow: 0 0 10px rgba(255, 255, 255, 0.35); }
.kicker { color: #b8fff2; letter-spacing: .14em; text-transform: uppercase; font-size: .74rem; font-weight: 700; margin: 18px 0 2px; }
.match { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px; }
.match .stat b { font-size: 1.45rem; }
@media (max-width: 900px) {
  .stats, .match, .forecast-row { grid-template-columns: 1fr 1fr; }
  .title { font-size: 2.15rem; }
  .hero { padding: 18px 16px 14px; }
}
</style>
""",
    unsafe_allow_html=True,
)

APP = "#3ee0c5"
WORK = "#f0b15a"
OFFER = "#b6a4ff"
ENDC = "#79b8ff"
INK = "#d5e2f2"
GRID = "rgba(255,255,255,0.08)"

WORK_LABELS = {
    "W_Afhandelen leads": "Follow up on enquiry",
    "W_Beoordelen fraude": "Review fraud risk",
    "W_Completeren aanvraag": "Complete application details",
    "W_Nabellen incomplete dossiers": "Follow up on missing documents",
    "W_Nabellen offertes": "Follow up on offer",
    "W_Valideren aanvraag": "Validate application",
}
APPLICATION_LABELS = {
    "A_SUBMITTED": "Application submitted",
    "A_PARTLYSUBMITTED": "Application partly submitted",
    "A_PREACCEPTED": "Initial acceptance",
    "A_ACCEPTED": "Application accepted",
    "A_DECLINED": "Application declined",
    "A_CANCELLED": "Application cancelled",
    "A_FINALIZED": "Application finalised",
    "A_APPROVED": "Application approved",
    "A_REGISTERED": "Application registered",
    "A_ACTIVATED": "Application activated",
}
OFFER_LABELS = {
    "O_CREATED": "Offer created",
    "O_SELECTED": "Offer selected",
    "O_SENT": "Offer sent",
    "O_SENT_BACK": "Signed offer returned",
    "O_ACCEPTED": "Offer accepted",
    "O_DECLINED": "Offer declined",
    "O_CANCELLED": "Offer cancelled",
}
SHORT = {
    "A_SUBMITTED": "Submitted",
    "A_PARTLYSUBMITTED": "Partly submitted",
    "A_PREACCEPTED": "Initial acceptance",
    "A_ACCEPTED": "Accepted",
    "A_DECLINED": "Declined",
    "A_CANCELLED": "Cancelled",
    "A_FINALIZED": "Finalised",
    "A_APPROVED": "Approved",
    "A_REGISTERED": "Registered",
    "A_ACTIVATED": "Activated",
    "O_CREATED": "Offer created",
    "O_SELECTED": "Offer selected",
    "O_SENT": "Offer sent",
    "O_SENT_BACK": "Offer returned",
    "O_ACCEPTED": "Offer accepted",
    "O_DECLINED": "Offer declined",
    "O_CANCELLED": "Offer cancelled",
    "W_Afhandelen leads": "Enquiry follow-up",
    "W_Beoordelen fraude": "Fraud review",
    "W_Completeren aanvraag": "Complete details",
    "W_Nabellen incomplete dossiers": "Chase documents",
    "W_Nabellen offertes": "Offer follow-up",
    "W_Valideren aanvraag": "Validate",
    "<END>": "Completed",
    "<START>": "Start",
}


class NextEventLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        return self.output(hidden[-1])


@st.cache_resource
def load_model():
    import json

    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    model = NextEventLSTM(len(metadata["tokens"]), metadata["embedding_dim"], metadata["hidden_dim"])
    model.load_state_dict(torch.load(ROOT / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, metadata


@st.cache_data
def load_cases():
    frame = pd.read_csv(ROOT / "demo_cases.csv.gz", dtype=str)
    grouped = {
        str(case): group["event"].astype(str).tolist()
        for case, group in frame.groupby("case", sort=False)
    }
    return grouped


model, metadata = load_model()
TOKENS = metadata["tokens"]
REVERSE = {index: name for name, index in TOKENS.items()}
CASES = load_cases()
SPECIAL = {"<PAD>", "<START>"}
CASE_IDS = list(CASES)
MAX_UPLOAD_CASES = 200
LOG_PHRASE = "the included extract"
CASE_HEADERS = ("caseconceptname", "caseid", "case")
EVENT_HEADERS = ("conceptname", "activityname", "activity", "event", "taskname", "task")


def build_successors(cases):
    successors = defaultdict(Counter)
    for trace in cases.values():
        previous = None
        for event in trace:
            successors[previous][event] += 1
            previous = event
        successors[previous]["<END>"] += 1
    return successors


def cases_from_frame(frame):
    columns = {normalise_name(name): name for name in frame.columns}
    case_col = next((columns[key] for key in CASE_HEADERS if key in columns), None)
    event_col = next((columns[key] for key in EVENT_HEADERS if key in columns), None)
    if case_col is None or event_col is None or case_col == event_col:
        raise ValueError(
            "The file needs a case column and an activity column. "
            "Accepted names include case, case_id, case:concept:name, activity, event, and concept:name."
        )
    usable = frame[[case_col, event_col]].copy()
    usable[case_col] = usable[case_col].astype(str).str.strip()
    usable[event_col] = usable[event_col].astype(str).str.strip()
    usable = usable[(usable[case_col] != "") & (usable[event_col] != "") & (usable[case_col].str.lower() != "nan")]
    if usable.empty:
        raise ValueError("The file has no events.")
    grouped = {}
    unknown = set()
    for case, group in usable.groupby(case_col, sort=False):
        events = []
        for raw in group[event_col].tolist():
            token = ALIASES.get(normalise_name(raw), raw)
            events.append(token)
            if token not in TOKENS:
                unknown.add(raw)
        if events:
            grouped[str(case)] = events
        if len(grouped) >= MAX_UPLOAD_CASES:
            break
    if not grouped:
        raise ValueError("The file has no events.")
    truncated = usable[case_col].nunique() > MAX_UPLOAD_CASES
    return grouped, sorted(unknown), truncated


@st.cache_data(show_spinner=False)
def read_uploaded(raw, _name):
    text = raw.decode("utf-8-sig")
    first = text.splitlines()[0] if text else ""
    separator = ";" if first.count(";") > first.count(",") else ","
    frame = pd.read_csv(io.StringIO(text), dtype=str, sep=separator)
    return cases_from_frame(frame)


def friendly(event):
    if event == "<END>":
        return "Process completed"
    return WORK_LABELS.get(event) or APPLICATION_LABELS.get(event) or OFFER_LABELS.get(event) or event


def short(event):
    return SHORT.get(event, friendly(event))


def normalise_name(value):
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def activity_aliases():
    aliases = {}
    for token in TOKENS:
        if token in SPECIAL:
            continue
        aliases[normalise_name(token)] = token
        aliases[normalise_name(friendly(token))] = token
        aliases[normalise_name(short(token))] = token
    return aliases


ALIASES = activity_aliases()


def category(event):
    if event.startswith("A_"):
        return "Application"
    if event.startswith("O_"):
        return "Offer"
    if event.startswith("W_"):
        return "Work task"
    if event == "<END>":
        return "Completion"
    if event in TOKENS:
        return "Completion"
    return "Other"


def kind_class(event):
    return {"Application": "app", "Offer": "offer", "Work task": "work", "Completion": "end", "Other": "other"}[category(event)]


def color_of(event):
    return {"Application": APP, "Offer": OFFER, "Work task": WORK, "Completion": ENDC, "Other": "#9eb0c8"}[category(event)]


def predict(prefix):
    if any(event not in TOKENS for event in prefix):
        return [], 0.0
    sequence = [TOKENS["<START>"]] + [TOKENS[event] for event in prefix]
    sequence = sequence[-metadata["max_len"] :]
    # Pad on the right: pack_padded_sequence reads the first `length` positions,
    # so left padding would hide the real history behind padding tokens.
    padded = sequence + [0] * (metadata["max_len"] - len(sequence))
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.tensor([padded], dtype=torch.long))[0], dim=-1).tolist()
    invalid = sum(float(probabilities[TOKENS[token]]) for token in SPECIAL)
    candidates = [
        (REVERSE[index], float(probability))
        for index, probability in enumerate(probabilities)
        if REVERSE[index] not in SPECIAL
    ]
    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return candidates, invalid


def collapse(events):
    runs = []
    for event in events:
        if runs and runs[-1][0] == event:
            runs[-1][1] += 1
        else:
            runs.append([event, 1])
    return runs


def classify(trace):
    seen = set(trace)
    if "A_ACTIVATED" in seen or "A_APPROVED" in seen:
        return "approved"
    return "unsuccessful"


def directly_follows(traces):
    counts = Counter()
    for trace in traces:
        counts.update(zip(trace, trace[1:]))
    return counts


def normalise(counts):
    total = sum(value for value in counts.values() if value > 0)
    if total <= 0:
        return {}
    return {key: value / total for key, value in counts.items() if value > 0}


def js_divergence(left, right):
    keys = set(left) | set(right)
    mixture = {key: 0.5 * (left.get(key, 0.0) + right.get(key, 0.0)) for key in keys}

    def kl(source, target):
        total = 0.0
        for key, value in source.items():
            if value <= 0:
                continue
            total += value * math.log2(value / target[key])
        return total

    return 0.5 * kl(left, mixture) + 0.5 * kl(right, mixture)


def cosine(left, right):
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    dot = sum(left.get(key, 0.0) * right.get(key, 0.0) for key in keys)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def activity_window(traces, opening):
    counts = Counter()
    for trace in traces:
        counts.update(trace[:5] if opening else trace[-5:])
    return normalise(counts)


def relation_shifts(approved_counts, other_counts, limit=7, minimum=20):
    approved_rates = normalise(approved_counts)
    other_rates = normalise(other_counts)
    shifts = []
    for relation in set(approved_counts) | set(other_counts):
        combined = approved_counts[relation] + other_counts[relation]
        if combined < minimum:
            continue
        approved_rate = approved_rates.get(relation, 0.0)
        other_rate = other_rates.get(relation, 0.0)
        shifts.append((abs(approved_rate - other_rate), approved_rate, other_rate, relation, combined))
    shifts.sort(key=lambda item: item[0], reverse=True)
    return shifts[:limit]


def activity_shifts(opening, closing, limit=6):
    shifts = []
    for event in set(opening) | set(closing):
        early = opening.get(event, 0.0)
        late = closing.get(event, 0.0)
        shifts.append((abs(early - late), early, late, event))
    shifts.sort(reverse=True)
    return shifts[:limit]


APPROVED_TRACES = [trace for trace in CASES.values() if classify(trace) == "approved"]
OTHER_TRACES = [trace for trace in CASES.values() if classify(trace) == "unsuccessful"]
APPROVED_DF = directly_follows(APPROVED_TRACES)
OTHER_DF = directly_follows(OTHER_TRACES)
OUTCOME_JS = js_divergence(normalise(APPROVED_DF), normalise(OTHER_DF))
OPENING_DIST = activity_window(list(CASES.values()), True)
CLOSING_DIST = activity_window(list(CASES.values()), False)
LIFECYCLE_JS = js_divergence(OPENING_DIST, CLOSING_DIST)
OUTCOME_SHIFTS = relation_shifts(APPROVED_DF, OTHER_DF)
LIFECYCLE_SHIFTS = activity_shifts(OPENING_DIST, CLOSING_DIST)

SUCCESSORS = build_successors(CASES)


def pct(value, digits=0):
    return f"{value:.{digits}%}"


def apply_theme(fig, height):
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=18, t=36, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, Segoe UI, sans-serif", color=INK, size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#10182a", bordercolor="#31445f", font=dict(color="#f4f7fb", size=13)),
    )
    return fig


def journey_summary(prefix):
    if not prefix:
        return "Nothing has been recorded yet. The forecast can only use the start of an application."
    runs = collapse(prefix)
    longest = max(runs, key=lambda run: run[1])
    distinct = len(set(prefix))
    text = f"So far: {len(prefix)} recorded steps, {distinct} distinct activities."
    if longest[1] >= 3:
        text += f" The longest repeat is {friendly(longest[0])}, {longest[1]} times in a row."
    return text


def timeline_html(prefix, options, show_forecast, revealed, actual):
    limit = 5 if show_forecast else 6
    runs = collapse(prefix)
    hidden_events = 0
    if len(runs) > limit:
        hidden_events = sum(count for _, count in runs[:-limit])
        runs = runs[-limit:]
    parts = ['<div class="rail-wrap"><div class="rail">']
    if hidden_events:
        parts.append(f'<div class="card earlier"><span class="meta">{hidden_events} earlier steps</span></div>')
    step_index = hidden_events + 1
    for position, (event, count) in enumerate(runs):
        end_index = step_index + count - 1
        when = f"step {step_index}" if count == 1 else f"steps {step_index}–{end_index}"
        if count > 1:
            repeat = f'<span class="meta">×{count} in a row · {escape(when)}</span>'
        else:
            repeat = f'<span class="meta">{escape(when)}</span>'
        now = " now" if position == len(runs) - 1 else ""
        parts.append(
            f'<div class="card {kind_class(event)}{now}">'
            f'<span class="tag">{escape(category(event))}</span>'
            f"<strong>{escape(short(event))}</strong>"
            f"{repeat}</div>"
        )
        step_index = end_index + 1
    if not runs:
        parts.append(
            '<div class="card end"><span class="tag">Start</span><strong>No events yet</strong>'
            '<span class="meta">The application has just opened</span></div>'
        )
    parts.append("</div></div>")
    if show_forecast:
        parts.append('<div class="forecast-row">')
        shown = options[:3]
        recorded_shown = False
        for rank, (event, probability) in enumerate(shown):
            state = ""
            tag = f"Forecast · {pct(probability)}"
            if revealed and event == actual:
                state = " hit"
                tag = f"Recorded · {pct(probability)}"
                recorded_shown = True
            elif revealed:
                state = " miss"
            meter = APP if rank == 0 else OFFER if rank == 1 else WORK
            parts.append(
                f'<div class="card forecast {kind_class(event)}{state}">'
                f'<span class="tag">{escape(tag)}</span>'
                f"<strong>{escape(friendly(event))}</strong>"
                f'<div class="meter"><span style="width:{max(4, probability * 100):.1f}%;background:{meter}"></span></div>'
                "</div>"
            )
        if revealed and not recorded_shown:
            given = next((probability for event, probability in options if event == actual), 0.0)
            parts.append(
                f'<div class="card {kind_class(actual)} hit">'
                '<span class="tag">Recorded next step</span>'
                f"<strong>{escape(friendly(actual))}</strong>"
                f'<span class="meta">Model score {pct(given, 1)}</span></div>'
            )
        parts.append("</div>")
    return "".join(parts)


# Fixed places keep the loan process readable: application on the top lane,
# staff work in the middle, offers along the bottom.
STAGE = {
    "A_SUBMITTED": (0.0, 2.0),
    "A_PARTLYSUBMITTED": (1.8, 2.0),
    "A_PREACCEPTED": (3.6, 2.0),
    "W_Afhandelen leads": (3.6, 0.9),
    "W_Completeren aanvraag": (5.4, 0.9),
    "A_ACCEPTED": (5.4, 2.0),
    "A_DECLINED": (5.4, 3.1),
    "A_CANCELLED": (7.2, 3.1),
    "A_FINALIZED": (7.2, 2.0),
    "O_SELECTED": (7.2, -0.2),
    "O_CREATED": (9.0, -0.2),
    "O_SENT": (10.8, -0.2),
    "W_Nabellen offertes": (10.8, 0.9),
    "O_SENT_BACK": (12.6, -0.2),
    "W_Nabellen incomplete dossiers": (12.6, 0.9),
    "W_Beoordelen fraude": (12.6, 2.0),
    "W_Valideren aanvraag": (14.4, 0.9),
    "O_CANCELLED": (14.4, -0.2),
    "A_REGISTERED": (14.4, 2.0),
    "O_DECLINED": (16.2, -0.2),
    "A_APPROVED": (16.2, 2.0),
    "O_ACCEPTED": (16.2, 0.9),
    "A_ACTIVATED": (18.0, 2.0),
    "<END>": (18.0, 0.9),
}


def lane_y(event):
    if event == "<END>":
        return 0.9
    return {"Application": 2.0, "Work task": 0.9, "Offer": -0.2, "Completion": 0.9, "Other": 0.9}[category(event)]


def map_places(prefix, options, show_forecast, revealed, actual):
    positions = {}
    for event in prefix:
        if event not in positions:
            positions[event] = STAGE.get(event, (float(len(positions)) * 1.8, lane_y(event)))
    if show_forecast:
        occupied = Counter()
        anchor = max((point[0] for point in positions.values()), default=0.0)
        for event, _probability in options[:3]:
            if event in positions:
                continue
            if event in STAGE:
                positions[event] = STAGE[event]
                continue
            lane = lane_y(event)
            positions[event] = (anchor + 1.8 + occupied[lane] * 1.8, lane)
            occupied[lane] += 1
    if revealed and actual not in positions:
        positions[actual] = STAGE.get(
            actual,
            (max((point[0] for point in positions.values()), default=0.0) + 1.8, lane_y(actual)),
        )
    return positions


def process_map(prefix, options, show_forecast, revealed, actual, height=420):
    visits = Counter(prefix)
    positions = map_places(prefix, options, show_forecast, revealed, actual)
    fig = go.Figure()
    if not positions:
        fig.add_annotation(text="The map appears once a step is recorded.", showarrow=False, font=dict(color=INK, size=15))
        apply_theme(fig, height)
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return fig

    edges = Counter(zip(prefix, prefix[1:]))
    widest = max(edges.values()) if edges else 1
    for (source, target), count in edges.items():
        if source == target or source not in positions or target not in positions:
            continue
        x1, y1 = positions[source]
        x2, y2 = positions[target]
        if x2 < x1:
            y1 -= 0.16
            y2 -= 0.16
        fig.add_annotation(
            x=x2,
            y=y2,
            ax=x1,
            ay=y1,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.2 + 3.4 * (count / widest),
            arrowcolor=color_of(source),
            standoff=13,
            startstandoff=11,
        )
    if show_forecast and prefix:
        last = prefix[-1]
        linked = {target for (source, target), _count in edges.items() if source == last and source != target}
        for event, probability in options[:3]:
            if event == last or event in linked or event not in positions or last not in positions:
                continue
            x1, y1 = positions[last]
            x2, y2 = positions[event]
            fig.add_trace(
                go.Scatter(
                    x=[x1, x2],
                    y=[y1, y2],
                    mode="lines",
                    line=dict(color="#f4efe4", width=1.5, dash="dash"),
                    hovertemplate=f"Forecast {pct(probability)} · {friendly(event)}<extra></extra>",
                    showlegend=False,
                )
            )
    node_x, node_y, node_color, node_size, node_width, node_edge, hover = [], [], [], [], [], [], []
    for event, (x_pos, y_pos) in positions.items():
        seen = visits[event]
        node_x.append(x_pos)
        node_y.append(y_pos)
        node_color.append(color_of(event))
        current = bool(prefix) and event == prefix[-1]
        node_size.append(28 if current else 20)
        node_width.append(2.6 if current else 0)
        node_edge.append("#f7fbff" if current else color_of(event))
        hover.append(f"{friendly(event)}<br>{seen} visits" if seen else f"{friendly(event)}<br>Not yet in this history")
        label = short(event)
        if " " in label:
            label = label.replace(" ", "<br>", 1)
        if seen > 1:
            label = f"{label}<br>{seen} visits"
        fig.add_annotation(
            x=x_pos,
            y=y_pos,
            text=label,
            showarrow=False,
            yanchor="bottom" if y_pos >= 1.6 else "top",
            yshift=10 if y_pos >= 1.6 else -10,
            font=dict(color="#eef4ff", size=12),
            align="center",
        )
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            marker=dict(size=node_size, color=node_color, line=dict(color=node_edge, width=node_width)),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    if revealed and actual in positions:
        x_pos, y_pos = positions[actual]
        fig.add_trace(
            go.Scatter(
                x=[x_pos],
                y=[y_pos],
                mode="markers",
                marker=dict(size=48, color="rgba(0,0,0,0)", symbol="circle", line=dict(color=ENDC, width=3)),
                hovertemplate=f"Recorded next step · {friendly(actual)}<extra></extra>",
                showlegend=False,
            )
        )
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    apply_theme(fig, height)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=36), showlegend=False)
    fig.update_xaxes(range=[min(xs) - 0.95, max(xs) + 0.95], visible=False)
    fig.update_yaxes(range=[min(ys) - 0.95, max(ys) + 0.95], visible=False)
    return fig


def probability_chart(options, revealed, actual):
    shown = list(reversed(options[:6]))
    colours = []
    for event, _probability in shown:
        if revealed and event == actual:
            colours.append(ENDC)
        else:
            colours.append(color_of(event))
    figure = go.Figure(
        go.Bar(
            x=[probability for _, probability in shown],
            y=[short(event) for event, _ in shown],
            orientation="h",
            marker_color=colours,
            text=[pct(probability, 1) for _, probability in shown],
            textposition="outside",
            hovertemplate="%{y}<br>%{x:.1%}<extra></extra>",
            cliponaxis=False,
        )
    )
    apply_theme(figure, 320)
    figure.update_layout(margin=dict(l=8, r=56, t=12, b=8), showlegend=False)
    leading = shown[-1][1] if shown else 0.2
    figure.update_xaxes(range=[0, max(leading * 1.28, leading + 0.08)], tickformat=".0%", gridcolor=GRID, zeroline=False)
    figure.update_yaxes(automargin=True)
    return figure


def concentration_chart(prefix):
    leading, combined, labels = [], [], []
    for index in range(len(prefix) + 1):
        options, _invalid = predict(prefix[:index])
        if not options:
            figure = go.Figure()
            figure.add_annotation(
                text="This history includes an activity the network was not trained on, so the line stops.",
                showarrow=False,
                font=dict(color=INK, size=15),
            )
            apply_theme(figure, 220)
            figure.update_xaxes(visible=False)
            figure.update_yaxes(visible=False)
            return figure
        leading.append(options[0][1])
        combined.append(sum(probability for _, probability in options[:3]))
        labels.append(short(options[0][0]))
    xs = list(range(len(prefix) + 1))
    marker_size = [7] * (len(xs) - 1) + [13]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=leading,
            mode="lines+markers",
            name="Leading score",
            line=dict(color=APP, width=3),
            marker=dict(size=marker_size, color=APP),
            customdata=labels,
            hovertemplate="After %{x} events<br>Leading score %{y:.1%}<br>%{customdata}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=xs,
            y=combined,
            mode="lines",
            name="Top three combined",
            line=dict(color=OFFER, width=2, dash="dot"),
            hovertemplate="After %{x} events<br>Top three %{y:.1%}<extra></extra>",
        )
    )
    apply_theme(figure, 300)
    figure.update_xaxes(title="Events observed", gridcolor="rgba(0,0,0,0)", zeroline=False)
    figure.update_yaxes(range=[0, 1.02], tickformat=".0%", gridcolor=GRID, zeroline=False)
    return figure


def grouped_bars(labels, left_values, right_values, left_name, right_name, left_color, right_color, height=390):
    figure = go.Figure()
    figure.add_trace(go.Bar(name=left_name, y=labels, x=left_values, orientation="h", marker_color=left_color, hovertemplate="%{y}<br>" + left_name + " %{x:.1%}<extra></extra>"))
    figure.add_trace(go.Bar(name=right_name, y=labels, x=right_values, orientation="h", marker_color=right_color, hovertemplate="%{y}<br>" + right_name + " %{x:.1%}<extra></extra>"))
    apply_theme(figure, height)
    figure.update_layout(barmode="group", margin=dict(l=8, r=18, t=36, b=8))
    figure.update_xaxes(tickformat=".0%", gridcolor=GRID, zeroline=False)
    figure.update_yaxes(automargin=True, categoryorder="array", categoryarray=labels)
    return figure


def relation_label(relation):
    source, target = relation
    if source == target:
        return f"{short(source)} repeats"
    return f"{short(source)} → {short(target)}"


def forecast_reading(options):
    (first_event, first_p), (second_event, second_p) = options[0], options[1]
    margin = first_p - second_p
    top3 = sum(probability for _, probability in options[:3])
    if margin >= 0.22 and first_p >= 0.4:
        tone = f"The distribution is concentrated on {friendly(first_event)}."
    elif margin < 0.08:
        tone = f"The distribution is split between {friendly(first_event)} and {friendly(second_event)}."
    else:
        tone = f"The distribution leans toward {friendly(first_event)}, with {friendly(second_event)} still plausible."
    return (
        f"{tone} The top three possibilities hold {pct(top3)} of the score. "
        "These percentages are the network's own scores, before any calibration."
    )


def resemblance(case_id, prefix):
    full = CASES[case_id]
    own = directly_follows([full])
    references = {}
    for name, base in (("approved", APPROVED_DF), ("unsuccessful", OTHER_DF)):
        counts = base.copy()
        if classify(full) == name:
            counts.subtract(own)
        references[name] = normalise(counts)
    observed = normalise(directly_follows([prefix]))
    return {name: cosine(observed, distribution) for name, distribution in references.items()}


def legend_html(prefix):
    parts = [
        '<div class="legend">',
        f'<span><i class="swatch" style="background:{APP}"></i>Application</span>',
        f'<span><i class="swatch" style="background:{WORK}"></i>Work task</span>',
        f'<span><i class="swatch" style="background:{OFFER}"></i>Offer</span>',
        f'<span><i class="swatch" style="background:{ENDC}"></i>Recorded outcome</span>',
    ]
    if any(category(event) == "Other" for event in prefix):
        parts.append('<span><i class="swatch" style="background:#9eb0c8"></i>Outside the trained vocabulary</span>')
    parts.append("</div>")
    return "".join(parts)


def render_journey(prefix, options, show_forecast, revealed, actual, presentation):
    st.markdown('<div class="kicker">01 · The journey</div>', unsafe_allow_html=True)
    st.subheader("What has happened so far")
    st.markdown(f'<div class="panel">{escape(journey_summary(prefix))}</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(legend_html(prefix), unsafe_allow_html=True)
    st.markdown(timeline_html(prefix, options, show_forecast, revealed, actual), unsafe_allow_html=True)
    if len(prefix) > metadata["max_len"] - 1:
        st.caption("The network is reading the latest 39 steps. The cards and the map still include the earlier history.")
    st.caption(
        "Cards run in time order. A × mark is the same step happening again immediately. "
        "Earlier steps stay in the count when the row is shortened to fit the screen."
    )
    st.plotly_chart(
        process_map(prefix, options, show_forecast, revealed, actual, 390 if presentation else 440),
        width="stretch",
        config={"displayModeBar": False},
        key="process-map" if not presentation else "process-map-talk",
    )
    st.caption(
        "Application steps sit on the top lane, staff work in the middle, and offers along the bottom. "
        "Arrow thickness follows how often that hand-off happened. A repeat is written on the step instead of drawn as a loop. "
        "A dashed line is a forecast toward a step that did not just happen."
    )


def render_forecast(case_id, prefix, options, invalid, revealed, actual, presentation, threshold):
    st.markdown('<div class="kicker">02 · The forecast</div>', unsafe_allow_html=True)
    st.subheader("What the network expects next")
    show_key = f"show_model_{case_id}_{len(prefix)}"
    if revealed:
        st.session_state[show_key] = True
    last = prefix[-1] if prefix else None
    successors = SUCCESSORS[last].most_common(4)
    choices = [event for event, _count in successors]

    voted = False
    if presentation:
        reminder = friendly(prefix[-1]) if prefix else "the opening of the application"
        st.markdown(
            f'<div class="panel">The last recorded step is <b>{escape(reminder)}</b>. '
            "Make your own guess before the model is shown. "
            f"The options are the continuations that follow this step most often in {LOG_PHRASE}.</div>",
            unsafe_allow_html=True,
        )
        st.write("")
        if len(choices) >= 2:
            guess = st.radio(
                "What do you think happens next?",
                choices,
                format_func=friendly,
                key=f"guess_{case_id}_{len(prefix)}",
                disabled=revealed,
            )
            voted = True
        else:
            guess = None
            st.caption(f"At this point {LOG_PHRASE} continues in only one way, so there is little to guess. Move further into the case")
        if not st.session_state.get(show_key, False):
            if st.button("Show the model's forecast", type="secondary"):
                st.session_state[show_key] = True
                st.rerun()
            return
    else:
        guess = None

    if not options:
        st.markdown(
            '<div class="panel">This history includes an activity outside the trained vocabulary, so the network has no score for the next step. '
            "It knows the 24 BPIC 2012 activity names, and the English labels used on this page. The recorded journey above still stands.</div>",
            unsafe_allow_html=True,
        )
        return

    first_event, first_p = options[0]
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.markdown(
            f'<div class="stat"><span class="tag" style="color:#8dffe9">Most likely next step</span>'
            f'<b style="font-size:1.42rem">{escape(friendly(first_event))}</b>'
            f"<span>{pct(first_p, 1)} of the model's score · dataset label {escape(first_event)}</span></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(f'<div class="panel">{escape(forecast_reading(options))}</div>', unsafe_allow_html=True)
    with right:
        st.plotly_chart(
            probability_chart(options, revealed, actual),
            width="stretch",
            config={"displayModeBar": False},
            key="probability-chart",
        )
    if invalid > 0.01:
        st.warning(f"{pct(invalid, 1)} of the score sat on reserved input tokens. Read this forecast with extra care.")

    st.markdown("##### Decision graph")
    graph, blocked = decision_chart(prefix, threshold)
    st.plotly_chart(graph, width="stretch", config={"displayModeBar": False}, key="decision-graph")
    if blocked and not model_continuation(prefix, threshold)[0]:
        st.caption("Lower the decision-graph threshold to let a continuation through. The dissertation used 0.8 on models whose leading scores often reached that height.")
    else:
        st.caption(
            "Each arrow is the activity with the highest score, and the number on it is that score. "
            "This is a single-case graph, drawn with the network running on this page."
        )

    st.markdown("##### How concentrated the forecast has been")
    st.plotly_chart(
        concentration_chart(prefix),
        width="stretch",
        config={"displayModeBar": False},
        key="concentration-chart",
    )
    st.caption(
        "Each point uses only the history up to that moment. A falling line means the next step became harder to separate. "
        "The dotted line is the share held by the three leading possibilities."
    )
    reveal_key = f"reveal_{case_id}_{len(prefix)}"
    if st.button("Reveal the recorded next step", type="primary", disabled=revealed):
        st.session_state[reveal_key] = True
        st.rerun()
    if revealed:
        actual_p = next((probability for event, probability in options if event == actual), 0.0)
        rank = next((index for index, (event, _probability) in enumerate(options, start=1) if event == actual), None)
        if rank is None:
            st.info(f"The recorded next step is {friendly(actual)}. That activity is outside the trained vocabulary, so it has no score.")
            return
        correct = options[0][0] == actual
        if correct:
            st.success(f"The leading forecast matches the record: {friendly(actual)}.")
        else:
            st.info(
                f"The recorded next step is {friendly(actual)}. "
                f"The model gave more of its score to {friendly(options[0][0])}."
            )
        st.caption(f"It was choice {rank} in the model distribution, with a score of {pct(actual_p, 1)}.")
        if presentation and voted and successors:
            usual = successors[0][0]
            human = "Your guess matched the record." if guess == actual else f"You guessed {friendly(guess)}."
            usual_text = (
                "That is also the most common continuation in the extract."
                if usual == actual
                else f"The most common continuation in the extract is {friendly(usual)}."
            )
            st.write(human, usual_text)


def shift_sentence(shifts):
    if not shifts:
        return "The two groups are too small to compare transition by transition."
    _gap, approved_rate, other_rate, relation, _count = shifts[0]
    source, target = relation
    if source == target:
        return (
            f"The sharpest difference is a repeated step. In unsuccessful applications, "
            f"{pct(other_rate, 1)} of transitions are {friendly(source)} following itself. "
            f"In applications that reached approval, that share is {pct(approved_rate, 1)}."
        )
    return (
        f"The sharpest difference is the move from {friendly(source)} to {friendly(target)}: "
        f"{pct(approved_rate, 1)} of transitions where the case reached approval, "
        f"and {pct(other_rate, 1)} where it did not."
    )


def lifecycle_sentence(shifts):
    if not shifts:
        return "The opening and the close of these applications use a similar mix of activities."

    def share(value):
        if value < 0.005:
            return "almost none"
        return pct(value, 1)

    pieces = []
    for _gap, early, late, event in shifts[:3]:
        name = friendly(event)
        if early >= late:
            pieces.append(f"{name} is {share(early)} of the opening and {share(late)} of the close")
        else:
            pieces.append(f"{name} is {share(late)} of the close and {share(early)} of the opening")
    return "The opening and the close are different stages. " + ". ".join(pieces) + "."


def model_continuation(prefix, threshold, horizon=5):
    """Chapter 3: repeatedly take the highest-scoring next activity."""
    path = list(prefix)
    steps = []
    blocked = False
    for _ in range(horizon):
        options, _invalid = predict(path)
        if not options:
            break
        event, probability = options[0]
        if probability < threshold:
            blocked = True
            break
        source = path[-1] if path else "<START>"
        steps.append((source, event, probability))
        if event == "<END>" or (path and event == path[-1]):
            break
        path.append(event)
    return steps, blocked


def decision_chart(prefix, threshold):
    steps, blocked = model_continuation(prefix, threshold)
    figure = go.Figure()
    if not steps:
        figure.add_annotation(
            text="The leading score is under the threshold, so there is no arrow to draw.",
            showarrow=False,
            font=dict(color=INK, size=15),
        )
        apply_theme(figure, 180)
        figure.update_xaxes(visible=False)
        figure.update_yaxes(visible=False)
        return figure, True
    nodes = []
    links = []
    for source, event, probability in steps:
        if not nodes:
            nodes.append(source)
        if event == nodes[-1]:
            links.append((len(nodes) - 1, len(nodes) - 1, probability))
        else:
            nodes.append(event)
            links.append((len(nodes) - 2, len(nodes) - 1, probability))
    for left, right, probability in links:
        if left == right:
            continue
        figure.add_annotation(
            x=right,
            y=0,
            ax=left,
            ay=0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=2,
            arrowwidth=2.2,
            arrowcolor="#f4efe4",
            standoff=16,
            startstandoff=14,
        )
        figure.add_annotation(
            x=(left + right) / 2,
            y=0.18,
            text=pct(probability),
            showarrow=False,
            font=dict(color="#f7fbff", size=13),
        )
    stay = next((probability for left, right, probability in links if left == right), None)
    figure.add_trace(
        go.Scatter(
            x=list(range(len(nodes))),
            y=[0] * len(nodes),
            mode="markers",
            marker=dict(
                size=[30] + [24] * (len(nodes) - 1),
                color=[color_of(event) for event in nodes],
                line=dict(color=["#f7fbff"] + ["#08111d"] * (len(nodes) - 1), width=[2.4] + [0] * (len(nodes) - 1)),
            ),
            text=[friendly(event) for event in nodes],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )
    for index, event in enumerate(nodes):
        label = short(event)
        if " " in label:
            label = label.replace(" ", "<br>", 1)
        if index == len(nodes) - 1 and stay is not None:
            label = f"{label}<br>stays · {pct(stay)}"
        figure.add_annotation(
            x=index,
            y=0,
            text=label,
            showarrow=False,
            yanchor="top",
            yshift=-16,
            font=dict(color="#eef4ff", size=12),
            align="center",
        )
    apply_theme(figure, 250)
    figure.update_layout(margin=dict(l=20, r=20, t=28, b=56), showlegend=False)
    figure.update_xaxes(range=[-0.45, max(len(nodes) - 1, 1) + 0.45], visible=False)
    figure.update_yaxes(range=[-0.85, 0.55], visible=False)
    return figure, blocked


results = metadata["results"]
accuracy = results["test"]["accuracy"]
baseline = results["majority_class_baseline_train"]
lift = accuracy / baseline if baseline else 0

with st.sidebar:
    st.header("Explore an application")
    dataset = st.radio(
        "Dataset",
        ["Included loan applications", "Upload a log"],
        help="The included set is a BPIC 2012 extract. A CSV is read in file order. The network scores the 24 BPIC 2012 activity names, including the English labels on this page.",
    )
    cases = CASES
    dataset_key = "included"
    unknown = []
    truncated = False
    upload = None
    if dataset == "Upload a log":
        upload = st.file_uploader(
            "Event log",
            type=["csv"],
            help="One row per event, in time order. Include a case column and an activity column.",
        )
        if upload is not None:
            try:
                cases, unknown, truncated = read_uploaded(upload.getvalue(), upload.name)
                dataset_key = f"upload-{upload.name}-{upload.size}"
            except ValueError as error:
                st.error(str(error))
                st.caption("The screen is still showing the included loan applications.")
            except Exception:
                st.error("That file could not be read as a CSV event log.")
                st.caption("The screen is still showing the included loan applications.")
        else:
            st.caption("Add a CSV to switch the screen to it. Until then, the included loan applications stay on screen.")
    if dataset_key == "included":
        st.caption("Real sequences from the public BPIC 2012 loan-application log. Names on screen are English display labels.")
        LOG_PHRASE = "the included extract"
    else:
        st.caption(f"{len(cases):,} cases from {upload.name}. Row order in the file is the order of events.")
        LOG_PHRASE = "this uploaded log"
        if truncated:
            st.caption(f"The live view keeps the first {MAX_UPLOAD_CASES} cases so the comparison stays responsive.")
        if unknown:
            preview = ", ".join(unknown[:6])
            extra = "…" if len(unknown) > 6 else ""
            st.caption(
                f"{len(unknown)} activity names are outside this network ({preview}{extra}). "
                "Those steps still appear on the map. Scores need the BPIC 2012 names or the English labels used here."
            )
    SUCCESSORS = build_successors(cases)
    case_ids = list(cases)
    default_case = case_ids.index("115") if dataset_key == "included" and "115" in cases else 0
    case_label = "Application" if dataset_key == "included" else "Case"
    case_id = st.selectbox(
        "Example case",
        case_ids,
        index=default_case,
        format_func=lambda case, known=cases, label=case_label: f"{label} {case} · {len(known[case])} events",
        key=f"case_{dataset_key}",
    )
    trace = cases[case_id]
    max_prefix = len(trace)
    default_prefix = min(12, max_prefix)
    observed = st.slider(
        "Events observed so far",
        0,
        max_prefix,
        default_prefix,
        key=f"prefix_{dataset_key}_{case_id}",
        help="The network reads at most the latest 39 events. The map still shows the whole history you have revealed.",
    )
    threshold = st.slider(
        "Decision-graph threshold",
        0.0,
        0.8,
        0.0,
        0.05,
        help="Predicted transitions below this score are left off the decision graph. The thesis used 0 to keep every transition and 0.8 to keep only the most likely paths.",
    )
    presentation = st.toggle(
        "Guess first",
        value=False,
        help="Shows one section at a time and hides the forecast until you have made your own guess.",
    )
    st.divider()
    st.markdown("**Network**")
    st.caption(
        f"Embedding {metadata['embedding_dim']} · LSTM hidden {metadata['hidden_dim']} · "
        f"input length {metadata['max_len']}. Trained on {results['train_cases']:,} cases, "
        f"tested on {results['test_cases']:,} held-out cases."
    )
    with st.expander("How to read the screen"):
        st.write("The cards are the recorded case in order. The map is that same case: colour is the kind of step, and arrow thickness is how often that hand-off happened.")
        st.write("The decision graph is the continuation the network would write, one highest-scoring step at a time. The number on each arrow is that score. Raising the threshold removes the less likely arrows.")
    with st.expander("Research notes and limits"):
        st.write("The dissertation asks four questions: how to build a stronger LSTM for process mining, how to draw a graph of that model's decisions, how deep learning can detect sudden concept drift, and how to localise the activities involved. The aim is to keep the accuracy of a neural model and the readability of a process graph.")
        st.write("Two architectures are reported. Model 1 is a unidirectional LSTM. Model 2 is bidirectional. Both use an embedding, one recurrent layer, and a dense output with softmax. On the test sets, Model 1 reaches 91.2% next-activity accuracy on the Helpdesk log and 85.4% on BPIC 2012. Model 2 reaches 90.7% and 85.3%. On those two logs the comparison figures are Tax et al. 71.2% and 76.0%, Camargo et al. 78.9% and 78.6%, and Lin et al. 80.8% and 81.4%. Model 1 is preferred in the thesis because Model 2's gain is small beside the extra training time.")
        st.write("The network on this page is a smaller live demonstration of the same next-activity task: embedding size 32, one LSTM of hidden size 64, and a linear output. Its reported held-out top-1 accuracy is 52.5% on 39,716 prefixes, against a 17.3% majority-class baseline. It is not the dissertation model, and its scores are uncalibrated.")
        st.write("Friendly labels are translations. The pictures describe recorded paths and output scores.")

prefix = trace[:observed]
options, invalid = predict(prefix)
next_event = trace[observed] if observed < len(trace) else "<END>"
widget_case = f"{dataset_key}_{case_id}"
reveal_key = f"reveal_{widget_case}_{observed}"
revealed = st.session_state.get(reveal_key, False)
subject = "a real loan application" if dataset_key == "included" else "the case you selected"

st.markdown(
    f"""
<div class="hero">
  <div class="eyebrow">Process intelligence · a live model</div>
  <div class="title">What happens next?</div>
  <div class="sub">A neural network reads {subject} and gives every possible next step a score.</div>
</div>
<div class="stats">
  <div class="stat"><b>{pct(accuracy, 1)}</b><span>Top-1 accuracy on held-out cases</span></div>
  <div class="stat"><b>{pct(baseline, 1)}</b><span>Majority-class baseline</span></div>
  <div class="stat"><b>{lift:.1f}×</b><span>Accuracy against that baseline</span></div>
  <div class="stat"><b>{results['cases']:,}</b><span>Applications in the source log</span></div>
</div>
<div class="fine">My PhD research achieved 85.4% accuracy in predicting the next activity in a business process. This interactive demonstration uses a smaller, separately trained AI model that correctly predicted the next activity in approximately {pct(accuracy, 1)} of its test examples. The percentages shown represent the model's estimates, not guaranteed outcomes.</div>
""",
    unsafe_allow_html=True,
)
st.write("")

if presentation:
    chapter = st.radio(
        "Chapter",
        ["The journey", "The forecast"],
        horizontal=True,
        label_visibility="collapsed",
        key="talk_chapter",
    )
    if chapter == "The journey":
        render_journey(prefix, options, False, False, next_event, True)
    else:
        render_forecast(widget_case, prefix, options, invalid, revealed, next_event, True, threshold)
else:
    render_journey(prefix, options, bool(options), revealed, next_event, False)
    st.write("")
    render_forecast(widget_case, prefix, options, invalid, revealed, next_event, False, threshold)
