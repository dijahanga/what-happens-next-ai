# What Happens Next? — Version 2

A presentation-ready Streamlit demonstration of a PyTorch LSTM trained to predict the next event in BPIC2012 loan application histories. Inspired by the research *A Deep Learning Approach to Business Process Mining*.

## What's new

- Connected observed-history path and probability-weighted prediction branches
- A revealable recorded outcome on the same graph
- Human-readable English activity labels while preserving original model tokens
- Presentation mode with an audience guessing experiment
- Clear explanation of the data, limitations and uncalibrated probabilities

## Update your GitHub repository

Replace `app.py` and `README.md` with the versions in this ZIP. The included `model.pt`, `metadata.json`, `demo_cases.csv.gz` and `requirements.txt` are unchanged from Version 1 and may be left in place. Streamlit Community Cloud will redeploy automatically after your GitHub commit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Showcase script (2–3 minutes)

1. “Have you ever applied for something and wondered what happens behind the scenes?”
2. Choose an application and explain the recorded journey.
3. Ask the audience to guess the next step.
4. Show the LSTM's probabilities; emphasise uncertainty.
5. Reveal the actual next event. Whether right or wrong, discuss why a probabilistic forecast is useful but not a guarantee.
6. Close: “The next step is to help people see not only what might happen, but where processes change.”

## Limitations

- Test accuracy is the reported 52.54% top-1 value, not newly verified here.
- This is next-event prediction, not causal explanation, time prediction or drift detection.
- Friendly labels are display translations, not changes to model data.
- Test the deployed app before presenting; keep a backup screen recording.
