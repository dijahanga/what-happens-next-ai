# What Happens Next? — Version 2

A presentation-ready Streamlit demonstration of a PyTorch LSTM trained to predict the next event in BPIC2012 loan application histories. Inspired by the research *A Deep Learning Approach to Business Process Mining*.

## What's new

- Connected observed-history path and probability-weighted prediction branches
- A revealable recorded outcome on the same graph
- Human-readable English activity labels while preserving original model tokens
- Clear explanation of the data, limitations and uncalibrated probabilities

## Update your GitHub repository

Replace `app.py` and `README.md` with the versions in this ZIP. The included `model.pt`, `metadata.json`, `demo_cases.csv.gz` and `requirements.txt` are unchanged from Version 1 and may be left in place. Streamlit Community Cloud will redeploy automatically after your GitHub commit.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```


