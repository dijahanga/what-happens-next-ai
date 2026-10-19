# What Happens Next? — Streamlit demo

An interactive showcase based on a real PyTorch LSTM trained on BPIC2012.

## Run locally

1. Install Python 3.10+.
2. Extract this ZIP into a folder.
3. In a terminal in that folder, run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (normally http://localhost:8501).

## Deploy

Upload the folder's files to a GitHub repository and create a new Streamlit Community Cloud app pointing at `app.py`. Keep `model.pt`, `metadata.json`, and `demo_cases.csv.gz` alongside `app.py`. The dashboard runs inference on CPU; it does not retrain the model.

## Provenance and caveats

- Trained weights and metadata come from the uploaded `what_happens_next_trained_model.zip`.
- Demo histories are sampled from the uploaded BPIC2012.csv (original row order preserved; original case IDs replaced with demo IDs).
- 52.54% top-1 accuracy is the value reported by the training metadata, not a new independent benchmark.
- The network's output probabilities are uncalibrated. The process graphic visualizes prediction options, not neural reasoning or causality.
- This version deliberately does not estimate delays or run counterfactual interventions.

## Showcase tip

Move the slider through a case, reveal the next event, and repeat with another case. Emphasise that the model sometimes gets it wrong, and that transparent limitations matter.
