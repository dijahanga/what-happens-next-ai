# What Happens Next?

A presentation-ready Streamlit demonstration of a PyTorch LSTM trained to predict the next event in BPIC2012 loan-application histories. Inspired by doctoral research on deep learning for business process mining.

## What the screen shows

- A readable journey: event cards in order, plus a lane map of the same path
- A next-step distribution, and how concentrated that distribution has been as the case unfolds
- An audience guess that stays hidden until you ask for the model, when presentation mode is on

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `model.pt`, `metadata.json`, and `demo_cases.csv.gz` beside `app.py`.

## Showcase script

Stay on **Application 115**. Turn on **Presentation mode**.

1. Open on the journey at 12 events. Read the cards left to right, then the map: colour is the kind of step, thickness is how often one step followed another.
2. Move the slider to 18 events. Offer follow-up starts to repeat, and several next steps are plausible. Ask the room what happens next.
3. Open **The forecast**. Let people vote, then show the model. Say clearly that the scores are uncalibrated, and that 52.5% top-1 accuracy is about three times the 17.3% majority baseline — still not a promise.
4. Reveal the recorded step. If the model is wrong, that is part of the point.

## Limits

- The live network's 52.5% is its own reported held-out top-1 accuracy. The dissertation's Model 1 reached 85.4% on BPIC 2012 and 91.2% on Helpdesk. Those are different models.
- Scores on this page are the live network's own outputs. They are not calibrated probabilities.
- Friendly labels are display translations. The model still uses the original activity identifiers.
- The pictures describe the recorded path and the forecast. They do not explain a cause inside the LSTM, a waiting time, or the effect of changing the process.
