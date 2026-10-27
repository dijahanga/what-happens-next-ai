# What Happens Next?

An interactive Streamlit demonstration of a PyTorch LSTM trained to predict the next event in BPIC 2012 loan-application histories. Inspired by doctoral research on deep learning for business process mining.

## What the screen shows

- A readable journey: event cards in order, plus a lane map of the same path
- A next-step distribution, and how concentrated that distribution has been as the case unfolds
- A decision graph: the continuation the network would write, one highest-scoring step at a time
- An optional **Guess first** mode that keeps the forecast hidden until you have made your own guess

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Keep `model.pt`, `metadata.json`, and `demo_cases.csv.gz` beside `app.py`.

## Try it

1. Pick an application and move the **Events observed so far** slider to choose how much of its history the model sees.
2. Read the journey: colour is the kind of step, arrow thickness is how often one step followed another.
3. Turn on **Guess first**, open **The forecast**, pick what you think happens next, then show the model.
4. Reveal the recorded step. When the model is wrong, the ranked scores show how close it came.

A good starting point is **Application 115**: at 15 events the model picks a less common continuation and gets it right; at 35 events (offer accepted) it is split almost evenly between two outcomes.

## Limits

- The live network's 52.5% is its own reported held-out top-1 accuracy, against a 17.3% majority-class baseline. The dissertation's Model 1 reached 85.4% on BPIC 2012 and 91.2% on Helpdesk. Those are different models.
- Scores on this page are the live network's own outputs. They are not calibrated probabilities.
- Friendly labels are display translations. The model still uses the original activity identifiers.
- The pictures describe the recorded path and the forecast. They do not explain a cause inside the LSTM, a waiting time, or the effect of changing the process.

## References

- Hanga, K.M., Kovalchuk, Y. and Gaber, M.M. (2020). A graph-based approach to interpreting recurrent neural networks in process mining. *IEEE Access*, 8, pp. 172923–172938. https://doi.org/10.1109/ACCESS.2020.3025999
- van Dongen, B.F. (2012). BPI Challenge 2012. 4TU.ResearchData. https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f

© 2026 Khadijah Muzzammil Hanga. All rights reserved. This code is shared for viewing and demonstration only. Please contact me before reusing any part of it.
