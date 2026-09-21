# Sequence Models

> Autoregressive time-series prediction in PyTorch — one-step vs. multi-step forecasting on a synthetic sine wave, and why the second one falls apart.

A linear regression model is trained to predict the next value of a noisy sine
wave from the previous `τ` values. The interesting part isn't the model — it's
the comparison between predicting one step ahead with real history, and
predicting far ahead by feeding the model its own output.

## Run it

Everything lives in [`script`](script) — a single 46-line file.

```bash
pip install d2l torch
```

Paste it into a notebook or run it directly. It produces three plots: the raw
series, one-step predictions, and one-step vs. multi-step.

## The data

A synthetic series of 1,000 points:

```python
self.time = torch.arange(1, T + 1, dtype=torch.float32)
self.x = torch.sin(0.01 * self.time) + torch.randn(T) * 0.2
```

A sine wave plus Gaussian noise. The first 600 points train; the rest are held
out.

## The setup

| Parameter | Value |
|---|---|
| `T` (series length) | 1000 |
| `num_train` | 600 |
| `tau` (context window) | 4 |
| `batch_size` | 16 |
| Model | `d2l.LinearRegression`, lr 0.01 |
| Epochs | 5 |

The autoregressive framing turns a sequence into a supervised problem: each
training example is the four previous values as features, and the next value as
the label. That's what `get_dataloader` builds by stacking `tau` shifted slices
of the series.

## One-step vs. multi-step

**One-step prediction** uses four *real* observations to predict the fifth. It
looks excellent — the predicted curve sits almost on top of the labels.

**Multi-step prediction** only gets real data up to the end of training, then
has to feed its own predictions back in as input:

```python
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(multistep_preds[i - data.tau:i].reshape((1, -1)))
```

This degrades fast. Each prediction carries a small error; the next prediction
takes that error as input and adds its own, and the compounding is exponential
rather than linear. The forecast drifts away from the true series and usually
collapses toward a constant.

That gap is the point of the exercise. A model that looks accurate on one-step
metrics can be useless for actual forecasting, and the only way to see it is to
run the closed loop.

## Credit

Built on the [d2l](https://d2l.ai/) library from *Dive into Deep Learning*,
which supplies `DataModule`, `LinearRegression`, `Trainer`, and the plotting
helpers.
