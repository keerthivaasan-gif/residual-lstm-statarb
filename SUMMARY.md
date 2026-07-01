# What this project is 

This started as a simple question I had while learning quant finance: **can a neural
network beat the classic textbook mean-reversion strategy?** The textbook approach
(Avellaneda-Lee) is linear and has been around forever. LSTMs are the fancy modern
tool everyone talks about. A few papers claim the LSTM wins. I wanted to check that
for myself instead of taking it on faith.

So I built a full pipeline from scratch to test it honestly.

## The idea in plain terms

Stat-arb (statistical arbitrage) doesn't bet on the market going up or down. It bets
on stocks moving *relative to each other*. If two similar stocks usually move
together and one suddenly lags, you buy the laggard and short the leader, betting the
gap closes. To do this you first have to strip out the "everybody moved together"
part (the market and sector effects) and look at what's left over for each stock —
the **residual**. That leftover is what mean-reverts, and that's what we trade.

I cleaned out the common factors using Random Matrix Theory (a neat trick that tells
you which correlations are real signal and which are just noise), then fed the
leftover residuals to three models:

1. **s-score** — the classic linear textbook model.
2. **Gradient-boosted trees** — nonlinear, but has no memory of the sequence.
3. **LSTM** — nonlinear *and* remembers the recent path of the residual.

All three trade on the exact same residuals, so it's a fair fight.

## The part I'm actually proud of: not fooling myself

The easiest thing in the world is to build a backtest that looks amazing because it's
secretly cheating (using future information it wouldn't have had in real life). Most
of my effort went into *not* doing that:

- I wrote a test that literally checks the model's residual for a given day is
  identical whether or not future data exists — proving no future leaks in.
- I used **purged, embargoed walk-forward validation**, which puts gaps between the
  training and testing periods so overlapping data can't leak either.
- I built a fake "synthetic" market where I *know* the true answer, so I could check
  each step was doing what it claimed. This actually caught two real bugs I'd have
  never found otherwise.

## What I found (and it surprised me)

**On US large-cap stocks:** honestly, nobody won convincingly. The linear model was
weak, the LSTM was the only one that made money after trading costs — but just barely,
and it fell apart if I assumed costs were even a bit higher. Large-cap US stocks are
just really efficient; there's not much free money lying around.

**On crypto:** the story flipped. The linear textbook model *failed completely* — it
actually lost money before costs. The nonlinear models were the only ones that found
anything. This lines up with the idea that fancier models help most in messier, less
efficient markets — not on the most heavily-traded stocks in the world.

**A mistake I'm keeping in the writeup on purpose:** I tried an extra "cleaning" step
that I was sure would help. It made everything worse. It scrubbed out real signal
along with the noise. I left it in the paper as a negative result because that's more
honest than quietly deleting it.

## The biggest lesson: gross vs. net

The single most important thing I learned is that a good *signal* and a good *strategy*
are not the same thing. The trees model had the best raw predictions of all three —
but it traded so often that trading fees turned a winning signal into a losing
strategy. The LSTM had weaker predictions but traded less, so it actually kept its
edge. **How often you trade can matter more than how right you are.**

## What I know is still missing

I'm not going to pretend this is production-ready:

- My data is free and only includes companies still alive today (survivorship bias),
  so the real numbers are probably a bit optimistic.
- My trading-cost model is simple. When I added a realistic one, the strategy's
  capacity turned out to be tiny (~$10M before it stops working).
- The datasets are small by real industry standards, and I'd want to run more seeds
  to be sure the rankings are stable.

## TL;DR

I built an honest head-to-head of a classic quant strategy vs. machine learning,
went out of my way to avoid the usual backtest traps, and found that the ML edge
shows up in messy markets (crypto) but mostly not in efficient ones (US stocks) —
and that trading costs quietly decide the winner more than the models do. I learned
more from the negative results than I expected to.
