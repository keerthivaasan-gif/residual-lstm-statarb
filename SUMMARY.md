# Project Idea

This started as a simple question I had while learning quant finance: **can a neural
network beat the classic textbook mean-reversion strategy?** The textbook approach
(Avellaneda-Lee) is linear and has been around forever, while LSTMs are the fancy
modern tool everyone talks about. A few papers claim the LSTM has a strong edge as an
alpha source. I wanted to check that for myself on crypto and equity markets.

So I built a full pipeline from scratch to test it.


Stat-arb (statistical arbitrage) doesn't bet on the market going up or down. It bets
on stocks moving *relative to each other*. If two similar stocks usually move
together and one suddenly lags, you buy the laggard and short the leader, betting the
gap closes. To do this you first strip out the "everybody moved together" part (the
market and sector effects) and look at what's left over for each stock, which is
the **residual**. That leftover is what mean-reverts, and that's what we trade.

I cleaned out the common factors using Random Matrix Theory (a neat trick that tells
you which correlations are real signal and which are just noise), then fed the
leftover residuals to three models:

1. **s-score**: the classic linear textbook model.
2. **Gradient-boosted trees**: nonlinear, but with no memory of the sequence.
3. **LSTM**: nonlinear, and it remembers the recent path of the residual.

All three trade on the exact same residuals, so it's a fair fight.

## Steps

The easiest thing in the world is to build a backtest that looks amazing because it's
secretly cheating (using future information it wouldn't have had in real life). Most
of my effort went into *not* doing that:

- I wrote a test that checks a given day's residual is identical whether or not future
  data exists, which proves no future information leaks in.
- I used **purged, embargoed walk-forward validation**, which puts gaps between the
  training and testing periods so overlapping data can't leak either.
- I built a fake "synthetic" market where I *know* the true answer, so I could check
  each step was doing what it claimed. This actually caught two real bugs I'd have
  never found otherwise.

## Results

**On US large-cap stocks:** honestly, nobody won convincingly. The linear model was
weak. The LSTM was the only one that made money after trading costs, and only barely,
and it fell apart once I assumed costs were even a bit higher. Large-cap US stocks are
just really efficient, so there's not much free money lying around.

**On crypto:** the story flipped. The linear textbook model *failed completely* and
actually lost money before costs. The nonlinear models were the only ones that found
anything. This lines up with the idea that fancier models help most in messier, less
efficient markets, not on the most heavily-traded stocks in the world.

**A mistake I'm keeping in the writeup on purpose:** I tried an extra "cleaning" step
that I was sure would help. It made everything worse. It scrubbed out real signal
along with the noise. I left it in the paper as a negative result because that's more
honest than quietly deleting it.

## Conclusion

The single most important thing I learned is that a good *signal* and a good *strategy*
are not the same thing. The trees model had the best raw predictions of all three, but
it traded so often that trading fees turned a winning signal into a losing strategy.
The LSTM had weaker predictions but traded less, so it actually kept its edge. **How
often you trade can matter more than how right you are.**

## Future Directions

- My data is free and only includes companies still alive today (survivorship bias),
  so the real numbers are probably a bit optimistic.
- My trading-cost model is simple. When I added a realistic one, the strategy's
  capacity turned out to be tiny (~$10M before it stops working).
- The datasets are small by real industry standards, and I'd want to run more seeds
  to be sure the rankings are stable.
