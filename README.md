# NVIDIA CEO quantile forecast

An entry for the make.fun bounty `polymarket-nvidia-ceo-forecast-01`: forecast the Yes price of every "Next CEO of NVIDIA?" Polymarket market minute by minute for the next 4 hours, as the 5%, 25%, 50%, 75% and 95% quantiles. Scoring is the mean pinball loss.

## How the forecast works

Both paths share the same volatility features, computed from the market's own recent prices (the first 6 hours of a market's life are skipped, because the opening auction is not a guide to later volatility):

- the mean absolute h-minute move over windows of 2, 6 and 24 hours;
- exponentially weighted averages of absolute 1-minute moves (half-lives 15, 60 and 240 minutes), scaled by the square root of the horizon.

Offsets are computed on a grid of horizons and interpolated linearly in between. Upper quantiles are added to the median. Lower ones are multiplicative (`m * exp(-offset / m)`), so they never cross zero.

### Mature markets (at least 1500 prices in the case)

- **Median:** the last observed price (persistence).
- **Fan width:** a non-negative linear blend of the volatility features. The weights were fitted offline by non-negative linear quantile regression (pinball loss) on rolling history of this event up to 2026-11-13 13:40 UTC, and are stored as constants.

### Young markets (fewer than 1500 prices, under about 25 hours of trading)

A market in its opening phase behaves differently: prices decay from wide opening quotes, the sum of all Yes prices in the event is far from 1 and snaps back, and moves are much larger. These markets get their own parameters, fitted offline on the six practice cases, with separate sets for markets under 500, 900 and 1500 prices.

- **Median:** the last price plus a fitted shift that grows with the horizon. The shift depends on the price level, on how far the medians of the last 5, 30 and 120 prices sit from the last price, and on how far the event's Yes prices sum from 1.
- **Fan width:** separate fitted blends for each of the four outer quantiles. The features are the six volatility features plus a price-level term and a constant floor, both scaled by the square root of the horizon.
- **Newborn markets** (too little history for the features, after the opening skip): a fan proportional to the price that widens with the square root of the horizon and is skewed towards the downside.

In the final, every market is expected to have at least 1598 prices, so the young path only matters on practice-like cases. Final scores come from the mature path.

### Safety

Every row is clamped to [0, 1], rounded to 6 decimals and made non-decreasing. If anything goes wrong for one market, that market falls back to a fixed square-root-of-time fan around its last price, and the rest of the case is unaffected.

The program uses the Python standard library only. It has no randomness, clock or network access, so two runs write identical bytes.

## Running locally

```sh
MAKEFUN_INPUT_DIR=path/to/input MAKEFUN_OUTPUT_DIR=out python main.py
python validate.py path/to/input out
```

`main.py` reads every `*.in` file in the input directory (default `/input`) and writes one `<id>.out` per case to the output directory (default `/output`). `validate.py` checks each output against the bounty's shape rules and exits non-zero on any violation.

## License

MIT, see [LICENSE](LICENSE).
