# gate-length-pilot

![Past their trained length, the models stop retrieving: accuracy falls from 99.9% to 1.6% (toy models) and from 84.5% to 6.5% (TinyLlama-NoPE-1.1B)](readme/fig1_cliff.png)

![Each value stops looking back at its key: the binding heads' attention to the key falls from 0.71 to 0.02 (toy) and from 0.46 to 0.11 (TinyLlama)](readme/fig2_lookback.png)

![Put the look-back back, and retrieval comes back: the oracle patch raises accuracy to 65.0%, 85.5% and 76.1%; the sham lowers it](readme/fig3_rescue.png)

![Nothing else we tried came close: eight other fixes gained at most 1.4 points; the patch gained 74.5](readme/fig4_alternatives.png)

![Rebuilt from scratch, the rescue reappears: independent re-implementations reach 75.7% (toy) and 100% on 18 inputs (TinyLlama)](readme/fig5_reproduced.png)

[Overview, verdicts and how to reproduce](OVERVIEW.md) · figures: `python readme/build.py`
