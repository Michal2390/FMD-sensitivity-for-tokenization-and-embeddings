# Paper-Oriented FMD Benchmark Summary

## Pairwise comparisons

All valid rows: **896**
Real-only rows: **896**

## Ranking stability by reference dataset

- `maestro`: 0.6933
- `midicaps`: 0.6757
- `pop909`: 0.6215
- `jazz`: 0.6164
- `rock`: 0.5565
- `classical`: 0.5804
- `pop`: 0.6106
- `rap`: 0.6532

## Expected-order agreement

- variant `tok=REMI|model=CLaMP-1|vel=on|quant=off`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=on|quant=off`, ref `pop909` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=off|quant=off`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=off|quant=off`, ref `pop909` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=on|quant=on`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=on|quant=on`, ref `pop909` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=off|quant=on`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-1|vel=off|quant=on`, ref `pop909` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-2|vel=on|quant=off`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-2|vel=on|quant=off`, ref `pop909` -> spearman=1.0, kendall=1.0
- variant `tok=REMI|model=CLaMP-2|vel=off|quant=off`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-2|vel=off|quant=off`, ref `pop909` -> spearman=1.0, kendall=1.0
- variant `tok=REMI|model=CLaMP-2|vel=on|quant=on`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-2|vel=on|quant=on`, ref `pop909` -> spearman=1.0, kendall=1.0
- variant `tok=REMI|model=CLaMP-2|vel=off|quant=on`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=REMI|model=CLaMP-2|vel=off|quant=on`, ref `pop909` -> spearman=1.0, kendall=1.0
- variant `tok=TSD|model=CLaMP-1|vel=on|quant=off`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=TSD|model=CLaMP-1|vel=on|quant=off`, ref `pop909` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=TSD|model=CLaMP-1|vel=off|quant=off`, ref `maestro` -> spearman=0.5, kendall=0.33333333333333337
- variant `tok=TSD|model=CLaMP-1|vel=off|quant=off`, ref `pop909` -> spearman=0.5, kendall=0.33333333333333337

## Special genre-pair separability

- `classical vs rap` -> mean FMD=0.3207, std=0.2518, ratio=1.219
- `rock vs classical` -> mean FMD=0.3133, std=0.2310, ratio=1.191
- `jazz vs classical` -> mean FMD=0.3026, std=0.2433, ratio=1.150
- `rap vs pop` -> mean FMD=0.2759, std=0.2070, ratio=1.049
- `rock vs rap` -> mean FMD=0.2746, std=0.2195, ratio=1.044
- `jazz vs rock` -> mean FMD=0.2632, std=0.2330, ratio=1.001
- `jazz vs rap` -> mean FMD=0.2393, std=0.1843, ratio=0.910
- `jazz vs pop` -> mean FMD=0.2349, std=0.1897, ratio=0.893
- `rock vs pop` -> mean FMD=0.2259, std=0.1651, ratio=0.859
- `classical vs pop` -> mean FMD=0.1800, std=0.1304, ratio=0.684

## Top separating variants per special pair

- `classical vs pop` rank 1: `tok=Octuple|model=CLaMP-2|vel=on|quant=off` (FMD=0.4331)
- `classical vs pop` rank 2: `tok=Octuple|model=CLaMP-2|vel=on|quant=on` (FMD=0.4331)
- `classical vs pop` rank 3: `tok=Octuple|model=CLaMP-2|vel=off|quant=off` (FMD=0.4189)
- `classical vs pop` rank 4: `tok=Octuple|model=CLaMP-2|vel=off|quant=on` (FMD=0.4189)
- `classical vs pop` rank 5: `tok=TSD|model=CLaMP-2|vel=on|quant=off` (FMD=0.3155)
- `classical vs rap` rank 1: `tok=TSD|model=CLaMP-2|vel=off|quant=off` (FMD=0.8195)
- `classical vs rap` rank 2: `tok=TSD|model=CLaMP-2|vel=off|quant=on` (FMD=0.8195)
- `classical vs rap` rank 3: `tok=TSD|model=CLaMP-2|vel=on|quant=off` (FMD=0.8040)
- `classical vs rap` rank 4: `tok=TSD|model=CLaMP-2|vel=on|quant=on` (FMD=0.8040)
- `classical vs rap` rank 5: `tok=Octuple|model=CLaMP-2|vel=on|quant=off` (FMD=0.5033)
- `jazz vs classical` rank 1: `tok=TSD|model=CLaMP-2|vel=on|quant=off` (FMD=0.8007)
- `jazz vs classical` rank 2: `tok=TSD|model=CLaMP-2|vel=on|quant=on` (FMD=0.8007)
- `jazz vs classical` rank 3: `tok=TSD|model=CLaMP-2|vel=off|quant=off` (FMD=0.7396)
- `jazz vs classical` rank 4: `tok=TSD|model=CLaMP-2|vel=off|quant=on` (FMD=0.7396)
- `jazz vs classical` rank 5: `tok=Octuple|model=CLaMP-2|vel=on|quant=off` (FMD=0.5601)
- `jazz vs pop` rank 1: `tok=TSD|model=CLaMP-2|vel=on|quant=off` (FMD=0.6641)
- `jazz vs pop` rank 2: `tok=TSD|model=CLaMP-2|vel=on|quant=on` (FMD=0.6641)
- `jazz vs pop` rank 3: `tok=TSD|model=CLaMP-2|vel=off|quant=off` (FMD=0.6321)
- `jazz vs pop` rank 4: `tok=TSD|model=CLaMP-2|vel=off|quant=on` (FMD=0.6321)
- `jazz vs pop` rank 5: `tok=Octuple|model=CLaMP-2|vel=on|quant=off` (FMD=0.3323)
- `jazz vs rap` rank 1: `tok=TSD|model=CLaMP-2|vel=off|quant=off` (FMD=0.5928)
- `jazz vs rap` rank 2: `tok=TSD|model=CLaMP-2|vel=off|quant=on` (FMD=0.5928)
- `jazz vs rap` rank 3: `tok=TSD|model=CLaMP-2|vel=on|quant=off` (FMD=0.5696)
- `jazz vs rap` rank 4: `tok=TSD|model=CLaMP-2|vel=on|quant=on` (FMD=0.5696)
- `jazz vs rap` rank 5: `tok=Octuple|model=CLaMP-2|vel=off|quant=off` (FMD=0.4651)
- `jazz vs rock` rank 1: `tok=TSD|model=CLaMP-2|vel=off|quant=off` (FMD=0.7773)
- `jazz vs rock` rank 2: `tok=TSD|model=CLaMP-2|vel=off|quant=on` (FMD=0.7773)
- `jazz vs rock` rank 3: `tok=TSD|model=CLaMP-2|vel=on|quant=off` (FMD=0.7737)
- `jazz vs rock` rank 4: `tok=TSD|model=CLaMP-2|vel=on|quant=on` (FMD=0.7737)
- `jazz vs rock` rank 5: `tok=Octuple|model=CLaMP-2|vel=off|quant=off` (FMD=0.4045)

## Variant effects (delta FMD)

Tokenizer deltas rows (all): **48**
Model deltas rows (all): **16**
Tokenizer deltas rows (real-only): **48**
Model deltas rows (real-only): **16**
- model `CLaMP-1` (False/False): `MIDI-Like` - `Octuple` = -0.0389
- model `CLaMP-1` (False/False): `MIDI-Like` - `REMI` = 0.0039
- model `CLaMP-1` (False/False): `MIDI-Like` - `TSD` = -0.0029
- model `CLaMP-1` (False/False): `Octuple` - `REMI` = 0.0428
- model `CLaMP-1` (False/False): `Octuple` - `TSD` = 0.0360
- model `CLaMP-1` (False/False): `REMI` - `TSD` = -0.0068
- model `CLaMP-1` (False/True): `MIDI-Like` - `Octuple` = -0.0389
- model `CLaMP-1` (False/True): `MIDI-Like` - `REMI` = 0.0039
- tokenizer `MIDI-Like` (False/False): `CLaMP-1` - `CLaMP-2` = -0.3360
- tokenizer `MIDI-Like` (False/True): `CLaMP-1` - `CLaMP-2` = -0.3360
- tokenizer `MIDI-Like` (True/False): `CLaMP-1` - `CLaMP-2` = -0.3318
- tokenizer `MIDI-Like` (True/True): `CLaMP-1` - `CLaMP-2` = -0.3318
- tokenizer `Octuple` (False/False): `CLaMP-1` - `CLaMP-2` = -0.3724
- tokenizer `Octuple` (False/True): `CLaMP-1` - `CLaMP-2` = -0.3724
- tokenizer `Octuple` (True/False): `CLaMP-1` - `CLaMP-2` = -0.3639
- tokenizer `Octuple` (True/True): `CLaMP-1` - `CLaMP-2` = -0.3639
