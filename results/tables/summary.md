## Primary spec (pre-declared)

```
                         spec        x   coef     se      t   p    n     r2  adj_r2  n_firms  status
PRIMARY: FinBERT -> CAR[0,+1] fb_net_z 0.0088 0.0017 5.1454 0.0 2050 0.0428  0.0174       60 primary
```

## Horse race, CAR[0,+1] (z-scored tone; SE clustered by firm)

```
        spec        x   coef     se      t      p    n  adj_r2
FinBERT only fb_net_z 0.0088 0.0017 5.1454 0.0000 2050  0.0174
     LM only lm_net_z 0.0100 0.0019 5.3179 0.0000 2050  0.0224
        Both fb_net_z 0.0028 0.0019 1.4873 0.1369 2050  0.0227
        Both lm_net_z 0.0079 0.0022 3.6203 0.0003 2050  0.0227
```
