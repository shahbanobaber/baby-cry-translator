# Baby cry translator

Predicts why a baby is crying (hunger, discomfort, loneliness) from a log-mel spectrogram, with a small CNN. Tested on babies the model has never heard.

## Data

Lockhart-Bouron et al. 2023, [OSF ru7na](https://osf.io/ru7na/). 39,201 cries, 612 crying episodes, 24 babies aged 0.5 to 3.5 months, recorded at home.

| Cause | Cries | Episodes | Babies |
|---|---|---|---|
| Loneliness | 15,609 | 193 | 24 |
| Hunger | 13,095 | 255 | 23 |
| Discomfort | 10,497 | 164 | 23 |

## Settings

### Data
| Setting | Choice | Why |
|---|---|---|
| Label | Action that stopped the cry | Less of a guess than the parent's belief |
| Classes | Hunger, discomfort, loneliness | Pain (95 cries) and "other" (76) are too few; "don't know" has no label |
| Prediction unit | Episode, averaged over its cries | What a parent actually hears |

### Audio
| Setting | Choice | Why |
|---|---|---|
| Sample rate | 16 kHz | Keeps everything up to 8 kHz, where cry harmonics fade |
| Filter, loudness | Done by the authors (100 Hz high-pass, peak-normalised) | No need to repeat |
| Length | 1.5 s | Covers about 90% of cries whole |
| Short cries | Zero-padded | Silence adds no false signal |
| Long cries | Random 1.5 s crop in training, centre crop in testing | Uses the whole cry over many epochs |

### Spectrogram
| Setting | Choice | Why |
|---|---|---|
| Window | Hann | Lowest leakage between harmonics on a log scale |
| Window length | 512 samples (32 ms) | 31 Hz bins resolve harmonics about 350 Hz apart |
| Hop | 160 samples (10 ms) | Follows pitch changes; standard for speech |
| Mel bands | 64, from 100 Hz to 8 kHz | Common middle ground; matches the filter |
| Decibels | Relative to each cry's peak, 80 dB floor | Mic distance should not change the input |
| Standardise | Mean and spread per band, from training babies only | Test babies stay unseen |
| Input shape | 1 x 64 x 151 | One channel, 64 bands, 151 frames |

### Augmentation (training only)
| Setting | Choice | Why |
|---|---|---|
| Frequency masks | 2, up to 8 bands each | Stops reliance on one narrow band |
| Time masks | 2, up to 20 frames each | Stops reliance on one moment |
| Pitch shift | Off | Pitch may be the very signal |

### Model
| Setting | Choice | Why |
|---|---|---|
| Blocks | 4 x (3x3 conv, batch norm, ReLU, 2x2 max-pool) | Enough depth for a 64 x 151 picture |
| Channels | 16, 32, 64, 128 | About 100k weights; 612 episodes can't support more |
| Head | Global average pool, dropout 0.3, linear to 3 | Same answer wherever the cry sits in time |

### Training
| Setting | Choice | Why |
|---|---|---|
| Loss | Cross-entropy, class-weighted | Hunger has 255 episodes, discomfort 164 |
| Cries per episode per epoch | Up to 32, drawn at random | Each episode counts equally |
| Optimiser | AdamW, learning rate 0.001, weight decay 0.0001 | Reliable default for small CNNs |
| Schedule | Cosine decay, 30 epochs | Smooth finish without tuning |
| Batch size | 64 | Stable batch norm on CPU |
| Early stopping | Best balanced accuracy on validation babies | Validation babies sit outside training |
| Hardware | CPU, 1 thread, low priority | Keeps the laptop usable |

### Evaluation
| Setting | Choice | Why |
|---|---|---|
| Split A | Paper's: by recording session, 66/34, 5 repeats | Direct comparison with their 36% |
| Split B | By baby: 6 folds, 16 train, 4 validation, 4 test babies | The honest test for a new family |
| Main metric | Balanced accuracy per episode; chance is 33% | Fair across unequal classes |
| Also reported | Recall per class, confusion matrix | Shows which causes get confused |
| Uncertainty | 95% interval, bootstrapped over babies | Babies are the independent units |
| Baselines | Chance; majority class; the paper's random forest on the same splits | The CNN has to beat these |
| Experiment 2 | Standardise per baby instead of globally | Removes identity, so only cause is left |
