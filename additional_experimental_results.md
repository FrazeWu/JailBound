# Additional Experimental Results

## Matched Optimization Baselines

### Octopus-SEval-14B

| Source | Init | GCG | PEZ | ZOL-only | High-Value Branch | Safety-Sensitivity Branch |
|---|---:|---:|---:|---:|---:|---:|
| HarmBench | 11.8 | 35.5 | 30.8 | 47.3 | **53.8** | 48.5 |
| JailBound | 18.3 | 44.4 | 36.7 | 60.4 | **64.5** | 58.0 |
| S-Eval | 20.1 | 41.4 | 37.9 | 56.2 | **60.9** | 58.6 |

### Qwen3-72B Judge

| Source | Init | GCG | PEZ | ZOL-only | High-Value Branch | Safety-Sensitivity Branch |
|---|---:|---:|---:|---:|---:|---:|
| HarmBench | 11.2 | 34.3 | 29.6 | 45.6 | **53.3** | 47.3 |
| JailBound | 16.6 | 43.8 | 36.7 | 60.4 | **62.7** | 57.4 |
| S-Eval | 20.1 | 39.6 | 37.3 | 55.0 | **59.8** | 56.8 |

## Human Evaluation

| Source | Init | ZOL-only | High-Value Branch |
|---|---:|---:|---:|
| JailBound | 11.2 |     36.7 |          **49.7** |
| S-Eval    |  8.9 |     32.5 |          **45.6** |

## Behavioral Flip-Rate Diagnostic

| Source | Radius | Low-FOL BFR | High-FOL BFR | Difference |
|---|---:|---:|---:|---:|
| JailBound | 0.05 | 10.7% | 14.3% | +3.6 pp |
| JailBound | 0.10 | 8.9% | 16.1% | +7.1 pp |
| JailBound | 0.20 | 10.7% | 16.1% | +5.4 pp |
| JailBound | 0.40 | 8.9% | 17.9% | +8.9 pp |
| JailBound | 0.80 | 7.1% | 12.5% | +5.4 pp |
| JailBound | 1.60 | 8.9% | 12.5% | +3.6 pp |
| JailBound | 3.20 | 8.9% | 10.7% | +1.8 pp |
| JailBound | 6.40 | 12.5% | 39.3% | +26.8 pp |
| S-Eval | 0.05 | 7.1% | 21.4% | +14.3 pp |
| S-Eval | 0.10 | 7.1% | 28.6% | +21.4 pp |
| S-Eval | 0.20 | 10.7% | 21.4% | +10.7 pp |
| S-Eval | 0.40 | 17.9% | 17.9% | 0.0 pp |
| S-Eval | 0.80 | 8.9% | 26.8% | +17.9 pp |
| S-Eval | 1.60 | 16.1% | 26.8% | +10.7 pp |
| S-Eval | 3.20 | 8.9% | 32.1% | +23.2 pp |
| S-Eval | 6.40 | 16.1% | 44.6% | +28.6 pp |

## Controlled Automated Evaluation

| Judge | Source | Init | ZOL-only | High-Value Branch |
|---|---|---:|---:|---:|
| Octopus-SEval-14B | JailBound | 18.3 | 60.4 | **64.5** |
| Octopus-SEval-14B | S-Eval | 20.1 | 56.2 | **60.9** |
| Qwen3-72B | JailBound | 16.6 | 60.4 | **62.7** |
| Qwen3-72B | S-Eval | 20.1 | 55.0 | **59.8** |
