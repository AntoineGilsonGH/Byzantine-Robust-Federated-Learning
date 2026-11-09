# Byzantine-Robust Federated Learning via Spectral Methods

Can a server detect malicious clients from the **spectrum of the submitted gradients**? This project studies a PCA-based aggregation rule for federated learning under gradient-poisoning (Byzantine) attacks, and benchmarks it against standard robust aggregators using the [ByzFL](https://github.com/LPD-EPFL/byzfl) library.

Research project, École Polytechnique (M2 Data Science), 2026.
**Authors:** Clément Gadeau, Antoine Gilson, Maxime Coppa, Marama Simoneau.

## Idea

Honest gradients concentrate around their mean; Byzantine gradients that are crafted to shift the aggregate leave a signature along the top principal directions of the centred gradients. The **PCA Eigenvalue Aggregator (PCAEA)** projects the client updates on their leading principal components (computed by power iteration or randomised SVD, without forming the covariance matrix), flags the most outlying clients, and averages the rest.

On the theory side, the project analyses detection guarantees in a Gaussian mixture model of honest vs. malicious gradients, in the population and finite-sample regimes. `main_bound_challenge.py` tests the resulting bound empirically: attacks are rescaled so that the gap between honest and malicious means sits at a chosen multiple of the detection threshold, and detection rates (TPR / FNR) are tracked along training.

## What is implemented

- Three spectral aggregators in `utils/aggregators.py`: `PCAEigenvalueAggregator`, `PCAEigenvalueAggregatorV2` (randomised SVD, variance threshold) and `RobustPCAEigenvalueAggregator` (iterative removal)
- Comparison with Average, Krum, Multi-Krum, Trimmed Mean, Median, SMEA, CAF and MDA
- Attacks from ByzFL: sign flipping, Gaussian noise, inner product manipulation (IPM), optimal IPM
- IID and non-IID (Dirichlet, pathological) data splits; MNIST, Fashion-MNIST, CIFAR-10

## Example result

MNIST, 3 honest clients and 1 Byzantine client, inner product manipulation attack:

![Test accuracy under IPM attack](results/defenses/fl_comparison_ALL_CUSTOM_InnerProductManipulation_SGD_1_3.png)

Plain averaging collapses to chance level, while all robust rules, including the three PCA-based ones, recover about 96% test accuracy. In this setting the spectral aggregators are competitive with the state of the art rather than clearly better; more plots (other attacks, non-IID splits) are in `results/`.

## Usage

```bash
pip install -r requirements.txt

python main.py                  # compare aggregators under one attack   (config.py)
python main_attacks.py          # compare attacks for given aggregators  (config_attacks.py)
python run_all_attacks.py       # sweep over attacks
python main_bound_challenge.py  # bound-challenging experiment for PCAEA
```

All parameters (clients, rounds, dataset, attack, aggregators) are set in `config.py` and `config_attacks.py`. Note that `config.py` sets `DEVICE = "cuda"`; change it to `"cpu"` if no GPU is available.
