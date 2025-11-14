# Optimal Auctions through Deep Learning
Implementation of "Optimal Auctions through Deep Learning" (https://arxiv.org/pdf/1706.03459.pdf)

## Getting Started

### Prerequisites
- Python 3.8+ (tested with Python 3.12.8)
- PyTorch (>=2.1.0, tested with PyTorch 2.9.0)
- Numpy and Matplotlib packages
- Easydict - `pip install easydict`

### Installation

#### Using Virtual Environment
```bash
# Create and activate virtual environment
source venv/bin/activate
```

#### Installation
```bash
# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Running the experiments

### RegretNet

#### For Gradient-Based approach:
Default hyperparameters are specified in regretNet/cfgs/.  

#### For Sample-Based approach:
Modify the following hyperparameters in the config file specified in regretNet/cfgs/.
```
cfg.train.gd_iter = 0
cfg.train.num_misreports = 100
cfg.val.num_misreports = 100 # Number of val-misreports is always equal to the number of train-misreports
```

For training the network, testing the mechanism learnt and computing the baselines, run:
```bash
cd regretNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```

#### Resuming Training from Checkpoint
To resume training from a saved checkpoint, modify `restore_iter` in the config file:
```python
__C.train.restore_iter = 180000  # Resume from iteration 180000
```
Then run training as usual. Note: This requires `save_data = True` in the config file to load training data files (`X.npy` and `ADV_*.npy`).

#### Data Saving for Restore
Set `save_data = True` in the config file to save training data, enabling checkpoint resumption:
```python
__C.save_data = True  # Saves X.npy and ADV_*.npy files
```

setting\_no  |      setting\_name |
 :---:   | :---: |
  (a)    |  additive\_1x2\_uniform |
  (b)   | unit\_1x2\_uniform\_23 |
  (c\)  | additive\_2x2\_uniform |
  (d)   | CA\_sym\_uniform\_12 |
  (e)    | CA\_asym\_uniform\_12\_15 |
  (f)   | additive\_3x10\_uniform |
  (g)  | additive\_5x10\_uniform |
  (h) |   additive\_1x2\_uniform\_416\_47
  (i) |   additive\_1x2\_uniform\_triangle
  (j) |   unit\_1x2\_uniform
  (k) |  additive\_1x10\_uniform
  (l) |   additive\_1x2\_uniform\_04\_03
  (m) |   unit\_2x2\_uniform
  (n) |   additive\_1x2\_beta\_11
  (o) |   additive\_1x2\_gamma\_11
  (p) |   additive\_1x2\_gamma\_21
  (q) |   additive\_1x2\_gamma\_31
  (r) |   additive\_1x3\_constrained\_c0\_1 | 3 bidders, 3 items, constrained allocation (c=0.1) |
  (s) |   additive\_1x3\_constrained\_c0\_5 | 3 bidders, 3 items, constrained allocation (c=0.5) |
  (t) |   additive\_1x3\_constrained\_c1 | 3 bidders, 3 items, constrained allocation (c=1.0) |
  (u) |   additive\_1x3\_constrained\_c10 | 3 bidders, 3 items, constrained allocation (c=10.0) |


### RochetNet (Single Bidder Auctions)

Default hyperparameters are specified in rochetNet/cfgs/.  
For training the network, testing the mechanism learnt and computing the baselines, run:
```bash
cd rochetNet
python run_train.py [setting_name]
python run_test.py [setting_name]
python run_baseline.py [setting_name]
```

setting\_no  |      setting\_name | Description |
 :---:  | :---: | :--- |
  (a)   |  additive\_1x2\_uniform | Single additive bidder, 2 items, U[0,1] |
  (b)   |   additive\_1x2\_uniform\_416\_47 | Single additive bidder, 2 items, U[4,16]×U[4,7] |
  \(c\) |   additive\_1x2\_uniform\_triangle | Single additive bidder, 2 items, uniform triangle |
  (d)   |   additive\_1x2\_uniform\_04\_03 | Single additive bidder, 2 items, U[0,4]×U[0,3] |
  (e)   |  additive\_1x10\_uniform | Single additive bidder, 10 items, U[0,1] |
  (f)   |   additive\_1x3\_uniform\_0\_1 | Single additive bidder, 3 items, U[0,1] |
  (g)   |   unit\_1x2\_uniform | Single unit-demand bidder, 2 items, U[0,1] |
  (h)   |   unit\_1x2\_uniform\_23 | Single unit-demand bidder, 2 items, U[2,3] |
  (i)   |   additive\_1x2\_beta\_11 | Single additive bidder, 2 items, Beta(1,1) |
  (j)   |   additive\_1x2\_beta\_12 | Single additive bidder, 2 items, Beta(1,2) |
  (k)   |   additive\_1x2\_beta\_13 | Single additive bidder, 2 items, Beta(1,3) |
  (l)   |   additive\_1x2\_beta\_21 | Single additive bidder, 2 items, Beta(2,1) |
  (m)   |   additive\_1x2\_beta\_31 | Single additive bidder, 2 items, Beta(3,1) |
  (n)   |   additive\_1x2\_gamma\_0\_1 | Single additive bidder, 2 items, Gamma(0.1, 1.0) |
  (o)   |   additive\_1x2\_gamma\_0\_2 | Single additive bidder, 2 items, Gamma(0.2, 1.0) |
  (p)   |   additive\_1x2\_gamma\_0\_3 | Single additive bidder, 2 items, Gamma(0.3, 1.0) |
  (q)   |   additive\_1x2\_gamma\_0\_4 | Single additive bidder, 2 items, Gamma(0.4, 1.0) |
  (r)   |   additive\_1x2\_gamma\_0\_5 | Single additive bidder, 2 items, Gamma(0.5, 1.0) |
  (s)   |   additive\_1x2\_gamma\_0\_6 | Single additive bidder, 2 items, Gamma(0.6, 1.0) |
  (t)   |   additive\_1x2\_gamma\_0\_7 | Single additive bidder, 2 items, Gamma(0.7, 1.0) |
  (u)   |   additive\_1x2\_gamma\_0\_8 | Single additive bidder, 2 items, Gamma(0.8, 1.0) |
  (v)   |   additive\_1x2\_gamma\_0\_9 | Single additive bidder, 2 items, Gamma(0.9, 1.0) |
  (w)   |   additive\_1x2\_gamma\_1\_0 | Single additive bidder, 2 items, Gamma(1.0, 1.0) |
  (x)   |   additive\_1x2\_gamma\_1\_1 | Single additive bidder, 2 items, Gamma(1.1, 1.0) |
  (y)   |   additive\_1x2\_gamma\_1\_2 | Single additive bidder, 2 items, Gamma(1.2, 1.0) |
  (z)   |   additive\_1x2\_gamma\_1\_3 | Single additive bidder, 2 items, Gamma(1.3, 1.0) |
  (aa)  |   additive\_1x2\_gamma\_1\_4 | Single additive bidder, 2 items, Gamma(1.4, 1.0) |
  (ab)  |   additive\_1x2\_gamma\_1\_5 | Single additive bidder, 2 items, Gamma(1.5, 1.0) |
  (ac)  |   additive\_1x2\_gamma\_1\_6 | Single additive bidder, 2 items, Gamma(1.6, 1.0) |
  (ad)  |   additive\_1x2\_gamma\_1\_7 | Single additive bidder, 2 items, Gamma(1.7, 1.0) |
  (ae)  |   additive\_1x2\_gamma\_1\_8 | Single additive bidder, 2 items, Gamma(1.8, 1.0) |
  (af)  |   additive\_1x2\_gamma\_1\_9 | Single additive bidder, 2 items, Gamma(1.9, 1.0) |
  (ag)  |   additive\_1x2\_gamma\_2\_0 | Single additive bidder, 2 items, Gamma(2.0, 1.0) |
  (ah)  |   additive\_1x2\_gamma\_11 | Single additive bidder, 2 items, Gamma(1, 1) |
  (ai)  |   additive\_1x2\_gamma\_21 | Single additive bidder, 2 items, Gamma(2, 1) |
  (aj)  |   additive\_1x2\_gamma\_22 | Single additive bidder, 2 items, Gamma(2, 2) |
  (ak)  |   additive\_1x2\_gamma\_31 | Single additive bidder, 2 items, Gamma(3, 1) |
  (al)  |   additive\_1x2\_gamma\_41 | Single additive bidder, 2 items, Gamma(4, 1) |
  (am)  |   additive\_1x2\_gamma\_101 | Single additive bidder, 2 items, Gamma(10, 1) |
  (an)  |   additive\_1x2\_gamma\_20\_1 | Single additive bidder, 2 items, Gamma(20, 1) |
  (ao)  |   additive\_1x3\_gamma\_1\_0\_1\_0 | Single additive bidder, 3 items, Gamma(1.0, 1.0) |
  (ap)  |   additive\_1x3\_constrained\_c1 | Single additive bidder, 3 items, constrained allocation (c=1.0) |

#### Data Saving for Restore
All config files have `save_data = True` by default, which saves training data (`X.npy`) to enable checkpoint resumption and visualization.

#### Hyperparameter Updates for Gamma Distributions
All gamma distribution configurations have been updated with improved hyperparameters for better accuracy:
- **num_hidden_units**: 1000 → **2000** (increased network capacity)
- **max_iter**: 200000 → **400000** (longer training for better convergence)
- **num_batches**: 5000 → **10000** (more diverse training data)
- **batch_size**: 128 → **256** (more stable gradient estimation)

These updates apply to all gamma distribution settings (e.g., `additive_1x2_gamma_0_1` through `additive_1x2_gamma_2_0`, `additive_1x2_gamma_11`, `additive_1x2_gamma_21`, etc.).

#### Visualization
Visualization notebooks are available for all settings. All visualization notebooks follow a consistent naming convention:

**Naming Convention**: `visualize_{agent_type}_{auction_size}_{distribution}_{params}.ipynb`

Where:
- `agent_type`: `additive` or `unit` (for unit_demand)
- `auction_size`: `1x2`, `1x10`, etc. (number of agents × number of items)
- `distribution`: `gamma`, `uniform`, `beta`, etc.
- `params`: Distribution parameters in decimal format with underscores (e.g., `gamma_1_0_1_0` for Gamma(1.0, 1.0), `uniform_0_0_4_0_0_0_3_0` for asymmetric uniform [0,4]×[0,3])

**Examples**:
- `visualize_additive_1x2_uniform_0_0_1_0.ipynb` - Additive bidder, 1×2 auction, uniform [0,1]
- `visualize_additive_1x2_gamma_1_0_1_0.ipynb` - Additive bidder, 1×2 auction, Gamma(1.0, 1.0)
- `visualize_additive_1x2_gamma_20_0_1_0.ipynb` - Additive bidder, 1×2 auction, Gamma(20.0, 1.0)
- `visualize_additive_1x2_uniform_0_0_4_0_0_0_3_0.ipynb` - Additive bidder, 1×2 auction, asymmetric uniform [0,4]×[0,3]
- `visualize_additive_1x2_uniform_4_0_16_0_4_0_7_0.ipynb` - Additive bidder, 1×2 auction, asymmetric uniform [4,16]×[4,7]
- `visualize_unit_1x2_uniform_2_0_3_0.ipynb` - Unit-demand bidder, 1×2 auction, uniform [2,3]
- `visualize_additive_1x2_beta_1_0_1_0.ipynb` - Additive bidder, 1×2 auction, Beta(1.0, 1.0)

**Available visualization notebooks**:
- Uniform distributions:
  - `visualize_additive_1x2_uniform_0_0_1_0.ipynb`
  - `visualize_additive_1x2_uniform_4_0_16_0_4_0_7_0.ipynb`
  - `visualize_additive_1x2_uniform_triangle.ipynb`
  - `visualize_additive_1x2_uniform_0_0_4_0_0_0_3_0.ipynb`
  - `visualize_additive_1x3_uniform_0_1.ipynb`
  - `visualize_unit_1x2_uniform_2_0_3_0.ipynb`
- Beta distributions:
  - `visualize_additive_1x2_beta_1_0_1_0.ipynb`
  - `visualize_additive_1x2_beta_1_0_2_0.ipynb`
  - `visualize_additive_1x2_beta_2_0_1_0.ipynb`
  - `visualize_additive_1x2_beta_1_0_3_0.ipynb`
  - `visualize_additive_1x2_beta_3_0_1_0.ipynb`
- Gamma distributions (extensive coverage):
  - `visualize_additive_1x2_gamma_0_1_1_0.ipynb` through `visualize_additive_1x2_gamma_2_0_1_0.ipynb` (k=0.1 to 2.0)
  - `visualize_additive_1x2_gamma_1_0_1_0.ipynb`
  - `visualize_additive_1x2_gamma_2_0_2_0.ipynb`
  - `visualize_additive_1x2_gamma_3_0_1_0.ipynb`
  - `visualize_additive_1x2_gamma_4_0_1_0.ipynb`
  - `visualize_additive_1x2_gamma_10_0_1_0.ipynb`
  - `visualize_additive_1x2_gamma_20_0_1_0.ipynb`
  - `visualize_additive_1x3_gamma_1_0_1_0.ipynb`
- Constrained allocations:
  - `visualize_additive_1x3_constrained_c1.ipynb`
  
### MyersonNet (Single Item Auctions)
  
Default hyperparameters are specified in utils/cfg.py.  
For training the network, testing the mechanism learnt and computing the baselines, run:
```
cd myersonNet
python main.py -distr [setting_name] or
bash myerson.sh
```
setting\_no  |      setting\_name |
 :---:  | :---: |
  (a)   |  exponential 
  (b)   |   uniform
  \(c\) |   asymmetric\_uniform 
  (d)   |   irregular

 
## Settings

### Single Bidder
- **additive\_1x2\_uniform**: A single bidder with additive valuations over two items, where the items is drawn from U\[0, 1\].

- **unit\_1x2\_uniform\_23**: A single bidder with unit-demand valuations over two items, where the item values are drawn from U\[2, 3\].

- **additive\_1x2\_uniform\_416\_47**: Single additive bidder with preferences over two non-identically distributed items, where v<sub>1</sub> ∼ U\[4, 16\]and v<sub>2</sub> ∼ U\[4, 7\].

- **additive\_1x2\_uniform\_triangle**: A single additive bidder with preferences over two items, where (v<sub>1</sub>, v<sub>2</sub>) are drawn jointly and uniformly from a unit-triangle with vertices (0, 0), (0, 1) and (1, 0).

- **unit\_1x2\_uniform**: A single unit-demand bidder with preferences over two items, where the item values from U\[0, 1\]

- **additive\_1x2\_uniform\_04\_03**: A Single additive bidder with preferences over two items, where the item values v<sub>1</sub> ∼ U\[0, 4], v<sub>2</sub> ∼ U\[0, 3]

- **additive\_1x10\_uniform**: A single additive bidder and 10 items, where bidders draw their value for each item from U\[0, 1\].

- **additive\_1x2\_beta\_11**: A single additive bidder with preferences over two items, where the item values are drawn from Beta(1, 1), which is equivalent to U\[0, 1\].

- **additive\_1x2\_gamma\_11**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=1, θ=1) (mean=1, variance=1).

- **additive\_1x2\_gamma\_21**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=2, θ=1) (mean=2, variance=2).

- **additive\_1x2\_gamma\_22**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=2, θ=2) (mean=4, variance=8).

- **additive\_1x2\_gamma\_31**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=3, θ=1) (mean=3, variance=3).

- **additive\_1x2\_gamma\_41**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=4, θ=1) (mean=4, variance=4).

- **additive\_1x2\_gamma\_0\_1** through **additive\_1x2\_gamma\_2\_0**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k, 1.0) for k ranging from 0.1 to 2.0 in 0.1 increments. These settings allow for fine-grained exploration of the gamma distribution parameter space.

- **additive\_1x2\_gamma\_101**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=10, θ=1) (mean=10, variance=10).

- **additive\_1x2\_gamma\_20\_1**: A single additive bidder with preferences over two items, where the item values are drawn from Gamma(k=20, θ=1) (mean=20, variance=20).

- **additive\_1x3\_uniform\_0\_1**: A single additive bidder with preferences over three items, where the item values are drawn from U[0, 1].

- **additive\_1x3\_gamma\_1\_0\_1\_0**: A single additive bidder with preferences over three items, where the item values are drawn from Gamma(1.0, 1.0).

### Constrained Allocation (制約付き配分)

**RegretNet**:
- **additive\_1x3\_constrained\_c0\_1** (c=0.1): 3人のadditive bidderと3財のオークション。財1と財2の価値はU\[0, 1\]に従い、財3の価値はU\[0, 0.1\]に従う。
- **additive\_1x3\_constrained\_c0\_5** (c=0.5): 3人のadditive bidderと3財のオークション。財1と財2の価値はU\[0, 1\]に従い、財3の価値はU\[0, 0.5\]に従う。
- **additive\_1x3\_constrained\_c1** (c=1.0): 3人のadditive bidderと3財のオークション。財1と財2の価値はU\[0, 1\]に従い、財3の価値はU\[0, 1\]に従う。
- **additive\_1x3\_constrained\_c10** (c=10.0): 3人のadditive bidderと3財のオークション。財1と財2の価値はU\[0, 1\]に従い、財3の価値はU\[0, 10\]に従う。

**RochetNet**:
- **additive\_1x3\_constrained\_c1** (c=1.0): 単一のadditive bidderと3財のオークション。財1と財2の価値はU\[0, 1\]に従い、財3の価値はU\[0, 1\]に従う。

すべてのconstrained設定において、財3の配分確率には以下の制約が課される：
  - 下界制約: 財3の配分確率 ≥ max(0, 財1の配分確率 + 財2の配分確率-1)
  - 上界制約: 財3の配分確率 ≤ min(財1の配分確率, 財2の配分確率)
  
制約違反に対してAugmented Lagrangian法を用いて罰則項を追加し、制約を満たしながらrevenueを最大化する。実装は既存クラスを変更せず、新規クラス（`constrained_additive_net.py`、`constrained_trainer.py`など）として追加されている。

### Multiple Bidders
- **additive\_2x2\_uniform**: Two additive bidders and two items, where bidders draw their value for each item from U\[0, 1\]. 

- **unit\_2x2\_uniform**: Two unit-demand bidders and two items, where the bidders draw their value for each item from identical U\[0, 1\].

- **additive\_2x3\_uniform**: Two additive bidders and three items, where bidders draw their value for each item from U\[0, 1\]. 

- **CA\_sym\_uniform\_12**: Two bidders and two items, with v<sub>1,1</sub>, v<sub>1,2</sub>, v<sub>2,1</sub>, v<sub>2,2</sub> ∼ U\[1, 2\], v<sub>1,{1,2}</sub> = v<sub>1,1</sub> + v<sub>1,2</sub> + C<sub>1</sub> and v<sub>2,{1,2}</sub> = v<sub>2,1</sub> + v<sub>2,2</sub> + C<sub>2</sub>, where C<sub>1</sub>, C<sub>2</sub> ∼ U\[−1, 1\].

- **CA\_asym\_uniform\_12\_15**: Two bidders and two items, with v<sub>1,1</sub>, v<sub>1,2</sub> ∼ U\[1, 2\], v<sub>2,1</sub>, v<sub>2,2</sub> ∼ U\[1, 5\], v<sub>1,{1,2}</sub> = v<sub>1,1</sub> + v<sub>1,2</sub> + C<sub>1</sub> and v<sub>2,{1,2}</sub> = v<sub>2,1</sub> + v<sub>2,2</sub> + C<sub>2</sub>, where C<sub>1</sub>, C<sub>2</sub> ∼ U\[−1, 1].

- **additive\_3x10\_uniform**: 3 additive bidders and 10 items, where bidders draw their value for each item from U\[0, 1\].

- **additive\_5x10\_uniform**: 5 additive bidders and 10 items, where bidders draw their value for each item from U\[0, 1\].

## Visualization

Allocation Probabilty plots for **unit\_1x2\_uniform_23** setting learnt by **regretNet**:

<img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/regretNet/plots/visualization/unit_1x2_uniform_23_alloc1.png" width="300"> <img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/regretNet/plots/visualization/unit_1x2_uniform_23_alloc2.png" width="300">

Allocation Probabilty plots for **additive\_1x2\_uniform\_416\_47** setting learnt by **rochetNet**:

<img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/rochetNet/plots/visualization/additive_1x2_uniform_416_47_alloc1.png" width="300"> <img src="https://github.com/saisrivatsan/deep-opt-auctions/blob/master/rochetNet/plots/visualization/additive_1x2_uniform_416_47_alloc2.png" width="300">

For other allocation probability plots, check-out the ipython notebooks in `regretNet` or `rochetNet` folder.
