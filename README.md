# MaskDGNN: Self-Supervised Dynamic Graph Neural Networks with Activeness-aware Temporal Masking

This repository contains the PyTorch implementation of **MaskDGNN**, a novel architecture designed to handle dynamic graphs by addressing the challenges of redundant information and distribution shifts. MaskDGNN achieves state-of-the-art performance on multiple dynamic graph datasets using its **Self-Supervised Activeness-aware Temporal Masking Module** and **Adaptive Frequency Enhancing Graph Representation Learner**.

## Key Features
- **Self-Supervised Activeness-aware Temporal Masking Module:** Focuses on highly active nodes to retain critical temporal features while reducing redundancy.
- **Adaptive Frequency Enhancing Graph Representation Learner:** Leverages spectral-domain transformations to capture intrinsic features under distribution shifts.

## Requirements

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

### Tested Environment
- `dgl==0.9.1`
- `dgl-cu116==0.9.1.post1`
- `numpy==1.21.6`
- `scikit-learn==1.0.2`
- `torch==1.12.1+cu116`
- `torch-scatter==2.0.9`
  
### Running MaskDGNN
You can run **MaskDGNN** on the provided datasets using the following commands:

#### BitcoinAlpha
```bash
python main.py --dataset bitcoinalpha --lr 0.06 --maml_lr 0.002 --drop_rate 0.1 --mask 0.2 --window_num 8 --early_stop 15
```

#### BitcoinOTC
```bash
python main.py --dataset bitcoinotc --lr 0.01 --maml_lr 0.006 --drop_rate 0.2 --mask 0.2 --window_num 4 --early_stop 15
```

#### UCI Message
```bash
python main.py --dataset uci-msg --lr 0.01 --maml_lr 0.008 --drop_rate 0.16 --mask 0.1 --window_num 8 --early_stop 10
```

#### MOOC
```bash
python main.py --dataset mooc --lr 0.1 --maml_lr 0.003 --drop_rate 0.1 --mask 0.2 --window_num 8 --early_stop 35
```

#### Wiki-Talk
```bash
python main.py --dataset wiki-talk-temporal --lr 0.03 --maml_lr 0.001 --drop_rate 0.1 --window_num 4 --num_layers 1 --num_hidden 32 --out_dim 16 --mask 0.2 --early_stop 10
```

## Dataset Overview

MaskDGNN has been evaluated on the following datasets:

| Dataset        | Nodes   | Edges       | Snapshots |
|----------------|---------|-------------|-----------|
| Bitcoin-Alpha  | 3,783   | 24,186      | 226       |
| Bitcoin-OTC    | 5,881   | 35,592      | 262       |
| UCI Message    | 1,899   | 59,835      | 28        |
| MOOC           | 7,144   | 411,749     | 30        |
| Wiki-Talk      | 1,140,149 | 7,833,140 | 73        |



