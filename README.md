# GCN example

## 1. Installation

```bash
# CUDA-10.1
$ pip install tensorflow==2.3.1
$ pip install spektral==0.6.2
```

### 1-1. Prepare dataset

__Dataset must be downloaded (not public)__

1. Move downloaded dataset to `Data` directory

2. `cd Data`

3. `tar -zxvf gdp_dataset.tgz`

4. Run `python preprocess_dataset_v3.py`

## 2. Training GCN

### 2-1. Train Model with Single Node

Run `python train_gcn_v3.py` at the root dir.

### 2-2. [All-reduce] Distributed Learning

1. Set nodes' IP addresses in `dist_gcn_v3.py` file

2. Run `dist_gcn_v3.py` in each node. The chief node uses `0` for argument, and the others use increasing number from `1`.

```bash
# Chief node
$ python dist_gcn_v3.py 0

# Other worker nodes
$ python dist_gcn_v3.py 1
$ python dist_gcn_v3.py 2
...
```

### 2-3. [Parameter Server] Distributed Learning

- __1 Chief node & 2 Worker nodes & 1 PS node__

```
# Node 1
$ python dist_ps_gcn_v2.py 0

# Node 2
$ python dist_ps_gcn_v2.py 1

# Node 3
$ python dist_ps_gcn_v2.py 2

# Node 4
$ python dist_ps_gcn_v2.py 3
```

- __1 Chief node & 3 [Worker + PS] nodes__

```
# Node 1
$ python dist_ps_gcn_v3.py chief
 
# Node 2
$ python dist_ps_gcn_v3.py worker 1
$ python dist_ps_gcn_v3.py ps 1

# Node 3
$ python dist_ps_gcn_v3.py worker 2
$ python dist_ps_gcn_v3.py ps 2

# Node 4
$ python dist_ps_gcn_v3.py worker 3
$ python dist_ps_gcn_v3.py ps 3
```

## 3. Model Checkpoint & Logging

Results of training saved in to path `Model_v3/[datetime]/FOLD-[CV]/` or `Model_dist_v3/[datetime]/FOLD-[CV]/`.

Logging file (`train.log`) will be saved in to the same path with model.

Only chief node saves results in distributed training.
