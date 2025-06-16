# ExoGym: Technical Documentation (Simplified)

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Key Components](#3-key-components)
4. [Execution Flow](#4-execution-flow)
5. [Dataset Handling](#5-dataset-handling)
6. [Optimization Strategies](#6-optimization-strategies)
7. [Usage Examples](#7-usage-examples)

## 1. Introduction

ExoGym is an open-source framework designed to simulate distributed training methods on a single machine. Instead of requiring multiple physical machines or GPUs for distributed training, ExoGym creates a simulated distributed environment using multiple processes on a single machine.

```
Single Machine
+------------------------------------------+
|                                          |
|   +--------+    +--------+    +--------+ |
|   | Node 1 |    | Node 2 |    | Node 3 | |
|   +--------+    +--------+    +--------+ |
|                                          |
+------------------------------------------+
```

The framework supports various distributed training methods including:
- AllReduce (equivalent to PyTorch DDP)
- FedAvg (Federated Averaging)
- DiLoCo (Distributed Local Contrastive)
- SPARTA
- DeMo

ExoGym is particularly useful for researchers and developers who want to experiment with distributed training methods without the need for expensive hardware setups.

## 2. System Architecture

ExoGym follows a modular architecture with clear separation of concerns:

```
+-------------+
|   Trainer   |
+-------------+
      |  creates
      v
+-------------+       uses       +-------------+
|  TrainNode  |----------------->|  Strategy   |
+-------------+                  +-------------+
      |  uses                          |  uses
      v                                v
+---------------------------+    +-------------+
| PyTorch DataLoader/Sampler|    | PyTorch Dist|
+---------------------------+    +-------------+
```

## 3. Key Components

### 3.1 Trainer and LocalTrainer

The `Trainer` class is the main entry point for training. It's responsible for:
- Holding the model and datasets
- Setting up the distributed environment
- Creating and managing TrainNode instances
- Collecting and averaging results from multiple nodes

The `LocalTrainer` extends `Trainer` and implements the connection setup to establish PyTorch distributed communication.

### 3.2 TrainNode

The `TrainNode` class represents a single node in the distributed training process:
- Has its own copy of the model
- Works with a subset of the data
- Uses a strategy for optimization and communication

```
TrainNode
+----------------------------------+
| Model Copy                       |
| Data Subset                      |
| Strategy                         |
| Training Loop                    |
|   - Forward pass                 |
|   - Backward pass                |
|   - Strategy step (communication)|
+----------------------------------+
```

### 3.3 Strategy Hierarchy

The `Strategy` class is an abstract base class that defines the interface for optimization and communication strategies:

```
                  +-------------+
                  |  Strategy   |
                  +-------------+
                        ^
                        |
        +---------------+---------------+
        |               |               |
+----------------+ +-----------+ +-------------+
| SimpleReduce   | | DiLoCo    | | FedAvg      |
| (DDP)          | | Strategy  | | Strategy    |
+----------------+ +-----------+ +-------------+
```

#### 3.3.1 SimpleReduceStrategy (DDP)

This strategy implements the standard Distributed Data Parallel approach:
- Gradients are synchronized after every backward pass
- All nodes maintain identical model parameters

#### 3.3.2 DiLoCoStrategy

DiLoCo (Distributed Local Contrastive) strategy:
- Uses a two-level optimization approach
- Inner optimizer for local updates
- Outer optimizer for global updates
- Synchronizes after H local steps

#### 3.3.3 FedAvg

Federated Averaging strategy:
- Each node trains locally for E steps
- Models are averaged periodically
- No gradient synchronization, only model averaging

## 4. Execution Flow

### 4.1 Training Process Flow

```
User creates model and datasets
           |
           v
User creates a trainer
           |
           v
User calls fit with a strategy
           |
           v
Trainer spawns multiple processes
           |
           v
Each process initializes a TrainNode
           |
           v
Each process initializes distributed communication
           |
           v
Each process trains for the specified number of epochs
           |
           v
Results are collected and averaged
           |
           v
Main process updates the original model
```

### 4.2 Strategy Execution Flow

Different strategies have different execution flows:

**SimpleReduceStrategy (DDP)**:
```
Forward pass → Backward pass → All-reduce gradients → Update model
```

**FedAvg**:
```
For E steps:
  Forward pass → Backward pass → Update model locally
Average models across nodes
```

**DiLoCo**:
```
For H steps:
  Forward pass → Backward pass → Update model locally
Synchronize master model
Update with outer optimizer
```

## 5. Dataset Handling

ExoGym supports two approaches for dataset handling:

1. **Direct Dataset**: The same dataset is provided to all nodes, and a `DistributedSampler` ensures each node gets a different subset.

2. **Dataset Factory**: A function that creates a different dataset for each node based on its rank.

```
Dataset Distribution
+-------------------+
| Full Dataset      |
+-------------------+
          |
          v
+-------------------+
| DistributedSampler|
+-------------------+
    /        |       \
   /         |        \
+-----+   +-----+   +-----+
|Node1|   |Node2|   |Node3|
+-----+   +-----+   +-----+
```

## 6. Optimization Strategies

### 6.1 SimpleReduceStrategy (DDP)

Standard distributed data parallel approach:
- Synchronizes gradients after every backward pass
- All nodes maintain identical model parameters
- Equivalent to PyTorch's DistributedDataParallel

### 6.2 DiLoCoStrategy

DiLoCo (Distributed Local Contrastive) strategy:
- Two-level optimization approach
- Inner optimizer for local updates
- Outer optimizer for global updates
- Synchronizes after H local steps

### 6.3 FedAvg

Federated Averaging strategy:
- Each node trains locally for E steps
- Models are averaged periodically
- No gradient synchronization, only model averaging

### 6.4 SPARTA

SPARTA (Sparse Training with Approximate Gradient Aggregation):
- Uses sparse communication
- Only communicates important gradients
- Reduces communication overhead

## 7. Usage Examples

Basic usage of ExoGym:

```python
# Create model and datasets
model = Net()
train_dataset = datasets.MNIST('./data', train=True, download=True)
test_dataset = datasets.MNIST('./data', train=False)

# Create trainer
trainer = LocalTrainer(model, train_dataset, test_dataset)

# Create strategy
strategy = FedAvgStrategy(inner_optim=optim_spec, H=5)

# Train with 3 nodes
trainer.fit(num_epochs=2, strategy=strategy, num_nodes=3, device="cpu")
```

## Conclusion

ExoGym's architecture allows it to simulate distributed training on a single machine by:

1. Creating separate processes for each simulated node
2. Partitioning the data across these processes
3. Using PyTorch's distributed package for communication
4. Implementing different strategies for optimization and synchronization

This design makes it possible to experiment with distributed training algorithms without needing multiple physical machines or GPUs.