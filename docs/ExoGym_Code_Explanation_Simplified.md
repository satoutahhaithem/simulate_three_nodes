# ExoGym Code Explanation (Simplified)

This document provides a simplified explanation of how ExoGym works internally, focusing on the core components and their interactions.

## Overview of ExoGym Architecture

ExoGym is designed to simulate distributed training on a single machine. It achieves this through a layered architecture:

```
+------------------+
|  Trainer Layer   |
+------------------+
         |
         v
+------------------+
|  TrainNode Layer |
+------------------+
         |
         v
+------------------+
|  Strategy Layer  |
+------------------+
         |
         v
+------------------+
| PyTorch Dist.    |
+------------------+
```

1. **Trainer Layer**: Manages the overall training process
2. **TrainNode Layer**: Represents individual nodes in the distributed system
3. **Strategy Layer**: Implements different optimization and communication strategies
4. **PyTorch Distributed Layer**: Handles the actual communication between processes

## 1. Trainer Layer

The Trainer layer is the highest-level abstraction in ExoGym. It's responsible for:

- Setting up the distributed environment
- Spawning multiple processes (nodes)
- Collecting and aggregating results

There are two main trainer classes:
- `Trainer`: Base class with common functionality
- `LocalTrainer`: Implementation for local simulation

The `fit` method is the main entry point for training. It:
1. Creates a configuration object
2. Launches distributed training
3. Updates the model with the averaged state

### Process Spawning

ExoGym uses PyTorch's multiprocessing to spawn multiple processes:

```
+-------------+     spawns     +-----------+
| Main Process|--------------->| Process 0 |
|             |                +-----------+
|             |     spawns     +-----------+
|             |--------------->| Process 1 |
|             |                +-----------+
|             |     spawns     +-----------+
|             |--------------->| Process 2 |
+-------------+                +-----------+
```

This creates `num_nodes` separate processes, each running with the same configuration.

## 2. TrainNode Layer

The TrainNode layer represents an individual node in the distributed system. Each node:

- Has its own copy of the model
- Works with a subset of the data
- Communicates with other nodes according to a strategy

The `train` method is the main training loop for each node. It:
1. Sets up distributed communication
2. Trains for the specified number of epochs
3. Returns the final model state

### Data Partitioning

Each node works with a different subset of the data:

```
Full Dataset
[1,2,3,4,5,6,7,8,9]
      |
      v
+---------+  +---------+  +---------+
| Node 1  |  | Node 2  |  | Node 3  |
| [1,2,3] |  | [4,5,6] |  | [7,8,9] |
+---------+  +---------+  +---------+
```

This ensures that each node processes a different part of the dataset, simulating how distributed training would work across multiple machines.

## 3. Strategy Layer

The Strategy layer implements different optimization and communication strategies. This is where the actual distributed algorithms are implemented.

### Example: FedAvgStrategy

The Federated Averaging strategy implements local training followed by periodic model averaging:

```
+-------+    +-------+    +-------+
| Node1 |    | Node2 |    | Node3 |
+-------+    +-------+    +-------+
    |            |            |
    v            v            v
Train locally  Train locally  Train locally
for H steps    for H steps    for H steps
    |            |            |
    +------+-----+------+-----+
           |
           v
    Average models
    |      |      |
    v      v      v
+-------+    +-------+    +-------+
| Node1 |    | Node2 |    | Node3 |
+-------+    +-------+    +-------+
```

The key aspects of this strategy are:
1. Each node trains locally for `H` steps
2. After `H` steps, models are averaged across all nodes
3. All nodes continue training from the averaged model

## 4. PyTorch Distributed Layer

At the lowest level, ExoGym uses PyTorch's distributed package for communication between processes.

### Communication Primitives

ExoGym uses several PyTorch distributed primitives:

- **all_reduce**: Sums tensors across all processes and distributes the result
- **broadcast**: Sends a tensor from one process to all others
- **all_gather**: Collects tensors from all processes

## Execution Flow

Let's trace the execution flow of a typical ExoGym run:

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

## Key Classes and Their Relationships

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

- **Trainer**: Manages the overall training process
- **TrainNode**: Represents a single node in the distributed system
- **Strategy**: Implements the distributed algorithm
- **PyTorch Distributed**: Provides the communication primitives

## Conclusion

ExoGym's architecture allows it to simulate distributed training on a single machine by:

1. Creating separate processes for each simulated node
2. Partitioning the data across these processes
3. Using PyTorch's distributed package for communication
4. Implementing different strategies for optimization and synchronization

This design makes it possible to experiment with distributed training algorithms without needing multiple physical machines or GPUs.