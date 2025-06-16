# ExoGym Code Explanation

This document provides a detailed explanation of how ExoGym works internally, focusing on the core components and their interactions.

## Overview of ExoGym Architecture

ExoGym is designed to simulate distributed training on a single machine. It achieves this through a layered architecture:

1. **Trainer Layer**: Manages the overall training process
2. **TrainNode Layer**: Represents individual nodes in the distributed system
3. **Strategy Layer**: Implements different optimization and communication strategies
4. **PyTorch Distributed Layer**: Handles the actual communication between processes

Let's examine each layer in detail.

## 1. Trainer Layer

The Trainer layer is the highest-level abstraction in ExoGym. It's responsible for:

- Setting up the distributed environment
- Spawning multiple processes (nodes)
- Collecting and aggregating results

There are two main trainer classes:
- `Trainer`: Base class with common functionality
- `LocalTrainer`: Implementation for local simulation

### Key Components in Trainer

```python
# From exogym/trainer.py
class Trainer:
    def __init__(self, model, train_dataset, val_dataset, lazily_load_dataset=False):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.lazily_load_dataset = lazily_load_dataset
        
    def fit(self, num_epochs, strategy, num_nodes, max_steps=None, device=None, 
            devices=None, batch_size=16, minibatch_size=16, shuffle=True, 
            val_size=64, eval_interval=100, autocast=False, checkpoint_interval=100, **kwargs):
        # Configure training
        config = TrainerConfig(...)
        
        # Launch distributed training
        averaged_state_dict = launch(config)
        
        # Update model with averaged state
        self.model.load_state_dict(averaged_state_dict)
```

The `fit` method is the main entry point for training. It:
1. Creates a configuration object
2. Launches distributed training
3. Updates the model with the averaged state

### Process Spawning

ExoGym uses PyTorch's multiprocessing to spawn multiple processes:

```python
# From exogym/trainer.py
def launch(config):
    result_queue = mp.Queue()
    mp.spawn(
        _worker,
        args=(config, result_queue),
        nprocs=config.num_nodes,
        start_method="spawn",
        join=True,
    )
    
    # Collect results from all nodes
    model_states = {}
    for _ in range(config.num_nodes):
        rank, state_dict = result_queue.get()
        model_states[rank] = state_dict
    
    # Average the models
    return _average_model_states(model_states)
```

This creates `num_nodes` separate processes, each running the `_worker` function with the same configuration.

## 2. TrainNode Layer

The TrainNode layer represents an individual node in the distributed system. Each node:

- Has its own copy of the model
- Works with a subset of the data
- Communicates with other nodes according to a strategy

### Key Components in TrainNode

```python
# From exogym/train_node.py
class TrainNode:
    def __init__(self, rank, config):
        self.rank = rank
        self.num_nodes = config.num_nodes
        self.model = copy.deepcopy(config.model)
        self.strategy = config.strategy
        # ... other initialization
        
    def train(self):
        # Initialize distributed communication
        self._build_connection()
        
        # Train for specified number of epochs
        for epoch in range(self.num_epochs):
            self._train_epoch()
            
        # Return final model state
        return self.model.state_dict()
        
    def _train_epoch(self):
        # Iterate through batches
        for batch_idx, batch in enumerate(self.train_loader):
            # Forward pass
            loss = self.model(batch)
            
            # Backward pass
            loss.backward()
            
            # Update using strategy
            self.strategy.step()
```

The `train` method is the main training loop for each node. It:
1. Sets up distributed communication
2. Trains for the specified number of epochs
3. Returns the final model state

### Data Partitioning

Each node works with a different subset of the data using PyTorch's `DistributedSampler`:

```python
# From exogym/train_node.py
self.sampler = torch.utils.data.DistributedSampler(
    self.train_dataset, 
    num_replicas=self.num_nodes,
    rank=self.rank,
    shuffle=self.shuffle
)

self.train_loader = torch.utils.data.DataLoader(
    self.train_dataset,
    batch_size=self.batch_size,
    sampler=self.sampler
)
```

This ensures that each node processes a different part of the dataset, simulating how distributed training would work across multiple machines.

## 3. Strategy Layer

The Strategy layer implements different optimization and communication strategies. This is where the actual distributed algorithms are implemented.

### Base Strategy Class

```python
# From exogym/strategy/strategy.py
class Strategy(ABC, LogModule):
    def __init__(self, model, optimizer):
        self.model = model
        self.optim = optimizer
        
    @abstractmethod
    def step(self):
        """Perform one optimization step."""
        pass
        
    def zero_grad(self):
        """Zero gradients of parameters."""
        self.optim.zero_grad()
```

### Example: FedAvgStrategy

The Federated Averaging strategy implements local training followed by periodic model averaging:

```python
# From exogym/strategy/federated_averaging.py
class FedAvgStrategy(CommunicateOptimizeStrategy):
    def __init__(self, inner_optim=None, island_size=None, H=1, max_norm=None, **kwargs):
        super().__init__(
            inner_optim=inner_optim,
            outer_optim=None,
            island_size=island_size,
            H=H,
            max_norm=max_norm,
            **kwargs
        )
        
    def _communicate(self):
        # Average models across nodes
        self._average_models()
```

The key aspects of this strategy are:
1. Each node trains locally for `H` steps
2. After `H` steps, models are averaged across all nodes
3. All nodes continue training from the averaged model

## 4. PyTorch Distributed Layer

At the lowest level, ExoGym uses PyTorch's distributed package for communication between processes.

### Initialization

```python
# From exogym/trainer.py
def _build_connection(self):
    # Set environment variables for distributed communication
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(12355 + (10 if self.device == 'cpu' else 0))
    
    # Initialize the process group
    dist.init_process_group(
        "gloo",  # Communication backend
        rank=self.rank,
        world_size=self.num_nodes
    )
```

### Communication Primitives

ExoGym uses several PyTorch distributed primitives:

- **all_reduce**: Sums tensors across all processes and distributes the result
  ```python
  dist.all_reduce(tensor)  # In-place operation
  ```

- **broadcast**: Sends a tensor from one process to all others
  ```python
  dist.broadcast(tensor, src=0)  # Broadcast from rank 0
  ```

- **all_gather**: Collects tensors from all processes
  ```python
  gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
  dist.all_gather(gathered_tensors, tensor)
  ```

## Execution Flow

Let's trace the execution flow of a typical ExoGym run:

1. **User creates a model and datasets**:
   ```python
   model = Net()
   train_dataset = datasets.MNIST('./data', train=True, download=True)
   test_dataset = datasets.MNIST('./data', train=False)
   ```

2. **User creates a trainer**:
   ```python
   trainer = LocalTrainer(model, train_dataset, test_dataset)
   ```

3. **User calls fit with a strategy**:
   ```python
   strategy = FedAvgStrategy(inner_optim=optim_spec, H=1)
   trainer.fit(num_epochs=2, strategy=strategy, num_nodes=3, device="cpu")
   ```

4. **Trainer spawns multiple processes**:
   - Process 0 (Rank 0)
   - Process 1 (Rank 1)
   - Process 2 (Rank 2)

5. **Each process initializes a TrainNode**:
   - Creates a copy of the model
   - Sets up data loaders with its portion of the data
   - Initializes the strategy

6. **Each process initializes distributed communication**:
   - Joins the process group
   - Sets up device assignment

7. **Each process trains for the specified number of epochs**:
   - Forward pass
   - Backward pass
   - Strategy step (which may include communication)

8. **Results are collected and averaged**:
   - Each process returns its final model state
   - States are averaged
   - Main process updates the original model

## Key Classes and Their Relationships

- **Trainer**: Manages the overall training process
  - Creates and configures TrainNodes
  - Collects and aggregates results

- **TrainNode**: Represents a single node in the distributed system
  - Has its own model copy
  - Works with a subset of the data
  - Uses a strategy for optimization and communication

- **Strategy**: Implements the distributed algorithm
  - Defines how and when nodes communicate
  - Controls the optimization process

- **PyTorch Distributed**: Provides the communication primitives
  - Enables message passing between processes
  - Supports collective operations like all_reduce

## Conclusion

ExoGym's architecture allows it to simulate distributed training on a single machine by:

1. Creating separate processes for each simulated node
2. Partitioning the data across these processes
3. Using PyTorch's distributed package for communication
4. Implementing different strategies for optimization and synchronization

This design makes it possible to experiment with distributed training algorithms without needing multiple physical machines or GPUs.