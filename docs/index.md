# ExoGym Documentation (Simplified)

Welcome to the simplified ExoGym documentation. These documents provide clear explanations of ExoGym and its network simulation capabilities, with a focus on simplicity and visual aids.

## Available Documentation

1. [**ExoGym Documentation**](ExoGym_Documentation_Simplified.md) - Overview of ExoGym's architecture and components
2. [**ExoGym Code Explanation**](ExoGym_Code_Explanation_Simplified.md) - How ExoGym works internally
3. [**ExoGym Network Simulation Guide**](ExoGym_Network_Simulation_Guide_Simplified.md) - How to simulate network conditions in ExoGym
4. [**Network Simulation Usage Guide**](Network_Simulation_Usage_Guide_Simplified.md) - Practical instructions for using network simulation

## Quick Start

For a quick overview of ExoGym and network simulation, see the [README](../README.md).

## Key Concepts

### What is ExoGym?

ExoGym lets you simulate distributed training (training ML models across multiple computers) on a single machine. It creates multiple virtual "nodes" that work together to train a model.

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

### What is Network Simulation?

In real distributed training, nodes communicate over a network with limitations (speed, delays). Our network simulator adds these limitations to make the simulation more realistic.

```
Without Network Simulation:
Node 1 <==== FAST ====> Node 2

With Network Simulation:
Node 1 <--- SLOW -----> Node 2
       (bandwidth limits)
       (delays)
       (packet loss)
```

### Network Profiles

We provide preset network conditions to simulate different environments:

```
Speed & Delay Comparison:
                                                  
Fast ↑  local [1-10 GB/s, 0.1-1ms]  *             
      |                                           
      |  datacenter [100-1000 MB/s, 0.5-5ms]  *   
      |                                           
      |  wan [10-100 MB/s, 10-50ms]  *            
      |                                           
      |  mobile [1-10 MB/s, 50-200ms]  *          
      |                                           
Slow ↓  poor [0.1-1 MB/s, 200-1000ms]  *          
      +---------------------------------------→    
        Low                  Delay                High
```

### Network Topologies

You can also change how nodes are connected:

```
Fully Connected:    Ring:             Star:
                                        
  1 --- 2           1 --- 2             2
  |     |           |     |            /
  |     |           |     |           /
  3 --- 4           4 --- 3          1 --- 4
                                      \
                                       \
                                        3