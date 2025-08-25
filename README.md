# Multi-Agent Drone Simulation with MADDPG

## Overview
This project implements a multi-agent reinforcement learning system for disaster area coverage using UAVs. The system uses MADDPG (Multi-Agent Deep Deterministic Policy Gradient) for coordinated learning.

## Recent Updates - Reward Normalization v2.0

### What Changed
The reward normalization system has been completely overhauled to address stability and performance issues:

**Before (v1.0):**
- Global reward statistics across all UAVs
- Classic running mean/variance (non-adaptive)
- Sum-based variance calculation (numerical instability)
- Print statements every 50 steps (performance impact)
- No gradual transition after warmup

**After (v2.0):**
- **Per-agent reward statistics** - Each UAV maintains its own normalization stats
- **EMA-based normalization** - Adaptive to non-stationary reward distributions
- **Welford's algorithm** - Numerically stable variance calculation
- **Proper logging** - Configurable debug output (no performance impact by default)
- **Gradual ramp-up** - Smooth transition from raw to normalized rewards
- **Raw reward clipping** - Prevents heavy-tailed distributions from causing instability

### Configuration Parameters

```python
# Reward normalization settings
NORMALIZE_REWARDS = True          # Enable/disable normalization
NORM_WARMUP = 10                  # Steps before normalization starts
NORM_CLIP = 3.0                   # Z-score clipping threshold
NORM_EMA_BETA = 0.99             # EMA decay factor (0.99 = slow adaptation)
NORM_RAMP_STEPS = 20             # Gradual transition steps after warmup
RAW_REWARD_CLIP = 1000.0         # Clip raw rewards before normalization

# Logging settings
VERBOSE_LOGGING = False           # Enable debug output (impacts performance)
```

### Breaking Changes
⚠️ **WARNING**: The default behavior has changed from v1.0 to v2.0:
- `NORMALIZE_REWARDS = True` by default (was False in v1.0)
- This changes the training signal for existing models

### Migration Guide
1. **For existing models**: Set `NORMALIZE_REWARDS = False` to maintain v1.0 behavior
2. **For new training**: Use v2.0 defaults for better stability
3. **Compare performance**: Use fixed seeds to compare pre/post normalization curves

### Performance Impact
- **v1.0**: ~2-5% performance hit from print statements
- **v2.0**: <0.1% performance hit (logging disabled by default)
- **Memory**: Slight increase due to per-agent stats (negligible for 6 UAVs)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python main.py
```

### Viewing Results
```bash
python start_tensorboard.py
# Then open http://localhost:6006
```

## Architecture

### Environment
- `DisasterCoverageEnvironment`: Main simulation environment
- `DisasterZone`: Individual disaster zones with coverage tracking
- `Simulation`: Communication and physics simulation

### Agents
- `MADDPGAgent`: Multi-agent DDPG implementation
- `CentralizedReplayBuffer`: Shared experience replay
- `BaseRLAgent`: Common RL agent functionality

### Algorithms
- `FedProx`: Federated learning with proximal terms
- `FedNova`: Federated normalized averaging

## Key Features

1. **Multi-Agent Coordination**: UAVs learn to coordinate coverage
2. **Communication Modeling**: Realistic wireless communication simulation
3. **Federated Learning**: Support for distributed training
4. **Reward Normalization**: Stable training with adaptive normalization
5. **TensorBoard Integration**: Comprehensive training monitoring

## Contributing

When making changes to reward normalization:
1. Document the change in this README
2. Add configuration parameters for new features
3. Maintain backward compatibility or clearly mark breaking changes
4. Test with fixed seeds to ensure reproducibility
