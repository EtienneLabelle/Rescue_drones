# TensorBoard Integration for Drone Simulation

This project now uses TensorBoard for comprehensive training monitoring instead of the old `simulation_log.txt` file.

## Features

### Training Metrics Tracked
- **Average Reward**: Overall performance across all agents
- **Average Loss**: Neural network training loss
- **Average Coverage**: Disaster zone coverage percentage
- **Corner Rushing Rate**: Percentage of time drones spend in corners
- **Noise Scale**: Exploration noise decay over time
- **Individual Agent Rewards**: Per-agent performance tracking

### Final Simulation Metrics
- **Final Average Reward**: Performance in the final simulation
- **Final Average Coverage**: Coverage achieved in final simulation

## Usage

### 1. Run Training
```bash
python main.py
```

Training logs will be automatically saved to:
```
runs/drone_simulation_YYYYMMDD_HHMMSS/
```

### 2. View Training Logs

#### Option A: Using the provided script
```bash
python start_tensorboard.py
```

#### Option B: Manual TensorBoard command
```bash
tensorboard --logdir=runs --port=6006
```

#### Option C: Using Python module
```bash
python -m tensorboard.main --logdir=runs --port=6006
```

### 3. Access TensorBoard
Open your web browser and go to:
```
http://localhost:6006
```

## TensorBoard Dashboard

### Training Tab
- **Scalars**: All training metrics over time
- **Graphs**: Neural network architecture visualization
- **Distributions**: Weight and gradient distributions
- **Histograms**: Parameter value distributions

### Key Metrics to Monitor

1. **Average Reward**: Should increase over time
2. **Average Loss**: Should decrease and stabilize
3. **Corner Rushing Rate**: Should decrease over time
4. **Average Coverage**: Should increase over time

## Log Structure

```
runs/
└── drone_simulation_YYYYMMDD_HHMMSS/
    ├── events.out.tfevents.*
    └── ...
```

## Troubleshooting

### TensorBoard not found
```bash
pip install tensorboard
```

### No logs found
- Make sure you've run `python main.py` first
- Check that the `runs/` directory exists
- Verify that training completed successfully

### Port already in use
```bash
tensorboard --logdir=runs --port=6007
```

## Benefits Over Text Logging

1. **Real-time Monitoring**: View training progress in real-time
2. **Interactive Plots**: Zoom, pan, and explore metrics
3. **Multiple Metrics**: Compare different metrics simultaneously
4. **Historical Comparison**: Compare different training runs
5. **Professional Interface**: Clean, modern dashboard
6. **Export Capabilities**: Save plots and data for reports

## Migration from simulation_log.txt

The old `simulation_log.txt` file has been completely replaced with TensorBoard logging. All the information that was previously logged to text files is now available in a much more comprehensive and interactive format through TensorBoard. 