# SlicVision-F2I: Network Slice KPIs to Visual Representation Dataset

![Dataset Overview](https://raw.githubusercontent.com/AbidHasanRafi/SliceVision-F2I/main/assets/header.png)

SlicVision-F2I is a novel dataset that transforms network slice Key Performance Indicators (KPIs) into multiple visual representation patterns, designed for machine learning and deep learning applications in network slicing management.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Key Features](#key-features)
- [Dataset Structure](#dataset-structure)
- [Visual Representation Patterns](#visual-representation-patterns)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Dataset Statistics](#dataset-statistics)
- [License](#license)
- [Citation](#citation)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

## Dataset Description

SlicVision-F2I contains 30,000 samples of network slice KPIs converted into four distinct visual representation patterns. Each sample represents one of three network slice types:

1. **eMBB (Enhanced Mobile Broadband)**
2. **URLLC (Ultra-Reliable Low-Latency Communications)**
3. **mIoT (Massive Internet of Things)**

The dataset bridges the gap between network telemetry data and computer vision approaches by providing multiple visual representations of the same underlying KPI data.

## Key Features

- **Multi-pattern representations**: Each KPI sample is converted into four distinct visual patterns
- **Comprehensive KPI coverage**: 10 key network performance metrics
- **Realistic simulations**: Includes noise, missing values, and class imbalance
- **High-quality preprocessing**: Careful normalization and missing value handling
- **Ready-to-use**: Pre-generated NumPy arrays for easy integration with ML pipelines

## Dataset Structure

The dataset is organized as follows:

```
SlicVision-F2I/
├── numeric_data.csv            # Raw KPI measurements and labels
├── guided_patterns.npy         # Physically-guided pattern representations
├── perlin_patterns.npy         # Perlin noise-based patterns
├── wallpaper_patterns.npy      # Wallpaper-style patterns
└── fractal_patterns.npy        # Fractal branching patterns
```

### Data Fields in numeric_data.csv

| Field | Description | Range/Values |
|-------|------------|--------------|
| slice_type | Network slice type | eMBB, URLLC, mIoT |
| delay | Packet delay in seconds | 0-0.1s |
| jitter | Delay variation in seconds | 0-0.05s |
| loss | Packet loss rate | 0-0.1 |
| throughput | Data throughput in Mbps | 0-300Mbps |
| retransmissions | Packet retransmission rate | 0-0.1 |
| packet_discard_rate | Discarded packets rate | 0-0.1 |
| rssi | Received Signal Strength Indicator | -100 to -30 dBm |
| snr | Signal-to-Noise Ratio | 0-40 dB |
| cpu_util | CPU utilization percentage | 0-100% |
| mem_util | Memory utilization percentage | 0-100% |
| label | Encoded class label | 0 (eMBB), 1 (URLLC), 2 (mIoT) |

## Visual Representation Patterns

### 1. Physically-Guided Patterns
![Guided Patterns](https://raw.githubusercontent.com/AbidHasanRafi/SliceVision-F2I/main/assets/guided.png) 
- Incorporates physical relationships between KPIs
- Uses gaussian blobs, gradients, and wave patterns
- Channel assignments:
  - Red: Delay, jitter, loss
  - Green: Throughput, retransmissions
  - Blue: RSSI, SNR, CPU/Memory

### 2. Perlin Noise Patterns
![Perlin Patterns](https://raw.githubusercontent.com/AbidHasanRafi/SliceVision-F2I/main/assets/perlin.png)
- Generated using Perlin noise with KPI-parameterized settings
- Each channel uses different noise parameters based on related KPIs
- Provides organic, natural-looking patterns

### 3. Wallpaper Patterns
![Wallpaper Patterns](https://raw.githubusercontent.com/AbidHasanRafi/SliceVision-F2I/main/assets/wallpaper.png)
- Periodic and geometric patterns
- Combines sinusoidal waves, grids, and gradients
- Designed to highlight periodic behaviors in network traffic

### 4. Fractal Branching Patterns
![Fractal Patterns](https://raw.githubusercontent.com/AbidHasanRafi/SliceVision-F2I/main/assets/fractal.png)
- Recursive branching structures
- Branch characteristics determined by KPIs
- Represents network paths and connectivity

## Getting Started

### Loading the Dataset
```python
import numpy as np
import pandas as pd

# Load numeric data
df = pd.read_csv('numeric_data.csv')

# Load pattern data (example for guided patterns)
guided_patterns = np.load('guided_patterns.npy')
```

## Usage Examples
Training a Simple Classifier
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Prepare data
X = guided_patterns.reshape(len(guided_patterns), -1)  # Flatten images
y = df['label'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train classifier
clf = RandomForestClassifier()
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")
```

## Dataset Statistics

| Statistic | Value |
|-----------|-------|
| Total samples | 30,000 |
| eMBB samples | 6,000 (20%) |
| URLLC samples | 3,000 (10%) |
| mIoT samples | 21,000 (70%) |
| Image size | 16x16 pixels |
| Channels per image | 3 (RGB) |
| Missing values | ~5% (imputed) |

The dataset intentionally follows a realistic class imbalance reflecting expected slice distribution in real networks.

## License

This dataset is released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use SlicVision-F2I in your research, please cite it as:

```bibtex
@dataset{slicvision_f2i_2023,
  author = {Your Name},
  title = {SlicVision-F2I: Network Slice KPIs to Visual Representation Dataset},
  year = {2025},
  publisher = {GitHub},
  version = {1.0},
  url = {https://github.com/abidhasanrafi/slicvision-f2i}
}
```

## Contributing

Contributions to improve the dataset are welcome! Please open an issue or submit a pull request for:
- Additional visual representations
- Improved KPI simulations
- Documentation enhancements

## Acknowledgments

- Inspired by prior work in network slicing and visual representation learning
- Thanks to contributors who helped with testing and validation
- Built with Python scientific computing ecosystem