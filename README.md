
# 🧬 Genetic Algorithm for Set Covering Problem (SCP)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> A genetic algorithm implementation to solve the Set Covering Problem, optimizing for both coverage and efficiency.

## 📝 Table of Contents
- [About](#about)
- [Algorithm Details](#algorithm-details)
- [Performance Analysis](#performance-analysis)
- [Improvements](#improvements)
- [Getting Started](#getting-started)
- [Results](#results)
- [Future Work](#future-work)

## 🎯 About <a name="about"></a>
This project implements a genetic algorithm to tackle the Set Covering Problem (SCP). The solution uses binary string representation where each bit corresponds to a subset in the collection, with 1 indicating inclusion and 0 indicating exclusion.

## 🔧 Algorithm Details <a name="algorithm-details"></a>

### Solution Representation
```
[1 0 1 0 1 1 0] - Example chromosome
 ↑ ↑ ↑ ↑ ↑ ↑ ↑
 S₁S₂S₃S₄S₅S₆S₇  - Subsets
```

### Key Components
- **Encoding**: Binary string
- **Default Mutation Rate**: 0.01
- **Selection**: Fitness-based
- **Crossover**: Multi-point
- **Population**: Randomly initialized

## 📊 Performance Analysis <a name="performance-analysis"></a>

### Collection Size Impact
| Collection Size | Mean Fitness | Standard Deviation |
|----------------|--------------|-------------------|
| 50 subsets     | 48.10       | ±1.97            |
| 150 subsets    | 61.60       | ±1.54            |
| 250 subsets    | 71.09       | ±1.78            |
| 350 subsets    | 65.67       | ±2.01            |

## ⚡ Improvements <a name="improvements"></a>

### 1. Dual Child Insertion
- **Before**: Fitness = 58, Minimum subsets = 63
- **After**: Fitness = 65, Minimum subsets = 49
- **Implementation**:
  ```python
  def crossover(parent1, parent2):
      # Return two children instead of one
      point = random.randint(1, len(parent1)-1)
      child1 = parent1[:point] + parent2[point:]
      child2 = parent2[:point] + parent1[point:]
      return child1, child2
  ```

### 2. Adaptive Mutation Rate
- **Initial Rate**: 0.3
- **Final Rate**: 0.01
- **Generations**: Increased to 250
- **Results**: Fitness = 69, Minimum subsets = 45

### 3. Time-Constrained Execution
- **Time Limit**: 40 seconds
- **Results**: Fitness = 73, Minimum subsets = 39

### 4. Elitism Implementation
- **Final Results**: 
  - Fitness: 86
  - Minimum subsets: 21

## 🚀 Getting Started <a name="getting-started"></a>

### Prerequisites
```bash
python >= 3.7
numpy
pandas
```

### Installation
```bash
git clone https://github.com/KislayTandon22/Genetic-Algorithm-for-Set-Covering-Problem.git
cd scp-genetic-algorithm
pip install -r requirements.txt
```

## 📈 Results <a name="results"></a>

### Key Findings
- Optimal performance at 150 subsets (61.60 ± 1.54)
- Performance plateaus beyond 250 subsets
- Early generations show fastest improvement
- Smaller collections demonstrate more stability

### Trends
- Mean subset size decreases over generations
- Larger collections → larger subset sizes
- Rapid early-generation improvements
- Enhanced stability in smaller collections

## 🔮 Future Work <a name="future-work"></a>

### Planned Improvements
- [ ] Parameter tuning optimization
- [ ] Alternative genetic operators
- [ ] Large-scale problem instance handling
- [ ] Enhanced elitism strategies

## 📄 License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## ✨ Contributors
- Kislay Ranjan Nee Tandon([KislayTandon22](https://github.com/KislayTandon22))

---
<p align="center">Made with ❤️ for Artificial Intelligence Course</p>
