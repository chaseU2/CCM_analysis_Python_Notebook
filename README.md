# CCM Analysis for Python Notebooks
[![PyPI version](https://img.shields.io/pypi/v/ccm-analysis-pynb.svg)](https://pypi.org/project/ccm-analysis-pynb/)

Package implementing Convergent Cross Mapping for causality inference in dynamical systems as defined by ["Sugihara et al (2012)"](https://www.science.org/doi/10.1126/science.1227079).
> **Note:** This package is optimized for Jupyter Notebooks (`.ipynb`). If you need to perform CCM analysis in a Python script (`.py`), please use [CCM_analysis_Python](https://github.com/chaseU2/CCM_analysis_Python).

## Installation

You can install the package via pip:

```bash
pip install ccm_analysis_pynb
```

## Usage

To use this package in a Jupyter Notebook, import the function and run the analysis:

```python
from ccm_analysis_pynb import run_ccm_analysis_jupyter
import pandas as pd

data = pd.read_csv("example_data.tsv", delimiter="\t")

# Run CCM analysis
result_holder = run_ccm_analysis_jupyter(data, L=100, E=2, tau=1, THRESHOLD=0.8, save_output=True, output_dir="output")

# Access the final DataFrame of significant relationships
final_results = result_holder.result
print(final_results)
```

## Parameters

- `data_input`: Either a filepath (str) to tab-separated data or a pandas DataFrame.
- `L`: Length of time series to consider.
- `E`: Embedding dimension.
- `tau`: Time delay.
- `THRESHOLD`: Significance threshold for cross-map scores.
- `save_output`: Whether to save plots and protocol (default: True).
- `output_dir`: Directory to save outputs (default: current working directory).

## Output

The function returns an interactive widget for manual evaluation of relationships. Once the evaluation is completed, it generates a final heatmap and saves the results if `save_output=True`. You can access the output dataframe by using:

```python

result_holder.result

```
- result_holder is the variable you stored the function's output in. You can rename it as needed.

## Test Dataset

You can test the `ccm_analysis` package using the sample dataset provided in this repository. You can download the test dataset from the following link:

- [Test Dataset (ccm_test_data.txt)](https://github.com/chaseU2/ccm-analysis-tool/blob/master/ccm_test_data.txt)

  

### Data Format

The dataset is structured as follows:

- **Columns** represent the variables you want to analyze.
- **Rows** represent the time points for each variable.


## Interpreting Convergence Plots

The package generates diagnostic convergence plots that reveal causal relationships between variables. Below are the three characteristic patterns to analyze:





Here you have to take a look at the shown plot and decide in which directions you can see a convergence of the crossmap skill with increasing library size.
You have to click at one of the options shown and then on next. When you are done you have to klick on Finish Evaluation


---

### 1. No Significant Causality

![No Causal Relationship](https://raw.githubusercontent.com/chaseU2/CCM_analysis_Python_Notebook/main/ccm_analysis_pynb/Screenshot%2013.png)

- Neither directional curve (X→Y in blue, Y→X in red) show a clear convergence to a final crossmap score
- Example use case: Independent systems

---

### 2. Unidirectional Causality
![Unidirectional Causality](https://raw.githubusercontent.com/chaseU2/CCM_analysis_Python_Notebook/main/ccm_analysis_pynb/Screenshot%2012.png)

- One direction (X→Y) converges to a final crossmap score
- Reverse direction (Y→X) does ot show a clear convegence
- Interpretation: X drives Y but not vice versa
- Pay attention to whether convergence is present in the X→Y or Y→X plot, and enter 2 or 3 accordingly

---

### 3. Bidirectional Causality
![Bidirectional Causality](https://raw.githubusercontent.com/chaseU2/CCM_analysis_Python_Notebook/main/ccm_analysis_pynb/Screenshot%2011.png)

- Both directions show positive convergence
- Typical of feedback systems
- Convergence rates may differ (e.g., X→Y stronger than Y→X)


---

## Dependencies and Acknowledgements

Parts of this project are based on the Convergent Cross Mapping (CCM) implementation from the following repository from Prince Javier :

- [Convergent Cross Mapping GitHub Repository Prince Javier ](https://github.com/PrinceJavier/causal_ccm.git)

I have utilized parts of the CCM algorithm from this repository to help analyze causality in time series data in my own project.
