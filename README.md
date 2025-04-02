# CCM Analysis

A Python package for performing Convergent Cross Mapping (CCM) analysis on time series data.

## Installation

You can install the package via pip:

```bash
pip install ccm_analysis_pynb
```

## Usage

To use this package in a Jupyter Notebook, import the function and run the analysis:

```python
from ccm_analysis_pynb import run_ccm_analysis_jupyter

df = pd.DataFrame(data)

# Run CCM analysis
result_holder = run_ccm_analysis_jupyter(df, L=100, E=2, tau=1, THRESHOLD=0.8, save_output=False)

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

The function returns an interactive widget for manual evaluation of relationships. Once the evaluation is completed, it generates a final heatmap and saves the results if `save_output=True`.

---

For more details, check the source code and documentation!
