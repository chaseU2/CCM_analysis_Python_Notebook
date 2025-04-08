import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance
from scipy.stats import pearsonr
from matplotlib import colors
import os
from datetime import datetime
from ipywidgets import widgets, Layout
from IPython.display import display, clear_output

def run_ccm_analysis_jupyter(data_input, L=110, E=2, tau=1, THRESHOLD=0.8, save_output=True, output_dir=None):
    """
    Run Convergent Cross Mapping (CCM) analysis on time series data - optimized for Jupyter Notebooks.
    
    Parameters:
    - data_input: Either a filepath (str) to tab-separated data or a pandas DataFrame
    - L: Length of time series to consider
    - E: Embedding dimension 
    - tau: Time delay
    - THRESHOLD: Significance threshold for cross-map scores
    - save_output: Whether to save plots and protocol (default: True)
    - output_dir: Directory to save outputs (default: current working directory)
    
    Returns:
    - Final dataframe with significant CCM relationships
    """
    
    # Load data (accept either filepath or DataFrame)
    if isinstance(data_input, str):
        ccm_data = pd.read_csv(data_input, delimiter='\t')
    elif isinstance(data_input, pd.DataFrame):
        ccm_data = data_input.copy()
    else:
        raise ValueError("data_input must be either a filepath (str) or pandas DataFrame")
    
    # Set up output directory
    if save_output:
        if output_dir is None:
            output_dir = os.getcwd()
        
        # Create subdirectory with timestamp
        output_dir = os.path.join(output_dir, f"ccm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)
    
    # Helper functions
    def shadow_manifold(time_series_Y, L, E, tau):
        """Create shadow manifold from time series"""
        shadow_M = {}
        for t in range((E - 1) * tau, L):
            lag = []
            for t2 in range(0, E):
                lag.append(time_series_Y[t - t2 * tau])
            shadow_M[t] = lag
        return shadow_M

    def vec_dist_matrx(shadow_M):
        """Calculate distance matrix between vectors in shadow manifold"""
        vector = []
        steps = []
        vecs = []
        for a, b in shadow_M.items():
            vector.append((a, b))
        for i in vector:
            steps.append(i[0])
            vecs.append(i[1])
        steps = np.array(steps)
        vecs = np.array(vecs)
        distance_metrics = distance.cdist(vecs, vecs, metric="euclidean")
        return distance_metrics, steps

    def nearest_dist_and_step(timepoint_oi, steps, dist_matr):
        """Find nearest neighbors in shadow manifold"""
        index_timepoint = np.where(steps == timepoint_oi)
        dist_timepoint = dist_matr[index_timepoint].squeeze()
        nearest_indis = np.argsort(dist_timepoint)[1:E + 2]
        nearest_timesteps = steps[nearest_indis]
        nearest_distances = dist_timepoint[nearest_indis]
        return nearest_timesteps, nearest_distances

    def prediction(timepoint_oi, time_series_X, shadow_m):
        """Predict X values from Y's shadow manifold"""
        dist_matrix, steps = vec_dist_matrx(shadow_m)
        non_zero = 0.000001  # Small value to avoid division by zero
        nearest_timesteps, nearest_distances = nearest_dist_and_step(timepoint_oi, steps, dist_matrix)
        u = np.exp(-nearest_distances / np.max([non_zero, nearest_distances[0]]))
        w = u / np.sum(u)
        X_true = time_series_X[timepoint_oi]
        X_cor = np.array(time_series_X)[nearest_timesteps]
        X_hat = (w * X_cor).sum()
        return X_true, X_hat

    def find_causality(time_series_X, time_series_Y, L, E, tau):
        """Calculate cross-mapping skill between X and Y"""
        My = shadow_manifold(time_series_Y, L, E, tau)
        X_true_list = []
        X_hat_list = []
        
        # Check for constant time series
        if np.all(time_series_X == time_series_X[0]) or np.all(time_series_Y == time_series_Y[0]):
            return 0, 1  # Return zero correlation for constant series
            
        for t in list(My.keys()):
            X_true, X_hat = prediction(t, time_series_X, My)
            X_true_list.append(X_true)
            X_hat_list.append(X_hat)
        
        x, y = X_true_list, X_hat_list
        r, p = pearsonr(x, y)
        return r if not np.isnan(r) else 0, p

    # Initialize results dataframe
    results_df = pd.DataFrame(index=ccm_data.columns, columns=ccm_data.columns)
    
    # Calculate initial CCM matrix
    for species1 in ccm_data.columns:
        for species2 in ccm_data.columns:
            r, p = find_causality(ccm_data[species1], ccm_data[species2], L, E, tau)
            results_df.loc[species1, species2] = max(0, round(r, 4))  # Store only positive correlations
    
    results_df = results_df.fillna(0)

    # Create protocol file if output is enabled
    if save_output:
        protocol_path = os.path.join(output_dir, 'protocol.txt')
        with open(protocol_path, 'w', encoding='utf-8') as protocol:
            protocol.write(f"CCM Analysis Protocol - {datetime.now()}\n")
            protocol.write(f"Parameters used:\n")
            protocol.write(f"- L (time series length): {L}\n")
            protocol.write(f"- E (embedding dimension): {E}\n")
            protocol.write(f"- tau (time delay): {tau}\n")
            protocol.write(f"- THRESHOLD: {THRESHOLD}\n")
            protocol.write(f"- Output directory: {output_dir}\n")
            protocol.write(f"- Data source: {'DataFrame' if isinstance(data_input, pd.DataFrame) else data_input}\n\n")
            protocol.write("Initial CCM Matrix:\n")
            protocol.write(results_df.to_string())
            protocol.write("\n\n")

    # Convergence analysis
    L_range = range(5, L, 5)  # Range of library sizes to test
    convergence_results = pd.DataFrame(columns=['Species_X', 'Species_Y', 'X_to_Y_ρ', 'Y_to_X_ρ'])
    analyzed_pairs = set()
    
    # Store all pairs that need evaluation
    pairs_to_evaluate = []
    
    for i, species1 in enumerate(results_df.index):
        for j, species2 in enumerate(results_df.columns):
            if i == j or (species1, species2) in analyzed_pairs:
                continue
                
            score = results_df.iloc[i, j]
            if score <= THRESHOLD:
                continue
                
            analyzed_pairs.add((species1, species2))
            analyzed_pairs.add((species2, species1))
            
            # Calculate convergence
            x_to_y = []
            y_to_x = []
            for L_val in L_range:
                r_xy, _ = find_causality(ccm_data[species1], ccm_data[species2], L_val, E, tau)
                r_yx, _ = find_causality(ccm_data[species2], ccm_data[species1], L_val, E, tau)
                x_to_y.append(r_xy if not np.isnan(r_xy) else 0)
                y_to_x.append(r_yx if not np.isnan(r_yx) else 0)
            
            pairs_to_evaluate.append({
                'species1': species1,
                'species2': species2,
                'x_to_y': x_to_y,
                'y_to_x': y_to_x,
                'score': score,
                'decision': None  # Will store the user's decision
            })

    # Create a class to hold the final result
    class ResultHolder:
        def __init__(self):
            self.result = None
    
    result_holder = ResultHolder()

    # Create interactive evaluation interface
    current_index = 0
    
    def create_decision_buttons(pair):
        """Helper function to create decision buttons with current pair"""
        return widgets.RadioButtons(
            options=[
                ('None', 0),
                ('Both directions', 1),
                (f"{pair['species1']}→{pair['species2']} only", 2),
                (f"{pair['species2']}→{pair['species1']} only", 3)
            ],
            value=0,  # Always initialize to 'None'
            description='Decision:',
            disabled=False,
            layout={'width': 'max-content'}
        )
    
    # Create initial widgets
    status_label = widgets.Label(value=f"Pair 1 of {len(pairs_to_evaluate)}: {pairs_to_evaluate[0]['species1']} ↔ {pairs_to_evaluate[0]['species2']}")
    decision_buttons = create_decision_buttons(pairs_to_evaluate[0])
    
    prev_button = widgets.Button(
        description='Previous',
        disabled=True,
        button_style='',
        tooltip='Previous pair',
        layout=Layout(width='100px')
    )
    
    next_button = widgets.Button(
        description='Next',
        disabled=len(pairs_to_evaluate) <= 1,
        button_style='',
        tooltip='Next pair',
        layout=Layout(width='100px')
    )
    
    finish_button = widgets.Button(
        description='Finish Evaluation',
        disabled=False,
        button_style='success',
        layout=Layout(width='150px')
    )
    
    buttons_box = widgets.HBox([prev_button, next_button, finish_button])
    output = widgets.Output()
    
    # Create main container for all widgets
    main_container = widgets.VBox([status_label, decision_buttons, buttons_box, output])
    display(main_container)
    
    def update_display(index):
        nonlocal current_index, decision_buttons
        current_index = index
        pair = pairs_to_evaluate[index]
        
        # Update status label
        status_label.value = f"Pair {index+1} of {len(pairs_to_evaluate)}: {pair['species1']} ↔ {pair['species2']}"
        
        # Create new decision buttons with current pair and reset to 'None'
        new_decision_buttons = create_decision_buttons(pair)
        
        # Store the None selection in the pairs data
        pair['decision'] = 0
        
        # Set up event handler before displaying
        new_decision_buttons.observe(on_decision_change, names='value')
        
        # Replace the old buttons in the container
        main_container.children = (status_label, new_decision_buttons, buttons_box, output)
        decision_buttons = new_decision_buttons
        
        # Update button states
        prev_button.disabled = index <= 0
        next_button.disabled = index >= len(pairs_to_evaluate) - 1
        
        with output:
            clear_output(wait=True)
            
            # Create new plot
            plt.figure(figsize=(12, 7))
            plt.plot(L_range, pair['x_to_y'], 'b-o', label=f"{pair['species1']} → {pair['species2']}")
            plt.plot(L_range, pair['y_to_x'], 'r-s', label=f"{pair['species2']} → {pair['species1']}")
            plt.xlabel('Library Size (L)')
            plt.ylabel('Cross-Map Skill (ρ)')
            plt.title(f"CCM Convergence: {pair['species1']} ↔ {pair['species2']}\nScore = {pair['score']:.2f}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    def on_prev_clicked(b):
        if current_index > 0:
            update_display(current_index - 1)
    
    def on_next_clicked(b):
        if current_index < len(pairs_to_evaluate) - 1:
            update_display(current_index + 1)
    
    def on_decision_change(change):
        pairs_to_evaluate[current_index]['decision'] = change['new']
    
    def on_finish_clicked(b):
        # Check if all pairs have been evaluated
        unevaluated = [i for i, pair in enumerate(pairs_to_evaluate) if pair['decision'] is None]
        if unevaluated:
            print(f"Warning: {len(unevaluated)} pairs still unevaluated. Proceeding anyway.")
        
        # Process the results
        process_results()
    
    def process_results():
        # Clear the interactive widgets
        clear_output()
        
        # Process decisions
        for pair in pairs_to_evaluate:
            decision = pair['decision'] if pair['decision'] is not None else 0
            species1 = pair['species1']
            species2 = pair['species2']
            x_to_y = pair['x_to_y']
            y_to_x = pair['y_to_x']
            
            # Store results based on decision
            if decision == 1:
                convergence_results.loc[len(convergence_results)] = [species1, species2, x_to_y[-1], y_to_x[-1]]
            elif decision == 2:
                convergence_results.loc[len(convergence_results)] = [species1, species2, x_to_y[-1], 0]
            elif decision == 3:
                convergence_results.loc[len(convergence_results)] = [species1, species2, 0, y_to_x[-1]]
            
            # Save plot if output is enabled
            if save_output:
                plot_path = os.path.join(output_dir, f'convergence_{species1}_vs_{species2}.png')
                plt.figure(figsize=(12, 7))
                plt.plot(L_range, x_to_y, 'b-o', label=f"{species1} → {species2}")
                plt.plot(L_range, y_to_x, 'r-s', label=f"{species2} → {species1}")
                plt.xlabel('Library Size (L)')
                plt.ylabel('Cross-Map Skill (ρ)')
                plt.title(f"CCM Convergence: {species1} ↔ {species2}\nScore = {pair['score']:.2f}")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                with open(protocol_path, 'a', encoding='utf-8') as protocol:
                    protocol.write(f"\nPair: {species1} <-> {species2}\n")
                    protocol.write(f"ρ values: {x_to_y[-1]:.2f}, {y_to_x[-1]:.2f}\n")
                    protocol.write(f"Decision: {decision}\n")
        
        # Create final heatmap with only significant relationships
        significant_species = sorted(set(convergence_results[['Species_X', 'Species_Y']].values.flatten()))
        final_matrix = pd.DataFrame(0, index=significant_species, columns=significant_species)
        
        for _, row in convergence_results.iterrows():
            if row['X_to_Y_ρ'] >= THRESHOLD:
                final_matrix.loc[row['Species_X'], row['Species_Y']] = row['X_to_Y_ρ']
            if row['Y_to_X_ρ'] >= THRESHOLD:
                final_matrix.loc[row['Species_Y'], row['Species_X']] = row['Y_to_X_ρ']
        
        # Store the final result
        result_holder.result = final_matrix
        
        # Create final heatmap
        plt.figure(figsize=(12, 10))
        cmap = plt.cm.Reds
        cmap.set_under('white')
        norm = colors.Normalize(vmin=THRESHOLD, vmax=1)

        # Plot the heatmap
        im = plt.imshow(final_matrix, cmap=cmap, norm=norm, aspect='auto')

        # Add values to cells
        for i in range(len(final_matrix.index)):
            for j in range(len(final_matrix.columns)):
                if i != j and final_matrix.iloc[i, j] >= THRESHOLD:
                    plt.text(j, i, f"{final_matrix.iloc[i, j]:.2f}",
                            ha='center', va='center',
                            color='black' if final_matrix.iloc[i, j] < THRESHOLD+(1-THRESHOLD)/2 else 'white',
                            fontsize=9, weight='bold')

        # Set ticks and labels
        plt.xticks(range(len(final_matrix.columns)), final_matrix.columns, rotation=90, fontsize=10)
        plt.yticks(range(len(final_matrix.index)), final_matrix.index, fontsize=10)

        # Add grid lines
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, len(final_matrix.columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(final_matrix.index)), minor=True)
        ax.grid(which='minor', color='lightgray', linestyle='-', linewidth=0.5)

        # Add colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label(f'Cross-Map Score (ρ >={THRESHOLD})', rotation=270, labelpad=20)

        plt.title(f'Significant CCM Relationships (ρ >= {THRESHOLD})', pad=20, fontsize=14)
        plt.tight_layout()

        if save_output:
            final_heatmap_path = os.path.join(output_dir, 'significant_ccm_relationships.png')
            plt.savefig(final_heatmap_path, dpi=300, bbox_inches='tight')
            with open(protocol_path, 'a', encoding='utf-8') as protocol:
                protocol.write("\nFinal Results:\n")
                protocol.write(final_matrix.to_string())
        plt.show()

        if save_output:
            print(f"\nAnalysis complete. Results saved in: {output_dir}")
    
    # Set up event handlers
    prev_button.on_click(on_prev_clicked)
    next_button.on_click(on_next_clicked)
    decision_buttons.observe(on_decision_change, names='value')
    finish_button.on_click(on_finish_clicked)
    
    # Initial display
    update_display(0)
    
    # Return the result holder which will contain the final result
    return result_holder