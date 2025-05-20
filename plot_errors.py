import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle # For bounding boxes
import difflib
import random
import os

# --- Helper for Trigram Highlighting ---
def get_highlighted_trigram_prediction(source_str: str, target_str: str, pred_str: str) -> tuple[list, list, list]:
    """
    Compares target and prediction, highlights errors and their trigram context in red.
    Handles insertions and deletions with placeholders.
    Returns:
        - pred_display_parts: list of {'text': char, 'color': color_str} for prediction
        - target_display_parts: list of {'text': char, 'color': color_str} for target (context highlighting)
        - source_display_parts: list of {'text': char, 'color': 'black'} for source (no highlighting)
    """
    matcher = difflib.SequenceMatcher(None, target_str, pred_str, autojunk=False)
    opcodes = matcher.get_opcodes()

    pred_display_parts = []
    target_display_parts = [] # To show context in target as well

    # Initialize all characters to black
    # We will change colors based on error contexts
    pred_colors = ['black'] * len(pred_str)
    target_colors = ['black'] * len(target_str)

    # Mark error locations and their neighbors
    error_indices_pred = set()
    error_indices_target = set()

    # First pass: identify all directly affected indices by errors
    for tag, i1, i2, j1, j2 in opcodes:
        if tag != 'equal':
            # Mark characters involved in the error
            for k in range(j1, j2): # Prediction indices
                error_indices_pred.add(k)
            for k in range(i1, i2): # Target indices
                error_indices_target.add(k)

    # Second pass: expand to trigrams for marked error indices
    # For prediction string
    final_error_indices_pred = set(error_indices_pred) # Start with direct errors
    for idx in error_indices_pred:
        if idx - 1 >= 0:
            final_error_indices_pred.add(idx - 1)
        if idx + 1 < len(pred_str):
            final_error_indices_pred.add(idx + 1)
    
    # For target string
    final_error_indices_target = set(error_indices_target)
    for idx in error_indices_target:
        if idx - 1 >= 0:
            final_error_indices_target.add(idx - 1)
        if idx + 1 < len(target_str):
            final_error_indices_target.add(idx + 1)

    # Construct display parts for prediction
    current_pred_idx = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            for k in range(j1, j2):
                color = 'red' if k in final_error_indices_pred else 'black'
                pred_display_parts.append({'text': pred_str[k], 'color': color})
            current_pred_idx = j2
        elif tag == 'replace':
            for k in range(j1, j2): # Mismatched characters
                pred_display_parts.append({'text': pred_str[k], 'color': 'red'})
            current_pred_idx = j2
        elif tag == 'delete': # Character in target, not in prediction
            # Add placeholder for each deleted char in target
            for _ in range(i2 - i1):
                 # Placeholder and its potential neighbors in pred (if any) would be based on context.
                 # Here, we mark this slot as an error.
                pred_display_parts.append({'text': '·', 'color': 'red'}) # Placeholder for deletion
        elif tag == 'insert': # Character in prediction, not in target
            for k in range(j1, j2):
                pred_display_parts.append({'text': pred_str[k], 'color': 'red'})
            current_pred_idx = j2
            
    # Construct display parts for target
    for k in range(len(target_str)):
        color = 'red' if k in final_error_indices_target else 'black'
        target_display_parts.append({'text': target_str[k], 'color': color})
        
    source_display_parts = [{'text': char, 'color': 'black'} for char in source_str]

    return source_display_parts, target_display_parts, pred_display_parts


def plot_text_sequence_on_ax(ax, x_start, y_pos, parts, font_props, renderer, fig_width):
    """Plots a sequence of text parts (char, color) on a given axis."""
    current_x = x_start
    for part in parts:
        char_text_obj = ax.text(current_x, y_pos, part['text'],
                                color=part['color'],
                                va='center', ha='left',
                                fontsize=10, # Slightly smaller for fitting more
                                fontproperties=font_props)
        try:
            bbox = char_text_obj.get_window_extent(renderer)
            char_width_fig_coords = bbox.width / fig_width
            current_x += char_width_fig_coords
        except Exception:
            # Fallback width (less accurate)
            char_spacing_estimate_norm = 0.015
            spacing_factor = 1.1 if '\u0900' <= part['text'] <= '\u097F' else 1.0
            current_x += char_spacing_estimate_norm * spacing_factor
    return current_x # Return the final x position

def plot_collated_comparisons(samples_data, font_path_or_name=None):
    num_samples = len(samples_data)
    # Adjust figsize: width for text, height proportional to num_samples
    # Each sample needs about 0.8 - 1.0 units of height in figure coords.
    fig_height = max(5, num_samples * 1.0)
    fig_width_inches = 12
    fig, axes = plt.subplots(num_samples, 1, figsize=(fig_width_inches, fig_height), squeeze=False)
    axes = axes.flatten() # Ensure axes is always a 1D array

    # --- Font Selection (Simplified for brevity, reuse previous robust logic if needed) ---
    font_props_to_use = None
    final_font_description = "Default"
    if font_path_or_name and font_path_or_name.lower().endswith(('.ttf', '.otf')) and os.path.exists(font_path_or_name):
        try:
            font_props_to_use = fm.FontProperties(fname=font_path_or_name)
            final_font_description = os.path.basename(font_path_or_name)
            print(f"Using font file: '{final_font_description}' for all plots.")
        except Exception as e:
            print(f"Warning: Could not load specified font file. {e}. Using fallback.")
            font_props_to_use = fm.FontProperties(family='monospace') # Fallback
            final_font_description = "monospace (fallback)"
    else: # Fallback if no specific ttf is given or found
        font_props_to_use = fm.FontProperties(family='monospace')
        final_font_description = "monospace (default)"
        print(f"Using default font: {final_font_description}")
    # --- End Font Selection ---

    fig_abs_width = fig.get_window_extent().width # For char width normalization

    for i, data in enumerate(samples_data):
        ax = axes[i]
        source, target, prediction = data['source'], data['target'], data['prediction']

        source_parts, target_parts, pred_parts = get_highlighted_trigram_prediction(source, target, prediction)
        
        # Store text objects to get their extents for the bounding box
        all_text_objects_in_subplot = []

        y_positions = {"source_label": 0.80, "target_label": 0.55, "prediction_label": 0.30}
        text_y_offset = -0.03 # Offset for actual text from label y
        x_label_start = 0.01
        x_text_start = 0.15 # Where the actual sequences start

        renderer = fig.canvas.get_renderer()

        # Source
        all_text_objects_in_subplot.append(ax.text(x_label_start, y_positions["source_label"], "Source:", va='center', ha='left', fontsize=10, fontproperties=font_props_to_use, weight='bold'))
        final_x_source = plot_text_sequence_on_ax(ax, x_text_start, y_positions["source_label"] + text_y_offset,
                                                 [{'text': c, 'color': 'black'} for c in source], # Source always black
                                                 font_props_to_use, renderer, fig_abs_width)
        
        # Target
        all_text_objects_in_subplot.append(ax.text(x_label_start, y_positions["target_label"], "Target:", va='center', ha='left', fontsize=10, fontproperties=font_props_to_use, weight='bold'))
        final_x_target = plot_text_sequence_on_ax(ax, x_text_start, y_positions["target_label"] + text_y_offset,
                                                 target_parts, font_props_to_use, renderer, fig_abs_width)

        # Prediction
        all_text_objects_in_subplot.append(ax.text(x_label_start, y_positions["prediction_label"], "Prediction:", va='center', ha='left', fontsize=10, fontproperties=font_props_to_use, weight='bold'))
        final_x_pred = plot_text_sequence_on_ax(ax, x_text_start, y_positions["prediction_label"] + text_y_offset,
                                               pred_parts, font_props_to_use, renderer, fig_abs_width)

        ax.set_xlim(0, max(1.0, final_x_source, final_x_target, final_x_pred) + 0.05)
        ax.set_ylim(0, 1) # Each subplot is normalized from 0 to 1 in y
        ax.axis('off')

        # Add a bounding box around the content of this subplot
        # Get overall extent of text in this subplot for a tight box (optional, can be complex)
        # Simpler: use a fixed relative box or one based on y_positions
        rect_y_bottom = y_positions["prediction_label"] + text_y_offset - 0.10 # A bit below the prediction text
        rect_height = (y_positions["source_label"] + 0.05) - rect_y_bottom # From top of source label to bottom
        
        # For a simple box around each sample's area
        rect = Rectangle((0.005, 0.05), 0.99, 0.90, # (x,y), width, height relative to subplot
                         transform=ax.transAxes,
                         linewidth=0.5, edgecolor='gray', facecolor='none', linestyle='--')
        ax.add_patch(rect)


    # fig.suptitle(f"{num_samples} Transliteration Samples (Font: {final_font_description})", fontsize=16, fontproperties=font_props_to_use, y=0.99) # y slightly adjusted for suptitle
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout to make space for suptitle
    
    output_filename = "collated_transliteration_comparisons.png"
    plt.savefig(output_filename, dpi=150) # Increase DPI for better text
    print(f"Saved collated plot: {output_filename}")
    plt.close(fig)

# --- Main script execution ---
if __name__ == "__main__":
    csv_file_path = 'test_predictions_attention.csv'
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file_path}' was not found.")
        exit()

    if not {'source', 'target', 'prediction'}.issubset(df.columns):
        print("Error: CSV file must contain 'source', 'target', and 'prediction' columns.")
        exit()

    df = df.astype(str).fillna('')

    num_samples_to_plot = 10 # Fixed to 10 as requested
    if len(df) < num_samples_to_plot :
        print(f"Warning: Requested {num_samples_to_plot} samples, but CSV only has {len(df)}. Plotting all.")
        num_samples_to_plot = len(df)
        
    if num_samples_to_plot == 0:
        print("The CSV file is empty or resulted in 0 samples. No plot generated.")
        exit()
        
    # Randomly sample WITHOUT a fixed random_state for true randomness each run
    sampled_df = df.sample(n=num_samples_to_plot) 

    print(f"\nPlotting {num_samples_to_plot} random samples into a single figure...")

    font_file_to_use = "TiroDevanagariHindi-Regular.ttf"
    if not os.path.exists(font_file_to_use):
        print(f"ERROR: Font file '{font_file_to_use}' not found. Using system fallback.")
        font_file_to_use = None # Allow fallback

    samples_for_plot = []
    for _, row in sampled_df.iterrows():
        samples_for_plot.append({
            'source': row.get('source', ''),
            'target': row.get('target', ''),
            'prediction': row.get('prediction', '')
        })
    
    plot_collated_comparisons(
        samples_for_plot,
        font_path_or_name=font_file_to_use
    )
    print("Done plotting.")