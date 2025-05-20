import os
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.font_manager as fm 
matplotlib.use('Agg') # Non-interactive backend for scripts
import matplotlib.pyplot as plt
import numpy as np
import wandb # For wandb.Image
from typing import List, Tuple, Optional, Any

# Forward declaration for type hinting if model is complexly typed
# from model import Seq2Seq -> This would create a circular dependency if utils is imported by model
# Instead, we can pass model components or use 'Any' type for the model in function signatures.
FONT_FILE_PATH = "./TiroDevanagariHindi-Regular.ttf"


class BeamSearchNode:
    """Helper class for beam search tracking."""
    def __init__(self,
                 hidden_state: Any, # Can be Tensor or Tuple (for LSTM)
                 previous_node: Optional['BeamSearchNode'],
                 token_id: int,
                 log_prob: float,
                 length: int,
                 attention_step: Optional[torch.Tensor] = None): # (L_src)
        self.hidden_state = hidden_state
        self.previous_node = previous_node
        self.token_id = token_id
        self.log_prob = log_prob
        self.length = length
        self.attention_step = attention_step # Attention weights for this step

    def eval(self, alpha: float = 0.75) -> float:
        """Calculates the score for length penalty."""
        # Add small epsilon to prevent division by zero for length 1 (SOS only)
        return self.log_prob / float(self.length -1 + 1e-6) ** alpha

    def __lt__(self, other: 'BeamSearchNode') -> bool: # For sorting/priority queue
        return self.eval() < other.eval()


def plot_attention_heatmap(source_tokens: List[str],
                           predicted_tokens: List[str],
                           attention_weights: np.ndarray, # Should be (L_pred, L_src)
                           title: str = "Attention Heatmap",
                           font_path: Optional[str] = FONT_FILE_PATH) -> Optional[wandb.Image]: # Added font_path
    """
    Plots and returns a W&B Image of the attention heatmap using a specified font.
    Args:
        source_tokens (List[str]): List of source tokens (strings).
        predicted_tokens (List[str]): List of predicted target tokens (strings).
        attention_weights (np.ndarray): Attention matrix (target_len, source_len).
        title (str): Title for the plot.
        font_path (Optional[str]): Path to the .ttf or .otf font file.
    Returns:
        Optional[wandb.Image]: W&B Image object for logging, or None if plotting fails.
    """
    if not source_tokens or not predicted_tokens or attention_weights.size == 0:
        print(f"Warning (plot_attention): Cannot plot. Src: {len(source_tokens)}, Pred: {len(predicted_tokens)}, Attn shape: {attention_weights.shape}")
        return None

    valid_target_len = min(len(predicted_tokens), attention_weights.shape[0])
    valid_source_len = min(len(source_tokens), attention_weights.shape[1])

    if valid_target_len == 0 or valid_source_len == 0:
        print(f"Warning (plot_attention): Cannot plot. Valid Target Len: {valid_target_len}, Valid Source Len: {valid_source_len}")
        return None

    attention_to_plot = attention_weights[:valid_target_len, :valid_source_len]
    predicted_tokens_to_plot = predicted_tokens[:valid_target_len]
    source_tokens_to_plot = source_tokens[:valid_source_len]
    
    if not predicted_tokens_to_plot or not source_tokens_to_plot: # Should be redundant due to above check
        print("Warning (plot_attention): Empty tokens lists after slicing for plotting.")
        return None

    # --- Font Handling ---
    font_props = None
    font_description = "Default"
    if font_path and os.path.exists(font_path):
        try:
            font_props = fm.FontProperties(fname=font_path)
            font_description = os.path.basename(font_path)
            # print(f"Using font: {font_description} for attention heatmap.")
        except Exception as e:
            print(f"Warning (plot_attention): Could not load font at '{font_path}'. Error: {e}. Using matplotlib default.")
            # font_props will remain None, matplotlib will use its default
    else:
        if font_path: # Path was given but file doesn't exist
             print(f"Warning (plot_attention): Font file not found at '{font_path}'. Using matplotlib default.")
        # else: font_path was None, so use default (font_props remains None)
    # --- End Font Handling ---


    # Adjust figsize dynamically
    # Base size + size proportional to number of tokens
    fig_width = max(6, len(source_tokens_to_plot) * 0.3 + 2)  # Adjust factor 0.3 as needed
    fig_height = max(5, len(predicted_tokens_to_plot) * 0.3 + 2) # Adjust factor 0.3 as needed
    
    # Cap max size to prevent overly large figures
    max_fig_dim = 20 
    fig_width = min(fig_width, max_fig_dim)
    fig_height = min(fig_height, max_fig_dim)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    try:
        cax = ax.matshow(attention_to_plot, cmap='Blues', aspect='auto') # aspect='auto' can help with non-square matrices
        fig.colorbar(cax)

        ax.set_xticks(np.arange(len(source_tokens_to_plot)))
        ax.set_yticks(np.arange(len(predicted_tokens_to_plot)))

        # Apply font properties to tick labels
        ax.set_xticklabels(source_tokens_to_plot, rotation=90, ha="center", fontsize=8, fontproperties=font_props)
        ax.set_yticklabels(predicted_tokens_to_plot, fontsize=8, fontproperties=font_props)
        
        ax.xaxis.set_major_locator(plt.FixedLocator(np.arange(len(source_tokens_to_plot))))
        ax.yaxis.set_major_locator(plt.FixedLocator(np.arange(len(predicted_tokens_to_plot))))

        # Apply font properties to axis labels and title
        ax.set_xlabel("Source Sequence", fontsize=10, fontproperties=font_props)
        ax.set_ylabel("Predicted Sequence", fontsize=10, fontproperties=font_props)
        ax.set_title(f"{title} (Font: {font_description})", fontsize=12, fontproperties=font_props)
        
        plt.tight_layout(pad=0.5) # Add some padding
        
        wb_image = wandb.Image(fig)
    except Exception as e:
        print(f"Error during plotting attention map with font '{font_description}': {e}")
        import traceback
        traceback.print_exc()
        wb_image = None
    finally:
        plt.close(fig) 
    return wb_image

def beam_search_decode(model_s2s: Any, # Type: Seq2Seq
                       source_tensor: torch.Tensor,      # (L_src, 1)
                       source_length: torch.Tensor,      # (1)
                       target_sos_idx: int,
                       target_eos_idx: int,
                       beam_width: int,
                       max_len: int = 50,
                       device: torch.device = torch.device('cpu'),
                       length_penalty_alpha: float = 0.75
                      ) -> Tuple[List[int], Optional[torch.Tensor]]:
    """
    Performs beam search decoding for a single input sequence.
    Args:
        model_s2s (Seq2Seq): The trained Seq2Seq model.
        source_tensor (torch.Tensor): Encoded source sequence (L_src, 1).
        source_length (torch.Tensor): Length of the source sequence (1) on CPU.
        target_sos_idx (int): SOS token index for the target.
        target_eos_idx (int): EOS token index for the target.
        beam_width (int): Size of the beam.
        max_len (int): Maximum length of the decoded sequence.
        device (torch.device): Device to run decoding on.
        length_penalty_alpha (float): Alpha for length penalty.

    Returns:
        Tuple[List[int], Optional[torch.Tensor]]:
            - List of predicted token IDs for the best sequence (excluding SOS, including EOS if found).
            - Attention weights tensor (L_pred, L_src) for the best sequence if attention is used, else None.
    """
    model_s2s.eval() # Ensure model is in eval mode
    
    # Access encoder and decoder from the Seq2Seq model
    encoder = model_s2s.encoder
    decoder = model_s2s.decoder # This is the DecoderRNN module

    with torch.no_grad():
        # --- Encoder Pass ---
        encoder_all_outputs, encoder_final_full_hidden = encoder(source_tensor.to(device), source_length.to('cpu'))
        decoder_current_hidden_state = model_s2s._project_encoder_hidden_for_decoder_init(encoder_final_full_hidden)

        # Root node: Start with SOS token
        root_node = BeamSearchNode(
            hidden_state=decoder_current_hidden_state,
            previous_node=None,
            token_id=target_sos_idx, # The current token_id of this node
            log_prob=0.0,
            length=1, # SOS is the first token
            attention_step=None
        )
        
        live_hypotheses = [root_node]
        completed_hypotheses = []

        for step in range(max_len):
            if not live_hypotheses:
                break

            # Gather all current inputs and hidden states for potential batching (though this impl is iterative)
            num_live = len(live_hypotheses)
            
            # Store (new_node_score, new_node) for sorting later
            all_candidates_this_step = []

            for i in range(num_live):
                node = live_hypotheses[i]

                # Do not expand a node if it ends with EOS
                if node.token_id == target_eos_idx:
                    completed_hypotheses.append(node)
                    continue

                # Input to decoder is the token_id of the current node
                current_input_token = torch.tensor([node.token_id], dtype=torch.long, device=device)
                
                decoder_output_logits_step, new_hidden_state, attn_weights_step = decoder(
                    decoder_input_token=current_input_token,
                    decoder_prev_hidden_state=node.hidden_state,
                    encoder_all_outputs=encoder_all_outputs if decoder.attention else None
                ) # decoder_output_probs_step is (1, V_out), attn_weights_step is (1, L_src)
                
                decoder_output_probs_step = F.log_softmax(decoder_output_logits_step, dim=1)

                # Get top K next tokens (log_softmax output from decoder)
                top_k_log_probs, top_k_indices = torch.topk(decoder_output_probs_step.squeeze(0), beam_width)

                for k_idx in range(beam_width):
                    next_token_id = top_k_indices[k_idx].item()
                    log_prob_contrib = top_k_log_probs[k_idx].item()

                    new_node = BeamSearchNode(
                        hidden_state=new_hidden_state,
                        previous_node=node,
                        token_id=next_token_id,
                        log_prob=node.log_prob + log_prob_contrib,
                        length=node.length + 1,
                        attention_step=attn_weights_step.squeeze(0) if attn_weights_step is not None else None # (L_src)
                    )
                    all_candidates_this_step.append(new_node)
            
            # Prune candidates:
            # Sort all candidates from all beams by their score
            sorted_candidates = sorted(all_candidates_this_step, key=lambda x: x.eval(length_penalty_alpha), reverse=True)
            
            live_hypotheses = [] # Reset for this step
            for cand_node in sorted_candidates:
                if cand_node.token_id == target_eos_idx:
                    completed_hypotheses.append(cand_node)
                else:
                    live_hypotheses.append(cand_node)
                
                # Keep only top beam_width live hypotheses
                if len(live_hypotheses) >= beam_width:
                    break
            
            # Prune completed hypotheses if too many
            if len(completed_hypotheses) > beam_width: # Keep N best completed
                completed_hypotheses = sorted(completed_hypotheses, key=lambda x: x.eval(length_penalty_alpha), reverse=True)[:beam_width]

            if not live_hypotheses and completed_hypotheses: # All live hypotheses ended or were pruned
                break
        
        # If no completed hypotheses, use the best live ones
        if not completed_hypotheses:
            completed_hypotheses.extend(live_hypotheses)
        
        if not completed_hypotheses: # Still nothing (e.g., max_len=0 or beam_width=0)
            return [target_eos_idx], None # Return at least EOS

        # Select the best hypothesis from all completed ones
        best_hypothesis = sorted(completed_hypotheses, key=lambda x: x.eval(length_penalty_alpha), reverse=True)[0]

        # Reconstruct the sequence and attention weights by backtracking
        predicted_ids_rev = []
        attention_list_rev = []
        current_node = best_hypothesis
        while current_node.previous_node is not None: # Backtrack until the node *after* SOS
            predicted_ids_rev.append(current_node.token_id)
            if current_node.attention_step is not None:
                attention_list_rev.append(current_node.attention_step)
            current_node = current_node.previous_node
        
        # The loop stops when current_node is the SOS node.
        # We don't add SOS to predicted_ids.
        # If the best hypothesis only has SOS (e.g. max_len=1, beam forces EOS immediately)
        # handle this case.
        if not predicted_ids_rev and best_hypothesis.token_id == target_eos_idx and best_hypothesis.length == 1:
            # This case occurs if SOS immediately leads to EOS as the best option.
             predicted_ids_rev.append(target_eos_idx)


        predicted_ids = predicted_ids_rev[::-1] # Reverse to get correct order
        
        final_attention_weights = None
        if decoder.attention and attention_list_rev:
            # Stack attention weights from each step
            # attention_list_rev contains attentions for tokens t_1, t_2, ..., t_k (k=len(predicted_ids))
            # Each is (L_src). We stack to (L_pred, L_src)
            final_attention_weights = torch.stack(attention_list_rev[::-1]) # Reverse list then stack
            
        # Ensure EOS is appended if not naturally predicted and max_len is reached
        if predicted_ids and predicted_ids[-1] != target_eos_idx and len(predicted_ids) == max_len:
            predicted_ids.append(target_eos_idx)
            # Note: Attention for this appended EOS won't be available from beam search directly.

        return predicted_ids, final_attention_weights

def get_font_properties(font_path: Optional[str] = FONT_FILE_PATH) -> Optional[fm.FontProperties]:
    """Helper to load font properties."""
    font_props = None
    if font_path and os.path.exists(font_path):
        try:
            font_props = fm.FontProperties(fname=font_path)
        except Exception as e:
            print(f"Warning: Could not load font at '{font_path}'. Error: {e}. Using matplotlib default.")
    if not font_props: # Fallback
        try:
            font_props = fm.FontProperties(family='DejaVu Sans') # Common fallback
        except:
            font_props = fm.FontProperties(family='sans-serif') # Generic fallback
    return font_props

def plot_focused_attention_visualization(
    source_word: str,
    predicted_word: str,
    attention_weights: np.ndarray, # Shape: (predicted_len, source_len)
    focused_char_indices: Optional[List[int]] = None, # Optional: list of indices in predicted_word to focus on
    font_props: Optional[fm.FontProperties] = None,
    input_description: Optional[str] = None
) -> Optional[plt.Figure]:
    """
    Generates a plot visualizing attention for each (or selected) predicted character.

    Args:
        source_word (str): The input source word (e.g., "emilie").
        predicted_word (str): The predicted target word (e.g., "एमिली").
        attention_weights (np.ndarray): Attention matrix, shape (len(predicted_word), len(source_word)).
        focused_char_indices (Optional[List[int]]): If provided, only generate plots for these
                                                     indices of the predicted_word. Otherwise, plot for all.
        font_props (Optional[fm.FontProperties]): Font properties to use.
        input_description (Optional[str]): A description for the overall input, e.g., "INPUT: emilie".

    Returns:
        Optional[plt.Figure]: The matplotlib Figure object, or None if error.
    """
    if font_props is None:
        font_props = get_font_properties()

    if not source_word or not predicted_word or attention_weights.size == 0:
        print("Error: Missing source, prediction, or attention weights.")
        return None

    if attention_weights.shape[0] != len(predicted_word) or \
       attention_weights.shape[1] != len(source_word):
        print(f"Error: Attention weights shape {attention_weights.shape} "
              f"mismatch with predicted_word len {len(predicted_word)} "
              f"or source_word len {len(source_word)}.")
        return None

    indices_to_plot = focused_char_indices if focused_char_indices else range(len(predicted_word))
    
    num_rows = len(indices_to_plot)
    if num_rows == 0:
        print("No characters selected for focused attention plot.")
        return None

    # Dynamically adjust figure height based on number of rows
    # Each row needs space for text and the visualization
    row_height_inches = 0.8  # Approximate height per row
    header_height_inches = 0.7 if input_description else 0.2 # Space for overall title
    table_header_height_inches = 0.5 # Space for column headers
    fig_height = num_rows * row_height_inches + header_height_inches + table_header_height_inches
    
    fig, axes = plt.subplots(num_rows, 2, figsize=(8, fig_height),
                             gridspec_kw={'width_ratios': [1, 1]}) # Two columns

    # If only one row, axes might not be an array, so make it one
    if num_rows == 1:
        axes = np.array([axes])


    # Overall Title (Input Description)
    if input_description:
        fig.suptitle(input_description, fontsize=14, y=1.0 - (header_height_inches*0.5 / fig_height), fontproperties=font_props, weight='bold')


    # Column Headers
    if num_rows > 0 : # Add headers only if there are rows to plot
        # Create a dummy axes for the header if it's cleaner
        # Or, plot on the first row's axes, then adjust layout
        # For simplicity, let's use text on the figure, or adjust the first subplot's title area
        # A cleaner way is to add a new Axes object just for headers, but that's more complex.
        # Let's try to place them relative to the first subplot's available space or figure.
        
        # Using figure coordinates for headers for more control
        header_y_pos = 1.0 - (header_height_inches / fig_height) - (table_header_height_inches * 0.5 / fig_height)

        fig.text(0.05 + 0.45/2, header_y_pos, "Character in Prediction Focused",
                 ha='center', va='center', fontsize=12, fontproperties=font_props, weight='bold',
                 bbox=dict(boxstyle='square,pad=0.3', fc='lightgray', ec='black', lw=0.5))
        fig.text(0.5 + 0.45/2, header_y_pos, "Attention Visualization",
                 ha='center', va='center', fontsize=12, fontproperties=font_props, weight='bold',
                 bbox=dict(boxstyle='square,pad=0.3', fc='lightgray', ec='black', lw=0.5))


    for i, pred_char_idx in enumerate(indices_to_plot):
        ax_text = axes[i, 0]
        ax_viz = axes[i, 1]

        ax_text.axis('off')
        ax_viz.axis('off')

        # --- Column 1: Text Description ---
        focused_pred_char = predicted_word[pred_char_idx]
        # For better display of combining characters, join with ZWJ if needed, or just display
        
        desc_text = f"character at index {pred_char_idx} of {predicted_word}\n({focused_pred_char})"
        ax_text.text(0.5, 0.5, desc_text, ha='center', va='center', fontsize=11, fontproperties=font_props,
                     wrap=True)
        ax_text.axhline(0, color='gray', lw=0.5) # Bottom border for the cell
        ax_text.axvline(1, color='gray', lw=0.5) # Right border for the cell


        # --- Column 2: Attention Visualization ---
        attn_for_this_pred_char = attention_weights[pred_char_idx, :] # Shape: (source_len,)
        
        # Normalize for color intensity (0 to 1)
        # If all attentions are zero, this will result in NaNs or zeros.
        # Handle the case where sum is 0 to avoid division by zero.
        norm_attn = attn_for_this_pred_char
        if np.sum(attn_for_this_pred_char) > 0:
            norm_attn = attn_for_this_pred_char / np.max(attn_for_this_pred_char) # Scale by max for better visual range
            # norm_attn = attn_for_this_pred_char / np.sum(attn_for_this_pred_char) # Alternative: normalize to sum to 1

        # Plot source word with background color based on attention
        # We need to place each character and its background box manually
        char_box_width = 0.8 / len(source_word) # Normalized width for each char box
        char_box_height = 0.4
        start_x = 0.1 # Starting x position in ax_viz coordinates
        
        for j, src_char in enumerate(source_word):
            intensity = norm_attn[j]
            # Color: light yellow (low attention) to dark blue/green (high attention)
            # Using a sequential colormap (e.g., 'Greens', 'Blues', 'YlGn')
            # cmap = plt.cm.get_cmap('YlGn') # Example: Yellow to Green
            cmap = plt.cm.get_cmap('viridis') # Viridis is perceptually uniform

            # background_color = cmap(intensity * 0.8 + 0.1) # Scale intensity to use a good range of colormap
            background_color = cmap(intensity)
            
            # Text color: white for dark backgrounds, black for light backgrounds
            text_color = 'white' if intensity > 0.6 else 'black' # Adjust threshold

            char_x_center = start_x + (j + 0.5) * char_box_width
            
            ax_viz.text(char_x_center, 0.5, src_char,
                        ha='center', va='center', fontsize=12, color=text_color,
                        bbox=dict(boxstyle='square,pad=0.3', fc=background_color, ec='none'),
                        fontproperties=font_props if src_char.isalnum() else get_font_properties(FONT_FILE_PATH) # Ensure Devanagari font for source too if mixed
                        )
        ax_viz.set_xlim(0, 1)
        ax_viz.set_ylim(0.2, 0.8) # Adjust y-limits for the text visualization
        ax_viz.axhline(0, color='gray', lw=0.5) # Bottom border for the cell


    plt.tight_layout(rect=[0, 0.02, 1, 0.98 - (header_height_inches / fig_height) - (table_header_height_inches / fig_height) ]) # Adjust rect to make space for suptitle and headers
    # The rect for tight_layout is [left, bottom, right, top]
    # We need to reserve space at the top for suptitle and table headers.
    wb_image = wandb.Image(fig)
    plt.close(fig)
    return wb_image

def log_focused_attention_to_wandb(
    fabric_logger: Any, # Expects fabric.logger or fabric object directly
    source_str: str,
    predicted_str: str,
    attention_weights_np: np.ndarray, # Shape (pred_len, src_len)
    log_key_prefix: str, # E.g., "val" or "test"
    font_path: Optional[str] = FONT_FILE_PATH
):
    """
    Generates the focused attention plot and logs it to W&B.
    """
    if not isinstance(attention_weights_np, np.ndarray):
        print("log_focused_attention: attention_weights must be a NumPy array.")
        return

    if not source_str or not predicted_str or attention_weights_np.size == 0:
        print("log_focused_attention: Missing data for plotting.")
        return
    
    # Ensure predicted_str and attention_weights_np[0] are compatible
    # This might require predicted_str to be a list of characters if that's how attention_weights are generated
    # For simplicity, assuming predicted_str is a string and its length matches attention_weights_np.shape[0]

    # Ensure source_str and attention_weights_np[1] are compatible
    if len(predicted_str) != attention_weights_np.shape[0] or \
       len(source_str) != attention_weights_np.shape[1]:
        print(f"log_focused_attention: Shape mismatch. Pred: '{predicted_str}' (len {len(predicted_str)}), "
              f"Src: '{source_str}' (len {len(source_str)}), Attn: {attention_weights_np.shape}")
        # Try to truncate/pad if minor mismatch, or just skip. For now, skip.
        # Example: Truncate attention if prediction is shorter than attention's first dim
        if len(predicted_str) < attention_weights_np.shape[0]:
            attention_weights_np = attention_weights_np[:len(predicted_str), :]
        # Similar for source if needed, but usually source_len is fixed for the batch.
        # If still mismatch, skip.
        if len(predicted_str) != attention_weights_np.shape[0] or \
           len(source_str) != attention_weights_np.shape[1]:
            print("log_focused_attention: Skipping due to persistent shape mismatch after basic adjustment.")
            return


    font_props = get_font_properties(font_path)
    
    fig = plot_focused_attention_visualization(
        source_word=source_str,
        predicted_word=predicted_str,
        attention_weights=attention_weights_np,
        font_props=font_props,
        input_description=f"INPUT : {source_str}"
    )

    if fig and wandb.run is not None:
        try:
            # The fabric logger itself should handle logging the image
            if hasattr(fabric_logger, 'log_image'): # L.Fabric's WandbLogger integration
                 fabric_logger.log_image(key=f"{log_key_prefix}/focused_attention", images=[wandb.Image(fig)])
            elif hasattr(fabric_logger, 'log_dict'): # General L.Fabric logging
                 fabric_logger.log_dict({f"{log_key_prefix}/focused_attention": wandb.Image(fig)})
            else: # Fallback to direct wandb logging if fabric_logger isn't recognized
                wandb.log({f"{log_key_prefix}/focused_attention": wandb.Image(fig)})
            print(f"Logged focused attention for '{source_str}' -> '{predicted_str}' to W&B.")
        except Exception as e:
            print(f"Error logging focused attention to W&B: {e}")
    elif not fig:
        print("Focused attention plot was not generated.")