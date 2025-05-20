import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple, Optional, Union, Any

# Define Tensor_or_Tuple type alias for hidden states for clarity
Tensor_or_Tuple = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


class EncoderRNN(nn.Module):
    """
    Encoder RNN module.
    Encodes a sequence of input character IDs into a context vector (final hidden state)
    and a set of output hidden states for all time steps (for attention).
    """
    def __init__(self,
                 input_vocab_size: int,
                 embed_dim: int,
                 hidden_dim: int, # Hidden dim per direction
                 num_layers: int,
                 rnn_cell_type: str = 'GRU',
                 dropout_p: float = 0.1, # Single dropout param for consistency
                 bidirectional: bool = False):
        """
        Args:
            input_vocab_size (int): Size of the input vocabulary.
            embed_dim (int): Dimension of character embeddings.
            hidden_dim (int): Dimension of RNN hidden states (per direction).
            num_layers (int): Number of RNN layers.
            rnn_cell_type (str, optional): Type of RNN cell ('RNN', 'LSTM', 'GRU'). Defaults to 'GRU'.
            dropout_p (float, optional): Dropout probability. Applied to embeddings and between RNN layers. Defaults to 0.1.
            bidirectional (bool, optional): If True, becomes a bidirectional RNN. Defaults to False.
        """
        super().__init__()
        self.input_vocab_size = input_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn_cell_type = rnn_cell_type.upper()
        self.dropout_p = dropout_p
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(input_vocab_size, embed_dim)
        # Dropout after embedding layer
        self.embedding_dropout = nn.Dropout(dropout_p)

        rnn_class = getattr(nn, self.rnn_cell_type)
        # Dropout between RNN layers (if num_layers > 1) is handled by the `dropout` argument in RNN modules
        self.rnn = rnn_class(
            embed_dim,
            hidden_dim, # RNN hidden_dim is per direction
            num_layers,
            dropout=dropout_p if num_layers > 1 else 0, # Inter-layer dropout for multi-layer RNNs
            bidirectional=bidirectional,
            batch_first=False # Expects (seq_len, batch, feature)
        )

    def forward(self,
                input_seqs: torch.Tensor,
                input_lengths: torch.Tensor,
                hidden: Optional[Tensor_or_Tuple] = None) -> Tuple[torch.Tensor, Tensor_or_Tuple]:
        """
        Forward pass of the encoder.

        Args:
            input_seqs (torch.Tensor): Batch of input sequences (L_src, N).
            input_lengths (torch.Tensor): Lengths of input sequences (N). Must be on CPU.
            hidden (Optional[Tensor_or_Tuple]): Initial hidden state. Defaults to None (zero-initialized).

        Returns:
            Tuple[torch.Tensor, Tensor_or_Tuple]:
                - encoder_outputs (torch.Tensor): Output features from the RNN's last layer for each time step (L_src, N, H_out).
                                                  H_out is num_directions * hidden_dim. These are used by attention.
                - final_hidden (Tensor_or_Tuple): Final hidden state from all layers.
                    - For GRU/RNN: (num_layers * num_directions, N, H_encoder_layer). H_encoder_layer is hidden_dim.
                    - For LSTM: Tuple of (h_n, c_n), each of shape (num_layers * num_directions, N, H_encoder_layer).
        """
        # input_seqs: (L_src, N)
        embedded = self.embedding(input_seqs)  # (L_src, N, E)
        embedded = self.embedding_dropout(embedded) # Apply dropout to embeddings

        # Pack sequence (input_lengths must be on CPU for pack_padded_sequence)
        packed_embedded = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=False, enforce_sorted=False)

        # packed_outputs contains all hidden states from the last RNN layer for each time step
        # final_hidden is the final hidden state (and cell state for LSTM) from all layers
        packed_outputs, final_hidden = self.rnn(packed_embedded, hidden)

        # Unpack sequence
        # encoder_outputs will be (L_src, N, num_directions * hidden_dim)
        encoder_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=False)

        return encoder_outputs, final_hidden


class BahdanauAttention(nn.Module):
    """
    Bahdanau Attention mechanism (also known as Additive Attention).
    Calculates attention weights and a context vector.
    Energy_i = v_a^T * tanh(W_a * s_{t-1} + U_a * h_j)
    where s_{t-1} is the previous decoder hidden state (from its last layer)
    and h_j are encoder output states.
    """
    def __init__(self,
                 encoder_output_dim: int, # Dimension of each encoder output vector (num_directions * encoder_hidden_dim_per_layer)
                 decoder_hidden_dim: int, # Dimension of the decoder's hidden state (from its last layer, so decoder_hidden_dim_per_layer)
                 attention_dim: int):     # Internal dimension for the alignment model (MLP)
        """
        Args:
            encoder_output_dim (int): The dimension of each encoder output vector (e.g., self.encoder.hidden_dim * self.encoder.num_directions).
            decoder_hidden_dim (int): The dimension of the decoder's hidden state (e.g., self.decoder.hidden_dim).
            attention_dim (int): The dimension of the hidden layer in the attention MLP.
        """
        super().__init__()
        self.encoder_output_dim = encoder_output_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.attention_dim = attention_dim

        self.W_a = nn.Linear(decoder_hidden_dim, attention_dim, bias=False) # Projects decoder's previous hidden state
        self.U_a = nn.Linear(encoder_output_dim, attention_dim, bias=False) # Projects encoder outputs
        self.v_a = nn.Linear(attention_dim, 1, bias=False) # Computes the score from the combined projection

    def forward(self,
                decoder_prev_hidden_last_layer: torch.Tensor, # (N, decoder_hidden_dim)
                encoder_outputs: torch.Tensor                 # (L_src, N, encoder_output_dim)
               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            decoder_prev_hidden_last_layer (torch.Tensor): Previous decoder hidden state from its last layer (N, H_decoder_per_layer).
            encoder_outputs (torch.Tensor): All encoder hidden states (L_src, N, H_enc_out = D * H_enc_per_layer).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - context_vector (torch.Tensor): Context vector (N, H_enc_out).
                - attention_weights (torch.Tensor): Attention weights (N, L_src).
        """
        src_len = encoder_outputs.shape[0]
        batch_size = encoder_outputs.shape[1]

        proj_decoder_hidden = self.W_a(decoder_prev_hidden_last_layer).unsqueeze(0) # (1, N, attention_dim)
        proj_encoder_outputs = self.U_a(encoder_outputs) # (L_src, N, attention_dim)

        energy_sum = torch.tanh(proj_decoder_hidden + proj_encoder_outputs) # (L_src, N, attention_dim)
        scores = self.v_a(energy_sum).squeeze(2) # (L_src, N)

        attention_weights = F.softmax(scores.t(), dim=1) # (N, L_src)

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))
        context_vector = context_vector.squeeze(1) # (N, encoder_output_dim)

        return context_vector, attention_weights


class DecoderRNN(nn.Module):
    """
    Decoder RNN module with optional Bahdanau Attention.
    Generates an output sequence one character at a time.
    """
    def __init__(self,
                 output_vocab_size: int,
                 embed_dim: int,
                 decoder_hidden_dim: int, # Hidden dim of the decoder's RNN cells (per layer)
                 num_layers: int,
                 rnn_cell_type: str = 'GRU',
                 dropout_p: float = 0.1, # Single dropout param
                 attention_module: Optional[BahdanauAttention] = None,
                 encoder_output_dim: Optional[int] = None): # Dim of each encoder output vector
        """
        Args:
            output_vocab_size (int): Size of the output vocabulary.
            embed_dim (int): Dimension of character embeddings for the decoder.
            decoder_hidden_dim (int): Dimension of the decoder's RNN hidden states (per layer).
            num_layers (int): Number of RNN layers in the decoder.
            rnn_cell_type (str, optional): Type of RNN cell ('RNN', 'LSTM', 'GRU'). Defaults to 'GRU'.
            dropout_p (float, optional): Dropout probability. Applied to embeddings, between RNN layers, and before output FC. Defaults to 0.1.
            attention_module (BahdanauAttention, optional): Pre-initialized attention mechanism. Defaults to None.
            encoder_output_dim (int, optional): Dimension of encoder's output vectors (num_directions * enc_hidden_dim_per_layer).
                                                Required if attention_module is not None.
        """
        super().__init__()
        self.output_vocab_size = output_vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = decoder_hidden_dim # This is H_decoder_per_layer
        self.num_layers = num_layers
        self.rnn_cell_type = rnn_cell_type.upper()
        self.dropout_p = dropout_p
        self.attention = attention_module

        if self.attention and encoder_output_dim is None:
            raise ValueError("encoder_output_dim must be provided if attention_module is used.")
        self.encoder_output_dim = encoder_output_dim

        self.embedding = nn.Embedding(output_vocab_size, embed_dim)
        self.embedding_dropout = nn.Dropout(dropout_p) # Dropout after embedding

        rnn_input_dim = embed_dim
        if self.attention:
            rnn_input_dim += self.encoder_output_dim # Context vector concatenated

        rnn_class = getattr(nn, self.rnn_cell_type)
        self.rnn = rnn_class(
            rnn_input_dim,
            self.hidden_dim,
            num_layers,
            dropout=dropout_p if num_layers > 1 else 0, # Inter-layer dropout
            batch_first=False
        )

        # Dropout before the final fully connected layer
        self.output_dropout = nn.Dropout(dropout_p)
        self.fc_out = nn.Linear(self.hidden_dim, output_vocab_size) # Input to FC is H_decoder_per_layer
        self.log_softmax = nn.LogSoftmax(dim=1)

    def _get_last_layer_hidden(self, decoder_full_hidden_state: Tensor_or_Tuple) -> torch.Tensor:
        """
        Extracts the hidden state of the *last layer* from the decoder's full hidden state.
        This is used as s_{t-1} for the attention mechanism.
        Args:
            decoder_full_hidden_state (Tensor_or_Tuple): The full hidden state from the decoder RNN.
                - For GRU/RNN: (num_layers, N, H_decoder_per_layer)
                - For LSTM: Tuple of (h_n, c_n), each of shape (num_layers, N, H_decoder_per_layer)
        Returns:
            torch.Tensor: Hidden state of the last layer (N, H_decoder_per_layer).
        """
        if self.rnn_cell_type == 'LSTM':
            return decoder_full_hidden_state[0][-1]  # h_n from the last layer
        else: # GRU or RNN
            return decoder_full_hidden_state[-1]     # Last layer's hidden state

    def forward(self,
                decoder_input_token: torch.Tensor,       # (N) - current input token IDs for this step
                decoder_prev_hidden_state: Tensor_or_Tuple, # Previous full hidden state of decoder RNN
                encoder_all_outputs: Optional[torch.Tensor] = None # (L_src, N, enc_output_dim) - ALL encoder outputs
               ) -> Tuple[torch.Tensor, Tensor_or_Tuple, Optional[torch.Tensor]]:
        """
        Forward pass for a single decoding step.
        Returns:
            Tuple[torch.Tensor, Tensor_or_Tuple, Optional[torch.Tensor]]:
                - output_probs (torch.Tensor): Log-softmax probabilities for the next token (N, V_out).
                - new_decoder_hidden_state (Tensor_or_Tuple): Updated full decoder hidden state.
                - attention_weights_step (Optional[torch.Tensor]): Attention weights (N, L_src) if attention is used, else None.
        """
        decoder_input_unsqueezed = decoder_input_token.unsqueeze(0) # (1, N)
        embedded_token = self.embedding(decoder_input_unsqueezed)   # (1, N, E_decoder)
        embedded_token = self.embedding_dropout(embedded_token)     # Apply dropout

        attention_weights_step = None
        rnn_input = embedded_token

        if self.attention:
            if encoder_all_outputs is None:
                raise ValueError("encoder_all_outputs must be provided when attention is enabled.")
            
            prev_hidden_last_layer = self._get_last_layer_hidden(decoder_prev_hidden_state)
            context_vector, attention_weights_step = self.attention(prev_hidden_last_layer, encoder_all_outputs)
            rnn_input = torch.cat((embedded_token, context_vector.unsqueeze(0)), dim=2) # (1, N, E_decoder + enc_output_dim)

        rnn_output, new_decoder_hidden_state = self.rnn(rnn_input, decoder_prev_hidden_state) # rnn_output: (1, N, H_decoder_per_layer)
        
        rnn_output_squeezed = rnn_output.squeeze(0) # (N, H_decoder_per_layer)
        output_after_dropout = self.output_dropout(rnn_output_squeezed) # Apply dropout
        
        output_logits = self.fc_out(output_after_dropout) # (N, V_out)
        # output_probs = self.log_softmax(output_logits)   # (N, V_out)

        return output_logits, new_decoder_hidden_state, attention_weights_step


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model combining EncoderRNN and DecoderRNN, 
    with optional Attention and learnable projection for decoder initialization.
    """
    def __init__(self,
                 encoder: EncoderRNN,
                 decoder: DecoderRNN,
                 target_sos_idx: int,
                 target_eos_idx: int,
                 device: torch.device):
        """
        Args:
            encoder (EncoderRNN): The encoder module.
            decoder (DecoderRNN): The decoder module.
            target_sos_idx (int): Index of the SOS token in the target vocabulary.
            target_eos_idx (int): Index of the EOS token in the target vocabulary.
            device (torch.device): Device for tensor creation.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.target_sos_idx = target_sos_idx
        self.target_eos_idx = target_eos_idx
        self.device = device

        # Determine the input dimension for the projection layer
        # This will be the encoder's hidden_dim * num_directions (from its last layer)
        encoder_last_layer_hidden_dim = self.encoder.hidden_dim * self.encoder.num_directions
        
        # Learnable linear layer to project encoder's context to decoder's initial hidden state
        # Output dimension should be decoder.hidden_dim * decoder.num_layers
        # We'll project to decoder.hidden_dim and then repeat/reshape for decoder.num_layers
        self.fc_hidden_projection = nn.Linear(encoder_last_layer_hidden_dim, self.decoder.hidden_dim)
        
        # If decoder is LSTM, we need a similar projection for the cell state
        if self.decoder.rnn_cell_type == 'LSTM':
            self.fc_cell_projection = nn.Linear(encoder_last_layer_hidden_dim, self.decoder.hidden_dim)
        
        # Optional: Add an activation function after projection if desired, e.g., Tanh
        # self.projection_activation = nn.Tanh() # Uncomment if you want to use Tanh

    def _project_encoder_hidden_for_decoder_init(self, encoder_final_full_hidden: Tensor_or_Tuple) -> Tensor_or_Tuple:
        """
        Adapts the encoder's final hidden state (from all its layers) 
        to be suitable for initializing the decoder's hidden state,
        using a learnable linear projection.

        This method takes the hidden state of the *last layer* of the encoder
        (after handling bidirectionality) and projects it.

        Args:
            encoder_final_full_hidden (Tensor_or_Tuple): 
                - For GRU/RNN: (enc_layers * enc_directions, N, enc_H_per_layer)
                - For LSTM: Tuple (h_n, c_n), each (enc_layers * enc_directions, N, enc_H_per_layer)

        Returns:
            Tensor_or_Tuple: Decoder's initial hidden state.
                - For GRU/RNN: (dec_layers, N, dec_H_per_layer)
                - For LSTM: Tuple (h_0, c_0), each (dec_layers, N, dec_H_per_layer)
        """
        batch_size = encoder_final_full_hidden[0].size(1) if isinstance(encoder_final_full_hidden, tuple) else encoder_final_full_hidden.size(1)

        # Step 1: Get the hidden state(s) from the last layer of the encoder
        if self.encoder.rnn_cell_type == 'LSTM':
            # For LSTM, h_n and c_n are (enc_layers * enc_directions, N, enc_H_per_layer)
            # We take the states from the last layer: h_n[-self.encoder.num_directions:] and c_n[-self.encoder.num_directions:]
            # These represent the forward and backward (if bidirectional) states of the top encoder layer.
            
            h_enc_top = encoder_final_full_hidden[0].view(self.encoder.num_layers, self.encoder.num_directions, batch_size, self.encoder.hidden_dim)[-1]
            # h_enc_top is now (enc_directions, N, enc_H_per_layer)
            
            c_enc_top = encoder_final_full_hidden[1].view(self.encoder.num_layers, self.encoder.num_directions, batch_size, self.encoder.hidden_dim)[-1]
            # c_enc_top is now (enc_directions, N, enc_H_per_layer)

            if self.encoder.bidirectional:
                # Concatenate or sum forward and backward states of the top layer
                # Here, let's sum them. For concatenation, fc_hidden_projection input_dim would need to be 2*encoder_hidden_dim
                # Assuming encoder_last_layer_hidden_dim was defined based on summing or single direction.
                # If concatenating: encoder_last_layer_hidden_dim = self.encoder.hidden_dim * 2
                # If summing: encoder_last_layer_hidden_dim = self.encoder.hidden_dim
                # Current definition of encoder_last_layer_hidden_dim = self.encoder.hidden_dim * self.encoder.num_directions
                # implies concatenation if bidirectional and hidden_dim is per direction.
                # Let's refine to take concatenated top layer states if bidirectional.
                if self.encoder.num_directions == 2:
                    h_enc_context = torch.cat((h_enc_top[0], h_enc_top[1]), dim=1) # (N, 2 * enc_H_per_layer)
                    c_enc_context = torch.cat((c_enc_top[0], c_enc_top[1]), dim=1) # (N, 2 * enc_H_per_layer)
                else:
                    h_enc_context = h_enc_top.squeeze(0) # (N, enc_H_per_layer)
                    c_enc_context = c_enc_top.squeeze(0) # (N, enc_H_per_layer)
            else: # Unidirectional
                h_enc_context = h_enc_top.squeeze(0) # (N, enc_H_per_layer)
                c_enc_context = c_enc_top.squeeze(0) # (N, enc_H_per_layer)
        
        else: # GRU/RNN
            # hidden is (enc_layers * enc_directions, N, enc_H_per_layer)
            # Take the states from the last layer
            hidden_enc_top = encoder_final_full_hidden.view(self.encoder.num_layers, self.encoder.num_directions, batch_size, self.encoder.hidden_dim)[-1]
            # hidden_enc_top is (enc_directions, N, enc_H_per_layer)

            if self.encoder.bidirectional:
                 if self.encoder.num_directions == 2:
                    h_enc_context = torch.cat((hidden_enc_top[0], hidden_enc_top[1]), dim=1) # (N, 2 * enc_H_per_layer)
                 else: # Should not happen if bidirectional is True
                    h_enc_context = hidden_enc_top.squeeze(0)
            else: # Unidirectional
                h_enc_context = hidden_enc_top.squeeze(0) # (N, enc_H_per_layer)

        # Step 2: Project this context vector
        # h_enc_context is (N, encoder_last_layer_hidden_dim)
        
        projected_h = self.fc_hidden_projection(h_enc_context) # (N, dec_H_per_layer)
        # Optional: Add activation
        # if hasattr(self, 'projection_activation'):
        #     projected_h = self.projection_activation(projected_h)

        # Step 3: Repeat/expand for each decoder layer
        # We want (dec_layers, N, dec_H_per_layer)
        decoder_h_0 = projected_h.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)

        if self.decoder.rnn_cell_type == 'LSTM':
            projected_c = self.fc_cell_projection(c_enc_context) # (N, dec_H_per_layer)
            # if hasattr(self, 'projection_activation'):
            #     projected_c = self.projection_activation(projected_c)
            decoder_c_0 = projected_c.unsqueeze(0).repeat(self.decoder.num_layers, 1, 1)
            return (decoder_h_0, decoder_c_0)
        else: # GRU/RNN
            return decoder_h_0


    def forward(self,
                source_seqs: torch.Tensor,
                source_lengths: torch.Tensor,
                target_seqs: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5,
                inference_max_len: int = 50
               ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Seq2Seq model.
        """
        batch_size = source_seqs.size(1)
        source_len = source_seqs.size(0)

        encoder_all_outputs, encoder_final_full_hidden = self.encoder(source_seqs, source_lengths)
        
        # Use the new projection method for decoder's initial hidden state
        decoder_current_hidden_state = self._project_encoder_hidden_for_decoder_init(encoder_final_full_hidden)

        num_decoding_steps = (target_seqs.size(0) - 1) if target_seqs is not None else inference_max_len
        if num_decoding_steps <= 0:
            return torch.empty(0, batch_size, self.decoder.output_vocab_size, device=self.device), None

        decoder_current_input_token = torch.full((batch_size,), self.target_sos_idx, dtype=torch.long, device=self.device)
        # This will store raw logits from the decoder
        decoder_logits_batch = torch.zeros(num_decoding_steps, batch_size, self.decoder.output_vocab_size, device=self.device)
        
        all_attention_weights = None
        if self.decoder.attention:
            all_attention_weights = torch.zeros(num_decoding_steps, batch_size, source_len, device=self.device)

        for t in range(num_decoding_steps):
            # Decoder outputs raw logits per step
            decoder_step_logits, decoder_new_hidden_state, attn_weights_step = self.decoder(
                decoder_input_token=decoder_current_input_token,
                decoder_prev_hidden_state=decoder_current_hidden_state,
                encoder_all_outputs=encoder_all_outputs if self.decoder.attention else None
            )
            decoder_logits_batch[t] = decoder_step_logits # Store raw logits
            if self.decoder.attention and attn_weights_step is not None:
                all_attention_weights[t] = attn_weights_step

            decoder_current_hidden_state = decoder_new_hidden_state
            use_teacher_force = (target_seqs is not None) and (random.random() < teacher_forcing_ratio)

            if use_teacher_force:
                decoder_current_input_token = target_seqs[t+1]
            else:
                # Get next input from the highest probability LOGIT (argmax)
                _, top_idx = decoder_step_logits.max(1) # Argmax on raw logits
                decoder_current_input_token = top_idx.detach()

            if target_seqs is None and (decoder_current_input_token == self.target_eos_idx).all():
                decoder_logits_batch = decoder_logits_batch[:t+1]
                if all_attention_weights is not None:
                    all_attention_weights = all_attention_weights[:t+1]
                break
                
        return decoder_logits_batch, all_attention_weights # Return raw logits