import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#from transformer import MultiHeadedAttention


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size
    ):
        super(LSTM, self).__init__()
        # input size corresponds to 1 batch with dimension : (batch_size, sequence_length, hidden_size)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.w_ii = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_if = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_ig = nn.Parameter(torch.empty(hidden_size, input_size))
        self.w_io = nn.Parameter(torch.empty(hidden_size, input_size))

        self.b_ii = nn.Parameter(torch.empty(hidden_size))
        self.b_if = nn.Parameter(torch.empty(hidden_size))
        self.b_ig = nn.Parameter(torch.empty(hidden_size))
        self.b_io = nn.Parameter(torch.empty(hidden_size))

        self.w_hi = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hf = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_hg = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w_ho = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_hi = nn.Parameter(torch.empty(hidden_size))
        self.b_hf = nn.Parameter(torch.empty(hidden_size))
        self.b_hg = nn.Parameter(torch.empty(hidden_size))
        self.b_ho = nn.Parameter(torch.empty(hidden_size))

        for param in self.parameters():
            nn.init.uniform_(param, a=-(1/hidden_size)**0.5, b=(1/hidden_size)**0.5)

    def forward(self, inputs, hidden_states):
        """LSTM.
        This is a Long Short-Term Memory (LSTM) network.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the embedded sequences.

        hidden_states 
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states 
            The final hidden state. 
            - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
        """
        # # ==========================
        # # TODO: Write your code here
        # # ==========================
        
        # Initial hidden states
        h_prev, c_prev = hidden_states
        # Reshape hidden_states to (batch_size, hidden_size)
        h_prev = h_prev.squeeze(0)
        c_prev = c_prev.squeeze(0)

        outputs = []

        # Iterate over the elements in sequences
        for input in inputs.split(1, dim=1):
            input = input.squeeze(1) # shape (batch_size, hidden_size)

            # Input gate
            i_t = torch.sigmoid(
                torch.matmul(input, self.w_ii.t()) + 
                torch.matmul(h_prev, self.w_hi.t()) + 
                self.b_ii +  self.b_hi)
            # Forget gate
            f_t = torch.sigmoid(
                torch.matmul(input, self.w_if.t()) + 
                torch.matmul(h_prev, self.w_hf.t()) + 
                self.b_if + self.b_hf)
            # Candidate memory
            g_t = torch.tanh(
                torch.matmul(input, self.w_ig.t()) + 
                torch.matmul(h_prev, self.w_hg.t()) + 
                self.b_ig + self.b_hg)
            # Output gate
            o_t = torch.sigmoid(
                torch.matmul(input, self.w_io.t()) + 
                torch.matmul(h_prev, self.w_ho.t()) + 
                self.b_io + self.b_ho)

            # Update hidden_states
            c_t = f_t * c_prev + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t) # list of elements of shape (batch_size, hidden_size)
            h_prev, c_prev = h_t, c_t # shape (batch_size, hidden_size)
            
        # The encoded input sequence
        x = torch.stack(outputs, dim=1) # Concatenate over the sequences : shape (batch_size, sequence_length, hidden_size)
        hidden_states = (h_t.unsqueeze(0), c_t.unsqueeze(0)) # shape (1, batch_size, hidden_size)
        return x, hidden_states

class Encoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        dropout=0.0, 
    ):
        super(Encoder, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            vocabulary_size, embedding_size, padding_idx=0,
        )

        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size, 
                           batch_first=True, bidirectional=True)

    def forward(self, inputs, hidden_states):
        """LSTM Encoder.

        This is a Bidirectional Long-Short Term Memory network
        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        hidden_states 
            The (initial) hidden state.
            - h (`torch.FloatTensor` of shape `(2, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(2, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, 2 * hidden_size)`)
            A feature tensor encoding the input sentence. 

        hidden_states 
            The final hidden state. 
            - h (`torch.FloatTensor` of shape `(2, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(2, batch_size, hidden_size)`)
        """

        # # ==========================
        # # TODO: Write your code here
        # # ==========================
        # WRITE CODE HERE

        inputs = self.embedding(inputs) # shape (batch_size, sequence_length, embedding_size)
        inputs = self.dropout(inputs)

        # Use of nn.LSTM module : bidirectional
        x, hidden_states = self.rnn(inputs, hidden_states)

        return x, hidden_states


    def initial_states(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device
        shape = (2, batch_size, self.hidden_size)
        # The initial state is a constant here, and is not a learnable parameter
        h_0 = torch.zeros(shape, dtype=torch.float, device=device)
        return (h_0, h_0)
    

class DecoderAttn(nn.Module):
    def __init__(
        self,
        input_size=512, # default input_size : 2*256 because Encoder is bidirectional
        hidden_size=256,
        attn=None
    ):

        super(DecoderAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            batch_first=True
        )
        
        # Self attention
        self.mlp_attn = attn

    def forward(self, inputs, hidden_states, mask=None):
        """LSTM Decoder network with Soft attention

        This is a Unidirectional Long-Short Term Memory network
        
        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            The input tensor containing the encoded input sequence.

        hidden_states
            The hidden states from the forward pass of the encoder network.
            - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)

        Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`)
            A feature tensor decoding the orginally encoded input sentence. 

        hidden_states 
            The final hidden states. 
            - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
        """

        # # ==========================
        # # TODO: Write your code here
        # # ==========================
        if not self.mlp_attn is None:
            x  = self.mlp_attn(inputs, mask)
        else :
            x = inputs
        # Decoder with self attention or not
        outputs = self.rnn(x, hidden_states)
        return outputs
                
class EncoderDecoder(nn.Module):
    def __init__(
        self,
        vocabulary_size=30522,
        embedding_size=256,
        hidden_size=256,
        dropout = 0.0,
        encoder_only = False,
        attn = None
    ):
        super(EncoderDecoder, self).__init__()
        self.encoder_only = encoder_only
        self.encoder = Encoder(vocabulary_size, embedding_size, hidden_size, dropout=dropout)
        if not encoder_only:
          self.decoder = DecoderAttn(hidden_size*2, hidden_size, attn=attn)
        
    def forward(self, inputs, mask=None):
        """LSTM Encoder-Decoder network with Soft attention.

        This is a Long-Short Term Memory network for Sentiment Analysis. This
        module returns a decoded feature for classification. 

        Parameters
        ----------
        inputs (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences.

        mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The masked tensor containing the location of padding in the sequences.

         Returns
        -------
        x (`torch.FloatTensor` of shape `(batch_size, hidden_size)`)
            A feature tensor representing the input sentence for sentiment analysis

        hidden_states
            The final hidden states. This is a tuple containing
            - h (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
            - c (`torch.FloatTensor` of shape `(1, batch_size, hidden_size)`)
        """
        hidden_states = self.encoder.initial_states(inputs.shape[0])
        x, hidden_states = self.encoder(inputs, hidden_states)
        if self.encoder_only:
          x = x[:, 0]
          return x, hidden_states
        hidden_states = (hidden_states[0][0][None], hidden_states[1][0][None])
        x, hidden_states = self.decoder(x, hidden_states, mask)
        x = x[:, 0]
        return x, hidden_states
