import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

class CustomCNN(nn.Module):
    def __init__(self, ):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
    
    def forward(self, inputs):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        outputs: (Batch_size, Sequence_length, Hidden_dim)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem 1: design CNN forward path
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs
        

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2):
        # NOTE: you can freely add hyperparameters argument
        super(Encoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.cnn = CustomCNN()
        # NOTE: you can freely modify self.rnn module (ex. LSTM -> GRU)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc =
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, inputs, lengths):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths (Batch_size)
        output: (Batch_size, Sequence_length, Hidden_dim)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        x = self.cnn(inputs)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden_state = self.rnn(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        output = 
        # NOTE: you can utilize additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

        return output, hidden_state
        

class Decoder(nn.Module):
    def __init__(self, n_vocab=28, hidden_dim=64, num_layers=2, pad_idx=0, dropout=0.5):
        # NOTE: you can freely add hyperparameters argument
        super(Decoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.hidden_dim = hidden_dim
        self.n_layers = num_layers
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, hidden_dim, padding_idx=pad_idx)
        # NOTE: you can freely modify self.rnn module (ex. LSTM -> GRU)
        self.rnn = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.lm_head = nn.Linear(hidden_dim, n_vocab)
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

    def forward(self, input_seq, hidden_state):
        """
        input_seq: (Batch_size, Sequence_length)
        output: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem 2: design Decoder forward path
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return output, hidden_state


class Seq2SeqModel(nn.Module):
    def __init__(self, num_classes=28, hidden_dim=64, n_rnn_layers=2, rnn_dropout=0.5):
        # NOTE: you can freely add hyperparameters argument
        super(Seq2SeqModel, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.encoder = Encoder(hidden_dim=hidden_dim, num_layers=n_rnn_layers)
        self.decoder = Decoder(n_vocab=num_classes, hidden_dim=hidden_dim, num_layers=n_rnn_layers)
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
    
    def forward(self, inputs, lengths, inp_seq):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, Sequence_length)
        logits: (Batch_size, Sequence_length, N_vocab)
        hidden_state: ((Num_layers, Batch_size, Hidden_dim), (Num_layers, Batch_size, Hidden_dim)) (tuple of h_n and c_n)
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem 3: design Seq2SeqModel forward path using encoder and decoder
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return logits, hidden_state

    def generate(self, inputs, lengths, inp_seq, max_length):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, 1)
        max_length -> a single integer number (ex. 10)
        generated_tok: (Batch_size, max_length) -> long dtype tensor
        """
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem 4: design generate function of Seq2SeqModel
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return generated_tok
        

# NOTE: you can freely add and define other modules and models
# ex. TransformerDecoder, Seq2SeqTransformerModel, Attention, etc.
