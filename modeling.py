import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

from torchvision.models import resnet18

class CustomCNN(nn.Module):
    def __init__(self, hidden_dim):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Assuming input image size is 28x28
        self.fc2 = nn.Linear(128, 32)
        """

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.avgPool4 = nn.AvgPool2d(kernel_size=3, stride=1)
        height = 28
        width = 28
        self.fc1 = nn.Linear(256, hidden_dim)  # Adjust based on input size

        self.resnet = resnet18(num_classes=hidden_dim)
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
        batch_size, seq_length, height, width, channels = inputs.size()
        x = inputs.view(batch_size * seq_length, channels, height, width)

        """
        x = self.conv1(x)
        x = self.maxPool1(x)
        
        x = self.conv2(x)
        x = self.maxPool2(x)

        x = self.conv3(x)
        x = self.maxPool3(x)

        x = self.conv4(x)
        x = self.avgPool4(x)


        # Flatten the CNN output
        x = x.view(x.size(0), -1)
        #print(x.size)
        x = self.fc1(x)
        outputs = x.view(batch_size, seq_length, -1)
        """

        x = self.resnet(x)
        outputs = x.view(batch_size, seq_length, -1)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs
        

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, cnn_output_dim = None):
        # NOTE: you can freely add hyperparameters argument
        super(Encoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        if cnn_output_dim is None:
            cnn_output_dim = hidden_dim
        self.cnn = CustomCNN(cnn_output_dim)
        # NOTE: you can freely modify self.rnn module (ex. LSTM -> GRU)
        self.rnn = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, hidden_dim)
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
        # Embedding the input sequence
        embedded = self.embedding(input_seq)
        # Passing the embedded sequence through the RNN
        output, hidden_state = self.rnn(embedded, hidden_state)
        # Generating the output tokens
        output = self.lm_head(output)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return output, hidden_state


class Seq2SeqModel(nn.Module):
    def __init__(self, num_classes=28, hidden_dim=64, n_rnn_layers=2, rnn_dropout=0.5, device = torch.device("cuda:0")):
        # NOTE: you can freely add hyperparameters argument
        super(Seq2SeqModel, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.n_vocab = num_classes
        self.encoder = Encoder(hidden_dim=hidden_dim, num_layers=n_rnn_layers, cnn_output_dim=26)
        self.decoder = Decoder(n_vocab=num_classes, hidden_dim=hidden_dim, num_layers=n_rnn_layers)
        self.device = device
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
        
        # Encode the input sequence
        encoder_outputs, hidden_state = self.encoder(inputs, lengths)
        #print(hidden_state[0].shape)
        #print(hidden_state[1].shape)
        
        # Decode the encoded sequence
        logits, hidden_state = self.decoder(inp_seq, hidden_state)

        """
        # New
        Batch_size, Sequence_length = inp_seq.shape
        logits = torch.zeros(Batch_size, Sequence_length, self.n_vocab).to(self.device)
        input = inp_seq[:, :1]
        for t in range(1, Sequence_length):
            #print(f"{input.shape=}")
            output, hidden_state = self.decoder(input, hidden_state)
            #print(hidden_state[0].shape)
            #print(hidden_state[1].shape)
            logits[:, t, :] = output.squeeze(1)
            input = inp_seq[:, t-1 : t]
        """
        
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

        encoder_outputs, encoder_hidden = self.encoder(inputs, lengths)
        
        # Initialize the input for the decoder with the start token
        dec_input = inp_seq
        dec_hidden = encoder_hidden
        
        generated_tok = []
        
        for t in range(max_length):
            logits, dec_hidden = self.decoder(dec_input, dec_hidden)
            predicted_token = logits.argmax(2)
            generated_tok.append(predicted_token)
            
            dec_input = predicted_token
        
        generated_tok = torch.cat(generated_tok, dim=1)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return generated_tok
        

# NOTE: you can freely add and define other modules and models
# ex. TransformerDecoder, Seq2SeqTransformerModel, Attention, etc.
