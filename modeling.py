import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

from torchvision.models import resnet18

class CustomResnet(nn.Module):
    def __init__(self, hidden_dim):
        # NOTE: you can freely add hyperparameters argument
        super(CustomResnet, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model


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

        x = self.resnet(x)
        outputs = x.view(batch_size, seq_length, -1)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

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
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs
        
class CustomCNN2(nn.Module):
    def __init__(self, hidden_dim):
        # NOTE: you can freely add hyperparameters argument
        super(CustomCNN2, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # define cnn model

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.maxPool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        )
        self.maxPool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        )
        self.maxPool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
        )
        self.fc1 = nn.Linear(256, 256) 
        self.fc2 = nn.Linear(256, hidden_dim)  # Adjust based on input size

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

        x = self.conv1(x)
        x = self.maxPool1(x)
        
        x = self.conv2(x)
        x = self.maxPool2(x)

        x = self.conv3(x)
        x = self.maxPool3(x)

        x = self.conv4(x)

        # Flatten the CNN output
        x = x.view(x.size(0), -1)
        #print(x.size)
        x = self.fc1(x)
        x = self.fc2(x)
        outputs = x.view(batch_size, seq_length, -1)

        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return outputs

class Encoder(nn.Module):
    def __init__(self, hidden_dim=64, num_layers=2, cnn_output_dim = None, dropout = 0.5, customCNN = "CustomCNN", convld_dim=26):
        # NOTE: you can freely add hyperparameters argument
        super(Encoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        if cnn_output_dim is None:
            cnn_output_dim = hidden_dim
        func = globals()[customCNN]
        self.cnn = func(cnn_output_dim)
        # NOTE: you can freely modify self.rnn module (ex. LSTM -> GRU)
        self.rnn = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        #self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.conv1d = nn.Conv1d(in_channels=cnn_output_dim, out_channels=convld_dim, kernel_size=3, stride=1, padding=1)
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
        output = self.conv1d(x.permute(0, 2, 1)).permute(0, 2, 1)
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        #print(packed_input.data.shape)
        packed_output, hidden_state = self.rnn(packed_input)
        #output, _ = pad_packed_sequence(packed_output, batch_first=True)
        #output = self.fc(output)
        # NOTE: you can utilize additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################

        return output, hidden_state
        

class Decoder(nn.Module):
    def __init__(self, n_vocab=28, hidden_dim=64, num_layers=2, pad_idx=0, dropout=0.5,
                 embed_dim = None, nhead=None, fc_hidden = 3, conv1d_dim=26):
        # NOTE: you can freely add hyperparameters argument
        super(Decoder, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        if self.embed_dim is None:
            self.embed_dim = hidden_dim
        self.n_layers = num_layers
        self.n_vocab = n_vocab
        self.embedding = nn.Embedding(n_vocab, embed_dim, padding_idx=pad_idx)
        # NOTE: you can freely modify self.rnn module (ex. LSTM -> GRU)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
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

    def forward(self, input_seq, hidden_state, cnn3_outputs):
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
        #embedded = input_seq
        # Passing the embedded sequence through the RNN
        output, hidden_state = self.rnn(embedded, hidden_state)
        """
        output = self.fc1(output)
        output = torch.cat((output, cnn3_outputs), dim = 2)
        output = self.fc2(output)
        """
        # Generating the output tokens
        output = self.lm_head(output)
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return output, hidden_state


class Seq2SeqModel(nn.Module):
    def __init__(self, num_classes=28, hidden_dim=64, n_rnn_layers=2, embed_dim = 28, rnn_dropout=0.5, 
                 device = torch.device("cuda:0"), customCNN = "CustomCNN", nhead=8, conv1d_dim=26, dec_fc_dim=3):
        # NOTE: you can freely add hyperparameters argument
        super(Seq2SeqModel, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.n_vocab = num_classes
        self.encoder = Encoder(hidden_dim=hidden_dim, 
                               num_layers=n_rnn_layers, 
                               cnn_output_dim=26, 
                               dropout=rnn_dropout,
                               customCNN=customCNN,
                               convld_dim=conv1d_dim)
        self.decoder = Decoder(n_vocab=num_classes, 
                               hidden_dim=hidden_dim, 
                               num_layers=n_rnn_layers, 
                               embed_dim=embed_dim, 
                               dropout=rnn_dropout,
                               conv1d_dim=0,
                               fc_hidden=dec_fc_dim)
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
        cnn3_outputs, hidden_state = self.encoder(inputs, lengths)
        #print(f"{cnn3_outputs.shape=}")
        #print(inp_seq[0, :])
        
        # Decode the encoded sequence
        logits, _ = self.decoder(inp_seq, hidden_state, cnn3_outputs)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return logits

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

        cnn3_outputs, encoder_hidden = self.encoder(inputs, lengths)
        
        # Initialize the input for the decoder with the start token
        dec_input = inp_seq
        dec_hidden = encoder_hidden
        
        generated_tok = []
        
        for t in range(max_length):
            logits, dec_hidden = self.decoder(dec_input, dec_hidden, cnn3_outputs[:, t, :])
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

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # shape (1, max_len, embed_dim)
    
    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)


class TransformerDecoder(nn.Module):
    def __init__(self, n_vocab=28, hidden_dim=64, num_layers=2, pad_idx=0, dropout=0.5, embed_dim=None, nhead=8, dim_feedforward=2048):
        super(TransformerDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim or hidden_dim
        self.n_layers = num_layers
        self.n_vocab = n_vocab
        
        self.embedding = nn.Embedding(n_vocab, self.embed_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(self.embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.embed_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(self.embed_dim, n_vocab)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask
    
    def create_pad_mask(self, tgt: torch.Tensor, pad_idx: int) -> torch.Tensor:
        mask = (tgt == pad_idx).float()
        mask = mask.masked_fill(mask == 1, float('-inf'))
        mask = mask.masked_fill(mask == 0, float(0.0))
        return mask
        
    def forward(self, input_seq, memory, tgt_mask=None, tgt_key_padding_mask=None):
        embedded = self.embedding(input_seq)
        embedded = self.positional_encoding(embedded)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(input_seq.shape[1]).to(input_seq.device)
        
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self.create_pad_mask(input_seq, 0)
        
        output = self.transformer_decoder(
            tgt=embedded.permute(1, 0, 2),  # Transformer expects (Seq_len, Batch, Embed_dim)
            memory=memory,  # memory from encoder (Seq_len, Batch, Embed_dim)
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        output = self.lm_head(output.transpose(0, 1))  # back to (Batch, Seq_len, N_vocab)
        
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.5, customCNN="CustomCNN", nhead=2, dim_feedforward=2048):
        super(TransformerEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        func = globals()[customCNN]
        self.cnn = func(input_dim)
        #self.embedding = nn.Embedding(input_dim, hidden_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, inputs, lengths, max_seq_len=10):
        batch_size, seq_len, h, w, c = inputs.shape
        src_key_padding_mask = torch.ones((batch_size, max_seq_len), device=inputs.device, dtype=torch.bool)
        for i, length in enumerate(lengths.type(torch.int32)):
            src_key_padding_mask[i, :length] = False
        x = self.cnn(inputs)
        #packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        #embedded = self.embedding(x)  # (Batch_size, Seq_len, Hidden_dim)
        embedded = self.positional_encoding(x)
        
        output = self.transformer_encoder(embedded.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask)
        #output = output.permute(1, 0, 2)  # back to (Batch_size, Seq_len, Hidden_dim)
        
        return output

class TransformerSeq2Seq(nn.Module):
    def __init__(self, num_classes=28, hidden_dim=64, n_rnn_layers=2, embed_dim = 28, rnn_dropout=0.5, 
                 device = torch.device("cuda:0"), customCNN = "CustomCNN", nhead=8, dim_feedforward=2048):
        # NOTE: you can freely add hyperparameters argument
        super(TransformerSeq2Seq, self).__init__()
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        self.n_vocab = num_classes
        self.encoder = TransformerEncoder(hidden_dim=hidden_dim, 
                               num_layers=n_rnn_layers, 
                               input_dim=hidden_dim, 
                               dropout=rnn_dropout,
                               customCNN=customCNN,
                               nhead=nhead,
                               dim_feedforward=dim_feedforward)
        self.decoder = TransformerDecoder(n_vocab=num_classes, 
                               hidden_dim=hidden_dim, 
                               num_layers=n_rnn_layers, 
                               embed_dim=hidden_dim, 
                               dropout=rnn_dropout,
                               nhead=nhead,
                               dim_feedforward=dim_feedforward)
        self.device = device
        # NOTE: you can define additional parameters
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
    
    def forward(self, inputs, lengths, inp_seq, max_seq_len=10):
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
        encoder_outputs = self.encoder(inputs, lengths)
        
        # Decode the encoded sequence
        logits = self.decoder(inp_seq, encoder_outputs)

        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return logits

    def generate(self, inputs, lengths, inp_seq, max_length):
        """
        inputs: (Batch_size, Sequence_length, Height, Width, Channel)
        lengths: (Batch_size,)
        inp_seq: (Batch_size, 1)
        max_length -> a single integer number (ex. 10)
        generated_tok: (Batch_size, max_length) -> long dtype tensor
        
        ##############################################################################
        #                          IMPLEMENT YOUR CODE                               #
        ##############################################################################
        # Problem 4: design generate function of Seq2SeqModel

        encoder_outputs = self.encoder(inputs, lengths)
        
        # Initialize the input for the decoder with the start token
        dec_input = inp_seq
        
        generated_tok = []
        
        for t in range(max_length):
            logits = self.decoder(dec_input, encoder_outputs)
            predicted_token = logits.argmax(2)
            generated_tok.append(predicted_token)
            
            dec_input = predicted_token
        
        generated_tok = torch.cat(generated_tok, dim=1)
        """
        encoder_output = self.encoder(inputs, lengths)
        generated_tok = inp_seq

        for step in range(max_length):
            tgt_mask = self.decoder.generate_square_subsequent_mask(generated_tok.size(1)).to(inp_seq.device)
            tgt_key_padding_mask = self.decoder.create_pad_mask(generated_tok, 0).to(inp_seq.device)
            output = self.decoder(generated_tok, encoder_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
            predicted_token = output[:, -1:].argmax(2)
            generated_tok = torch.cat((generated_tok, predicted_token), dim=1)

        return generated_tok[:, 1:]
        
        ##############################################################################
        #                          END OF YOUR CODE                                  #
        ##############################################################################
        return generated_tok
    
    def pad_and_create_mask(self, inputs, lengths):
        batch_size, seq_len, h, w, c = inputs.shape
        max_seq_len = max(lengths)
        
        padded_inputs = torch.zeros((batch_size, max_seq_len, h, w, c), device=inputs.device)
        src_key_padding_mask = torch.ones((batch_size, max_seq_len), device=inputs.device, dtype=torch.bool)
        
        for i, length in enumerate(lengths):
            padded_inputs[i, :length] = inputs[i, :length]
            src_key_padding_mask[i, :length] = False
        
        return padded_inputs, src_key_padding_mask
    