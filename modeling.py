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
        seq_len = cnn3_outputs.shape[1]
        
        # Initialize the input for the decoder with the start token
        dec_input = inp_seq
        dec_hidden = encoder_hidden
        
        generated_tok = []
        
        for t in range(seq_len):
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
        src_key_padding_mask = torch.ones((batch_size, seq_len), device=inputs.device, dtype=torch.bool)
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

    
    
    def generate(self, inputs, lengths, inp_seq, max_length, num_beams=5, pad_token_id=0):

        encoder_output = self.encoder(inputs, lengths).permute(1, 0, 2)

        batch_size, seq_len, d_embed = encoder_output.size()
        encoder_output = encoder_output.unsqueeze(1).repeat(1, num_beams, 1, 1).view(batch_size * num_beams, seq_len, d_embed)

        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            max_length=max_length,
            num_beams=num_beams,
            device=inputs.device,
            length_penalty=1.0,
            do_early_stopping=False,
        )



        input_ids = torch.ones((batch_size * num_beams, 1), device=inputs.device).long()
        input_ids = input_ids * inp_seq[:, :1].repeat(num_beams, 1)
        
        

        beam_scores = torch.zeros((batch_size, num_beams), device=inputs.device)
        beam_scores[:, 1:] = -1e9

        beam_scores = beam_scores.view(-1)

        for step in range(max_length):

            tgt_mask = self.decoder.generate_square_subsequent_mask(input_ids.size(1)).to(inputs.device)
            tgt_key_padding_mask = self.decoder.create_pad_mask(input_ids, pad_token_id).to(inputs.device)


            outputs = self.decoder(input_ids, encoder_output.permute(1, 0, 2), tgt_mask=None, tgt_key_padding_mask=tgt_key_padding_mask)

            next_token_logits = outputs[:, -1, :]

            next_token_scores = F.log_softmax(next_token_logits, dim=-1)


            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)
            vocab_size = next_token_scores.size(-1)

            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)


            next_scores, next_tokens = torch.topk(next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
            next_indices = next_tokens // vocab_size
            next_tokens = next_tokens % vocab_size

            beam_outputs = beam_scorer.process(
                input_ids, next_scores, next_tokens, next_indices,
                pad_token_id=pad_token_id
            )

            input_ids = torch.cat([input_ids[beam_outputs["next_beam_indices"], :], beam_outputs["next_beam_tokens"].unsqueeze(-1)], dim=-1)
            beam_scores = beam_outputs["next_beam_scores"]

            if beam_scorer.is_done:
                break

        final_outputs = beam_scorer.finalize(input_ids, beam_scores, next_tokens, next_indices, pad_token_id=pad_token_id)

        return final_outputs[:, 1:]
    
from abc import *
from collections import UserDict



# This code is from huggingface
@abstractmethod
class BeamScorer(ABC):
    """
    Abstract base class for all beam scorers that are used for :meth:`~transformers.PretrainedModel.beam_search` and
    :meth:`~transformers.PretrainedModel.beam_sample`.
    """

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs
    ):
        raise NotImplementedError("This is an abstract method.")

    def finalize(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        **kwargs
    ) -> torch.LongTensor:
        raise NotImplementedError("This is an abstract method.")



class BeamSearchScorer(BeamScorer):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        num_beams: int,
        device: torch.device,
        length_penalty = 1.0,
        do_early_stopping  = False,
        num_beam_hyps_to_keep = 1,
    ):
        self.max_length = max_length + 1
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.do_early_stopping = do_early_stopping
        self.num_beam_hyps_to_keep = num_beam_hyps_to_keep

        self._is_init = False
        self._beam_hyps = [
            BeamHypotheses(
                num_beams=self.num_beams,
                max_length=self.max_length,
                length_penalty=self.length_penalty,
                early_stopping=self.do_early_stopping,
            )
            for _ in range(batch_size)
        ]
        self._done = torch.tensor([False for _ in range(batch_size)], dtype=torch.bool, device=self.device)

        if not isinstance(num_beams, int) or num_beams <= 1:
            raise ValueError(
                f"`num_beams` has to be an integer strictly greater than 1, but is {num_beams}. For `num_beams` == 1, one should make use of `greedy_search` instead."
            )

    @property
    def is_done(self) -> bool:
        return self._done.all()

    def process(
        self,
        input_ids: torch.LongTensor,
        next_scores: torch.FloatTensor,
        next_tokens: torch.LongTensor,
        next_indices: torch.LongTensor,
        pad_token_id = None,
        eos_token_id = None,
    ):
        cur_len = input_ids.shape[-1]
        batch_size = len(self._beam_hyps)
        assert batch_size == (input_ids.shape[0] // self.num_beams)

        device = input_ids.device
        next_beam_scores = torch.zeros((batch_size, self.num_beams), dtype=next_scores.dtype, device=device)
        next_beam_tokens = torch.zeros((batch_size, self.num_beams), dtype=next_tokens.dtype, device=device)
        next_beam_indices = torch.zeros((batch_size, self.num_beams), dtype=next_indices.dtype, device=device)

        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                assert (
                    len(beam_hyp) >= self.num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(self.num_beams)
                assert (
                    eos_token_id is not None and pad_token_id is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                # pad the batch
                next_beam_scores[batch_idx, :] = 0
                next_beam_tokens[batch_idx, :] = pad_token_id
                next_beam_indices[batch_idx, :] = 0
                continue

            # next tokens for this sentence
            beam_idx = 0
            for beam_token_rank, (next_token, next_score, next_index) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx], next_indices[batch_idx])
            ):
                batch_beam_idx = batch_idx * self.num_beams + next_index
                # add to generated hypotheses if end of sentence
                if (eos_token_id is not None) and (next_token.item() == eos_token_id):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= self.num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    beam_hyp.add(
                        input_ids[batch_beam_idx].clone(),
                        next_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_beam_scores[batch_idx, beam_idx] = next_score
                    next_beam_tokens[batch_idx, beam_idx] = next_token
                    next_beam_indices[batch_idx, beam_idx] = batch_beam_idx
                    beam_idx += 1

                # once the beam for next step is full, don't add more tokens to it.
                if beam_idx == self.num_beams:
                    break

            if beam_idx < self.num_beams:
                raise ValueError(
                    f"At most {self.num_beams} tokens in {next_tokens[batch_idx]} can be equal to `eos_token_id: {eos_token_id}`. Make sure {next_tokens[batch_idx]} are corrected."
                )

            # Check if we are done so that we can save a pad step if all(done)
            self._done[batch_idx] = self._done[batch_idx] or beam_hyp.is_done(
                next_scores[batch_idx].max().item(), cur_len
            )

        return UserDict(
            {
                "next_beam_scores": next_beam_scores.view(-1),
                "next_beam_tokens": next_beam_tokens.view(-1),
                "next_beam_indices": next_beam_indices.view(-1),
            }
        )


    def finalize(
        self,
        input_ids: torch.LongTensor,
        final_beam_scores: torch.FloatTensor,
        final_beam_tokens: torch.LongTensor,
        final_beam_indices: torch.LongTensor,
        pad_token_id = None,
        eos_token_id = None,
    ) -> torch.LongTensor:
        batch_size = len(self._beam_hyps)

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx, beam_hyp in enumerate(self._beam_hyps):
            if self._done[batch_idx]:
                continue

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(self.num_beams):
                batch_beam_idx = batch_idx * self.num_beams + beam_id
                final_score = final_beam_scores[batch_beam_idx].item()
                final_tokens = input_ids[batch_beam_idx]
                beam_hyp.add(final_tokens, final_score)

        # select the best hypotheses
        sent_lengths = input_ids.new(batch_size * self.num_beam_hyps_to_keep)
        best = []

        # retrieve best hypotheses
        for i, beam_hyp in enumerate(self._beam_hyps):
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0])
            for j in range(self.num_beam_hyps_to_keep):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[self.num_beam_hyps_to_keep * i + j] = len(best_hyp)
                best.append(best_hyp)

        # prepare for adding eos
        sent_max_len = min(sent_lengths.max().item() + 1, self.max_length)
        decoded: torch.LongTensor = input_ids.new(batch_size * self.num_beam_hyps_to_keep, sent_max_len)
        # shorter batches are padded if needed
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, "`pad_token_id` has to be defined"
            decoded.fill_(pad_token_id)

        # fill with hypotheses and eos_token_id if the latter fits in
        for i, hypo in enumerate(best):
            decoded[i, : sent_lengths[i]] = hypo
            if sent_lengths[i] < self.max_length:
                decoded[i, sent_lengths[i]] = eos_token_id
        return decoded



class BeamHypotheses:
    def __init__(self, num_beams: int, max_length: int, length_penalty: float, early_stopping: bool):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.LongTensor, sum_logprobs: float):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: int) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated can become better than the worst
        one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


    