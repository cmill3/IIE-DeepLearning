import torch
from torch import nn
from torch import Tensor
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class InputAttentionEncoder(nn.Module):
    def __init__(self, input_features, encoder_units, time_steps, device, stateful=False):
        super(InputAttentionEncoder, self).__init__()
        self.input_features = input_features
        self.encoder_units = encoder_units
        self.time_steps = time_steps
        self.device = device
        
        self.encoder_lstm = nn.LSTMCell(input_size=self.input_features, hidden_size=self.encoder_units)
        
        # equation 8 matrices
        self.W_e = nn.Linear(2 * self.encoder_units, self.time_steps)
        self.U_e = nn.Linear(self.time_steps, self.time_steps, bias=False)
        self.v_e = nn.Linear(self.time_steps, 1, bias=False)
    
    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.size(0), self.time_steps, self.encoder_units)).to(self.device)
        
        # initialize hidden states
        h_tm1 = torch.zeros((inputs.size(0), self.encoder_units)).to(self.device)
        s_tm1 = torch.zeros((inputs.size(0), self.encoder_units)).to(self.device)
        
        for t in range(self.time_steps):
            # concatenate hidden states
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)
            
            # attention weights for each k in N (equation 8)
            x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.input_features, 1)
            y = self.U_e(inputs.permute(0, 2, 1))
            z = torch.tanh(x + y)
            e_k_t = torch.squeeze(self.v_e(z))
        
            # normalize attention weights (equation 9)
            alpha_k_t = F.softmax(e_k_t, dim=1)
            
            # weight inputs (equation 10)
            weighted_inputs = alpha_k_t * inputs[:, t, :] 
    
            # calculate next hidden states (equation 11)
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            
            encoded_inputs[:, t, :] = h_tm1
        return encoded_inputs
    

class TemporalAttentionDecoder(nn.Module):
    def __init__(self, encoder_units, decoder_units, time_steps, num_classes, device, stateful=False):
        super(TemporalAttentionDecoder, self).__init__()
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.time_steps = time_steps
        self.stateful = stateful
        self.device = device
        self.num_classes = num_classes
        
        self.decoder_lstm = nn.LSTMCell(input_size=encoder_units, hidden_size=self.decoder_units)
        
        # equation 12 matrices
        self.W_d = nn.Linear(2 * self.decoder_units, self.encoder_units)
        self.U_d = nn.Linear(self.encoder_units, self.encoder_units, bias=False)
        self.v_d = nn.Linear(self.encoder_units, 1, bias=False)
        
        # equation 22 matrices
        self.W_y = nn.Linear(self.decoder_units + self.encoder_units, self.decoder_units)
        self.v_y = nn.Linear(self.decoder_units, self.num_classes)
        
    def forward(self, encoded_inputs):
        # initializing hidden states
        d_tm1 = torch.zeros((encoded_inputs.size(0), self.decoder_units)).to(self.device)
        s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.decoder_units)).to(self.device)
        
        for t in range(self.time_steps):
            # concatenate hidden states
            d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
            
            # temporal attention weights (equation 12)
            x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)
            l_i_t = self.v_d(z1)
            
            # normalized attention weights (equation 13)
            beta_i_t = F.softmax(l_i_t, dim=1)
            
            # create context vector (equation 14)
            c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)
            
            # calculate next hidden states (equation 16)
            d_tm1, s_prime_tm1 = self.decoder_lstm(c_t, (d_tm1, s_prime_tm1))
        
        # concatenate context vector at step T and hidden state at step T
        d_c_concat = torch.cat((d_tm1, c_t), dim=1)

        # calculate output (softmax for classification)
        y_Tp1 = F.softmax(self.v_y(self.W_y(d_c_concat)), dim=1)
        return y_Tp1
    

class DARNN(nn.Module):
    def __init__(self, input_features, encoder_units, decoder_units, time_steps, num_classes, device, stateful_encoder=False, stateful_decoder=False):
        super(DARNN, self).__init__()
        self.encoder = InputAttentionEncoder(input_features, encoder_units, time_steps, device, stateful_encoder).to(device)
        self.decoder = TemporalAttentionDecoder(encoder_units, decoder_units, time_steps, num_classes, device, stateful_decoder).to(device)

    def forward(self, X_history):
        encoded_inputs = self.encoder(X_history)
        out = self.decoder(encoded_inputs)
        return out



# class InputAttentionEncoder(nn.Module):
#     def __init__(self, input_features, encoder_units, time_steps, device, stateful=False):
#         """
#         :param: N: int
#             number of time serieses
#         :param: M:
#             number of LSTM units
#         :param: T:
#             number of timesteps
#         :param: stateful:
#             decides whether to initialize cell state of new time window with values of the last cell state
#             of previous time window or to initialize it with zeros
#         """
#         super(self.__class__, self).__init__()
#         self.input_features = input_features
#         self.encoder_units = encoder_units
#         self.time_steps = time_steps
#         self.device = device
        
#         self.encoder_lstm = nn.LSTMCell(input_size=self.input_features, hidden_size=self.encoder_units)
        
#         #equation 8 matrices
        
#         self.W_e = nn.Linear(2*self.encoder_units, self.time_steps)
#         self.U_e = nn.Linear(self.time_steps, self.time_steps, bias=False)
#         self.v_e = nn.Linear(self.time_steps, 1, bias=False)
    
#     def forward(self, inputs):
#         encoded_inputs = torch.zeros((inputs.size(0), self.time_steps, self.encoder_units)).to(self.device)
        
#         #initiale hidden states
#         h_tm1 = torch.zeros((inputs.size(0), self.encoder_units)).to(self.device)
#         s_tm1 = torch.zeros((inputs.size(0), self.encoder_units)).to(self.device)
        
#         for t in range(self.time_steps):
#             #concatenate hidden states
#             h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)
            
#             #attention weights for each k in N (equation 8)
#             x = self.W_e(h_c_concat).unsqueeze_(1).repeat(1, self.input_features, 1)
#             y = self.U_e(inputs.permute(0, 2, 1))
#             z = torch.tanh(x + y)
#             e_k_t = torch.squeeze(self.v_e(z))
        
#             #normalize attention weights (equation 9)
#             alpha_k_t = F.softmax(e_k_t, dim=1)
            
#             #weight inputs (equation 10)
#             weighted_inputs = alpha_k_t * inputs[:, t, :] 
    
#             #calculate next hidden states (equation 11)
#             h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))
            
#             encoded_inputs[:, t, :] = h_tm1
#         return encoded_inputs
    

# class TemporalAttentionDecoder(nn.Module):
#     def __init__(self, encoder_units, decoder_units, time_steps, device, stateful=False):
#         """
#         :param: M: int
#             number of encoder LSTM units
#         :param: P:
#             number of deocder LSTM units
#         :param: T:
#             number of timesteps
#         :param: stateful:
#             decides whether to initialize cell state of new time window with values of the last cell state
#             of previous time window or to initialize it with zeros
#         """
#         super(self.__class__, self).__init__()
#         self.encoder_units = encoder_units
#         self.decoder_units = decoder_units
#         self.time_steps = time_steps
#         self.stateful = stateful
#         self.device = device
        
#         self.decoder_lstm = nn.LSTMCell(input_size=1, hidden_size=self.decoder_units)
        
#         #equation 12 matrices
#         self.W_d = nn.Linear(2*self.decoder_units, self.encoder_units)
#         self.U_d = nn.Linear(self.encoder_units, self.encoder_units, bias=False)
#         self.v_d = nn.Linear(self.encoder_units, 1, bias = False)
        
#         #equation 15 matrix
#         self.w_tilda = nn.Linear(self.encoder_units + 1, 1)
        
#         #equation 22 matrices
#         self.W_y = nn.Linear(self.decoder_units + self.encoder_units, self.decoder_units)
#         self.v_y = nn.Linear(self.decoder_units, 1)
        
#     def forward(self, encoded_inputs, y):
        
#         #initializing hidden states
#         d_tm1 = torch.zeros((encoded_inputs.size(0), self.decoder_units)).to(self.device)
#         s_prime_tm1 = torch.zeros((encoded_inputs.size(0), self.decoder_units)).to(self.device)
#         for t in range(self.time_steps):
#             #concatenate hidden states
#             d_s_prime_concat = torch.cat((d_tm1, s_prime_tm1), dim=1)
#             #print(d_s_prime_concat)
#             #temporal attention weights (equation 12)
#             x1 = self.W_d(d_s_prime_concat).unsqueeze_(1).repeat(1, encoded_inputs.shape[1], 1)
#             y1 = self.U_d(encoded_inputs)
#             z1 = torch.tanh(x1 + y1)
#             l_i_t = self.v_d(z1)
            
#             #normalized attention weights (equation 13)
#             beta_i_t = F.softmax(l_i_t, dim=1)
            
#             #create context vector (equation_14)
#             c_t = torch.sum(beta_i_t * encoded_inputs, dim=1)
            
#             #concatenate c_t and y_t
#             y_c_concat = torch.cat((c_t, y[:, t, :]), dim=1)
#             #create y_tilda
#             y_tilda_t = self.w_tilda(y_c_concat)
            
#             #calculate next hidden states (equation 16)
#             d_tm1, s_prime_tm1 = self.decoder_lstm(y_tilda_t, (d_tm1, s_prime_tm1))
        
#         #concatenate context vector at step T and hidden state at step T
#         d_c_concat = torch.cat((d_tm1, c_t), dim=1)

#         #calculate output
#         y_Tp1 = self.v_y(self.W_y(d_c_concat))
#         return y_Tp1
    

# class DARNN(nn.Module):
#     def __init__(self, input_features, encoder_units, decoder_units, time_steps, device, stateful_encoder=False, stateful_decoder=False):
#         super(self.__class__, self).__init__()
#         self.encoder = InputAttentionEncoder(input_features, encoder_units, time_steps, stateful_encoder).to(device)
#         self.decoder = TemporalAttentionDecoder(encoder_units, decoder_units, time_steps, stateful_decoder).to(device)
#     def forward(self, X_history, y_history):
#         out = self.decoder(self.encoder(X_history), y_history)
#         return out