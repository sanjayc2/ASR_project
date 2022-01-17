from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
        TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
import numpy as np
import tensorflow as tf
import math

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='batchnorm_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model
        
def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='batchnorm_rnn')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid', 'causal'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    elif border_mode == 'causal':
        output_length = input_length - dilated_filter_size + 1
        
    return (output_length + stride - 1) // stride

'''
class cnn_rnn_model(nn.Module):
    def __init__(self, input_dim, output_dim=29, rnn_dim=32, num_layers=1, dropout=0.2, \
                 filters, kernel_size, conv_stride, conv_border_mode, batch_first=True):
        super(simple_rnn_model, self).__init__()

        model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride)
        self.BiGRU = nn.GRU(
            input_size=input_dim, hidden_size=rnn_dim,
            num_layers=num_layers, batch_first=batch_first, bidirectional=True)
        self.layer_norm  = nn.LayerNorm(input_dim)
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(2*rnn_dim, output_dim)
    
    def forward(self, x):
        #print("fwd x shape", x.shape)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
'''

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    for i in range(recur_layers):
        if i == 0:
            input = input_data
        simp_rnn = GRU(units, activation='relu', return_sequences=True, implementation=2, name='rnn_'+str(i))(input)
        bn_rnn = BatchNormalization(name='batchnorm_rnn'+str(i))(simp_rnn)
        input = bn_rnn  # for next loop
        
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn1'), merge_mode='concat')(input_data)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

'''
class bidirectional_rnn_model(nn.Module):
    def __init__(self, input_dim, output_dim=29, rnn_dim=128, hidden_size=128, num_layers=1, dropout=0.2, batch_first=True):
        super(bidirectional_rnn_model, self).__init__()
        
        self.output_length = lambda x: x    # create instance attribute
        self.BiGRU = nn.GRU(
            input_size=input_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm  = nn.LayerNorm(input_dim)           # normalization is over number of features
        self.dropout     = nn.Dropout(dropout)
        self.classifier  = nn.Linear(2*rnn_dim, output_dim)
    
    def forward(self, x):
        x = x.transpose(2, 3).contiguous().squeeze(1)    # input has extra (channel) dimension, with feature and time dim interchanged
        #print("fwd x shape", x.shape)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class BidirectionalGRU(nn.Module):

    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = nn.LayerNorm(rnn_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = F.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x
    
class CNNLayerNorm(nn.Module):
    """Layer normalization built for cnns input"""
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous() # (batch, channel, time, feature)
        x = self.layer_norm(x)
        return x.transpose(2, 3).contiguous() # (batch, channel, feature, time) 


class ResidualCNN(nn.Module):
    """Residual CNN inspired by https://arxiv.org/pdf/1603.05027.pdf
        except with layer norm instead of batch norm
    """
    def __init__(self, in_channels, out_channels, n_feats, kernel_size, stride, padding, dilation, dropout):
        super(ResidualCNN, self).__init__()

        #self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=kernel//2)
        #self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel, stride, padding=kernel//2)
        self.cnn1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation)
        self.cnn2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        #print("n_feats", n_feats)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = F.gelu(x)
        x = self.dropout1(x)
        #print("res_cnn fwd cnn1 input shape:", x.shape)
        x = self.cnn1(x)
        #print("res_cnn fwd layer_norm2 input shape:", x.shape)
        x = self.layer_norm2(x)
        x = F.gelu(x)
        x = self.dropout2(x)
        #print("res_cnn fwd cnn2 input shape:", x.shape)
        x = self.cnn2(x)
        x += residual
        return x # (batch, channel, feature, time)

class final_model(nn.Module):
    
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, padding=0, dilation=1, dropout=0.1):
        super(final_model, self).__init__()
        
        kernel_size = 3
        #self.cnn = nn.Conv2d(1, 32, kernel_size, stride=stride, padding=3//2)  # cnn for extracting heirachal features
        self.cnn = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=stride, padding=padding, dilation=dilation)

        #get number of features to pass to res cnn layers.. (based on # output features used in Conv2d above). 
        #Note, dim=0 for features, 1 for sequence (time)
        #n_feats = n_feats//2
        n_feats = int(np.floor((n_feats + 2*padding[0] - dilation[0]*(kernel_size - 1) - 1)// stride[0] + 1))
        print("num features:", n_feats)
        # n residual cnn layers with filter size of 32
        self.rescnn_layers = nn.Sequential(*[
            ResidualCNN(in_channels=32, out_channels=32, n_feats=n_feats, kernel_size=3, stride=1, 
                        padding=padding, dilation=dilation, dropout=dropout) 
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i==0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i==0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        #print("final model fwd rescnn input shape:", x.shape)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2) # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x

'''

#@tf.function(experimental_relax_shapes=True)
def final_model(input_dim, filters, kernel_size, conv_stride, conv_border_mode, cnn_recur_layers,
                rnn_recur_layers, rnn_units, recurrent_dropout, cnn_dropout, dilation_rate, output_dim=29):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add 1D convolutional and dropout layers
    conv_1d_model = Sequential()
    for i in range(cnn_recur_layers):
            conv_1d_model.add(Conv1D(filters, kernel_size, strides=conv_stride, padding=conv_border_mode,
                     activation='relu', dilation_rate=dilation_rate, name='conv1d_'+str(i)))
            conv_1d_model.add(Dropout(rate=cnn_dropout, name='dropout_'+str(i)))
    conv1d = conv_1d_model(input_data)
    # Add bidirectional recurrent layers
    bidir_rnn_model = Sequential()
    for i in range(rnn_recur_layers):
            bidir_rnn_model.add(Bidirectional(GRU(rnn_units, activation='tanh', return_sequences=True, recurrent_dropout=recurrent_dropout,
                                      implementation=2, name='rnn_'+str(i)), merge_mode='concat'))
            # Add batch normalization
            bidir_rnn_model.add(BatchNormalization(name='bn_rnn_'+str(i)))
    bidir_rnn = bidir_rnn_model(conv1d)
    # Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    model = Model(inputs=input_data, outputs=y_pred)
    # Specify model.output_length
    model.output_length = lambda x: cnn_output_length(x, kernel_size, conv_border_mode, conv_stride, dilation_rate)
    print(model.summary())
    return model