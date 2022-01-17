"""
Defines a functions for training a NN.
"""

from data_generator import AudioGenerator
import _pickle as pickle

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Lambda)
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
import os
#import torch
#import torch.nn as nn
#import torch.nn.utils as utils
#import torch.optim as optim
#import torch.nn.functional as F
#import torchaudio
#from torch.utils.tensorboard import SummaryWriter
import numpy as np

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def add_ctc_loss(input_to_softmax):
    the_labels = Input(name='the_labels', shape=(None,), dtype='float32')
    input_lengths = Input(name='input_length', shape=(1,), dtype='int64')
    label_lengths = Input(name='label_length', shape=(1,), dtype='int64')
    output_lengths = Lambda(input_to_softmax.output_length)(input_lengths)
    # CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [input_to_softmax.output, the_labels, output_lengths, label_lengths])
    model = Model(
        inputs=[input_to_softmax.input, the_labels, input_lengths, label_lengths], 
        outputs=loss_out)
    return model

def train_model(input_to_softmax, 
                pickle_path,
                save_model_path,
                train_json='train_corpus.json',
                valid_json='valid_corpus.json',
                minibatch_size=20,
                spectrogram=True,
                mfcc_dim=13,
                optimizer=SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5),
                epochs=20,
                verbose=1,
                sort_by_duration=False,
                max_duration=10.0):
    
    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size=minibatch_size, 
        spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration,
        sort_by_duration=sort_by_duration)
    # add the training data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)
    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size
    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths) 
    validation_steps = num_valid_samples//minibatch_size
    
    # add CTC loss to the NN specified in input_to_softmax
    model = add_ctc_loss(input_to_softmax)

    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')

    # add checkpointer
    checkpointer = ModelCheckpoint(filepath='results/'+save_model_path, verbose=0)

    # train the model
    hist = model.fit(x=audio_gen.next_train(), steps_per_epoch=steps_per_epoch,
        epochs=epochs, validation_data=audio_gen.next_valid(), validation_steps=validation_steps,
        callbacks=[checkpointer], verbose=verbose)

    # save model loss
    with open('results/'+pickle_path, 'wb') as f:
        pickle.dump(hist.history, f)

'''
train_audio_transforms = nn.Sequential(
    #torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
    torchaudio.transforms.TimeMasking(time_mask_param=100)
)
#valid_audio_transforms = torchaudio.transforms.MelSpectrogram()

def data_processing(data, data_type="train"):
    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []
    # data is a dict.. get the waveforms and utterances ndarrays
    specgrams = data['the_input']                
    utterances = data['the_labels']
    for specgram, utterance in zip(specgrams, utterances):
        if data_type == 'train':
            spec = torch.Tensor(specgram) #train_audio_transforms(torch.Tensor(specgram))
        elif data_type == 'valid':
            spec = torch.Tensor(specgram)
        else:
            raise Exception('data_type should be train or valid')
        spectrograms.append(spec)
        label = torch.Tensor(utterance)
        labels.append(label)
        input_lengths.append(spec.shape[0]//2)
        label_lengths.append(len(label))

    #print("spectrogram[0] shape:", spectrograms[0].shape)
    #print("labels[0] shape:", labels[0].shape)

    spectrograms = utils.rnn.pad_sequence(spectrograms, batch_first=True)
    labels = utils.rnn.pad_sequence(labels, batch_first=True)

    return spectrograms, labels, input_lengths, label_lengths

def train_model_pytorch(model, 
                        pickle_path,
                        save_model_path,
                        optimizer,
                        train_json='train_corpus.json',
                        valid_json='valid_corpus.json',
                        minibatch_size=20,
                        spectrogram=True,
                        mfcc_dim=13,
                        epochs=20,
                        verbose=1,
                        sort_by_duration=False,
                        max_duration=10.0):
    
    
    # make results/ directory, if necessary
    if not os.path.exists('results'):
        os.makedirs('results')
    file_writer = SummaryWriter(log_dir='results/'+save_model_path+'/metrics')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("device used:", device)
    model.to(device)
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    #best_val_acc = 0 # for model check pointing
    
    # create a class instance for obtaining batches of data
    audio_gen = AudioGenerator(minibatch_size=minibatch_size, 
        spectrogram=spectrogram, mfcc_dim=mfcc_dim, max_duration=max_duration,
        sort_by_duration=sort_by_duration)
    # add the training data to the generator
    audio_gen.load_train_data(train_json)
    audio_gen.load_validation_data(valid_json)
    # calculate steps_per_epoch
    num_train_examples=len(audio_gen.train_audio_paths)
    steps_per_epoch = num_train_examples//minibatch_size
    #print("steps per epoch:", steps_per_epoch)    
    # calculate validation_steps
    num_valid_samples = len(audio_gen.valid_audio_paths) 
    validation_steps = num_valid_samples//minibatch_size
    
    # add CTC loss to the NN specified in input_to_softmax
    loss_criterion = nn.CTCLoss(blank=28).to(device)          # blank maps to 28 (see char_map.py)
    
    # loop over epochs
    start.record()
    train_loss = np.zeros(epochs)
    valid_loss = np.zeros(epochs)
    for epoch in range(epochs):
        # loop over batches to train the model
        model.train()
        for batch_idx, _data in enumerate(audio_gen.next_train()): 
            inputs, outputs = _data
            spectrograms, labels, input_lengths, label_lengths = data_processing(inputs, "train")
            # for cnn models, add extra dimension for input channel. Also, transpose features and seq length order
            spectrograms = spectrograms.unsqueeze(1).transpose(2,3)
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            #print("spectrograms shape:", spectrograms.shape)

            optimizer.zero_grad()
            
            output = model(spectrograms)  # output shape is (batch, time, n_classes)
            output = F.log_softmax(output, dim=2)
            output = output.transpose(0, 1) # change output shape to (time, batch, n_classes) for use in CTC loss calc
            
            #print("output shape:", output.shape)
            #print("labels shape:", labels.shape)
            #print("input lengths:", input_lengths)
            #print("label lengths:", label_lengths)
            
            loss = loss_criterion(output, labels, input_lengths, label_lengths)
            loss.backward()

            utils.clip_grad_norm_(model.parameters(), 5)   # gradient clipping 
            optimizer.step()
            train_loss[epoch] += loss.item() / steps_per_epoch                     # training loss per batch
        
            #if batch id reaches number per epoch, do validaiton and break out of the training for loop
            #print("batch_idx:", batch_idx)
            if ((batch_idx+1) % steps_per_epoch == 0):
                model.eval()
                with torch.no_grad():
                    for idx, val_data in enumerate(audio_gen.next_valid()):
                        inputs, outputs = val_data
                        spectrograms, labels, input_lengths, label_lengths = data_processing(inputs, "valid")
                        # for cnn models, add extra dimension for input channel. Also, transpose features and seq length order
                        spectrograms = spectrograms.unsqueeze(1).transpose(2,3)
                        spectrograms, labels = spectrograms.to(device), labels.to(device)

                        output = model(spectrograms)  # (batch, time, n_class)
                        output = F.log_softmax(output, dim=2)
                        output = output.transpose(0, 1) # (time, batch, n_class)

                        val_loss = loss_criterion(output, labels, input_lengths, label_lengths)
                        valid_loss[epoch] += val_loss.item() / validation_steps     # validation loss per batch
                        #print("valid batch idx:", idx)
                        if ((idx+1) % validation_steps == 0):
                            break
                break
        
        print("epoch, train and val loss", epoch, train_loss[epoch], valid_loss[epoch])
        # save train and valid losses for each epoch
        for epoch in range(epochs):
            file_writer.add_scalar('Loss/train', train_loss[epoch], epoch)
            file_writer.add_scalar('Loss/validation', valid_loss[epoch], epoch)

    # after all epochs, save model loss using pytorch save
    # add checkpoint
    filepath='results/'+save_model_path+'.pt'
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            'valid_loss': valid_loss,
            }, filepath)
    
    end.record()
    torch.cuda.synchronize()
    print("total time taken:", start.elapsed_time(end))

    with open('results/'+pickle_path, 'wb') as f:
        torch.save({'loss': train_loss,
                   'val_loss': valid_loss}, f)
'''