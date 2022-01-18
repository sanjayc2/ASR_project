# ASR_project_CNN+RNN_CTC
This is a repository for my ASR project for Udacity's Natural Language Processing Nanodegree capstone project. It was built using Tensorflow2, and was run on my gaming laptop (which has one GPU and six CPU cores).
It uses the LibriSpeech dataset. An audio spectrogram feature representation is used. Much of the data processing functionality was provided in the data_generator.py file. It also contains a data processing function which uses data augmentation, i.e. time and frequency masking from tensorflow I/O.
The train_utils.py contains the utility function for training the model. Note that, while the file contains a pytorch-based training function, that was not eventally used.

The sample_models.py contains all the models built/tested. The final model architecture comprises a single CNN layer, along with two GRU layers followed by a time-distributed dense layer, and the final softmax layer to get the probabilities of each character in the output sequence. The reason for the choice of CNN and bidirectional GRU were based on earlier models that I tested. As noted earlier, overfitting is a huge problem, in part since the data set size is pretty small. I tested a large number of architectures (with a few different choices of kernel size and stride) using both regular dropout (for the CNN part) and recurrent dropout for the RNN part (regualr dropout doesnt work for RNNs, as explained by Gal and Ghahramani) to control overfitting. I also tried changing the GRU units and number of layer (2-3). However, even if the overfitting could be controlled, I was not able to get very good generalization. I even tried to add causal dilation (as in Google's WaveNet), but could not get improved generalization, possibly because I was still using a single CNN layer. I eventually decided to skip dilation, and instead use Google Brain's SpecAugment methodology for ASR, which applies time and frequency masking to the audio spectrogram, which has been shown to help prevent overfitting and improve generalization. Although this definitely improved the results, once the training loss became low enough, I could see overfitting again on my training dataset. I first added back regular dropout, and then recurrent dropout (for GRU layers) and was finally able to come up with a good solution (called model_end_8) for which both the training and validaiton loss were close to 100. btw, I should note that in order to get good convergence, I had to lower the learning rate to 0.01 for the final 50 epochs.. (although a learning schedule would have been a better solution) I should note that I changed the decoder (greedy-False in ctc_decode) to use beam search instead of greedy search, since I found that improved the predictions.

Results can be improved by pairing the CTC with a language model trained on a larger corpus, as in Baidu's DeepSpeech2. For a PyTorch implementation of that, see https://github.com/SeanNaren/deepspeech.pytorch
