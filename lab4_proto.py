import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from pyctcdecode import build_ctcdecoder



# DT2119, Lab 4 End-to-end Speech Recognition

# mapping between integers and lowercase letters of the alphabet
alphabet_dict = {
    0: "'",
    1: " ",
    2: 'a',
    3: 'b',
    4: 'c',
    5: 'd',
    6: 'e',
    7: 'f',
    8: 'g',
    9: 'h',
    10: 'i',
    11: 'j',
    12: 'k',
    13: 'l',
    14: 'm',
    15: 'n',
    16: 'o',
    17: 'p',
    18: 'q',
    19: 'r',
    20: 's',
    21: 't',
    22: 'u',
    23: 'v',
    24: 'w',
    25: 'x',
    26: 'y',
    27: 'z'
}




# Variables to be defined --------------------------------------
''' 
train-time audio transform object, that transforms waveform -> spectrogram, with augmentation
''' 
train_audio_transform = ...
'''
test-time audio transform object, that transforms waveform -> spectrogram, without augmentation 
'''
test_audio_transform = ...

# Functions to be implemented ----------------------------------

def intToStr(labels):
    '''
        convert list of integers to string
    Args: 
        labels: list of ints
    Returns:
        string with space-separated characters
    '''
    chars = [alphabet_dict[label] for label in labels]
    return ''.join(chars)

def strToInt(text):
    '''
        convert string to list of integers
    Args:
        text: string
    Returns:
        list of ints
    '''
    text=text.lower()
    return [list(alphabet_dict.keys())[list(alphabet_dict.values()).index(i)] for i in text]

def dataProcessing(data, transform):
    '''
    process a batch of speech data
    arguments:
        data: list of tuples, representing one batch. Each tuple is of the form
            (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        transform: audio transform to apply to the waveform
    returns:
        a tuple of (spectrograms, labels, input_lengths, label_lengths) 
        -   spectrograms - tensor of shape B x C x T x M 
            where B=batch_size, C=channel, T=time_frames, M=mel_band.
            spectrograms are padded the longest length in the batch.
        -   labels - tensor of shape B x L where L is label_length. 
            labels are padded to the longest length in the batch. 
        -   input_lengths - list of half spectrogram lengths before padding
        -   label_lengths - list of label lengths before padding
    '''

    spectrograms = []
    labels = []
    input_lengths = []
    label_lengths = []

    for waveform, _, utterance, _, _, _ in data:
        # Apply audio transformation to waveform
        spectrogram = transform(waveform)

        # Convert utterance string to integer label sequence
        label_sequence = strToInt(utterance)
        label_sequence = torch.tensor(label_sequence)

        # Rearrange spectrogram tensor
        spectrogram = spectrogram.squeeze(0).transpose(0, 1)

        spectrograms.append(spectrogram)
        labels.append(label_sequence)
        input_lengths.append(spectrogram.shape[0] // 2)
        label_lengths.append(len(label_sequence))

    # Pad spectrograms and labels
    spectrograms = pad_sequence(spectrograms, batch_first=True)
    labels = pad_sequence(labels, batch_first=True)

    # Rearrange spectrograms tensor to (batch, channel, mel bands, time)
    spectrograms = spectrograms.unsqueeze(1).transpose(2, 3)

    return spectrograms, labels, input_lengths, label_lengths

    
def greedyDecoder(output, blank_label=28):
    """
    Decode a batch of utterances using greedy decoding with a language model.

    Args:
        output (Tensor): Network output tensor of shape B x T x C, where B is the batch size,
            T is the number of time steps, and C is the number of characters.
        alphabet_dict (dict): Dictionary mapping label indices to characters.
        blank_label (int): ID of the blank label token.

    Returns:
        list: List of decoded strings.
    """
    decoded_strings = []
    #get list of characters from alphabet_dict
    characters = list(alphabet_dict.values())
    decoder = build_ctcdecoder(characters, 'wiki-interpolate.3gram.arpa', alpha=0.5, beta=1.0)

    for i in range(output.shape[0]):
        # Convert network output to numpy array and decode with the language model
        text = decoder.decode(output[i].cpu().detach().numpy())

        # Remove repeated characters and blank labels
        decoded_string = ""
        prev_char = ""
        for char in text:
            if char != alphabet_dict[blank_label] and char != prev_char:
                decoded_string += char
                prev_char = char
        decoded_strings.append(decoded_string)
    return decoded_strings

def levenshteinDistance(sequence1, sequence2):
    '''
    calculate levenshtein distance (edit distance) between two sequences
    arguments:
        ref: reference sequence
        hyp: sequence to compare against the reference
    output:
        edit distance (int)
    '''

    m = len(sequence1)
    n = len(sequence2)

    # Create the distance matrix
    distance = [[0] * (n + 1) for _ in range(m + 1)]

    # Initialize the first row and column
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j

    # Calculate the distance
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if sequence1[i - 1] == sequence2[j - 1]:
                cost = 0
            else:
                cost = 1

            distance[i][j] = min(
                distance[i - 1][j] + 1,  # Deletion
                distance[i][j - 1] + 1,  # Insertion
                distance[i - 1][j - 1] + cost  # Substitution
            )

    return distance[m][n]