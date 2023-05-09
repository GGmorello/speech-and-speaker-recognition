import numpy as np
from lab3_tools import *
from lab2_tools import *
from lab2_proto import *

def words2phones(wordList, pronDict, addSilence=True, addShortPause=True):
    """ word2phones: converts word level to phone level transcription adding silence

    Args:
       wordList: list of word symbols
       pronDict: pronunciation dictionary. The keys correspond to words in wordList
       addSilence: if True, add initial and final silence
       addShortPause: if True, add short pause model "sp" at end of each word
    Output:
       list of phone symbols
    """
    result = []

    if addSilence:
        result.append('sil')

    for word in wordList:
        if word in pronDict:
            phones = pronDict[word]

            for phone in phones:
                result.append(phone)

            if addShortPause:
                result.append('sp')

    if addSilence:
        # Remove the last 'sp' if it exists
        if result[-1] == 'sp':
            result.pop()
        result.append('sil')
    return result
    
def forcedAlignment(lmfcc, phoneHMMs, phoneTrans):
    """ forcedAlignmen: aligns a phonetic transcription at the state level

    Args:
       lmfcc: NxD array of MFCC feature vectors (N vectors of dimension D)
              computed the same way as for the training of phoneHMMs
       phoneHMMs: set of phonetic Gaussian HMM models
       phoneTrans: list of phonetic symbols to be aligned including initial and
                   final silence

    Returns:
       list of strings in the form phoneme_index specifying, for each time step
       the state from phoneHMMs corresponding to the viterbi path.
    """
    phoneme_index_list = []

    # Concatenate HMMs for the phone transcription
    utterance_HMM = concatHMMs(phoneHMMs, phoneTrans)

    # Generate state transition list
    state_transition = [f"{phone}_{state_id}" for phone in phoneTrans
                        for state_id in range(phoneHMMs[phone]['means'].shape[0])]

    # Calculate observation log likelihoods
    observation_log_likelihood = log_multivariate_normal_density_diag(lmfcc, utterance_HMM['means'], utterance_HMM['covars'])

    # Apply Viterbi algorithm
    _, viterbi_path = viterbi(observation_log_likelihood, np.log(utterance_HMM['startprob']), np.log(utterance_HMM['transmat']), forceFinalState=True)

    # Convert state path to phoneme_index format
    for index in viterbi_path:
        phoneme_index_list.append(state_transition[int(index)])

    return phoneme_index_list