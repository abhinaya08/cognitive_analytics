import numpy as np
import tensorflow as tf
from nltk.translate.bleu_score import sentence_bleu


def minibatches(inputs, targets, minibatch_size):
    """batch generator. yields x and y batch.
    """
    x_batch, y_batch = [], []
    for inp, tgt in zip(inputs, targets):
        if len(x_batch) == minibatch_size and len(y_batch) == minibatch_size:
            yield x_batch, y_batch
            x_batch, y_batch = [], []
        x_batch.append(inp)
        y_batch.append(tgt)

    if len(x_batch) != 0:
        for inp, tgt in zip(inputs, targets):
            if len(x_batch) != minibatch_size:
                x_batch.append(inp)
                y_batch.append(tgt)
            else:
                break
        yield x_batch, y_batch


def pad_sequences(sequences, pad_tok, tail=True):
    """Pads the sentences, so that all sentences in a batch have the same length.
    """

    max_length = max(len(x) for x in sequences)

    sequence_padded, sequence_length = [], []

    for seq in sequences:
        seq = list(seq)
        if tail:
            seq_ = seq[:max_length] + [pad_tok] * max(max_length - len(seq), 0)
        else:
            seq_ = [pad_tok] * max(max_length - len(seq), 0) + seq[:max_length]

        sequence_padded += [seq_]
        sequence_length += [min(len(seq), max_length)]

    return sequence_padded, sequence_length


def sample_results(preds, ind2word, word2ind, converted_summaries, converted_texts, use_bleu=False):
    """Plots the actual text and summary and the corresponding created summary.
    takes care of whether beam search or greedy decoder was used.
    """
    beam = False

    if len(np.array(preds).shape) == 4:
        beam = True

    '''Bleu score is not used correctly here, but serves as reference.
    '''
    if use_bleu:
        bleu_scores = []

    for pred, summary, text, seq_length in zip(preds[0],
                                               converted_summaries,
                                               converted_texts,
                                               [len(inds) for inds in converted_summaries]):
        print('\n\n\n', 100 * '-')
        if beam:
            actual_text = [ind2word[word] for word in text if
                           word != word2ind["<SOS>"] and word != word2ind["<EOS>"]]
            actual_summary = [ind2word[word] for word in summary if
                              word != word2ind['<EOS>'] and word != word2ind['<SOS>']]

            created_summary = []
            for word in pred:
                if word[0] != word2ind['<SOS>'] and word[0] != word2ind['<EOS>']:
                    created_summary.append(ind2word[word[0]])
                    continue
                else:
                    continue

            print('Actual Text:\n{}\n'.format(' '.join(actual_text)))
            print('Actual Summary:\n{}\n'.format(' '.join(actual_summary)))
            print('Created Summary:\n{}\n'.format(' '.join(created_summary)))
            if use_bleu:
                bleu_score = sentence_bleu([actual_summary], created_summary)
                bleu_scores.append(bleu_score)
                print('Bleu-score:', bleu_score)

            print()


        else:
            actual_text = [ind2word[word] for word in text if
                           word != word2ind["<SOS>"] and word != word2ind["<EOS>"]]
            actual_summary = [ind2word[word] for word in summary if
                              word != word2ind['<EOS>'] and word != word2ind['<SOS>']]
            created_summary = [ind2word[word] for word in pred if
                               word != word2ind['<EOS>'] and word != word2ind['<SOS>']]

            print('Actual Text:\n{}\n'.format(' '.join(actual_text)))
            print('Actual Summary:\n{}\n'.format(' '.join(actual_summary)))
            print('Created Summary:\n{}\n'.format(' '.join(created_summary)))
            if use_bleu:
                bleu_score = sentence_bleu([actual_summary], created_summary)
                bleu_scores.append(bleu_score)
                print('Bleu-score:', bleu_score)

    if use_bleu:
        bleu_score = np.mean(bleu_scores)
        print('\n\n\nTotal Bleu Score:', bleu_score)


def reset_graph(seed=97):
    """helper function to reset the default graph. this often
       comes handy when using jupyter noteboooks.
    """
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)
