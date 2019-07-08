import os
import pickle
import numpy as np


import tokenization
import csv
import collections


# inputs
root = './preprocess/SST'

# intermediate output
sst_encoded = 'data/sst_bert.pkl'

def read_sentence(path):

    lines = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            lines.append([line[2:].replace('\n', ''), '', line[0]])
    return lines


def read_raw(pm, hp, lb):
    lines = []
    f_pm = open(pm, encoding='utf-8', errors='ignore')
    f_hp = open(hp, encoding='utf-8', errors='ignore')
    f_lb = open(lb, encoding='utf-8', errors='ignore')

    for (line_pm, line_hp, line_lb) in zip(f_pm, f_hp, f_lb):
        line = [filter(line_pm), filter(line_hp), filter(line_lb)]

        lines.append(line)

    return lines

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id

def read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

def create_examples(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    dict = {'0': 'negative', '1': 'positive'}
    for (i, line) in enumerate(lines):

        guid = "%s-%s" % (set_type, str(i))
        # print("---line[8]", line[8])
        text_a = tokenization.convert_to_unicode(line[0])
        # print("---tex_a", text_a)
        #text_b = tokenization.convert_to_unicode(line[1])
        label = tokenization.convert_to_unicode(dict[line[-1]])
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {}  # label
    for (i, label) in enumerate(label_list):  # ['0', '1']
        label_map[label] = i

    features = []  # feature
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)  # text_a tokenize

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)  # text_b tokenize

        if tokens_b:  # if has b
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)  # truncate
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because  # (?)
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)  # mask

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        # if ex_index < 5:  # output
        #   tf.logging.info("*** Example ***")
        #   tf.logging.info("guid: %s" % (example.guid))
        #   tf.logging.info("tokens: %s" % " ".join(
        #       [tokenization.printable_text(x) for x in tokens]))
        #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        #   tf.logging.info(
        #       "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        #   tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(  # object
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features

def feature_list(features):
    # ids, mask, sgid, label = [],[],[],[]
    # for i in range(len(features)):
    #     ids.append(features[i].input_ids)
    #     mask.append(features[i].input_mask)
    #     sgid.append(features[i].segment_ids)
    #     label.append(features[i].label_id)
    # return ids, mask, sgid, label
    l = len(features)
    ids = np.zeros(shape=(l, max_seq_length), dtype=np.int32)
    mask = np.zeros(shape=(l, max_seq_length), dtype=np.int32)
    sgid = np.zeros(shape=(l, max_seq_length), dtype=np.int32)
    label = np.zeros(shape=(l), dtype=np.int32)
    for i in range(len(features)):
        ids[i] = features[i].input_ids
        mask[i] = features[i].input_mask
        sgid[i] = features[i].segment_ids
        label[i] = features[i].label_id
    return ids, mask, sgid, label

def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  vocab_inverse = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r") as reader:
    while True:
      token = reader.readline()
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      vocab_inverse[index] = token
      index += 1
  return vocab, vocab_inverse



if __name__ == "__main__":

    # config
    vocab_file = 'uncased_L-12_H-768_A-12/vocab.txt'
    max_seq_length      = 128        # 128
    do_lower_case       = True         # True
    label_list          = ["negative", "positive"]

    train_sent_path = os.path.join(root, 'stsa.binary.train')
    val_sent_path = os.path.join(root, 'stsa.binary.dev')
    test_sent_path = os.path.join(root, 'stsa.binary.test')

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)

    print('encoding dataset...')

    print("***** Read train_data *****")
    train_examples = create_examples(read_sentence(train_sent_path), "train")  # [exp1, exp2...] example X is a class object: {guid, text_a(str), text_b(str), label(str)}
    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer)
    train_ids, train_mask, train_sgid, train_label = feature_list(train_features)
    print("train:", len(train_ids))
    print(np.sum(train_label))

    print("***** Read validation_data *****")
    val_examples = create_examples(read_sentence(val_sent_path), "validation")
    val_features = convert_examples_to_features(val_examples, label_list, max_seq_length, tokenizer)
    val_ids, val_mask, val_sgid, val_label = feature_list(val_features)
    print("validation:", len(val_ids))
    print(np.sum(val_label))

    print("***** Read test_data *****")
    test_examples = create_examples(read_sentence(test_sent_path), "test")
    test_features = convert_examples_to_features(test_examples, label_list, max_seq_length, tokenizer)
    test_ids, test_mask, test_sgid, test_label = feature_list(test_features)
    print("test:", len(test_ids))
    print(np.sum(test_label))

    ids   = np.concatenate([train_ids,  val_ids, test_ids], axis=0)
    mask  = np.concatenate([train_mask, val_mask, test_mask], axis=0)
    sgid  = np.concatenate([train_sgid, val_sgid, test_sgid], axis=0)
    label = np.concatenate([train_label, val_label, test_label], axis=0)
    print("all:", len(ids))

    with open(sst_encoded, 'wb') as sst_dmp:
        pickle.dump({'sentence': ids, 'mask': mask, 'sgid': sgid, 'label': label}, sst_dmp)

