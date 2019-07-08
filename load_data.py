import numpy as np
import pickle
from scipy import sparse

nli_pkl = './data/snli_bert.pkl'


def array(x, dtype=np.int32):
    return np.array(x, dtype=dtype)


def load_pkl(file):
    # load pickle file
    f = open(file, 'rb')
    data = pickle.load(f)
    f.close()
    return data


class DataLoader:

    def __init__(self, max_word=128, frac=1.0):

        self.max_word = max_word

        dataset = load_pkl(nli_pkl)
        self.sentences = dataset['sentence']
        self.mask = dataset['mask']
        self.sgid = dataset['sgid']
        self.label = dataset['label']

        self.sentences  = self.sentences.reshape((-1, 128))
        self.mask       = self.mask.reshape((-1, 128))
        self.sgid       = self.sgid.reshape((-1, 128))
        self.label      = self.label.reshape((-1))
        assert self.max_word == self.sentences.shape[1]

        self.frac = frac
        self.n_train = int(549367 * frac)
        print('Training Samples: {} ({}%) Loaded'.format(self.n_train, 100*frac))

        self.pos = 0

        self.train_index = np.arange(0, 549367)[0:self.n_train]
        self.val_index = np.arange(549367, 549367+9842)
        self.test_index = np.arange(549367+9842, 549367+9842+9824)

        
    def iter_reset(self, shuffle=True):
        self.pos = 0
        if shuffle:
            np.random.shuffle(self.train_index)

    def sampled_batch(self, batch_size, phase='train'):

        # batch iterator, shuffle if train
        if phase == 'train':
            n = len(self.train_index)
            self.iter_reset(shuffle=True)
            index = self.train_index
        elif phase == 'validation':
            n = len(self.val_index)
            self.iter_reset(shuffle=False)
            index = self.val_index
        elif phase == 'test':
            n = len(self.test_index)
            self.iter_reset(shuffle=False)
            index = self.test_index


        while self.pos < n:

            X_batch = []
            Y_batch = []
            sgid_batch = []
            M_batch = []

            for i in range(batch_size):
                X_batch.append(self.sentences[index[self.pos]])
                Y_batch.append(self.label[index[self.pos]])
                sgid_batch.append(self.sgid[index[self.pos]])
                M_batch.append(self.mask[index[self.pos]])

                self.pos += 1
                if self.pos >= n:
                    break

            yield array(X_batch), array(M_batch), array(sgid_batch), array(Y_batch)


    def get_data(self, index = 0):
        return self.sentences[index], array(self.label[index]), self.mask[index], self.sgid[index]


if __name__ == "__main__":
    iterator = DataLoader()
    for x, m, i, y in iterator.sampled_batch(16, 'test'):
        print(x.shape, m.shape, i.shape,y.shape)
