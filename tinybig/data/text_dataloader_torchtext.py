# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

from abc import abstractmethod

import torch, torchtext
from torchtext.datasets import SST2, IMDB, AG_NEWS
from torchtext import transforms
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer

from tinybig.data.base_data import dataloader, dataset


class text_dataloader(dataloader):

    def __init__(self, name='text_dataloader', train_batch_size=64, test_batch_size=64, max_seq_len=256):
        super().__init__(name=name)
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_seq_len = max_seq_len

    @abstractmethod
    def load_datapipe(self, cache_dir='./data/', *args, **kwargs):
        pass

    @abstractmethod
    def get_class_number(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_train_number(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_test_number(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_idx_to_label(self, *args, **kwargs):
        pass

    @staticmethod
    def load_transform(max_seq_len, padding_idx: int = 1, bos_idx: int = 0, eos_idx: int = 2, *args, **kwargs):
        xlmr_vocab_path = r"https://download.pytorch.org/models/text/xlmr.vocab.pt"
        xlmr_spm_model_path = r"https://download.pytorch.org/models/text/xlmr.sentencepiece.bpe.model"

        text_transform = transforms.Sequential(
            transforms.SentencePieceTokenizer(xlmr_spm_model_path),
            transforms.VocabTransform(load_state_dict_from_url(xlmr_vocab_path)),
            transforms.Truncate(max_seq_len - 2),
            transforms.AddToken(token=bos_idx, begin=True),
            transforms.AddToken(token=eos_idx, begin=False),
            transforms.ToTensor(),
            transforms.PadTransform(max_seq_len, padding_idx),
        )
        return text_transform

    @staticmethod
    def get_embedding_number(*args, **kwargs):
        return 768

    @staticmethod
    def load_encoder(cache_dir='./data/', *args, **kwargs):
        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        encoder = xlmr_base.get_model()
        return encoder

    def load_tfidf_vectorizer(self, sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english", *args, **kwargs):
        vectorizer = TfidfVectorizer(sublinear_tf=sublinear_tf, max_df=max_df, min_df=min_df, stop_words=stop_words)
        return vectorizer

    def load_text(self, *args, **kwargs):
        return self.load(load_type='text', *args, **kwargs)

    def load_tfidf(self, *args, **kwargs):
        return self.load(load_type='tfidf', *args, **kwargs)

    def load_token(self, *args, **kwargs):
        return self.load(load_type='token', *args, **kwargs)

    def load_embedding(self, *args, **kwargs):
        return self.load(load_type='embedding', *args, **kwargs)

    def load(
            self,
            cache_dir='./data/',
            load_type: str = 'tfidf',
            max_seq_len: int = None,
            xy_reversed: bool = False,
            *args, **kwargs
    ):
        max_seq_len = max_seq_len if max_seq_len is not None else self.max_seq_len
        train_datapipe, test_datapipe = self.load_datapipe(cache_dir=cache_dir)
        transform = self.load_transform(max_seq_len=max_seq_len)
        idx_to_label = self.get_idx_to_label()

        def collect_data(datapipe):
            X = []
            Y = []
            if load_type in ['text', 'tfidf']:
                # for text and tfidf, no transform is needed
                for x, y in datapipe:
                    if not xy_reversed:
                        X.append(x)
                        Y.append(idx_to_label[y])
                    else:
                        X.append(y)
                        Y.append(idx_to_label[x])
                return X, Y
            else:
                # for token and embedding, the transform is applied
                for x, y in datapipe:
                    if not xy_reversed:
                        X.append(transform(x).tolist())
                        Y.append(idx_to_label[y])
                    else:
                        X.append(transform(y).tolist())
                        Y.append(idx_to_label[x])
                return torch.tensor(X, dtype=torch.int32), torch.tensor(Y)

        X_train, y_train = collect_data(train_datapipe)
        X_test, y_test = collect_data(test_datapipe)

        if load_type in ['text', 'token', 'embedding']:
            # for load_type = 'embedding', the encoder needs to be loaded from the cache dir
            encoder = self.load_encoder(cache_dir=cache_dir) if load_type == 'embedding' else None
            train_dataset = dataset(X=X_train, y=y_train, encoder=encoder)
            test_dataset = dataset(X=X_test, y=y_test, encoder=encoder)
        elif load_type == 'tfidf':
            vectorizer = self.load_tfidf_vectorizer()
            X_train = torch.tensor(vectorizer.fit_transform(X_train).toarray(), dtype=torch.float32)
            X_test = torch.tensor(vectorizer.transform(X_test).toarray(), dtype=torch.float32)
            train_dataset = dataset(X_train, y_train)
            test_dataset = dataset(X_test, y_test)
        else:
            raise ValueError('Unrecognized load type {}, current text data loader can only load raw text, token, tfidf, and embeddings...'.format(load_type))

        train_loader = DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)
        return {'train_loader': train_loader, 'test_loader': test_loader}


class imdb(text_dataloader):

    def __init__(self, name='imdb', train_batch_size=64, test_batch_size=64):
        super().__init__(name=name, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size)

    def load(self, *args, **kwargs):
        kwargs['xy_reversed'] = True
        return super().load(*args, **kwargs)

    @staticmethod
    def load_datapipe(cache_dir='./data/', *args, **kwargs):
        train_datapipe = IMDB(root=cache_dir, split="train")
        test_datapipe = IMDB(root=cache_dir, split="test")
        return train_datapipe, test_datapipe

    @staticmethod
    def get_class_number(*args, **kwargs):
        return 2

    @staticmethod
    def get_train_number(*args, **kwargs):
        return 25000

    @staticmethod
    def get_test_number(*args, **kwargs):
        return 25000

    @staticmethod
    def get_idx_to_label(*args, **kwargs):
        return {
            1: 0,
            2: 1,
        }


class sst2(text_dataloader):

    def __init__(self, name='sst2', train_batch_size=64, test_batch_size=64):
        super().__init__(name=name, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size)

    @staticmethod
    def load_datapipe(cache_dir='./data/', *args, **kwargs):
        train_datapipe = SST2(root=cache_dir, split="train")
        test_datapipe = SST2(root=cache_dir, split="dev")
        return train_datapipe, test_datapipe

    @staticmethod
    def get_class_number(*args, **kwargs):
        return 2

    @staticmethod
    def get_train_number(*args, **kwargs):
        return 67349

    @staticmethod
    def get_test_number(*args, **kwargs):
        return 872

    @staticmethod
    def get_idx_to_label(*args, **kwargs):
        return {
            0: 0,
            1: 1,
        }


class agnews(text_dataloader):

    def __init__(self, name='ag_news', train_batch_size=64, test_batch_size=64):
        super().__init__(name=name, train_batch_size=train_batch_size,
                         test_batch_size=test_batch_size)

    def load(self, *args, **kwargs):
        kwargs['xy_reversed'] = True
        return super().load(*args, **kwargs)

    @staticmethod
    def load_datapipe(cache_dir='./data/', *args, **kwargs):
        train_datapipe = AG_NEWS(root=cache_dir, split="train")
        test_datapipe = AG_NEWS(root=cache_dir, split="test")
        return train_datapipe, test_datapipe

    @staticmethod
    def get_class_number(*args, **kwargs):
        return 4

    @staticmethod
    def get_train_number(*args, **kwargs):
        return 120000

    @staticmethod
    def get_test_number(*args, **kwargs):
        return 7600

    @staticmethod
    def get_idx_to_label(*args, **kwargs):
        return {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }


if __name__ == '__main__':
    import time

    print(agnews.get_class_number())
    dataloader = sst2()
    data = dataloader.load(load_type='tfidf')
    print(dataloader.get_class_number())
    start = time.time()
    for x, y in data['test_loader']:
        print(x.shape, y.shape)
