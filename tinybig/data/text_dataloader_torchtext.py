# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis

######################
# Text Dataloader #
######################

from abc import abstractmethod
from collections import Counter

import torch, torchtext
from torchtext.datasets import SST2, IMDB, AG_NEWS
from torchtext import transforms
from torchtext.vocab import build_vocab_from_iterator, GloVe
from torchtext.data import get_tokenizer
from torch.hub import load_state_dict_from_url
from torch.utils.data import DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer

from tinybig.data.base_data import dataloader, dataset


class text_dataloader(dataloader):
    """
    The base text dataloader class.

    This class provides methods for handling text data, including tokenization, embedding generation, and data loading.

    Attributes
    ----------
    name : str, default = 'text_dataloader'
        The name of the text dataloader.
    train_batch_size : int
        The batch size for training data.
    test_batch_size : int
        The batch size for testing data.
    max_seq_len : int, default = 256
        The maximum sequence length for text data.
    min_freq : int, default = 10
        The minimum frequency for including tokens in the vocabulary.

    Methods
    ----------
    __init__
        Initializes the base text dataloader.
    load_datapipe
        Abstract method to load data pipelines for training and testing data.
    get_class_number
        Abstract method to retrieve the number of classes in the dataset.
    get_train_number
        Abstract method to retrieve the number of training examples.
    get_test_number
        Abstract method to retrieve the number of testing examples.
    get_idx_to_label
        Abstract method to retrieve the mapping from indices to labels.
    load_transform
        Loads and returns the text transformation pipeline.
    get_embedding_dim
        Retrieves the embedding dimension for the text encoder.
    load_encoder
        Loads a pre-trained text encoder model.
    load_tfidf_vectorizer
        Loads and returns a TF-IDF vectorizer.
    load_text
        Loads raw text data.
    load_tfidf
        Loads TF-IDF representations of the data.
    load_token
        Loads tokenized representations of the data.
    load_embedding
        Loads embeddings generated from the text data.
    load
        Loads data based on the specified type (e.g., TF-IDF, tokens, embeddings).
    load_xlmr
        Loads data using the XLM-R model for tokenization and embeddings.
    load_glove
        Loads data using GloVe embeddings.
    """
    def __init__(self, train_batch_size: int, test_batch_size: int, name='text_dataloader', max_seq_len: int = 256, min_freq: int = 10):
        """
        Initializes the text dataloader with configuration options for sequence length and token frequency.

        Parameters
        ----------
        train_batch_size : int
            The batch size for training data.
        test_batch_size : int
            The batch size for testing data.
        name : str, default = 'text_dataloader'
            The name of the dataloader.
        max_seq_len : int, default = 256
            The maximum sequence length for text data.
        min_freq : int, default = 10
            The minimum frequency for tokens to be included in the vocabulary.
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
        self.max_seq_len = max_seq_len
        self.min_freq = min_freq

    @abstractmethod
    def load_datapipe(self, cache_dir='./data/', *args, **kwargs):
        """
        Abstract method to load the data pipeline.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            The directory where the data is cached.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Notes
        -----
        This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def get_class_number(self, *args, **kwargs):
        """
        Abstract method to retrieve the number of classes in the dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of classes.

        Notes
        -----
        This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def get_train_number(self, *args, **kwargs):
        """
        Abstract method to retrieve the number of training samples.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of training samples.

        Notes
        -----
        This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def get_test_number(self, *args, **kwargs):
        """
        Abstract method to retrieve the number of testing samples.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of testing samples.

        Notes
        -----
        This method must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def get_idx_to_label(self, *args, **kwargs):
        """
        Abstract method to retrieve the mapping from indices to labels.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary mapping indices to labels.

        Notes
        -----
        This method must be implemented in subclasses.
        """
        pass

    @staticmethod
    def load_transform(max_seq_len, padding_idx: int = 1, bos_idx: int = 0, eos_idx: int = 2, *args, **kwargs):
        """
        Loads the text transformation pipeline for preprocessing.

        Parameters
        ----------
        max_seq_len : int
            The maximum sequence length for the transformation.
        padding_idx : int, default = 1
            The index used for padding tokens.
        bos_idx : int, default = 0
            The index used for beginning-of-sequence tokens.
        eos_idx : int, default = 2
            The index used for end-of-sequence tokens.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torchtext.transforms.Sequential
            A sequential transformation pipeline for text preprocessing.
        """
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
    def get_embedding_dim(*args, **kwargs):
        """
        Retrieves the embedding dimension for the text encoder.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The embedding dimension (768).
        """
        return 768

    @staticmethod
    def load_encoder(cache_dir='./data/', *args, **kwargs):
        """
        Loads the pre-trained XLM-R encoder for text embeddings.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            The directory where the encoder is cached.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        torch.nn.Module
            The XLM-R encoder model.
        """

        xlmr_base = torchtext.models.XLMR_BASE_ENCODER
        encoder = xlmr_base.get_model()
        encoder.eval()
        return encoder

    def load_tfidf_vectorizer(self, sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english", *args, **kwargs):
        """
        Loads a TF-IDF vectorizer for text data.

        Parameters
        ----------
        sublinear_tf : bool, default = True
            Whether to apply sublinear term frequency scaling.
        max_df : float, default = 0.5
            The maximum document frequency for inclusion in the vocabulary.
        min_df : int, default = 5
            The minimum document frequency for inclusion in the vocabulary.
        stop_words : str, default = "english"
            The language of stopwords to exclude.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        TfidfVectorizer
            A TF-IDF vectorizer instance.
        """
        vectorizer = TfidfVectorizer(sublinear_tf=sublinear_tf, max_df=max_df, min_df=min_df, stop_words=stop_words)
        return vectorizer

    def load_text(self, *args, **kwargs):
        """
        Loads raw text data.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        return self.load(load_type='text', *args, **kwargs)

    def load_tfidf(self, *args, **kwargs):
        """
        Loads TF-IDF representations of text data.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        return self.load(load_type='tfidf', *args, **kwargs)

    def load_token(self, *args, **kwargs):
        """
        Loads tokenized text data.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        return self.load(load_type='token', *args, **kwargs)

    def load_embedding(self, *args, **kwargs):
        """
        Loads pre-trained embeddings for text data.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        return self.load(load_type='embedding', *args, **kwargs)

    def load(
        self,
        cache_dir: str = './data/',
        load_type: str = 'tfidf',
        max_seq_len: int = None,
        xy_reversed: bool = False,
        max_vocab_size: int = 25000,
        min_freq: int = 10,
        *args, **kwargs
    ):
        """
        General method to load text data in various formats.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            The directory where the data is cached.
        load_type : str, default = 'tfidf'
            The format of the data to load. Options include:
            - 'tfidf': Load TF-IDF representations.
            - 'text': Load raw text.
            - 'token': Load tokenized text.
            - 'embedding': Load pre-trained embeddings.
            - 'xlmr_embedding': Load XLM-R embeddings.
        max_seq_len : int, optional
            The maximum sequence length for the text data.
        xy_reversed : bool, default = False
            Whether to reverse the order of features (X) and labels (Y).
        max_vocab_size : int, default = 25000
            The maximum size of the vocabulary.
        min_freq : int, default = 10
            The minimum frequency for tokens to be included in the vocabulary.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        max_seq_len = max_seq_len if max_seq_len is not None else self.max_seq_len

        if load_type in ['tfidf', 'text', 'token', 'xlmr_embedding']:
            return self.load_xlmr(cache_dir=cache_dir, load_type=load_type, max_seq_len=max_seq_len, xy_reversed=xy_reversed)
        elif load_type in ['embedding']:
            return self.load_glove(cache_dir=cache_dir, max_seq_len=max_seq_len, min_freq=min_freq, max_vocab_size=max_vocab_size, xy_reversed=xy_reversed)
        else:
            raise ValueError(f'load_type {load_type} not supported')

    def load_xlmr(
        self,
        cache_dir='./data/',
        load_type: str = 'tfidf',
        max_seq_len: int = None,
        xy_reversed: bool = False,
        *args, **kwargs
    ):
        """
        Loads text data using XLM-R (Cross-lingual Model Representations) pipeline.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            The directory where the data is cached.
        load_type : str, default = 'tfidf'
            The format of the data to load. Options include:
            - 'text': Load raw text.
            - 'tfidf': Load TF-IDF representations.
            - 'token': Load tokenized text.
            - 'xlmr_embedding': Load XLM-R embeddings.
        max_seq_len : int, optional
            The maximum sequence length for the text data.
        xy_reversed : bool, default = False
            Whether to reverse the order of features (X) and labels (Y).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
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

        if load_type in ['text', 'token', 'xlmr_embedding']:
            # for load_type = 'embedding', the encoder needs to be loaded from the cache dir
            encoder = self.load_encoder(cache_dir=cache_dir) if load_type == 'xlmr_embedding' else None
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

    def load_glove(
        self,
        cache_dir: str = './data/',
        max_vocab_size: int = 25000,
        min_freq: int = 10,
        max_seq_len: int = 150,
        xy_reversed: bool = False,
        *args, **kwargs
    ):
        """
        Loads text data using GloVe embeddings.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            The directory where the data is cached.
        max_vocab_size : int, default = 25000
            The maximum size of the vocabulary.
        min_freq : int, default = 10
            The minimum frequency for tokens to be included in the vocabulary.
        max_seq_len : int, default = 150
            The maximum sequence length for the text data.
        xy_reversed : bool, default = False
            Whether to reverse the order of features (X) and labels (Y).
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        train_iter, test_iter = self.load_datapipe()

        # Create a tokenizer
        tokenizer = get_tokenizer('basic_english')

        # Build a vocabulary using the training data
        def yield_tokens(data_iter, tokenizer):
            if xy_reversed:
                for _, line in data_iter:
                    yield tokenizer(line)
            else:
                for line, _ in data_iter:
                    yield tokenizer(line)

        # Create the vocabulary from the training data iterator and add special tokens
        vocab = build_vocab_from_iterator(
            yield_tokens(train_iter, tokenizer),
            max_tokens=max_vocab_size,
            min_freq=min_freq,
            specials=['<unk>', '<pad>']
        )
        vocab.set_default_index(vocab['<unk>'])  # Set default index for unknown tokens

        # Load GloVe vectors and associate them with the vocabulary directly
        glove_vectors = GloVe(name='6B', dim=300, cache=cache_dir)

        # Function to efficiently assign GloVe vectors to the vocab
        def assign_glove_vectors_to_vocab(vocab, glove_vectors):
            vocab_size = len(vocab)
            embedding_dim = glove_vectors.dim
            vectors = torch.zeros(vocab_size, embedding_dim)

            glove_indices = []
            for i, word in enumerate(vocab.get_itos()):  # get_itos() provides a list of words in order of their indices
                if word in glove_vectors.stoi:
                    glove_indices.append(i)

            glove_tensor_indices = torch.tensor([glove_vectors.stoi[vocab.get_itos()[i]] for i in glove_indices])
            vectors[glove_indices] = glove_vectors.vectors.index_select(0, glove_tensor_indices)

            return vectors

        vocab.vectors = assign_glove_vectors_to_vocab(vocab, glove_vectors)
        train_iter, test_iter = self.load_datapipe()
        self.vocab = vocab

        def collate_batch(batch):
            text_pipeline = lambda x: vocab(tokenizer(x))
            idx_to_label = self.get_idx_to_label()

            text_list, label_list = [], []
            for (x, y) in batch:
                if xy_reversed:
                    text = y
                    label = x
                else:
                    text = x
                    label = y
                text_tokens = torch.tensor(text_pipeline(text), dtype=torch.int64)
                if len(text_tokens) > max_seq_len:
                    text_tokens = text_tokens[:max_seq_len]
                else:
                    text_tokens = torch.cat(
                        [text_tokens, torch.full((max_seq_len - len(text_tokens),), vocab['<pad>'], dtype=torch.int64)])

                text_vectors = vocab.vectors[text_tokens].view(-1)

                text_list.append(text_vectors)
                label_list.append(idx_to_label[label])

            # Stack all padded sequences into a single tensor
            text_tensor = torch.stack(text_list)

            return text_tensor, torch.tensor(label_list, dtype=torch.int64)

        # Create data loaders for train and test datasets with a fixed max length
        train_loader = DataLoader(list(train_iter), batch_size=self.train_batch_size, shuffle=True, collate_fn=collate_batch)
        test_loader = DataLoader(list(test_iter), batch_size=self.test_batch_size, shuffle=False, collate_fn=collate_batch)
        return {'train_loader': train_loader, 'test_loader': test_loader}


class imdb(text_dataloader):
    """
    A dataloader class for the IMDB dataset.

    This class provides methods to load and preprocess the IMDB dataset, which contains movie reviews labeled as positive or negative.

    Attributes
    ----------
    name : str, default = 'imdb'
        The name of the dataset.
    train_batch_size : int, default = 64
        The batch size for training data.
    test_batch_size : int, default = 64
        The batch size for testing data.
    max_seq_len : int, default = 512
        The maximum sequence length for text data.

    Methods
    ----------
    __init__
        Initializes the IMDB dataset dataloader.
    load
        Loads the IMDB dataset with reversed (label, text) ordering.
    load_datapipe
        Loads training and testing pipelines for the IMDB dataset.
    get_class_number
        Returns the number of classes in the IMDB dataset (2).
    get_train_number
        Returns the number of training examples (25,000).
    get_test_number
        Returns the number of testing examples (25,000).
    get_idx_to_label
        Returns the mapping from indices to labels.
    """
    def __init__(self, name='imdb', train_batch_size=64, test_batch_size=64, max_seq_len: int = 512):
        """
        Initializes the IMDB dataset dataloader.

        Parameters
        ----------
        name : str, default = 'imdb'
            The name of the dataset.
        train_batch_size : int, default = 64
            The batch size for training data.
        test_batch_size : int, default = 64
            The batch size for testing data.
        max_seq_len : int, default = 512
            The maximum sequence length for text data.
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size, max_seq_len=max_seq_len)

    def load(self, *args, **kwargs):
        """
        Loads the IMDB dataset with reversed (label, text) ordering.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        kwargs['xy_reversed'] = True
        return super().load(*args, **kwargs)

    @staticmethod
    def load_datapipe(cache_dir='./data/', *args, **kwargs):
        """
        Loads training and testing pipelines for the IMDB dataset.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            Directory to store cached data.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing training and testing data pipelines.
        """
        train_datapipe = IMDB(root=cache_dir, split="train")
        test_datapipe = IMDB(root=cache_dir, split="test")
        return train_datapipe, test_datapipe

    @staticmethod
    def get_class_number(*args, **kwargs):
        """
        Returns the number of classes in the IMDB dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of classes (2).
        """
        return 2

    @staticmethod
    def get_train_number(*args, **kwargs):
        """
        Returns the number of training examples in the IMDB dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of training examples (25,000).
        """
        return 25000

    @staticmethod
    def get_test_number(*args, **kwargs):
        """
        Returns the number of testing examples in the IMDB dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of testing examples (25,000).
        """
        return 25000

    @staticmethod
    def get_idx_to_label(*args, **kwargs):
        """
        Returns the mapping from indices to labels for the IMDB dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary mapping indices to labels.
        """
        return {
            1: 0,
            2: 1,
        }


class sst2(text_dataloader):
    """
    A dataloader class for the SST-2 dataset.

    This class provides methods to load and preprocess the SST-2 dataset, which contains sentiment classification labels for sentences.

    Attributes
    ----------
    name : str, default = 'sst2'
        The name of the dataset.
    train_batch_size : int, default = 64
        The batch size for training data.
    test_batch_size : int, default = 64
        The batch size for testing data.
    max_seq_len : int, default = 32
        The maximum sequence length for text data.

    Methods
    ----------
    __init__
        Initializes the SST-2 dataset dataloader.
    load_datapipe
        Loads training and testing pipelines for the SST-2 dataset.
    get_class_number
        Returns the number of classes in the SST-2 dataset (2).
    get_train_number
        Returns the number of training examples (67,349).
    get_test_number
        Returns the number of testing examples (872).
    get_idx_to_label
        Returns the mapping from indices to labels.
    """
    def __init__(self, name='sst2', train_batch_size=64, test_batch_size=64, max_seq_len: int = 32):
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size, max_seq_len=max_seq_len)

    @staticmethod
    def load_datapipe(cache_dir='./data/', *args, **kwargs):
        """
        Loads training and testing pipelines for the SST-2 dataset.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            Directory to store cached data.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing training and testing data pipelines.
        """
        train_datapipe = SST2(root=cache_dir, split="train")
        test_datapipe = SST2(root=cache_dir, split="dev")
        return train_datapipe, test_datapipe

    @staticmethod
    def get_class_number(*args, **kwargs):
        """
        Returns the number of classes in the SST-2 dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of classes (2).
        """
        return 2

    @staticmethod
    def get_train_number(*args, **kwargs):
        """
        Returns the number of training examples in the SST-2 dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of training examples (67,349).
        """
        return 67349

    @staticmethod
    def get_test_number(*args, **kwargs):
        """
        Returns the number of testing examples in the SST-2 dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of testing examples (872).
        """
        return 872

    @staticmethod
    def get_idx_to_label(*args, **kwargs):
        """
        Returns the mapping from indices to labels for the SST-2 dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary mapping indices to labels.
        """
        return {
            0: 0,
            1: 1,
        }


class agnews(text_dataloader):
    """
    A dataloader class for the AG News dataset.

    This class provides methods to load and preprocess the AG News dataset, which contains news articles classified into four categories.

    Attributes
    ----------
    name : str, default = 'ag_news'
        The name of the dataset.
    train_batch_size : int, default = 64
        The batch size for training data.
    test_batch_size : int, default = 64
        The batch size for testing data.
    max_seq_len : int, default = 64
        The maximum sequence length for text data.

    Methods
    ----------
    __init__
        Initializes the AG News dataset dataloader.
    load
        Loads the AG News dataset with reversed (label, text) ordering.
    load_datapipe
        Loads training and testing pipelines for the AG News dataset.
    get_class_number
        Returns the number of classes in the AG News dataset (4).
    get_train_number
        Returns the number of training examples (120,000).
    get_test_number
        Returns the number of testing examples (7,600).
    get_idx_to_label
        Returns the mapping from indices to labels.
    """
    def __init__(self, name='ag_news', train_batch_size=64, test_batch_size=64, max_seq_len: int = 64):
        """
        Initializes the AG News dataset dataloader.

        Parameters
        ----------
        name : str, default = 'ag_news'
            The name of the dataset.
        train_batch_size : int, default = 64
            The batch size for training data.
        test_batch_size : int, default = 64
            The batch size for testing data.
        max_seq_len : int, default = 64
            The maximum sequence length for text data.
        """
        super().__init__(name=name, train_batch_size=train_batch_size, test_batch_size=test_batch_size, max_seq_len=max_seq_len)

    def load(self, *args, **kwargs):
        """
        Loads the AG News dataset with reversed (label, text) ordering.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary containing training and testing DataLoaders.
        """
        kwargs['xy_reversed'] = True
        return super().load(*args, **kwargs)

    @staticmethod
    def load_datapipe(cache_dir='./data/', *args, **kwargs):
        """
        Loads training and testing pipelines for the AG News dataset.

        Parameters
        ----------
        cache_dir : str, default = './data/'
            Directory to store cached data.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        tuple
            A tuple containing training and testing data pipelines.
        """
        train_datapipe = AG_NEWS(root=cache_dir, split="train")
        test_datapipe = AG_NEWS(root=cache_dir, split="test")
        return train_datapipe, test_datapipe

    @staticmethod
    def get_class_number(*args, **kwargs):
        """
        Returns the number of classes in the AG News dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of classes (4).
        """
        return 4

    @staticmethod
    def get_train_number(*args, **kwargs):
        """
        Returns the number of training examples in the AG News dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of training examples (120,000).
        """
        return 120000

    @staticmethod
    def get_test_number(*args, **kwargs):
        """
        Returns the number of testing examples in the AG News dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        int
            The number of testing examples (7,600).
        """
        return 7600

    @staticmethod
    def get_idx_to_label(*args, **kwargs):
        """
        Returns the mapping from indices to labels for the AG News dataset.

        Parameters
        ----------
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        dict
            A dictionary mapping indices to labels.
        """
        return {
            1: 0,
            2: 1,
            3: 2,
            4: 3,
        }



