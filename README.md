# Transformer Architecture

**Extract Sequence Features::**  WordSequences -> TokenizedSequences -> WordEmbedding + Position Embedding -> MultiHeadAttention(ScaledDotProduct, QKV) -> Normalization -> Dense (Feed Forward) -> Normalization -> Dense ...

Dataset :: CIFAR32(Images, Labels)

Tools :: Pytorch, Pytorch Lightings, GPU \
Architecture :: Transformer

Tokenization: We have to filter our sentences -> Vocabulary Creation -> convert it to tokens(sequence of word index).

Initial Embedding::
Word Embedding: We have to convert our word to a embedding(vector of n(512) dimension) to understand there features and meaning in computer.

Position Embedding: For finding word position in sequences we will get a positional embedding which will be same for all examples.

- **Encoder::**
  - MultiHeadAttention:: To extract word relational embedding according to context.
    - Word Embedding(For every word) -> Query(Linear Projection of Word Embedding), Key(Linear Projection of Word Embedding), Value(Original Embedding).
    - Query @ Key.T: For getting relation between diffrent positions word
    - (Query @ Key.T) + Value: For getting word embeddings according to context.
    - This will be done with n projections.(MultiHead)
    - We will concat all the heads or word embeddings.
    - We will get contextulized word embeddings.

  - Normalization: For error reduction(Gradient boosting, Gradient Decay)

  - Feed Forward: We will Dense Layers to WRT to word embeddings -> Normalization.

  - Add: MultiHeadAttention.Output + FeedForward.Output(Residual layer)

- **Decoder**
  - Intially we will predict our first sequence token to "SOS" Token only other are padding.

  - MultiHeadAttention:: We will use for decoder output and encoder output.

  - Normalization: For error reduction(Gradient boosting, Gradient Decay)

  - Feed Forward: We will Dense Layers to WRT to word embeddings -> Normalization.

  - Add: MultiHeadAttention.Output + FeedForward.Output(Residual layer)

  - Prediction: We will get a next word token now.

    We will repeat our decoder until we get "EOS" token.


Transformer Architecture:

<p align="center">
  <img src="images/transformer.png?raw=true" width="100%" title="Transformer">
</p>

Self Attention:

<p align="center">
  <img src="images/QKV_projection.png?raw=true" height="640px" title="Self Attention">
</p>

Multi Head Attention:

<p align="center">
  <img src="images/MultiHeadAttention.png?raw=true" height="720px" title="MultiHeadAttention">
</p>


Bert Architecture:


<p align="center">
  <img src="images/BertArchitecture.png?raw=true" height="1080px" title="BertArchitecture">
</p>


- Credits:
  - [Self Attention Step by Step](https://peltarion.com/blog/data-science/self-attention-video "Self Attention")
  - [How Transformers work](https://theaisummer.com/transformer/ "Transformer")
  - [Transformers and Multi-Head Attention](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html "Transformers and Multi-Head Attention")
