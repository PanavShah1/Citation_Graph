# Task 2: Machine Learning

## Problem Statement 

In this part, you have to train a model to look at a new unseen paper and return a ranked list of papers (from our dataset) which this unseen paper may have cited. This is essentially a link prediction task. Moreover, the evaluation of your predictions will be based on recall@K metric (Value of K will not be released). Specifically,

1. Update the evaluation.py file that takes as argument    `paper_folder_path`.
2. You are supposed to fill this file with your code and print your top-K predicted papers to the console in the specific format. `evaluation.py` will be called by a caller file `run_evaluation.py`. This file will feed papers to `evaluation.py` and get top-K predictions. If one of the top-K papers is indeed in the citations list, then you get a score.

## Approach

On a high level, the approach follows the following steps:
1. Citation Graph Construction (Step 1)
2. Text Embeddings, using LongFormer
3. Passing text embeddings to GraphSAGE to get a graph embedding
4. Using a projection MLP to get a lower-dimensional projection.
5. Using a Link Prediction MLP to score query-candidate pairs.


Each paper's `title` and `abstract` are concatenated and passed through a LongFormer model, and the final output is meanpooled to create a $768$-dimensional embedding for each paper.
These embeddings are saved in the `paper_embeddings.pkl` file for caching and reuse. Longformers were used because abstracts and titles are long, and LongFormers excel at long input sequences. 

A two-layer GraphSAGE model (often used for link prediction tasks) is used, with training happening using **contrastive loss** - the idea is to maximize similarity between edges that are connected by nodes (real citation pairs), and minimize similarity between random pairs. A GraphSAGE model learns node embeddings by message-passing.

Then we train a regressor model (projection MLP) to learn a mapping between the 768-dimensional LongFormer embeddings and the $128$-dimensional GraphSAGE embeddings. This is so that we can transform the test paper into the same space as the training dataset. Since there are no available citations (edges) for the test paper, we cannot expect to get a meaningful embedding by passing it through GraphSAGE, and so use this to learn a reasonable mapping that we may use for inference.

Then we use a Link Predictor, which is just an MLP - it takes two graph embeddings and predicts the probability of a citation link between them. The main idea here is also contrastive learning - learning to differentiate betwene positive (linked) pairs and negative (un-linked) pairs.

At inference time, the test paper and its abstract is concatenated, and then is embedded with a LongFormer, which is pretrained. It is then passed through the Projection MLP to get a reasonable embedding vector. The projection is scored against each candidate paper's GraphSAGE embedding using the Linked Predictor (we can also mix in cosine similarity with a weight alpha).
The papers are then ranked by decreasing score and printed.


## Possible Directions for future work

1. Replacing LongFormer with domain-specific models might help improve semantic extraction from papers.
2. Training the entire model pipeline end-to-end instead of stepwise.
3. Filter candidate set by year of publication, and use simple heurestics such as keywords to filter out extremely dissimilar papers (for example a paper on PyTorch implementation details would likely not cite a technical paper on game theory).




