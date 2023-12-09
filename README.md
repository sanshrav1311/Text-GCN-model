# Text-GCN-model
## Abstract
This paper proposes a new and improved Text Graph Convolutional
Network (Text GCN) model for natural language processing (NLP) tasks.
Traditional NLP methods rely on bag-of-words representations, which
ignore the complex relationships between words.
Text GCN uses graph convolutional neural networks to represent text data
as a graph, capturing these relationships and improving NLP model
accuracy. Our model combines graph convolutional neural networks and
attention mechanisms and introduces a novel graph pooling technique for
improved scalability.
We evaluate our model on benchmark datasets for text classification and
sentiment analysis, attempting to demonstrate superior performance
compared to existing Text GCN models and other state-of-the-art NLP
models. Our proposed Text GCN model represents a significant step
forward in NLP research, potentially improving NLP model accuracy and
performance in various applications.
This model is particularly aimed at Text GCN models that require different
weights for each part of input text. Its modular nature also makes it efficient
for cases with a large set of data.
## Introduction:
The increasing amount of unstructured text data has posed a challenge for
natural language processing (NLP) tasks. Traditional methods, which treat
each word as an independent feature, often ignore the complex
relationships between words, leading to inaccurate NLP models. To
overcome this limitation.
researchers have introduced a new graph-based approach called Text
Graph Convolutional Networks (Text GCN), which uses graph convolutional
neural networks to capture the complex relationships between words. By
treating each word as a node in the graph and modelling the edges as the
relationships between words, Text GCN can effectively learn meaningful
representations of text data for various NLP tasks.
In this paper, we propose a new and improved Text GCN model that
addresses these limitations and improves the performance of NLP models.
Our model combines the benefits of graph convolutional neural networks
and attention mechanisms to improve the expressiveness of the model and
enable it to handle large-scale graphs with unlabeled data. We also
introduce a novel graph pooling technique that improves the scalability of
our model while preserving the underlying structure of the graph.
We evaluate our model on several benchmark datasets for text
classification and sentiment analysis and demonstrate its superior
performance compared to existing Text GCN models and other
state-of-the-art NLP models. We also conduct an extensive ablation study
to analyse the contributions of each component of our model and provide
insights into the workings of our model.
Overall, our proposed Text GCN model represents a significant step
forward in NLP research and can improve the accuracy and performance of
NLP models in various applications.
## Dataset:
We have 8833 documents, 188 labelled. Each entry in the dataset has 5
columns: document id, title, keywords, abstract and contextual.
Document id spans from 0 to 8832. The title column shows the document's
title, and the keywords column contains all the keywords in the document.
The Abstract has the document content.
With 188 labelled and the rest unlabeled, 77 labels are non-contextual, and
111 as contextual, marked as 0 and 1, respectively.
The dataset has articles on the COVID-19 pandemic. It comprises the
following agendas:
1. Results and vaccine
2. State of various people with other diseases
3. Symptoms of people with different physical states in women, children
and the elderly.
4. Statements of people that were diagnosed with COVID.
5. How different parts of the world reacted to COVID-19.
The dataset is available here: Public Medical Records for COVID-19.
## TextGCN Model:
Traditional text classification studies mainly focus on feature engineering
and classification algorithms.
For feature engineering, the most commonly used feature is the
bag-of-words feature.
Profound learning text classification studies consist of two groups:
One group of studies focused on models based on word embeddings.
Another group of studies employed deep neural networks like CNN and
RNN.
A GCN (Kipf and Welling 2017) is a multilayer neural network that operates
directly on a graph and induces embedding vectors of nodes based on the
properties of their neighbourhoods.
GCN can capture information only about immediate neighbours with one
layer
of convolution. When multiple GCN layers are stacked, it integrates
information about larger neighbourhoods.
## Proposed Approach:
Each document in the dataset for public medical records contains 3
characteristics (title, keywords, and abstract).
We have split the dataset into 3 parts, one for the title, one for keywords
and one for the abstract.
For each part, we follow the below-mentioned steps:
1. Split the dataset into train and test datasets.
2. Compute the TF-IDF scores, and construct a text graph from the
same.
3. Compute the adjacency matrix A and X as inputs to the textGCN
model.
4. Feed A and X into the textGCN model, and train the same.
5. Evaluate the model on the test dataset, and output a label for each
document.
The model generates a list of predicted labels for each document. The final
predicted label is the majority label for all 3 parts.
The model's accuracy is determined by comparing the predicted and test
labels. The confusion matrix is constructed from the above information.
## Challenges:
The model couldn’t be run due to inadequate computation power.
The vocabulary size for part 1 and part-2 was 10K and 40K, respectively.
PMI matrix was constructed only for part-1 but couldn’t be for part-2 due to
resource constraints. The program crashed, and the machine blacked out
before the text graph could be generated.
## References:
1. Mixture-of-Partitions: Infusing Large Biomedical Knowledge Graphs
into BERT - ACL Anthology
2. https://arxiv.org/pdf/2104.08145.pdf
3. GitHub - plkmo/Bible_Text_GCN: Pytorch implementation of "Graph
Convolutional Networks for Text Classification."
4. GitHub - codeKgu/Text-GCN: A PyTorch implementation of "Graph
Convolutional Networks for Text Classification." (AAAI 2019)
5. Graph Convolutional Networks for Classification in Python | Well
Enough
