# Semantic-Textual-Similarity
Semantic Textual Similarity is the task of determining how similar two texts are. In our case, a relatively large-scale Persian dataset, called Farstail has been used. The train, validation, and test portions include 7,266, 1,537, and 1,564 instances, respectively.

First of all, we used tokenization and lemmatization. Then we removed stopwords and Punctuations on our data. These are our pre-processing step before we turn the data into our models. We follow deep learning approach. So, we finetuned Bert model according to our task. Then we get accuracy and F1-score as our evaluation metrics.
#### Results:
             
|                |   accuracy  |  f1score   |
| -------------- | ----------- | ---------- | 
| Validation Set |    0.647    |   0.642    |
| Test Set       |    0.664    |   0.661    |

