import advertools as adv
from parsivar import Normalizer, Tokenizer

class Preprocessing():

  def __init__(self, normalizer, tokenizer, stopwords:list, punc:list):
    """
        Class initializer
    Args:
        stopwords(list): list of stop words
        punc(list): list of punctuations
    Returns:
    """
    self.normalizer = normalizer
    self.tokenizer = tokenizer
    self.stopwords = stopwords
    self.punc = punc

  def normalize(self,X:str) -> str:
    """
        Normalizing
    Args:
        X(str): input string
    Return:
        X(str): string normalized
    """
    return self.normalizer.normalize(X)

  def tokenize(self,X:str) -> list:
    """
        Tokenization
    Args:
        X(str): input string
    Returns:
        tokens(list): list of words
    """
    return self.tokenizer.tokenize_words(X)

  def stopword_removal(self,X:list) -> list:
    """
        Stopword removal
    Args:
        X(list): list of words
    Returns:
        X(list): list of words that stopwords removed
    """
    return [token for token in X if token[0] not in self.stopwords]
  
  def join_sents(self,X:list) -> str:
    """
        Sentences creation
    Args:
        X(list): list of words
    Returns:
        X(str): sentence created
    """
    sents = ' '.join([token for token in X])
    return sents.strip()

  def punc_removal(self,X:str) -> str:
    """
        Punctuation removal
    Args:
        X(str): input string
    Returns:
        X(str): string that punctuations removed
    """
    for ele in X:
        if ele in self.punc:
            X = X.replace(ele, "")
    return X 
  
  def preprocessor(self,X:str) -> str:
    """
         Data Preprocessor
    Args:
        X(str): input string
    Return:
        X(str): preprocessed string
    """
    #Normalizing
    X = self.normalize(X)
    #Tokenization
    X = self.tokenize(X)
    #Stopword_removal
    X = self.stopword_removal(X)
    #Sentences creation
    X = self.join_sents(X)
    #Punchuation removal
    X = self.punc_removal(X)
    return X