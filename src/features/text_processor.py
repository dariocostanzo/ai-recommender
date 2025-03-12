"""
Text processing utilities for the recommendation system.
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextProcessor:
    """
    Class for processing text data for the recommendation system.
    """
    
    def __init__(self, lowercase=True, remove_punctuation=True, 
                 remove_stopwords=True, stemming=False, lemmatization=True):
        """
        Initialize the text processor.
        
        Parameters:
        -----------
        lowercase : bool, default=True
            Whether to convert text to lowercase
        remove_punctuation : bool, default=True
            Whether to remove punctuation
        remove_stopwords : bool, default=True
            Whether to remove stopwords
        stemming : bool, default=False
            Whether to apply stemming
        lemmatization : bool, default=True
            Whether to apply lemmatization
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text):
        """
        Preprocess a single text document.
        
        Parameters:
        -----------
        text : str
            The text to preprocess
            
        Returns:
        --------
        str
            The preprocessed text
        """
        if self.lowercase:
            text = text.lower()
        
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents):
        """
        Preprocess a list of text documents.
        
        Parameters:
        -----------
        documents : list of str
            The documents to preprocess
            
        Returns:
        --------
        list of str
            The preprocessed documents
        """
        return [self.preprocess_text(doc) for doc in documents]
    
    def create_tfidf_vectors(self, documents, max_features=None):
        """
        Create TF-IDF vectors from documents.
        
        Parameters:
        -----------
        documents : list of str
            The documents to preprocess and vectorize
        max_features : int, default=None
            Maximum number of features (words) to extract
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            TF-IDF vectors for the documents
        list of str
            Feature names (words)
        """
        # Preprocess documents
        preprocessed_docs = self.preprocess_documents(documents)
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=max_features)
        
        # Fit and transform documents
        tfidf_matrix = vectorizer.fit_transform(preprocessed_docs)
        
        return tfidf_matrix, vectorizer.get_feature_names_out()
    
    def create_bow_vectors(self, documents, max_features=None):
        """
        Create Bag-of-Words vectors from documents.
        
        Parameters:
        -----------
        documents : list of str
            The documents to preprocess and vectorize
        max_features : int, default=None
            Maximum number of features (words) to extract
            
        Returns:
        --------
        scipy.sparse.csr_matrix
            BoW vectors for the documents
        list of str
            Feature names (words)
        """
        # Preprocess documents
        preprocessed_docs = self.preprocess_documents(documents)
        
        # Create Count vectorizer
        vectorizer = CountVectorizer(max_features=max_features)
        
        # Fit and transform documents
        bow_matrix = vectorizer.fit_transform(preprocessed_docs)
        
        return bow_matrix, vectorizer.get_feature_names_out()
    
    def extract_keywords(self, document, top_n=5):
        """
        Extract the most important keywords from a document using TF-IDF.
        
        Parameters:
        -----------
        document : str
            The document to extract keywords from
        top_n : int, default=5
            Number of top keywords to extract
            
        Returns:
        --------
        list of tuple
            List of (keyword, score) tuples
        """
        # Create a single-document corpus
        corpus = [document]
        
        # Create TF-IDF vectors
        tfidf_matrix, feature_names = self.create_tfidf_vectors(corpus)
        
        # Get scores for the first (and only) document
        scores = zip(feature_names, tfidf_matrix.toarray()[0])
        
        # Sort by score and take top_n
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        return sorted_scores[:top_n]
    
    def compute_document_similarity(self, doc1, doc2, method='tfidf'):
        """
        Compute similarity between two documents.
        
        Parameters:
        -----------
        doc1 : str
            First document
        doc2 : str
            Second document
        method : str, default='tfidf'
            Method to use for vectorization ('tfidf' or 'bow')
            
        Returns:
        --------
        float
            Cosine similarity between the documents (0-1)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Create a corpus with both documents
        corpus = [doc1, doc2]
        
        # Vectorize based on method
        if method.lower() == 'tfidf':
            matrix, _ = self.create_tfidf_vectors(corpus)
        elif method.lower() == 'bow':
            matrix, _ = self.create_bow_vectors(corpus)
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'tfidf' or 'bow'.")
        
        # Compute cosine similarity
        similarity = cosine_similarity(matrix[0:1], matrix[1:2])[0][0]
        
        return similarity