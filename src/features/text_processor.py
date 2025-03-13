import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TextProcessor:
    """
    A class for processing text data for recommendation systems.
    
    This class implements various text processing techniques discussed in Lesson 2:
    - Text preprocessing (lowercase, punctuation removal, stopword removal, stemming, lemmatization)
    - Text representation methods (Bag of Words, TF-IDF)
    - Document similarity calculation
    - Keyword extraction
    
    The TextProcessor can be used in Jupyter notebooks to:
    1. Preprocess text data
    2. Convert text into numerical representations (vectors)
    3. Calculate similarity between documents
    4. Extract important keywords from documents
    
    Example usage:
    ```python
    # Create a TextProcessor instance
    processor = TextProcessor(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        stemming=False,
        lemmatization=True
    )
    
    # Preprocess documents
    preprocessed_docs = processor.preprocess_documents(documents)
    
    # Create TF-IDF vectors
    tfidf_matrix, feature_names = processor.create_tfidf_vectors(documents)
    
    # Calculate similarity between two documents
    similarity = processor.compute_document_similarity(doc1, doc2, method='tfidf')
    
    # Extract keywords from a document
    keywords = processor.extract_keywords(document, top_n=3)
    ```
    """
    
    def __init__(self, lowercase=True, remove_punctuation=True, remove_stopwords=True, 
                 stemming=False, lemmatization=False):
        """
        Initialize the TextProcessor with preprocessing options.
        
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
        lemmatization : bool, default=False
            Whether to apply lemmatization
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        
        # Initialize tools
        if self.remove_stopwords:
            # Download NLTK resources if needed
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            self.stop_words = set(stopwords.words('english'))
            
        if self.stemming:
            self.stemmer = PorterStemmer()
            
        if self.lemmatization:
            # Download NLTK resources if needed
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
            
        # Initialize vectorizers
        self.bow_vectorizer = None
        self.tfidf_vectorizer = None
    
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
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        tokens = nltk.word_tokenize(text)
        
        # Remove stopwords
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization
        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents):
        """
        Preprocess a collection of documents.
        
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
    
    def create_bow_vectors(self, documents, min_df=1, max_df=1.0):
        """
        Create Bag of Words vectors for a collection of documents.
        
        Parameters:
        -----------
        documents : list of str
            The documents to vectorize
        min_df : int or float, default=1
            Minimum document frequency for a term to be included
        max_df : float or int, default=1.0
            Maximum document frequency for a term to be included
            
        Returns:
        --------
        scipy.sparse.csr_matrix, list
            The BoW matrix and the list of feature names
        """
        # Preprocess documents
        preprocessed_docs = self.preprocess_documents(documents)
        
        # Create and fit the vectorizer
        self.bow_vectorizer = CountVectorizer(min_df=min_df, max_df=max_df)
        bow_matrix = self.bow_vectorizer.fit_transform(preprocessed_docs)
        
        return bow_matrix, self.bow_vectorizer.get_feature_names_out()
    
    def create_tfidf_vectors(self, documents, min_df=1, max_df=1.0):
        """
        Create TF-IDF vectors for a collection of documents.
        
        Parameters:
        -----------
        documents : list of str
            The documents to vectorize
        min_df : int or float, default=1
            Minimum document frequency for a term to be included
        max_df : float or int, default=1.0
            Maximum document frequency for a term to be included
            
        Returns:
        --------
        scipy.sparse.csr_matrix, list
            The TF-IDF matrix and the list of feature names
        """
        # Preprocess documents
        preprocessed_docs = self.preprocess_documents(documents)
        
        # Create and fit the vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(min_df=min_df, max_df=max_df)
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(preprocessed_docs)
        
        return tfidf_matrix, self.tfidf_vectorizer.get_feature_names_out()
    
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
            Method to use ('tfidf' or 'bow')
            
        Returns:
        --------
        float
            Similarity score between 0 and 1
        """
        # Preprocess documents
        preprocessed_doc1 = self.preprocess_text(doc1)
        preprocessed_doc2 = self.preprocess_text(doc2)
        
        if method == 'tfidf':
            # Create TF-IDF vectors if not already created
            if self.tfidf_vectorizer is None:
                self.tfidf_vectorizer = TfidfVectorizer()
                self.tfidf_vectorizer.fit([preprocessed_doc1, preprocessed_doc2])
            
            # Transform documents to TF-IDF vectors
            vec1 = self.tfidf_vectorizer.transform([preprocessed_doc1])
            vec2 = self.tfidf_vectorizer.transform([preprocessed_doc2])
        else:  # method == 'bow'
            # Create BoW vectors if not already created
            if self.bow_vectorizer is None:
                self.bow_vectorizer = CountVectorizer()
                self.bow_vectorizer.fit([preprocessed_doc1, preprocessed_doc2])
            
            # Transform documents to BoW vectors
            vec1 = self.bow_vectorizer.transform([preprocessed_doc1])
            vec2 = self.bow_vectorizer.transform([preprocessed_doc2])
        
        # Compute cosine similarity
        return cosine_similarity(vec1, vec2)[0, 0]
    
    def extract_keywords(self, document, top_n=5):
        """
        Extract the most important keywords from a document.
        
        Parameters:
        -----------
        document : str
            The document to extract keywords from
        top_n : int, default=5
            Number of keywords to extract
            
        Returns:
        --------
        list of tuple
            List of (keyword, score) tuples
        """
        # Preprocess document
        preprocessed_doc = self.preprocess_text(document)
        
        # Create TF-IDF vector
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_doc])
        
        # Get feature names and scores
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        
        # Create a list of (keyword, score) tuples
        keyword_scores = [(feature_names[i], scores[i]) for i in range(len(feature_names))]
        
        # Sort by score (descending) and return top_n
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        
        return keyword_scores[:top_n]