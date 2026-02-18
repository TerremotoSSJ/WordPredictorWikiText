import pandas as pd
from vocabulary import Vocabulary
import pyarrow.parquet as pq


def build_articles_chunk(chunk,chunksize,text_column='text'):
    """
    Docstring for build_articles_dataframe

    :param chunk: A pandas DataFrame containing a chunk of the parquet file, which is processed to build the articles and get the last article, which may be incomplete and will be used as a title for the next chunk to ensure that articles are not split across chunks and maintain the integrity of the data.
    :param chunksize: Number of rows to read at a time from the parquet file
    :param text_column: Name of the column containing the text data
    :return: A pandas DataFrame containing the articles and the last article in the chunk, which may be incomplete and will be used as a title for the next chunk to ensure that articles are not split across chunks.
    If "" is returned as the second element, it indicates that the last article in the chunk is complete and does not need to be carried over to the next chunk.
    """

    #Vectorized text cleaning using pandas string methods for efficiency
    texts=chunk["text"].astype(str).str.strip()
    
    #Identify titles based on the pattern of starting and ending with "=" return a boolean Series where True indicates a title and False indicates regular text
    is_title = texts.str.startswith("=") & texts.str.endswith("=") & (texts.str.count("=") == 2)

    #Use the cumulative sum of the boolean Series to assign a unique article ID to each group of text, effectively grouping the text into articles based on the identified titles
    chunk=chunk.copy()  # Create a copy of the chunk to avoid modifying the original DataFrame
    chunk["id_article"] = is_title.cumsum()

    #Group the text by the assigned article IDs and concatenate the text within each group into a single string, resulting in a Series where the index is the article ID and the value is the concatenated text of that article
    articles=chunk.groupby("id_article")[text_column].agg("\n".join)

    #If the chunk is full the last article may be incomplete, so we extract the first line of the last article to use as a title for the next chunk, ensuring that articles are not split across chunks and maintaining the integrity of the data
    if(len(chunk)>=chunksize):

        #We obtain the last article by splitting the text of the last article in the chunk by newline characters and taking the first line, which is assumed to be the title of the article. This title will be used as a reference for the next chunk to ensure that articles are not split across chunks.
        last_article_incomplete=articles.iloc[-1]
    
        #We remove the last article from the current chunk to prevent it from being split across chunks, ensuring that the integrity of the articles is maintained. This is done by dropping the last article from the articles Series, which contains all the articles in the current chunk.
        articles=articles.drop(articles.index[-1])

    else:
        #If the chunk is not full, we can safely assume that the last article is complete and does not need to be carried over to the next chunk, so we set the last_article_first_line to an empty string
        last_article_incomplete = ""

    return articles,last_article_incomplete

def build_articles_dataframe(route,chunksize=10000,text_column='text'):
    """
    Docstring for build_articles_dataframe
    
    :param route: Path to the parquet file containing the articles, which is read in chunks to efficiently process large datasets without loading the entire file into memory at once. The function uses a generator to yield articles one by one, allowing for memory-efficient processing of the articles while maintaining the integrity of the data by ensuring that articles are not split across chunks.
    :param chunksize: Number of rows to read at a time from the parquet file
    :param text_column: Name of the column containing the text data
    :yield: A generator that yields articles one by one from the parquet file, allowing for memory-efficient processing of the articles without 
    loading the entire dataset into memory at once. Each article is returned as a string, and the generator handles the reading and processing 
    of the parquet file in chunks to ensure that articles are not split across chunks and maintain the integrity of the data.

    """

    # Buffer to hold the incomplete last article from the previous chunk
    incomplete_buffer = ""  
    parquet_file = pq.ParquetFile(route)

    for chunk in parquet_file.iter_batches(batch_size=chunksize):
        chunk = chunk.to_pandas()  # Convert the chunk to a pandas DataFrame for processing

        #Preprocess the chunk to build the articles and get the last article, which may be incomplete and will be used as a title for the next 
        # chunk to ensure that articles are not split across chunks and maintain the integrity of the data
        if incomplete_buffer:

            # We divide it in lines
            text_incomplete=incomplete_buffer.split("\n")
            incomplete_df=pd.DataFrame({text_column:text_incomplete})

            # We concatenate the incomplete article with the current chunk to ensure that the articles are not split across chunks and maintain 
            # the integrity of the data
            chunk=pd.concat([incomplete_df,chunk],ignore_index=True)
            incomplete_buffer="" 
        
        articles,incomplete_buffer=build_articles_chunk(chunk,chunksize,text_column)

        # Yield each article in the current chunk one by one, allowing for memory-efficient processing of the articles without loading the entire 
        # dataset into memory at once. This is done using a generator, which allows for iterating over the articles in a lazy manner, yielding 
        # one article at a time as needed.
        for article in articles:
            yield article

    if incomplete_buffer:
        # If there is an incomplete article left in the buffer after processing all chunks, we yield it as well to ensure that no data is lost 
        # and all articles are processed.
        yield incomplete_buffer

def preprocess_article(article,vocabulary: Vocabulary)->list[int]:
    """
    Docstring for preprocess_article

    :param article: Input article to be preprocessed
    :return: Preprocessed article as a string, where the text is converted to lowercase, punctuation and special characters are removed, 
    leading and trailing whitespace is stripped, and empty strings are replaced with a special token "<UNK>" to indicate unknown or missing values.
    """

    article=vocabulary.text_to_sequence(article)

    # Replace empty strings with a special token
    if not article:
        article = [vocabulary.unk_index]

    return article

def preprocess_articles(articles,vocabulary: Vocabulary)->list[list[int]]:
    """
    Docstring for preprocess_articles

    :param articles: Input list of articles to be preprocessed
    :return: List of preprocessed articles, where each article is represented as a list of integers corresponding to the indices of the words in the vocabulary. The text is converted to lowercase, punctuation and special characters are removed, leading and trailing whitespace is stripped, and empty strings are replaced with a special token "<UNK>" to indicate unknown or missing values.
    """

    return [preprocess_article(article,vocabulary) for article in articles]

