import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from fuzzywuzzy import process 


def getdata():
    books = pd.read_csv('books.csv', encoding = "ISO-8859-1")
    book_data=books.drop(['book_id', 'isbn','isbn13','original_publication_year','title','language_code','small_image_url'], axis=1)
    book_data.fillna(' ', inplace=True)
    book_data['original_title']=book_data['original_title'].str.lower()
    return book_data

# for similar authors
def transformation(book_data):
    tf = TfidfVectorizer(lowercase=True, analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    tfidf_matrix = tf.fit_transform(book_data['authors'])
#pickle.dump(tfidf_matrix, open('tfidf_vectorizer.pickle', 'wb'))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

def transformation2(book_data):
    tf = TfidfVectorizer(lowercase=True, analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
    # for similar book name
    tfidf_matrix=tf.fit_transform(book_data['original_title'])
#pickle.dump(tfidf_matrix, open('tfidf_vectorizer.pickle', 'wb'))
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    return cosine_sim

 # for similar authors
def recommendation(title,data,transform):
# Build a 1-dimensional array with book titles
    indices = pd.Series(data.index, index=data['original_title'])
# Function that get book recommendations based on the cosine similarity score of book authors
    idx = indices[title]
    sim_scores = list(enumerate(transform[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    book_indices = [i[0] for i in sim_scores]
    book_title= data['original_title'].iloc[book_indices]
    book_url = data['image_url'].iloc[book_indices]
    book_rating= data['average_rating'].iloc[book_indices]
    book_author = data['authors'].iloc[book_indices]
    recommendation_data = pd.DataFrame(columns=['original_title','image_url', 'average_rating','authors'])
    recommendation_data['original_title'] = book_title.head(5)
    recommendation_data['image_url'] = book_url.head(5)
    recommendation_data['average_rating'] = book_rating.head(5)
    recommendation_data['authors'] = book_author.head(5)
    return recommendation_data
# for similar books
def recommendation2(title,data,transform):
# Build a 1-dimensional array with book titles
    indices = pd.Series(data.index, index=data['original_title'])
# Function that get book recommendations based on the cosine similarity score of book authors
    idx = indices[title]
    sim_scores = list(enumerate(transform[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    book_indices = [i[0] for i in sim_scores]
    book_title= data['original_title'].iloc[book_indices]
    book_url = data['image_url'].iloc[book_indices]
    book_rating= data['average_rating'].iloc[book_indices]
    book_author = data['authors'].iloc[book_indices]
    recommendation_data = pd.DataFrame(columns=['original_title','image_url', 'average_rating','authors'])
    recommendation_data['original_title'] = book_title.head(5)
    recommendation_data['image_url'] = book_url.head(5)
    recommendation_data['average_rating'] = book_rating.head(5)
    recommendation_data['authors'] = book_author.head(5)
    return recommendation_data
def SimilarAuthor(book):
  #  book=input()
    book= book.lower()
    find_book= getdata()
    transform_book = transformation(find_book)
    name2= find_book['original_title']
    search_book=process.extractOne(book, name2)
    (Name,choice,score)= search_book   
    if Name not in find_book['original_title'].unique():
        return print('Book not in Database')
    else:
        recommendations = recommendation(Name, find_book, transform_book)
        return recommendations.to_dict('records')
 
def SimilarBooks(book):
  #  book=input()
    book= book.lower()
    find_book= getdata()
    transform_book = transformation2(find_book)
    name2= find_book['original_title']
    search_book=process.extractOne(book, name2)
    (Name,choice,score)= search_book   
    if Name not in find_book['original_title'].unique():
        return print('Book not in Database')
    else:
        recommendations = recommendation(Name, find_book, transform_book)
        return recommendations.to_dict('records')
def top():
    find_book= getdata()
    rate=find_book.groupby(['original_title','authors','image_url'])['average_rating'].mean().sort_values(ascending = False).head(5)
    return rate.to_dict()
#name=input()
#result(name)
#authors_recommendations(title)
'''
images=authors_recommendations(name).head(5)
for i in images:
    response = requests.get(i)
    img = Image.open(BytesIO(response.content))
    plt.figure()
    print(plt.imshow(img))'''