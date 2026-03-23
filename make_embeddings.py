from semantic.retrieval import Retrieval
from app.helper_functions import get_csv

works = ['Old Testament', 'New Testament', 'Book of Mormon', 'Doctrine and Covenants', 'Pearl of Great Price']

for work in works:
    csv, embeddings = get_csv(work)
    retriever = Retrieval(csv, embeddings)