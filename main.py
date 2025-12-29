# import data
import pandas as pd
df = pd.read_csv("lds-scriptures.csv")

# import RAG functions
from RAG_module import search_index

# query
while True:
    query = input("What would you like to look for? (press space to quit): ")
    if query == " ":
        break
    rows = search_index(query, "scriptures")
    print("-----RESULTS-----")
    for row in rows[0]:
        title, verse = df.iloc[row][['verse_short_title', 'scripture_text']]
        print(f"{title} {verse}")
        print()