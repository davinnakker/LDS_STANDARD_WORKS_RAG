import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/search"

st.title("📖 Scripture Semantic Search")

query = st.text_input("Enter your query")

k = st.slider("Number of results", 1, 20, 5)

volumes = ['All', 'Old Testament', 'New Testament', 'Book of Mormon', 'Doctrine and Covenants', 'Pearl of Great Price']
book = st.selectbox("Select a Work", volumes)

if st.button("Search"):

    params = {
        "query": query,
        "k": k,
        "book": book
    }

    response = requests.get(API_URL, params=params)
    st.write(response.status_code)
    results = response.json()
    print(results)

    st.subheader("Results")

    for r in results:
        st.markdown(f"""
        **{r['citation']}**

        {r['text']}
        """)
        st.divider()

    # python -m uvicorn app.main:app --reload
    # streamlit run frontend/streamlit_app.py

    