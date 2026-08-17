## MVP 0:

1. ➡️ get hot-100-song data with song title and performer and store data ✅

* Extra 0: add popularity features: chart_weeks, wks_on_chart, peak_pos ✅
* Extra 1: add popularity features: periods_on_chart


2. ➡️ use hot-100-song title and performer to search lyrics on LrcLib and store data ✅


3. chunk lyrics and generate documents ✅


4. ➡️ embedding and vector search ✅

* bruteforce vector search ✅
* Extra 0 : index with minsearch
* Extra 1: index with sqlitesearch
* ➡️ Extra 2: HNSW index with PostgreSQL

5. ➡️ data ingestion into PostgreSQL ✅

6. ➡️ vector search in PostgreSQL ✅

7. rerank song results ✅

8. search evaluation (later)

9. ➡️ RAG

10. Docker Containerization

11. ➡️ Deploy

---

## datasource

* Billboard Hot 100 data: 
[UT RWD Billboard Dataset](https://github.com/utdata/rwd-billboard-data), contains Billboard weekly hot100 on chart from 1958-08-04 until current week.

* Lyrics:
[LRCLIB](https://lrclib.net), supports free API calls.

## files
src
 
 * download_hot100.py: 
 download Billboard Hot 100 datasets from [UT RWD Billboard Dataset](https://github.com/utdata/rwd-billboard-data) where charts are scrapted weekly to keep update.
 
 * clean_hot100.py: 
 clean, filter, and save hot 100 song data.

 * search_lyrics.py: 
 search lyrics from [LRCLIB](https://lrclib.net), generate and save hot 100 lyrics data with cleaning, checkpointing, and retries.

 * chunk_lyrics.py:
 apply custom chunking to preserve natural lyric sections as much as possible, with force-chunking fallback and minimum & maximum section-length controls, and save hot 100 chunks documents.

 * embeddings/download.py:
 fetch ONNX runtime model from HuggingFace. Copied from [`download.py`](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/embed/download.py) with minor configuration changes.

 * embeddings/embedder.py
 generate embeddings with the same encode interface as sentence-transformers but none of the PyTorch weight. Copied from [`embedder.py`](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/embed/embedder.py) with minor configuration changes.

 * embeddings/embed_texts.py
 embed lyric chunks and save their embeddings.

 * ingest_lyrics.py:

 * search_pgvector.py

 * rerank_songs.py




