# 🎵 LyricsLens:

<div align="left">
<span style="color: red;">Imagine this:💬</span>

You have a deadline in two days. 

You’re stressed, alone, and just want some music that understands how you feel, like a friend.

So you open Spotify and search “stressed”.

You get a generic mix of sad relationship songs.

Beautiful songs, but not what you need. 

Why does “stressed” suddenly mean heartbreak? I have a deadline!
</div>


<div align="right">
<span style="color: red;">Now imagine this:💬</span>

You’ve just gone through a peaceful breakup. 

You don’t want revenge, girl-power anthems, or songs about enjoying single life.

You want to sit with the sadness and grief.

So you search “post-breakup.”

And there they are: songs about being single, partying in Vegas, and, of course, cheating men.

No, no, no. Not right now.

You want a song that matches what you’re actually feeling.

But how do you find one?
</div>

Here comes <span style="background-color: yellow;">**LyricLens**</span>, your personal song recommender based on how you feel and what the lyrics actually say.

Drawing from Billboard Hot 100 songs from 1958 to today, simply describe your mood, feelings, occasion, or vibe. LyricLens uses RAG and semantic search to understand song lyrics, find songs that match what you're looking for, and recommend them with an explanation of why each song fits.



## 📊 Datasource

### 1. Billboard Hot 100 Songs
[UT RWD Billboard Dataset](https://github.com/utdata/rwd-billboard-data): weekly Hot 100 charts from 1958-08-04 to present.

### 2. Lyrics Search
[LRCLIB](https://lrclib.net): free lyrics API.



## ⛳️ Challenges & Approaches

### 1. Billboard Hot 100 data is messy, so lyrics search is hard.

🔴 **Duplicate and ambiguous records:** the same song can appear multiple times, including different releases in the same week (**Unchained Melody** by *The Righteous Brothers*, 1990), while unrelated songs can share a title.
> Songs are identified by <u>title and performer</u>, not title alone. Duplicates are consolidated and their popularity features <u>re-engineered</u> to preserve chart history.

🔴 **Popularity isn't comparable across eras:** Hot 100 chart rules have changed over time, making raw `wks_on_chart` an inconsistent popularity signal.
> Popularity features are <u>re-engineered and normalized</u> before reranking.

🔴 **Titles/artists aren't parsed cleanly:** some records embed extra info in the title field (**"Cherry Cherry" from Hot August Night** by *Neil Diamond*).
> Titles and artist names are <u>combined in the lyric search query</u>.

🔴 **Featured artists complicate matching:** the same song is credited differently across sources (**feat, featuring, x, &**).
> The <u>primary artist is extracted</u>, dropping featured-artist credits when it improves matching.

🔴 **Real song names are messy:** special characters (**punchin'.the.clock**), unusual capitalization (**thanK you aIMee**), and unconventional formatting (**g n f (Give No Fxk)**) break exact matching.
> <u>Failed searches are retried</u>; songs with no valid lyrics are <u>excluded and logged</u>.

### 2. Lyrics can't be chunked randomly.

🔴 Lyrics carry meaning through verses, choruses, bridges, and other natural sections — arbitrary fixed-length chunking breaks that context and hurts semantic retrieval.
> LyricLens <u>preserves LRCLIB's natural lyric sections whenever possible</u>, splitting oversized sections with a <u>controlled `force_chunk` fallback</u> and handling very short sections to avoid meaningless chunks.

🔴 Some songs have only a few lines of lyrics, making meaningful chunking impossible (**Beautiful Trip** by *Kid Cudi*).
> These edge cases are detected, logged, and excluded from chunking, since they're unlikely to produce meaningful matches.

### 3. Lyrics are copyrighted content.

🔴 Raw lyrics can't simply be redistributed with the project.
> The lyrics dataset and raw lyric text are **not included in this repository!** LyricLens searches real lyric chunks internally, but recommendations <u>describe the lyrical scene and explain the match in the assistant's own words</u>, without reproducing lyrics.



## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12+ |
| Package & environment management | `uv` |
| Data processing | `pandas`, `pyarrow`, `numpy`, `pydantic` |
| Embeddings | `onnxruntime` + [`Xenova/all-MiniLM-L6-v2`](https://huggingface.co/Xenova/all-MiniLM-L6-v2) (no PyTorch), `tokenizers` |
| Vector database | PostgreSQL + pgvector via `psycopg`, Dockerized |
| LLM / RAG | `openai`, `python-dotenv` |
| Web UI | `streamlit` |
| Utilities | `requests`, `tqdm` |
| Dev tools | `black`, `huggingface-hub`, `ipykernel`, `jupyter` |



## 📁 Files

| File | Description |
|---|---|
| `app.py` (repo root) | Streamlit UI: search box, side-by-side recommendation cards, and 👍/👎 feedback on each. |

All under `src/`:

| File | Description |
|---|---|
| `config.py` | Project-wide configuration. |
| `download_hot100.py` | Downloads Billboard Hot 100 data from the [UT RWD Billboard Dataset](https://github.com/utdata/rwd-billboard-data), scraped weekly. |
| `clean_hot100.py` | Cleans, filters, and saves Hot 100 song data. |
| `search_lyrics.py` | Searches lyrics via [LRCLIB](https://lrclib.net) and saves them, with cleaning, checkpointing, and retries. |
| `chunk_lyrics.py` | Chunks lyrics to preserve natural sections, with a force-chunking fallback and min/max length controls. |
| `embeddings/download.py` | Fetches the ONNX embedding model from Hugging Face. Adapted from [`download.py`](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/embed/download.py). |
| `embeddings/embedder.py` | Generates embeddings with a sentence-transformers-compatible interface, no PyTorch. Adapted from [`embedder.py`](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/02-vector-search/embed/embedder.py). |
| `embeddings/embed_texts.py` | Embeds lyric chunks and saves the embeddings. |
| `ingest_postgres.py` | Sets up PostgreSQL, ingests songs and documents, and builds the vector search index. |
| `search_pgvector.py` | Runs vector search with pgvector and aggregates results by song. |
| `rerank_songs.py` | Reranks top-k songs by a weighted score of popularity and vector similarity. |
| `feedback.py` | Logs 👍/👎 ratings from the Streamlit app to a `feedback` table in Postgres. |
| `rag.py` | Core RAG pipeline: retrieves songs, builds the LLM prompt, and generates recommendations. Based on [`rag_helper.py`](https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/01-agentic-rag/code/rag_helper.py) and [`vector_search_pgvector.ipynb`](https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/02-vector-search/code/vector_search_pgvector.ipynb). |

 

 ## 💻 Run local in CLI

 1. `git clone https://github.com/ChuqingShi/LyricLens.git`
 2. `uv sync`
 3. Customize `src/config.py`, or use the defaults.
 4. `uv run python -m src.download_hot100`
 5. `uv run python -m src.clean_hot100`
 6. `uv run python -m src.search_lyrics` (takes ~30-60 min with default config)
 7. `uv run python -m src.chunk_lyrics`
 8. Run PostgreSQL with pgvector:
    ```bash
    docker run -d \
    --name pgvector \
    -e POSTGRES_USER=lyricslens \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=lyricslensDB \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17
    ```
 9. `uv run python -m src.ingest_postgres`
 10. Add your `OPENAI_API_KEY` to a `.env` file in the project root.
 11. `uv run python -m src.rag` launches an interactive CLI. Start chatting with your assistant!
 12. Stop the Docker container when done: `docker stop pgvector`

 
 
 ## 🐳 Run with Docker

 Requires `data/processed/*` to already exist locally (steps 3-7 above) — Docker only handles ingest + serving, not the scraping/embedding pipeline.

 1. Add `OPENAI_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` to a `.env` file in the project root.
 2. `docker compose up --build`
 3. Open [localhost:8501](http://localhost:8501) and start chatting with your assistant! Rate each recommendation with 👍/👎 — ratings are logged to a `feedback` table in Postgres.
 4. `docker compose down` to stop (add `-v` to also wipe the Postgres volume).

 On first run, the app container waits for Postgres, then automatically runs the ingest step before serving — later runs skip straight to serving.

 
 
 ## 📋 MVP development checklist:

1. ➡️ get hot-100-song data with song title and performer and store data ✅

* Extra 0: add popularity features: chart_weeks, wks_on_chart, peak_pos ✅
* Extra 1: add popularity features: periods_on_chart

2. ➡️ use hot-100-song title and performer to search lyrics on LrcLib and store data ✅

3. chunk lyrics and generate documents ✅

4. ➡️ embedding and vector search ✅

* bruteforce vector search ✅
* Extra 0 : index with minsearch
* Extra 1: index with sqlitesearch
* ➡️ Extra 2: HNSW index with PostgreSQL ✅

5. ➡️ data ingestion into PostgreSQL ✅

6. ➡️ vector search in PostgreSQL ✅

7. rerank song results ✅

8. search evaluation (later)

9. ➡️ RAG ✅

10. ➡️ Docker Containerization ✅

11. ➡️ Deploy

12. monitoring



