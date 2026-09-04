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

Here comes 🎵 <span style="background-color: yellow;">**LyricLens**</span>, your personal song recommender based on how you feel and what the lyrics actually say.

Drawing from Billboard Hot 100 songs from 1958 to today, simply describe your mood, feelings, occasion, or vibe. LyricLens uses RAG and semantic search to understand song lyrics, find songs that match what you're looking for, and recommend them with an explanation of why each song fits.

⚠️ **English only:** The embedding model is intended for English text only, so queries should be written in English for best results.

💡 **Query quality matters:** Recommendations rely on your query’s meaning, emotional tone, and imagery, not genre. Be specific about your mood, situation, and what you want to feel for more relevant matches.



## 🎬 Demo (v0.1.0)

### Using the UI
https://github.com/user-attachments/assets/5350d178-f205-4256-8110-052847914146



### Local Feedback Tracking
https://github.com/user-attachments/assets/7fbaa01a-5fe9-4993-b860-e86bdb299f61


## 📊 Datasource

### 1. Billboard Hot 100 Songs
[UT RWD Billboard Dataset](https://github.com/utdata/rwd-billboard-data): weekly Hot 100 charts from 1958-08-04 to present.

### 2. Lyrics Search
[LRCLIB](https://lrclib.net): free lyrics API.



## ⛳️ Challenges & Approaches

### 1. Billboard Hot 100 data is messy, so lyrics search is hard.

🔴 **Duplicate and ambiguous records:** songs with the same title from different artists may be completely unrelated (**23** by *Chayce Beckham* VS by *Noah Kahan* VS by *Sam Hunt*). While different releases or remakes of the same song from the same artist can appear as separate records, even in the same chart week (**Unchained Melody** by *The Righteous Brothers*, 1990). 
> Songs are identified by <ins>title and performer</ins>, not title alone. For lyrics retrieval, releases with the same title and performer are <ins>treated as the same song</ins>. Duplicates are consolidated and popularity features <ins>re-engineered</ins> to reflect song-level rather than release-level popularity.

🔴 **Inconsistent title/performer records:** the same title or performer can appear with different formatting across records, which breaks exact-match deduplication (*BossMan Dlow* VS *BossMan DLow*, *RealestK* VS *Realestk*, *Jennie* VS *JENNIE*).
> Whitespace, case, and unicode <ins>formatting differences in title and performer are folded</ins> before comparison, then a deterministic hash of the (title, performer) pair is generated. Records sharing a `song_id` are <ins>treated as duplicates and removed</ins>.

🔴 **Popularity isn't comparable across eras:** Hot 100 chart rules have changed over time, making raw `wks_on_chart` an inconsistent popularity signal.
> Popularity features are <ins>re-engineered for more consistent interpretation</ins> across records.

🔴 **Titles/artists aren't parsed cleanly:** some records embed extra info in the title field (**"Cherry Cherry" from Hot August Night** by *Neil Diamond*). This can cause LRCLIB searches with `track_name` and `artist_name` to return no results.
> Titles and artist names are <ins>combined</ins> as parameter `q` <ins>in the lyric search query</ins>.

🔴 **Featured artists complicate matching:** performer credits may differ between the source dataset and the search database (**Feat., featuring, x, &**). Also, including too many featured artists in the search query can cause LRCLIB to return no results.
> <ins>Artist names are cleaned</ins> using a pattern before searching. As a last resort, the <ins>primary artist is extracted</ins>, omitting featured artists when it improves matching.

🔴 **Real song names are messy:** special characters (**punchin'.the.clock**), unusual capitalization (**thanK you aIMee**), and unconventional formatting (**g n f (Give No Fxk)**) make songs difficult to search reliably.
> Failed searches are <ins>retried</ins>; songs with no valid lyrics are <ins>logged and excluded</ins>.

### 2. Lyrics can't be chunked randomly.

🔴 Lyrics carry meaning through verses, choruses, bridges, and other natural sections. Arbitrary fixed-length chunking breaks that context and hurts semantic retrieval.
> LyricLens <ins>preserves LRCLIB's natural lyric sections whenever possible</ins>, splitting oversized sections with a <ins>controlled force_chunk fallback</ins> while preventing undersized remainder chunks.

🔴 Some songs have only a few lines of lyrics, making meaningful chunking impossible (**Beautiful Trip** by *Kid Cudi*).
> These edge cases are <ins>logged and excluded from chunking</ins>. They are preserved as-is and are very unlikely to produce meaningful matches.

### 3. Lyrics are copyrighted content.

🔴 Raw lyrics can't simply be redistributed with the project.
> The lyrics dataset and raw lyric text are **not included in this repository!** LyricLens searches real lyric chunks internally, but recommendations <ins>describe the lyrical scene and explain the match in the assistant's own words</ins>, without reproducing lyrics.



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

| File (project root) | Description |
|---|---|
| `app.py` | Streamlit UI: search box, side-by-side recommendation cards, with a Spotify search link, and 👍/👎 feedback on each. |
| `Dockerfile` | Builds the app image with `uv`, bakes in the ONNX embedding model at build time. |
| `docker-compose.yml` | Defines the `db` (pgvector/pgvector:pg17) and `app` services, volumes, and healthcheck. |
| `docker/entrypoint.sh` | Waits for Postgres, runs ingest only if the vector index is missing, then starts Streamlit. |
| `.dockerignore` | Excludes local data, models, and dev files from the image build context. |

| File (src/) | Description |
|---|---|
| `config.py` | Project-wide configuration. |
| `download_hot100.py` | Downloads Billboard Hot 100 data from the [UT RWD Billboard Dataset](https://github.com/utdata/rwd-billboard-data), scraped weekly. |
| `song_id.py` | Generate a deterministic song_id for each (title, performer) pair as a persisted identifier.|
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

 

 ## 💻 Run in CLI (local)

 ### First run:

 1. `git clone https://github.com/ChuqingShi/LyricLens.git`
 2. `uv sync`
 3. Customize `src/config.py`, or use the defaults
 4. `uv run python -m src.download_hot100`
 5. `uv run python -m src.clean_hot100`
 6. `uv run python -m src.search_lyrics` (takes ~30-60 min with default config)
 7. `uv run python -m src.chunk_lyrics`
 8. `uv run python -m src.embeddings.embed_texts`
 9. Run PostgreSQL with pgvector:
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
 10. `uv run python -m src.ingest_postgres`
 11. Add your `OPENAI_API_KEY` to a `.env` file in the project root
 12. `uv run python -m src.rag` launches an interactive CLI. <span style="background-color: yellow;"> Start chatting with your assistant!</span>
 13. Stop the Docker container when done: `docker stop pgvector`

 ### Later runs:
 1. `docker start pgvector` to restart the database
 2. `uv run python -m src.rag` to launch the app in CLI
 3. `docker stop pgvector` to stop the Docker container
   

 
 ## 🐳 Run with UI using Docker (local)

 This requires `data/processed/*` to already exist locally (steps 4-8 above), since Docker handles only ingestion + serving, not the data engineering + embedding pipeline.

 1. Add `OPENAI_API_KEY`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` to a `.env` file in the project root.
 2. `docker compose up --build`. On first run, the app container waits for Postgres, then automatically runs the ingestion before serving. Later runs skip straight to serving.
 3. Open [localhost:8501](http://localhost:8501) and <span style="background-color: yellow;"> start chatting with your assistant!</span> Each card has a **▶ Play on Spotify** link and 👍/👎 feedback buttons.
 4. To rebuild after any code change (including `src/config.py`), run `docker compose up --build` again.
 5. `docker compose down` to stop (add `-v` to also wipe the Postgres volume if you want to reset the database).


 
 ## 🎧 Interacting with the UI

 1. Open [localhost:8501](http://localhost:8501) and type your mood/vibe/situation (e.g. *"post-breakup, want to sit with the sadness"*), and click **Find songs**. The recommendations are displayed as cards, ranked from best match, with the *title*, *performer*, a poetic description of the matching *lyrical scene* (⚠️No raw lyrics are displayed for copyright reasons), and the *reason* for recommendation.
 3. On a card, click **▶ Play on Spotify**. It opens a new tab with Spotify's search results for the song.
 4. Click 👍/👎 on a card to submit feedback. The card remains visible after rating, and submitted feedback cannot be changed.
 5. Ready to run another query!
 
 
 <!-- ## 💾 Verifying Feedback Persistence

 Confirm a rating was saved to Postgres. In CLI, run

 ```bash
 docker compose exec db psql -U lyricslens -d lyricslensDB -c "select * from feedback;"
 ```
 
 This only applies when running with UI using Docker. For the CLI workflow's standalone `pgvector` container, use `docker exec pgvector psql ...` instead. -->




