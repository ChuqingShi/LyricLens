from pydantic import BaseModel
from .search_pgvector import search_sections_vector, aggregate_vector_results
from .rerank_songs import rerank_songs
from .config import TOP_K_SECTIONS, TOP_K_SONGS, NUM_SONG_RECOMMENDATIONS


class SongRecommendation(BaseModel):
    title: str
    performer: str
    lyric_scene: str
    reason: str


class SongRecommendations(BaseModel):
    recommendations: list[SongRecommendation]


INSTRUCTIONS = INSTRUCTIONS = f"""
You are a poetic and perceptive song recommender.

Select exactly {NUM_SONG_RECOMMENDATIONS} distinct songs from the provided context that best match the user's requested vibe, feeling, or occasion. 
Order the recommendations from best to worst match based on how well their lyrics match the user's request.

Choose the songs based ONLY on the meaning, emotional tone, themes, imagery, and overall vibe conveyed by their lyrics. 
Use only the provided context.
Do not use popularity, artist reputation, genre, musical style, cultural significance, or outside knowledge when deciding which songs to recommend.
Do not assume or introduce information that is not supported by the provided lyrics.

For each SongRecommendation, return:
- `title`: Must be copied exactly from the context.
- `performer`: Must be copied exactly from the context.
- `lyric_scene`: Paint a brief, vivid scene inspired by the part of the lyrics that best matches the user's request. 
Make the user feel as if they are inside the emotional moment or story of the song. 
Stay faithful to the lyrics, but do not quote or closely reproduce them. 
Keep it to 2–3 concise sentences.
- `reason`: Briefly explain why the song's lyrical meaning and emotional perspective make it a strong match for the user's request. 
Keep it to 1–2 concise sentences.
"""

PROMPT_TEMPLATE = """
REQUEST: {request}

CONTEXT:
{context}
""".strip()


class RAGPgVector:
    def __init__(
        self,
        conn,
        embedder,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model="gpt-5.4-mini",
    ):
        self.conn = conn
        self.embedder = embedder
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(
        self, query, num_section_results=TOP_K_SECTIONS, num_song_results=TOP_K_SONGS
    ) -> list[dict]:
        top_k_sections = search_sections_vector(
            query, self.embedder, self.conn, num_section_results
        )
        top_k_songs = aggregate_vector_results(top_k_sections, num_song_results)
        new_top_k_songs = rerank_songs(top_k_songs)
        # [
        #     {
        #         "song_id",
        #         "title",
        #         "performer",
        #         "wks_on_chart",
        #         "peak_pos",
        #         "best_vector_similarity",
        #         "num_matches",
        #         "overall_vs_score",
        #         "matched_sections":
        #         [
        #             {
        #                 "section_id",
        #                 "section",
        #                 "num_lines",
        #                 "vector_similarity"
        #             },...
        #         ]
        #     },...
        # ]
        return new_top_k_songs

    def build_context(self, search_results):
        lines = []

        for song in search_results:
            lines.append(f"title: {song["title"]}")
            lines.append(f"performer: {song["performer"]}")
            lines.append(
                f"lyrics: {song["lyrics"].strip().replace("\n\n", "; ").replace("\n", ", ")}"
            )
            lines.append("")

        return "\n".join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(request=query, context=context)

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        response = self.llm_client.responses.parse(
            model=self.model, input=input_messages, text_format=SongRecommendations
        )

        return response.output_parsed.recommendations

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer


def main():
    from dotenv import load_dotenv
    from openai import OpenAI
    from src.embeddings.embedder import Embedder
    from src.ingest_postgres import connect_db

    from src.rag import RAGPgVector

    load_dotenv()
    openai_client = OpenAI()
    embedder = Embedder()

    with connect_db() as conn:
        lyriclens_assistant = RAGPgVector(
            conn=conn,
            embedder=embedder,
            llm_client=openai_client,
        )
        query = input("How are you feeling, and what do you want to feel?\n")
        song_recommendations = lyriclens_assistant.rag(query)

    print("\n==========Running LyricsLens===========\n")
    for i, song_rec in enumerate(song_recommendations, start=1):
        print(f"Song Recommendation # {i}")
        print(f"title: {song_rec.title}")
        print(f"performer: {song_rec.performer}")
        print(f"lyric_scene: {song_rec.lyric_scene}")
        print(f"reason: {song_rec.reason}")
        print()


if __name__ == "__main__":
    main()
