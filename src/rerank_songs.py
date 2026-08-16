import numpy as np
from .config import TOP_K_SECTIONS

# popularity score:
#     wks_on_chart(1,2,...): larger-> better
#     peak_pos(1,2,...,100): smaller-> better
#
#   wks_on_chart(relative): song_wks_on_chart/ max(song_wks_on_chart): linearly increasing, (0,1]
#   peak_pos(absolute): 25/(peak_pos+25): decreasing, [1/5,~1)

# retrivial score: (more wight)
#     best_similarity [-1,1]: larger-> better
#     num_matches(1,2,...,TOP_K_SECTIONS): larger-> better
#
#   best_similarity(relative): linear map from [min, max] to [min,1], usually min>0
#   num_matches(relative): linear map from [min, max] to [min(num_matches)/TOP_K_SECTIONS, 1]


def minmax_linear_map(nums: list[float], min_goal: float, max_goal: float) -> np.ndarray:
    nums = np.asarray(nums)
    min_num = min(nums)
    max_num = max(nums)
    range_num = max_num - min_num
    range_goal = max_goal - min_goal

    if range_num == 0:
        return 1
    else:
        return min_goal + (nums - min_num) / range_num * range_goal


def best_sim_score(best_sim_s: list[float]) -> np.ndarray:
    return minmax_linear_map(best_sim_s, min(best_sim_s), 1)


def num_matches_score(num_matches_s: list[int], num_results: int = TOP_K_SECTIONS) -> np.ndarray:
    return minmax_linear_map(num_matches_s, min(num_matches_s) / num_results, 1)


def wks_on_chart_score(wks_on_chart_s: list[int]) -> np.ndarray:
    wks_on_chart_s = np.asarray(wks_on_chart_s)
    max_wks = max(wks_on_chart_s)
    return wks_on_chart_s / max_wks


def peak_pos_score(peak_pos_s: list[int], k=25) -> np.ndarray:
    peak_pos_s = np.asarray(peak_pos_s)
    return k / (k + peak_pos_s)


def weighted_total_score(
    song_results, w_pop_wks_on_chart=1, w_pop_peak_pos=1, w_sim_best=4, w_sim_num_matches=2
) -> np.ndarray:

    wks_on_chart_score_s = wks_on_chart_score([song["wks_on_chart"] for song in song_results])
    peak_pos_score_s = peak_pos_score([song["peak_pos"] for song in song_results])
    best_sim_score_s = best_sim_score([song["best_similarity"] for song in song_results])
    num_matches_score_s = num_matches_score([song["num_matches"] for song in song_results])

    total_score = (
        w_pop_wks_on_chart * wks_on_chart_score_s
        + w_pop_peak_pos * peak_pos_score_s
        + w_sim_best * best_sim_score_s
        + w_sim_num_matches * num_matches_score_s
    )
    weighted_total_score = total_score / (
        w_pop_wks_on_chart + w_pop_peak_pos + w_sim_best + w_sim_num_matches
    )

    return weighted_total_score


def rerank_songs(
    song_results, w_pop_wks_on_chart=1, w_pop_peak_pos=1, w_sim_best=4, w_sim_num_matches=2
) -> list[dict]:
    song_scores = weighted_total_score(
        song_results, w_pop_wks_on_chart, w_pop_peak_pos, w_sim_best, w_sim_num_matches
    )
    for i, song in enumerate(song_results):
        song["overall_score"] = song_scores[i].item()
    return sorted(song_results, key=lambda x: x["overall_score"], reverse=True)
