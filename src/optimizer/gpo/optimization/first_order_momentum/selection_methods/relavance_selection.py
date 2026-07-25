import os
import sys
import numpy as np

from sentence_transformers import SentenceTransformer
import torch

GPO_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
)


def calculate_sentence_similarity(x, string_list, k):
    # Use lightweight embedding model to avoid OOM (Llama2-7b too large when vLLM already loaded)
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Small model: ~90MB vs 7GB
    model.to('cpu')  # Use CPU to avoid GPU memory conflict with vLLM

    with torch.no_grad():
        embedding_x = model.encode([x], normalize_embeddings=True)
        embeddings_string_list = model.encode(string_list, normalize_embeddings=True)

    similarity = embedding_x @ embeddings_string_list.T

    top_k_similar_indices = np.argsort(similarity[0])[::-1][:k]

    return top_k_similar_indices


class Relavance_Selection:
    def select(self, history_list, select_num, momentum_para_name):
        if momentum_para_name in {'feedback'}:
            new = history_list[-1][0]
            selected_list = history_list[:-1][::-1]
            selected_list = [selected[0] for selected in selected_list]
            if len(selected_list) != 0:
                selected_idx = calculate_sentence_similarity(
                    new, selected_list, select_num - 1
                )
                selected_list = [selected_list[i] for i in selected_idx]
            selected_list.insert(0, new)
        else:
            assert momentum_para_name == "para"
            new = (history_list[-1][0], history_list[-1][1])
            selected_list = history_list[:-1][::-1]
            selected_sentence_list = [selected[0] for selected in selected_list]
            selected_list = [(selected[0], selected[1]) for selected in selected_list]
            if len(selected_sentence_list) != 0:
                selected_idx = calculate_sentence_similarity(
                    new[0], selected_sentence_list, select_num - 1
                )
                selected_list = [(selected_list[i][0], selected_list[i][1]) for i in selected_idx]
            selected_list.insert(0, new)

        return selected_list