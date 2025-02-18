import streamlit as st
from sklearn.metrics import ndcg_score


def calculate_ndcg(groundtruth_ids: list, predicted_ids: list, num_top: int):
    predicted_rels = []
    if num_top is None:
        k = len(predicted_ids)
    else:
        k = num_top
    st.write(f"Top k = {k}")
    for i in range(k):
        if predicted_ids[i] in groundtruth_ids:
            predicted_rels.append(1)
        else:
            predicted_rels.append(0)
    groundtruth_rels = sorted(predicted_rels, reverse=True)

    ndcg = ndcg_score([groundtruth_rels], [predicted_rels])
    return ndcg

def calculate_utility_score(groundtruth_ids: list, predicted_ids: list, num_top: int):
    predicted_rels = []
    if num_top is None:
        k = len(predicted_ids)
    else:
        k = num_top
    for i in range(k):
        if predicted_ids[i] in groundtruth_ids:
            predicted_rels.append(1)
        else:
            predicted_rels.append(0)
    utility = 0
    for i, r in enumerate(predicted_rels):
        if r == 1:
            utility += 1 / (2 ** ((i-1)/2))
    return utility