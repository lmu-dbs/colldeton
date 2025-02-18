import sys


# Insert functions path into working dir if they are not in the same working dir
sys.path.insert(1, "C:\\Absolute\\path\\to\\your\\folder")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from idendrogram_streamlit_component import StreamlitConverter
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cophenet, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from utils.func import calculate_utility_score, calculate_ndcg

converter = StreamlitConverter(release=False)


def import_preprocessed_data(path):
    dist_matrix = []
    students = []
    contents = path.decode("utf-8")
    contents = contents.splitlines()
    for content in contents:
        ref, dist = content.split(":")
        f, t = ref.split("->")
        f = int(f)
        if f not in students:
            students.append(f)
    students.sort()

    for _ in students:
        dist_matrix.append([0] * len(students))

    for content in contents:
        # print(content)
        ref, dist = content.split(":")
        dist = float(dist)
        f, t = ref.split("->")
        f = int(f)
        t = int(t)
        i = students.index(f)
        j = students.index(t)
        dist_matrix[i][j] = dist

    return np.array(dist_matrix), np.array(students)


def check_against_gt(y, x):
    st.info(f"Groundtruth has {len(y)} entries")
    # st.write(f"{len(x)} were detected: {sorted(set(x))}")
    intersect = sorted(set(y).intersection(set(x)))
    precision = len(intersect) / len(x)
    recall = len(intersect) / len(y)
    st.info(f"Precision: {precision:.2f}")
    st.info(f"Recall: {recall:.2f}")
    if not precision + recall == 0:
        f1 = 2 * precision * recall / (precision + recall)
        st.info(f"F1: {f1:.2f}")
    st.info(f"Intersection {len(intersect)}: {sorted(set(y).intersection(set(x)))}")


def linkage_to_dict(Z, t):
    # Initialize each data point as its own cluster
    n_points = Z.shape[0] + 1
    clusters = {i: {i} for i in range(n_points)}
    next_cluster_id = n_points  # To assign new cluster IDs starting after the initial points

    # Iterate through the linkage matrix
    for i, (cluster1, cluster2, dist, _) in enumerate(Z):
        # If the distance between merged clusters exceeds the threshold, stop merging
        if i >= n_points or dist >= t:
            break

        # Merge clusters below the threshold
        new_cluster = clusters[int(cluster1)].union(clusters[int(cluster2)])

        # Create a new cluster and assign it an ID
        clusters[next_cluster_id] = new_cluster

        # Remove old clusters that were merged
        del clusters[int(cluster1)]
        del clusters[int(cluster2)]

        next_cluster_id += 1
    return clusters


def get_recommended_threshold(linkage_matrix, distance_matrix):
    # Test different thresholds and calculate the silhouette score for each
    max_threshold = linkage_matrix[-1, 2]  # Max distance in the dendrogram
    thresholds = np.linspace(0.1, max_threshold, 50)
    best_threshold = 0
    best_score = -1
    scores = []

    for t in thresholds:
        clusters = fcluster(linkage_matrix, t, criterion='distance')  # Form clusters at threshold t
        if 1 < len(np.unique(clusters)) < len(
                distance_matrix):  # Ensure there is more than 1 cluster and less than the number of students
            score = silhouette_score(distance_matrix, clusters, metric="precomputed")
            scores.append(score)
            if score > best_score:
                best_score = score
                best_threshold = t
        else:
            scores.append(None)

    return round(best_threshold, 1)

def get_ranking(linkage_matrix, n_items):
    # st.info(linkage_matrix)
    st.info(f"Number of items: {n_items}")
    # Initialize ranking list
    ranking = []

    # Track clusters formed
    for row in linkage_matrix:
        cluster1, cluster2, distance, _ = row

        # Check if each cluster represents a single original item
        if cluster1 < n_items:  # cluster1 is an original item
            if int(cluster1) not in ranking:
                ranking.append(int(cluster1))
        if cluster2 < n_items:  # cluster2 is an original item
            if int(cluster2) not in ranking:
                ranking.append(int(cluster2))
    # st.info(f"Ranking: {ranking}")
    return ranking


def compute_dendrogram(distmatrix_path, color_thresh, linkage_criterion, use_recommendation, mydend, myheat):
    dist_matrix, students = import_preprocessed_data(distmatrix_path)

    dists = squareform(dist_matrix)
    with myheat:
        with st.expander("What is shown in the following plot?"):
            st.write("""
            This plot shows the distance matrix as a heatmap. The darker the color, the lower the distance between two students.
            """)
        plt.figure(figsize=(20, 20))
        sns.heatmap(dist_matrix, annot=False, cmap='Reds_r', square=True)
        plt.title("Distance Matrix Heatmap")
        plt.xlabel("Student IDs")
        plt.ylabel("Student IDs")
        st.pyplot(plt)

    linkage_matrix = linkage(dists, linkage_criterion)

    ranked_predicted_ids = get_ranking(linkage_matrix, len(students))
    ranked_predicted_ids = [students[i] for i in ranked_predicted_ids]

    recommended_threshold = get_recommended_threshold(linkage_matrix, dist_matrix)
    with mydend:
        title = f"Dendrogram: {linkage_criterion} link"
        plt.figure(figsize=(30, 6), dpi=1800)
        if use_recommendation:
            color_thresh = recommended_threshold
        st.info(f"Current threshold: {color_thresh}")
        fig = dendrogram(linkage_matrix, labels=students, color_threshold=color_thresh)
        clusters = fcluster(linkage_matrix, color_thresh, criterion='distance')
        cluster_dict = {}
        for idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = [students[idx]]
            else:
                cluster_dict[cluster_id].append(students[idx])
        clustering = []
        for cluster_id, student_ids in cluster_dict.items():
            if len(student_ids) > 1:
                # st.info(f"Cluster {cluster_id}: {student_ids}")
                clustering.append(student_ids)

        leaves_color_list = fig['leaves_color_list']
        default_color = 'C0'
        leaves_ids = fig['ivl']
        colored_leaves_ids = []
        colored_leaves_ids_dict = {}
        for idx, item in enumerate(leaves_color_list):
            if item != default_color:
                if item not in colored_leaves_ids_dict:
                    colored_leaves_ids_dict[item] = [idx]
                else:
                    colored_leaves_ids_dict[item].append(idx)

        plt.axhline(y=color_thresh, color='red', linestyle='--')
        plt.xlabel("Student IDs", fontsize=25)
        plt.ylabel("Distance", fontsize=25)
        plt.yticks(fontsize=25)
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)

        st.pyplot(plt)
        with st.expander("What does the cophenetic correlation coefficient mean?"):
            st.write("""
            The cophenetic correlation coefficient measures how well the dendrogram represents the original distances.
            A value close to 1.0 means that the dendrogram is a good representation of the distances.
            A value close to 0.0 or negative means that the dendrogram is not a good representation of the distances.
            """)
        st.info(f"Cophenetic Correlation Coefficient: {cophenet(linkage_matrix, dists)[0]:.3f}")
    plt.savefig(str(linkage_criterion)+"_"+str(color_thresh)+"_dendrogram.pdf", format='pdf', dpi=600)
    st.write("INFO: Saved dendrogram in current directory")

    return clustering, ranked_predicted_ids


if __name__ == '__main__':
    st.set_page_config(page_title="Collusion Detection via Dendrogram")

    st.title("Collusion Detection via HAC")

    st.write(
        "This demo uses Hierarchical Agglomerative Trace Clustering (HAC) to identify potential collusion in student submissions. It uses a dendrogram to visualize the clustering process and identify a recommendable threshold for separating colluding from non-colluding students.")

    mydend, myheat= st.tabs(["Dendrogram", "Heatmap"])

    # st.set_page_config(layout="wide")  # Set the layout to wide
    with st.sidebar:

        st.title("Collusion Detection via HAC")
        file_upload = st.file_uploader("Upload a text file", type="txt")

        # # Dropdown menu 1
        linkage_criterion = st.selectbox(
            'Select a linkage criterion type',
            ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward'),
            help="Choose the method for calculating the distance between clusters."
        )

        use_recommendation = st.checkbox("Use Recommended Threshold", value=False,
                                         help="Automatically select the optimal threshold for clustering.")
        color_thresh = st.slider('Select a color threshold', min_value=0.01, max_value=100.0, value=3.0, step=0.1,
                                 disabled=use_recommendation,
                                 help="Set the threshold for coloring clusters in the dendrogram.")
        topk = st.checkbox("Top-k", value=True)
        num_top = st.slider("Number of top relations to show", 1, 500, 50, 1, disabled=not topk)
        if not topk:
            num_top = None

        run = st.button("Run")

    if run:
        if file_upload is not None:
            file_contents = file_upload.read()
            found_ids, ranked_ids = compute_dendrogram(file_contents, color_thresh, linkage_criterion, use_recommendation, mydend, myheat)
            flattened_found_ids = [item for sublist in found_ids for item in sublist]

            # Insert ground truth ids here
            groundtruth_ids = [1,3] # new
            ranked_ids_cut = ranked_ids[:len(groundtruth_ids)]
            overlap = len(set(flattened_found_ids).intersection(set(groundtruth_ids)))
            recall = overlap / len(groundtruth_ids)
            st.info(f"Recall: {recall:.4f}")
            precision = overlap / len(flattened_found_ids)
            st.info(f"Precision: {precision:.4f}")
            f1 = 2 * (precision * recall) / (precision + recall)
            st.info(f"F1: {f1:.4f}")
            utility = calculate_utility_score(groundtruth_ids, ranked_ids, num_top)
            st.write(f"Utility: {utility:.4f}")
            ndcg = calculate_ndcg(groundtruth_ids, ranked_ids, num_top)
            st.write(f"NDCG: {ndcg:.4f}")

            with mydend:
                st.info(f"Detected groups of colludings students ({len(found_ids)}): {found_ids}")
                st.info(f"Detected single students ({len(flattened_found_ids)}): {flattened_found_ids}")
        else:
            st.error("Please upload a file")
