import os
import sys

# Insert functions path into working dir if they are not in the same working dir
sys.path.insert(1, "C:\\Absolute\\path\\to\\your\\folder")
import streamlit as st
from collections import defaultdict
import matplotlib.pyplot as plt
import squarify
from utils.func import calculate_utility_score, calculate_ndcg


def save_bar_chart(data_dict, title, file_name, folder="charts"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    plt.figure(figsize=(12, 4))
    plt.bar(data_dict.keys(), data_dict.values(), color="#3d606e")
    plt.xticks(rotation=90)
    plt.xlabel("Student Relations", fontsize=20)
    plt.ylabel("Z-Score", fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)

    plt.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False)

    st.info("INFO: Saving bar chart...")
    plt.savefig(file_name, format='pdf', dpi=600, bbox_inches='tight')
    st.info("INFO: Saved bar chart")
    # plt.savefig(os.path.join(folder, file_name))
    plt.close()

def save_violin_plot(data, title, file_name, folder="charts"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots(figsize=(10, 6))
    parts = ax.violinplot(data, showmeans=True, showextrema=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Temporal Dimension", "Content Dimension"], fontsize=20)
    # ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_ylim(-3, 6)
    plt.tick_params(axis='x', pad=10)
    plt.ylabel("Z-Score", fontsize=36)
    plt.yticks(fontsize=36)
    plt.xticks(fontsize=36)

    plt.savefig(os.path.join(folder, file_name), format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

def save_boxplot(data, title, file_name="boxplot.pdf", folder="charts"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig, ax = plt.subplots(1, 2, figsize=(10, 8))
    ax[0].set_ylim(-3, 0.3)
    ax[1].set_ylim(-3, 0.3)
    box = ax[0].boxplot(data[0], vert=True, patch_artist=True, whiskerprops=dict(linewidth=2))
    bottom_points = len(box["fliers"][0].get_data()[1])
    ax[0].text(1, -2.93, f"{bottom_points} outliers", ha='center', fontsize=25)
    for flier in box['fliers']:
        flier.set(marker='o', markersize=12, markerfacecolor='red', markeredgecolor='black')
    ax[0].set_title(title[0], fontsize=34, pad=15)
    ax[0].set_ylabel('Z-Score', fontsize=36)
    ax[0].tick_params(axis='y', labelsize=36)

    box = ax[1].boxplot(data[1], vert=True, patch_artist=True, whiskerprops=dict(linewidth=2))
    bottom_points = len(box["fliers"][0].get_data()[1])
    ax[1].text(1, -2.93, f"{bottom_points} outliers", ha='center', fontsize=25)
    for flier in box['fliers']:
        flier.set(marker='o', markersize=6, markerfacecolor='red', markeredgecolor='black', alpha=0.25)
    ax[1].set_title(title[1], fontsize=34, pad=15)
    ax[1].yaxis.set_tick_params(labelleft=False)
    ax[0].xaxis.set_tick_params(labelbottom=False)
    ax[1].xaxis.set_tick_params(labelbottom=False)

    plt.savefig(os.path.join(folder, file_name), format='pdf', dpi=600, bbox_inches='tight')
    plt.close()

def find_connected_components(pairs):
    graph = defaultdict(list)
    # Build graph
    for a, b in pairs:
        graph[a].append(b)
        graph[b].append(a)

    import networkx as nx
    import matplotlib.pyplot as plt

    G = nx.Graph()
    for a, b in pairs:
        G.add_edge(a, b)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    plt.show()

    visited = set()
    groups = []

    # Depth-First Search function to explore a group
    def dfs(node, group):
        visited.add(node)
        group.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor, group)

    # Visit all nodes
    for node in graph:
        if node not in visited:
            group = []
            dfs(node, group)
            groups.append(group)
    return groups


def normalize_and_combine(value1, range1, value2, range2):
    """
    Normalize two values from different ranges to a common scale and combine them.

    Parameters:
    - value1: The first value to be combined.
    - range1: A tuple (min1, max1) representing the range of value1.
    - value2: The second value to be combined.
    - range2: A tuple (min2, max2) representing the range of value2.

    Returns:
    - A single combined value on a scale from 0 to 1.
    """
    # Normalize value1 and value2 to a 0-1 scale
    normalized_value1 = (value1 - range1[0]) / (range1[1] - range1[0])
    normalized_value2 = (value2 - range2[0]) / (range2[1] - range2[0])

    # Combine the normalized values
    combined_value = (normalized_value1 + normalized_value2) / 2
    return combined_value


def visualize_groups_as_treemap(groups):
    group_sizes = [len(group) for group in groups]
    labels = [f"Group {i + 1}\n{group}\nSize: {len(group)}" for i, group in enumerate(groups)]
    colors = plt.cm.Spectral([i / float(len(groups)) for i in range(len(groups))])

    plt.figure(figsize=(8, 6))
    squarify.plot(sizes=group_sizes, label=labels, color=colors, alpha=0.7)
    plt.title("Treemap of Colluding Groups")
    plt.axis('off')
    st.pyplot(plt)

    # plt.savefig("./treemap.pdf", format='pdf', dpi=600)
    # st.write("INFO: Saved treemap in current directory")


def get_ranking(metrics):
    min_abs_time = min(metrics.values(), key=lambda x: x[0])[0]
    max_abs_time = max(metrics.values(), key=lambda x: x[0])[0]
    min_abs_lev = min(metrics.values(), key=lambda x: x[1])[1]
    max_abs_lev = max(metrics.values(), key=lambda x: x[1])[1]
    combined_metrics = {}
    for k, v in metrics.items():
        combined_metrics[k] = normalize_and_combine(v[0], (min_abs_time, max_abs_time), v[1],
                                                    (min_abs_lev, max_abs_lev))

    sorted_metrics = sorted(combined_metrics.items(), key=lambda x: x[1], reverse=True)
    sorted_metrics_dict = dict(sorted_metrics)

    return sorted_metrics_dict


def visualize_groups(relations):
    filtered_student_rels = [(int(k.split("->")[0]), int(k.split("->")[1])) for k in relations.keys()]
    groups = find_connected_components(filtered_student_rels)
    st.write(f"Number of connected components: {len(groups)}")
    st.write(f"Groups: {groups}")
    visualize_groups_as_treemap(groups)


def get_z_scores(metrics, time_zscore_threshold, levenshtein_zscore_threshold):
    if time_zscore_threshold is not None and levenshtein_zscore_threshold is not None:
        filtered = {k: v for k, v in metrics.items() if
                    v[3] < time_zscore_threshold and v[4] < levenshtein_zscore_threshold}
    else:
        filtered = metrics
    colluding_students = set()
    for k in filtered.keys():
        colluding_students.add(int(k.split("->")[0]))
        colluding_students.add(int(k.split("->")[1]))
    st.write(f"Number of colluding students: {len(colluding_students)}")

    visualize_groups(filtered)

    time_dict = {k: v[3] for k, v in filtered.items()}
    levenshtein_dict = {k: v[4] for k, v in filtered.items()}

    # st.header("Boxplot of Z-Scores")
    # st.write("Below are boxplots showing the distribution of Time and Levenshtein Z-Scores. "
    #          "The boxplots provide a visual summary of the key statistics, including the median (the thick black line), "
    #          "quartiles (the bottom and top of the box), and potential outliers (the points outside of the box). "
    #          "The box itself represents the interquartile range (IQR), which is the range of values between the 25th and 75th percentile "
    #          "of the data. The whiskers (the thin lines extending from the box) represent the range of values that are within 1.5 times the IQR "
    #          "from the median. Any points outside of the whiskers are considered outliers and may indicate unusual behavior. "
    #          "In this context, the boxplot can be used to quickly identify the range of Z-Scores that most relations fall within, "
    #          "and to identify any outliers that may be of interest.")


    # save_boxplot([list(time_dict.values()), list(levenshtein_dict.values())], ["Temporal Dim.", "Content Dim."], "boxplot.pdf")

    return filtered


if __name__ == '__main__':

    st.title("Collusion Detection via Z-Scoring")
    st.write(
        "This demo uses Z-scoring to identify pairs of students that have submitted similar assignments. It calculates the time and levenshtein distances between all pairs of students and computes the Z-scores for each pair. The Z-scores are then used to identify pairs with exceptionally high similarity.")

    with st.sidebar:
        st.title("Collusion Detection via Z-Scoring")
        file_upload_abs = st.file_uploader("Upload a text file with absolute time and levenshtein distances",
                                           type="txt")

        st.write("Select the threshold for identifying colluding students:")

        show_method = st.radio("Analysis", ("Top-k", "No Filter", "Filter"), index=2, key="show_method")
        topk = show_method == "Top-k"
        no_filter = show_method == "No Filter"
        filter = show_method == "Filter"
        time_zscore_threshold = st.slider("Time Z-Score Threshold", min_value=-3.0, max_value=0.0, value=-1.5, step=0.1,
                                          help="Choose the threshold for identifying colluding students based on time z-scores. A higher number will identify more colluding students.",
                                          disabled=not filter)
        levenshtein_zscore_threshold = st.slider("Levenshtein Z-Score Threshold", min_value=-3.0, max_value=0.0,
                                                 value=-2.2, step=0.1,
                                                 help="Choose the threshold for identifying colluding students based on levenshtein z-scores.  A higher number will identify more colluding students.",
                                                 disabled=not filter)
        num_top = st.slider("Number of top relations to show", 1, 500, 50, 1, disabled=not topk)
        if not topk:
            num_top = None

        if no_filter:
            levenshtein_zscore_threshold = None
            time_zscore_threshold = None

        run = st.button("Run")

    if run:
        if file_upload_abs is not None:
            file_contents = file_upload_abs.read().decode("utf-8").splitlines()
            time_zscores = []
            levenshtein_zscores = []
            abs_time_dists = []
            abs_lev_dists = []
            student_rels = []
            metrics = dict()
            filtered = dict()
            for line in file_contents:
                ids, dists = line.split(":")
                f, t = map(int, ids.split("->"))
                abs_time_dist, abs_lev_dist, time_lev_corr, time_zscore, levenshtein_zscore = map(float,
                                                                                                  dists.split(","))
                metrics["->".join([str(f), str(t)])] = (
                abs_time_dist, abs_lev_dist, time_lev_corr, time_zscore, levenshtein_zscore)
                student_rels.append((f, t))
                time_zscores.append(time_zscore)
                levenshtein_zscores.append(levenshtein_zscore)
                abs_time_dists.append(abs_time_dist)
                abs_lev_dists.append(abs_lev_dist)

            # Remove duplicates from metrics
            metrics_without_duplicates = {k: v for k, v in metrics.items() if "->".join(sorted(k.split("->"))) == k}
            metrics = metrics_without_duplicates
            ranked_metrics = sorted(metrics.items(), key=lambda x: (x[1][3], x[1][4]))
            ranked_metrics = dict(ranked_metrics)
            # print(ranked_metrics)
            all_ranked = []
            for k, v in ranked_metrics.items():
                all_ranked.append(int(k.split("->")[0]))
                all_ranked.append(int(k.split("->")[1]))
            unique_ranked = list(dict.fromkeys(all_ranked))
            print(unique_ranked)

            if topk:
                sorted_metrics_dict = get_ranking(metrics)
            else:
                sorted_metrics_dict = get_z_scores(metrics, time_zscore_threshold, levenshtein_zscore_threshold)

            if not no_filter:
                student_ids = set()
                for k in sorted_metrics_dict.keys():
                    student_ids.update(k.split("->"))

                ranked_ids_rel_students = list(set(map(int, student_ids)))
                st.write(f"Total number of retrieved students: {len(ranked_ids_rel_students)}")
                # Insert ground truth ids here
                groundtruth_ids = [1,3]  # new
                ranked_ids_cut = ranked_ids_rel_students[:len(groundtruth_ids)]
                overlap = len(set(ranked_ids_rel_students).intersection(set(groundtruth_ids)))
                recall = overlap / len(groundtruth_ids)
                st.info(f"Recall: {recall:.4f}")
                precision = overlap / len(ranked_ids_rel_students)
                st.info(f"Precision: {precision:.4f}")
                f1 = 2 * (precision * recall) / (precision + recall)
                st.info(f"F1: {f1:.4f}")
                utility = calculate_utility_score(groundtruth_ids, unique_ranked, num_top)
                st.write(f"Utility: {utility:.4f}")
                ndcg = calculate_ndcg(groundtruth_ids, unique_ranked, num_top)
                st.write(f"NDCG: {ndcg:.4f}")

