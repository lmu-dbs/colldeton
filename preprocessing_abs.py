import itertools
import os
import time
import editdistance
import numpy as np
import pandas as pd
import pm4py
import textdistance

import seaborn as sns
from matplotlib import pyplot as plt


def import_log_csv(filepath):
    pd.set_option('display.max_columns', None)
    df = pd.read_csv(filepath)
    dataframe = df[["userId", "taskId", "timeStamp", "additionaldata"]]
    dataframe.rename(columns={'userId': 'case:concept:name', 'taskId': 'concept:name', 'timeStamp': 'time:timestamp',
                              'additionaldata': 'answers'}, inplace=True)
    dataframe = pm4py.format_dataframe(dataframe, case_id='case:concept:name', activity_key='concept:name',
                                       timestamp_key='time:timestamp')
    for i, row in dataframe.iterrows():
        answer = row['answers']
        answer = answer.replace('\"', '').replace('[', '').replace(']', '').replace('}', '').split(":")[1]
        answer_parts = answer.split(",")
        dataframe.at[i, 'answers'] = answer_parts
    return dataframe


# Returns distance based on initial strict conditions
def get_strict_distance(submission_combination, time_threshold):
    if (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds() < time_threshold \
            and editdistance.eval(submission_combination[0][1], submission_combination[1][1]) == 0:
        return 0
    else:
        return 1


def get_text_based_distance(submission_combination):
    if editdistance.eval(submission_combination[0][1], submission_combination[1][1]) == 0:
        return 0
    else:
        return 1


def get_absolute_distances(submission_combination):
    time_dist = (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds()
    # 0 means equal, 1 different
    answer_dist = textdistance.Levenshtein().normalized_distance(submission_combination[0][1],
                                                                 submission_combination[1][1])
    return time_dist, answer_dist


# Returns distance based on whether the submission happened within a time range TIME_THRESHOLD only
def get_time_based_distance(submission_combination, time_threshold):
    if (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds() < time_threshold:
        return 0
    else:
        return 1


# Return inverse levenshtein distance weighted by time
def get_weighted_distance(submission_combination, time_threshold):
    time_diff = (abs(submission_combination[0][0] - submission_combination[1][0])).total_seconds()
    lev = textdistance.Levenshtein()
    if time_diff > time_threshold:
        return 1
    else:
        editdist = lev.normalized_distance(submission_combination[0][1], submission_combination[1][1])
        return editdist


# Add distance methods for different time ranges
def calculate_task_distance(t1, t2, dist, time_threshold=60):
    t1_tuple = [tuple(x) for x in t1[['time:timestamp', 'answers']].to_numpy()]
    t2_tuple = [tuple(x) for x in t2[['time:timestamp', 'answers']].to_numpy()]
    two_tuples = [t1_tuple, t2_tuple]
    combinations = [p for p in itertools.product(*two_tuples)]
    answer_distances = []
    time_distances = []
    for comb in combinations:
        time_dist, answer_dist = get_absolute_distances(comb)
        time_distances.append(time_dist)
        answer_distances.append(answer_dist)
    return min(time_distances), min(answer_distances)


def sum_norm(min_task_dists):
    return [float(i) / sum(min_task_dists) for i in min_task_dists]


def max_norm(min_task_dists):
    return [float(i) / max(min_task_dists) for i in min_task_dists]


def calculate_user_distance(df, u1, u2, tasks, dist, time_threshold):
    trace_u1 = df.loc[df['case:concept:name'] == u1]
    trace_u2 = df.loc[df['case:concept:name'] == u2]
    min_task_time_dists = []
    min_task_answer_dists = []
    for task in tasks:
        task_trace_u1 = trace_u1.loc[trace_u1['concept:name'] == task]
        task_trace_u2 = trace_u2.loc[trace_u2['concept:name'] == task]
        if not task_trace_u1.empty and not task_trace_u2.empty:
            mint_task_time_dist, min_task_answer_dist = calculate_task_distance(task_trace_u1, task_trace_u2, dist,
                                                                                time_threshold=time_threshold)
            min_task_time_dists.append(mint_task_time_dist)
            min_task_answer_dists.append(min_task_answer_dist)
    corr = np.corrcoef(min_task_time_dists, min_task_answer_dists)[0,1]
    # print(f"Correlation between time and answer distances: {corr}")
    return sum(min_task_time_dists), sum(min_task_answer_dists), corr


# Create a list of timestamps for each user following the order of taskIDs
def get_ordered_timestamps(group, task_order):
    # Sort the DataFrame by timestamp in descending order
    group = group.sort_values(by='time:timestamp', ascending=False)
    # Drop duplicates, keeping the first occurrence (greatest timestamp)
    df_unique = group.drop_duplicates(subset='concept:name').reset_index(drop=True)

    # Sort the group by the order defined in task_order
    ordered_timestamps = df_unique.set_index('concept:name').reindex(task_order)['time:timestamp'].tolist()
    ordered_timestamps = [int(ts.timestamp() * 1000) if pd.notna(ts) else -1 for ts in ordered_timestamps]
    return ordered_timestamps


def create_clustermap(df):
    # print(df)
    user_ts = dict()
    task_list = log['concept:name'].unique()
    user_list = log['case:concept:name'].unique()
    print(f"Task List: {task_list}\nStudent List: {user_list}\n")
    grouped = df.groupby('case:concept:name')
    for name, group in grouped:
        user_ts[name] = get_ordered_timestamps(group, task_list)
    print(user_ts)
    df = pd.DataFrame(user_ts, index=task_list)
    sns.clustermap(df, xticklabels=True, yticklabels=True, cmap="coolwarm", annot=False)
    plt.xlabel('Students')
    plt.ylabel('Tasks')
    plt.show()


def compute_distance_matrix(log, dist="strict", time_threshold=60):
    counter = 0
    user_count = 0
    user_dists = dict()
    task_list = log['concept:name'].unique()
    user_list = log['case:concept:name'].unique()
    combs = (len(user_list) - 1) ** 2
    start = time.time()
    for u1 in user_list:
        user_count += 1
        print(f"INFO: Calculating user {user_count} of {USER_SAMPLE_SIZE}")
        for u2 in user_list:
            counter += 1
            if counter % 1000 == 0:
                end = time.time()
                print(f"INFO: {end - start:.2f} seconds elapsed after {counter} combinations.")
                print(f"INFO: Calculating user combination {counter}/{combs}")
                start = time.time()
            if u1 != u2:
                user_dists[str(u1) + "->" + str(u2)] = calculate_user_distance(log, u1, u2, task_list, dist,
                                                                               time_threshold)
        if USER_SAMPLE_SIZE is not None and user_count >= USER_SAMPLE_SIZE:
            break

    return user_dists


def create_folder_and_save_file(folder_name, file_name, user_dists, user_zscores):
    # Create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    # Save a file in the folder
    with open(os.path.join(folder_name, file_name), 'w') as f:
        for k, v in user_dists.items():
            # abs time dist, abs levenshtein dist, corr, time zscore, levenshtein zscore
            f.write(k + ":" + str(v[0]) + "," + str(v[1]) + "," + str(v[2]) + "," + str(user_zscores[k][0]) + "," + str(user_zscores[k][1]) + "\n")
        print(f"File '{file_name}' saved in '{folder_name}'.")


OUTPUT_DIR = "preproc_output/"
log_dict = {"exam": "/example_log.csv"}

if __name__ == '__main__':
    DIST_FUNC = 'abs'
    # TIME_THRESHOLDS = [30, 60, 120, 180, 300]  # in seconds
    TIME_THRESHOLDS = [60]  # in seconds
    SAMPLE_SIZE = 50
    USER_SAMPLE_SIZE = None

    for i, t in enumerate(TIME_THRESHOLDS):
        for title, path in log_dict.items():
            t_start = time.time()
            print(f"Preprocessing log {i} -- {title}...")
            log = import_log_csv(path)
            user_dists = compute_distance_matrix(log, DIST_FUNC, t)

            user_zscores = dict()
            mean_time_dist = np.mean([v[0] for v in user_dists.values()])
            std_time_dist = np.std([v[0] for v in user_dists.values()])
            mean_answer_dist = np.mean([v[1] for v in user_dists.values()])
            std_answer_dist = np.std([v[1] for v in user_dists.values()])
            for k, v in user_dists.items():
                user_zscores[k] = ((v[0] - mean_time_dist) / std_time_dist, (v[1] - mean_answer_dist) / std_answer_dist)
            # print(user_dists)
            print(f"INFO: Distance computation took {time.time() - t_start}")
            OUTPUT_DIR = title + "/"
            OUTPUT_FILENAME = f"{DIST_FUNC}_distmatrix_{t}sec.txt"
            create_folder_and_save_file(OUTPUT_DIR, OUTPUT_FILENAME, user_dists, user_zscores)
