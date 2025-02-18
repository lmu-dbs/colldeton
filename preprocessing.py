import itertools
import os
import time
import editdistance
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
    distances = []
    for comb in combinations:
        distances.append(eval("get_" + dist + "_distance(comb, time_threshold)"))
    return min(distances)


def sum_norm(min_task_dists):
    return [float(i) / sum(min_task_dists) for i in min_task_dists]


def max_norm(min_task_dists):
    return [float(i) / max(min_task_dists) for i in min_task_dists]


def calculate_user_distance(df, u1, u2, tasks, dist, time_threshold):
    trace_u1 = df.loc[df['case:concept:name'] == u1]
    trace_u2 = df.loc[df['case:concept:name'] == u2]
    min_task_dists = []
    for task in tasks:
        task_trace_u1 = trace_u1.loc[trace_u1['concept:name'] == task]
        task_trace_u2 = trace_u2.loc[trace_u2['concept:name'] == task]
        if not task_trace_u1.empty and not task_trace_u2.empty:
            min_task_dists.append(
                calculate_task_distance(task_trace_u1, task_trace_u2, dist, time_threshold=time_threshold))
    return sum(min_task_dists)


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
    user_dists = dict()
    task_list = log['concept:name'].unique()
    user_list = log['case:concept:name'].unique()
    combs = (len(user_list) - 1) ** 2
    start = time.time()
    for u1 in user_list:
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
    return user_dists


def create_folder_and_save_file(folder_name, file_name, file_content):
    # Check if the folder already exists
    if not os.path.exists(folder_name):
        # Create the folder
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

        # Save a file in the newly created folder
        with open(os.path.join(folder_name, file_name), 'w') as f:
            for k, v in file_content.items():
                f.write(k + ":" + str(v) + "\n")
            print(f"File '{file_name}' saved in '{folder_name}'.")
    else:
        print(f"Folder '{folder_name}' already exists.")
        with open(os.path.join(folder_name, file_name), 'w') as f:
            for k, v in file_content.items():
                f.write(k + ":" + str(v) + "\n")
            print(f"File '{file_name}' saved in '{folder_name}'.")


OUTPUT_DIR = "preproc_output/"
log_dict = {"exam": "/example_log.csv"}

if __name__ == '__main__':
    DIST_FUNC = 'strict'  # "strict", "time_based", or "weighted"
    # TIME_THRESHOLDS = [30, 60, 120, 180, 300]  # in seconds
    TIME_THRESHOLDS = [60]  # in seconds
    SAMPLE_SIZE = 5

    file_path = "durations.txt"
    with open(file_path, "w") as file:
        for i, t in enumerate(TIME_THRESHOLDS):
            for title, path in log_dict.items():
                start_time = time.time()
                print(f"Preprocessing log {i} -- {title}...")
                log = import_log_csv(path)
                user_dists = compute_distance_matrix(log, DIST_FUNC, t)
                file.write(f"___{DIST_FUNC}_distmatrix_{t}sec.txt took {time.time() - start_time} seconds\n")
                OUTPUT_DIR = title + "/"
                OUTPUT_FILENAME = f"___{DIST_FUNC}_distmatrix_{t}sec.txt"
                create_folder_and_save_file(OUTPUT_DIR, OUTPUT_FILENAME, user_dists)
