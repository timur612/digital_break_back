import json
from os import listdir
from os.path import isfile, join
import pandas as pd


def all_json_to_df(all_path) -> pd.DataFrame:
    """

    :param all_path:
    :return:
    """
    all_files = [f for f in listdir(all_path) if isfile(join(all_path, f))]

    data_raw_all = []
    for a_file in all_files:
        try:
            with open("all/" + a_file) as f:
                q_a = json.loads(f.read())
        except:
            # print(a_file)
            pass
        q = q_a["question"]

        for a in q_a["answers"]:
            length_a = len(a["answer"])
            data_raw_all.append([a["answer"], q, length_a])
    return pd.DataFrame(data_raw_all, columns=["answer", "question", "len_answer"])

def file_2_df(file):
    answers = []
    with open(file, encoding='utf_8') as f:
        q_a = json.loads(f.read())
        q = q_a["question"]
        # print(q)
        for a in q_a["answers"]:
            answers.append([a["answer"], q])
    return pd.DataFrame(answers, columns=['answer', 'question'])