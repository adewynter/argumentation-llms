# Broad helper functions for evaluation (e.g., printing the number of examples)
# Some other functions aren't used and are here just for completeness.

import random
import numpy as np
from sklearn.metrics import f1_score
import json
from prompt_utils import collator

random.seed(123)
np.random.seed(123)

def print_am_baseline(processed):
    def get_response_idxs(actuals_bio, prompt):
        prediction = [random.randint(0,2) for _ in range(len(actuals_bio))]
        return prediction, actuals_bio, None

    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]

    failed = []
    num_tries = 5
    total_pads = 0

    for i, pt in enumerate(processed):
        for trial in range(num_tries):
            preds, actuals, switch = get_response_idxs(pt.review_bio, pt.review)
            tries[trial] += preds
            gtruths[trial] += actuals

    for i, pt in enumerate(processed):
        for trial in range(num_tries):
            preds, actuals, switch = get_response_idxs(pt.reply_bio, pt.reply)
            tries[trial] += preds
            gtruths[trial] += actuals

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(tri), np.array(gtuth), labels=[0,1,2], average="micro")*100.
        f1_avg_tries[trial] = round(f1, 2)


    arr = f1_avg_tries
    _arr = np.array(arr)
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))


def print_ape_baseline(processed):
    def get_response_idxs(tags):
        responses = []
        for row in tags:
            _res = [random.randint(0,2) for _ in range(len(row))]
            resp = [1 if i > 0 else 0 for i in _res]
            responses.append(resp)
        return responses


    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]
    failed = []
    num_tries = 5
    switches, points, total_pads = 0, 0, 0

    for i, (proc, response) in enumerate(zip(processed, processed)):
        for trial in range(num_tries):
            responses = get_response_idxs(proc.tags)
            for resp, tag_row in zip(responses, proc.tags):
                tries[trial] += resp
                gtruths[trial] += list(tag_row.numpy())

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(tri), np.array(gtuth))*100.
        f1_avg_tries[trial] = round(f1, 2)

    arr = f1_avg_tries #[round(f*100./points, 2) for f in f1_avg_tries]
    _arr = np.array(arr)
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))


def print_ape_x_baseline(processed):
    def get_response_idxs(tags):
        responses = []
        for row in tags:
            responses.append([random.randint(0,1) for _ in range(len(row))])
        return responses


    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]
    failed = []
    num_tries = 5
    switches, points, total_pads = 0, 0, 0

    for i, (proc, response) in enumerate(zip(processed, processed)):
        for trial in range(num_tries):
            responses = get_response_idxs(proc.tags)
            for resp, tag_row in zip(responses, proc.tags):
                tries[trial] += resp
                gtruths[trial] += list(tag_row.numpy())

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(tri), np.array(gtuth))*100.
        f1_avg_tries[trial] = round(f1, 2)

    arr = f1_avg_tries
    _arr = np.array(arr)
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))


def get_example_sizes_ape(fname_review, processed, suffix="APE/"):
    reviews = [json.loads(l) for l in open(suffix + fname_review, "r", encoding="utf-8").readlines()]
    examples = []
    for pt in reviews:
        examples.append(pt["prompt"].count("Answer:"))
    return examples


def get_example_sizes_am(fname_review, fname_response, processed, suffix="AM/"):
    reviews = [json.loads(l) for l in open(suffix + fname_review, "r", encoding="utf-8").readlines()]
    replies = [json.loads(l) for l in open(suffix + fname_response, "r", encoding="utf-8").readlines()]
    examples = []
    for pt in reviews:
        examples.append(pt["prompt"].count("Response:"))
    for pt in replies:
        examples.append(pt["prompt"].count("Response:"))

    return examples