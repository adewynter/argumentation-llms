# Code for printing concrete and symbolic scoring for AM.
# Each main method ("print_responses_etc") calls in the parsing functions defined here,
# and returns an array of responses and a dictionary with each prediction for statistical analysis.

from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
import json

from loading_code_rrv2 import *


def get_response_idxs_concrete_am(actuals_bio, prompt, responses, text, is_zero_shot=False):
    sentences = [l.strip() for l in responses.replace("-", "").split("\n")[1:]]
    _passage = prompt.split("Passage:")[-1].replace("|passage start|", "").replace("-", "").replace("|passage end|", "")
    passage = [p.strip() for p in _passage.split("\n") if p.strip() != ""]
    response = [r.replace("-", "").strip() for r in responses.split("\n")[1:]]

    actuals = [label2idx["O"] for _ in text]

    prediction = []
    ptr = 0
    for s in passage:
        if s == "":
            continue
        # Can't be shorter
        if ptr >= len(response):
            break
        # Can't be longer
        if ptr >= len(actuals):
            break

        _s = response[ptr].replace("|START|", "").strip()
        if s == _s:
            if "|START|" in response[ptr]:
                prediction.append(label2idx["B"])
            else:
                prediction.append(label2idx["I"])
            ptr += 1
        else:
            prediction.append(label2idx["O"])
    
    if len(prediction) < len(actuals):
        diff = len(actuals) - len(prediction)
        for i in range(diff):
            prediction.append(label2idx["O"])
    if len(prediction) > len(actuals):
        prediction = prediction[:len(actuals)]
    
    return prediction, actuals_bio


def get_response_idxs_cot_concrete_am(actuals_bio, prompt, responses, text, is_zero_shot=False):
    response = [r.replace("-", "").strip() for r in responses.split("\n")[1:]]
    
    prediction = [label2idx["O"] for _ in text]
    ptr = 0
    for s in response:
        if ptr > len(prediction) - 1:
            break
        if s == "":
            continue
        if "we skip it" in s.lower():
            prediction[ptr] = label2idx["O"]
            ptr += 1 
            continue
        if "|START|" in s:
            prediction[ptr] = label2idx["B"]
            ptr += 1 
            continue
        if "still reading an argument" in s.lower():
            prediction[ptr] = label2idx["I"]
            ptr += 1 
            continue
        if "so the answer should be" in s.lower():
            break

    return prediction, actuals_bio


def print_responses_concrete_am(fname_review, fname_response, processed, is_cot=False, is_zero_shot=False, use_new=False, collate=True, debug=False):
    
    suff = "AM/" if not is_cot else ""
    reviews = [json.loads(l) for l in open(suff + fname_review, "r", encoding="utf-8").readlines()]
    replies = [json.loads(l) for l in open(suff + fname_response, "r", encoding="utf-8").readlines()]
    reviews.sort(key=lambda x:x["index"])
    replies.sort(key=lambda x:x["index"])

    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]

    failed = []
    num_tries = 5
    perfs = []

    for i, (a_pt, pt) in enumerate(zip(processed, reviews)):
        if debug and i > 5:
            break
        text = a_pt.review
        for trial in range(num_tries):
            if "try" + str(trial) not in pt["model_responses"]:
                failed.append(("rev", i, trial))
            else:
                if is_cot:
                    preds, actuals = get_response_idxs_cot_concrete_am(a_pt.review_bio, pt["prompt"], pt["model_responses"]["try" + str(trial)], text, is_zero_shot)
                else:
                    if use_new:
                        preds, actuals = get_response_idxs_new_concrete_am(a_pt.review_bio, pt["prompt"], pt["model_responses"]["try" + str(trial)], text)
                    else:
                        preds, actuals = get_response_idxs_concrete_am(a_pt.review_bio, pt["prompt"], pt["model_responses"]["try" + str(trial)], text)
                tries[trial] += preds
                gtruths[trial] += actuals
                perfs.append(("rev", i, trial, f1_score(np.array(actuals), np.array(preds), labels=[0,1,2], average="micro")*100.))
    
    for i, (a_pt, pt) in enumerate(zip(processed, replies)):
        if debug and i > 5:
            break
        text = a_pt.reply
        for trial in range(num_tries):
            if "try" + str(trial) not in pt["model_responses"]:
                failed.append(("rep", i, trial))
            else:
                if is_cot:
                    preds, actuals = get_response_idxs_cot_concrete_am(a_pt.reply_bio, pt["prompt"], pt["model_responses"]["try" + str(trial)], text, is_zero_shot)
                else:
                    if use_new:
                        preds, actuals = get_response_idxs_new_concrete_am(a_pt.reply_bio, pt["prompt"], pt["model_responses"]["try" + str(trial)], text)
                    else:
                        preds, actuals = get_response_idxs_concrete_am(a_pt.reply_bio, pt["prompt"], pt["model_responses"]["try" + str(trial)], text)
                tries[trial] += preds
                gtruths[trial] += actuals
                perfs.append(("rev", i, trial, f1_score(np.array(actuals), np.array(preds), labels=[0,1,2], average="micro")*100.))

    raw_preds = {"predictions": tries, "actuals": gtruths}
    
    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(gtuth), np.array(tri), labels=[0,1,2], average="micro")*100.
        f1_avg_tries[trial] = round(f1, 2)

    if collate:
        perfs.sort(key= lambda x:x[-1])
        print(perfs[:5])

    arr = f1_avg_tries #[round(f*100./points, 2) for f in f1_avg_tries]
    _arr = np.array(arr)
    if collate:
        print(len(failed))
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))
    return round(_arr.mean(), 2), round(_arr.std(), 2), raw_preds



def get_response_idxs_indices_inline_cot_symbolic_am(actuals_bio, responses, text, symbols=True, is_zero_shot=False):
    is_reading = False
    response = [label2idx["O"] for _ in range(len(text))]
    
    if "|start of answer|" in responses:
        ix = responses.index("|start of answer|")
        responses = responses[ix:]
        
    if is_zero_shot: # GPT-3 outputs (a) - b -c d \n (e) -f -g... in zero-shot
        responses = responses.replace("-", "\n")
        
    ix = 0
    for line in responses.replace("-", "").replace("Response:", "").strip().split("\n"):
        line = line.strip()
        last_element = line.split(" ")[-1].replace(".", "").replace("\"", "").strip()
        if "answer should be" in line:
            break
        if symbols:
            if ix > len(response) - 1:
                break            
            if last_element in ["B", "I", "O"]:
                response[ix] = label2idx[last_element]
                ix += 1
            else:
                continue
        else:
            if "we skip it" in line:
                ix += 1
                continue
            else:
                if "(" in last_element and ")" in last_element:
                    last_element = last_element.replace("(", "").replace(")", "")
                    label = label2idx["B"]
                else:
                    label = label2idx["I"]

                try:
                    idx = int(last_element.strip())
                    response[idx] = label
                    ix += 1
                except:
                    continue

    return response, actuals_bio, 0


def get_response_idxs_indices_inline_symbolic_am(actuals_bio, responses, text, is_zero_shot=False):
    is_reading = False
    response = [label2idx["O"] for _ in range(len(text))]
    
    if "|start of answer|" in responses:
        ix = responses.index("|start of answer|")
        responses = responses[ix:]
        
    if is_zero_shot: # GPT-3 outputs (a) - b -c d \n (e) -f -g... in zero-shot
        responses = responses.replace("-", "\n")
    
    for line in responses.replace("-", "").replace("Response:", "").strip().split("\n"):
        line = line.strip()
        first_element = line.split(" ")[0].strip()
        if first_element.startswith("("):
            try:
                is_reading = True
                idx = int(first_element.replace("(", "").replace(")", "").strip())
                response[idx] = label2idx["B"]
            except:
                continue
        else:
            if first_element.strip().isnumeric():
                try:
                    idx = int(first_element.strip())
                    response[idx] = label2idx["I"]
                except:
                    continue
            else:
                continue
    return response, actuals_bio, 0


def get_response_idxs_indices_symbolic_am(actuals_bio, responses, text):
    is_reading = False
    response = [label2idx["O"] for _ in range(len(text))]
    for line in responses.replace("-", "").replace("Response:", "").strip().split("\n"):
        if "(" in line and ")" in line:
            try:
                is_reading = True
                idx = int(line.replace("(", "").replace(")", "").strip())
                response[idx] = label2idx["B"]
            except:
                continue
        else:
            if line.strip().isnumeric():
                if is_reading:
                    try:
                        idx = int(line.strip())
                        response[idx] = label2idx["I"]
                    except:
                        continue
            else:
                continue
    return response, actuals_bio, 0

def get_response_idxs_new_symbolic_am(actuals_bio, responses, text):
    response = [label2idx["O"] for _ in range(len(text))]
    is_reading=False
    for idx, line in enumerate(responses.replace("-", "").replace("Response:", "").strip().split("\n")):
        if "B" in line:
            try:
                is_reading = True
                response[idx] = label2idx["B"]
            except:
                continue
        else:
            if "I" in line:
                if is_reading:
                    try:
                        response[idx] = label2idx["I"]
                    except:
                        continue
            else:
                continue
    return response, actuals_bio, 0

def get_response_idxs_symbolic_am(actuals_bio, responses, text):
    sentences = [l.strip() for l in responses.replace("-", "").replace("Response:", "").strip().split("\n")[1:]]
    response = [label2idx[r] if r in label2idx else None for r in sentences]
    pads = 0
    # Handle predictions that are too short
    if len(text) > len(response):
        pads += 1
        diff = len(text) - len(response)
        for i in range(diff):
            response.append(label2idx["O"])
    prediction = []
    for i in range(len(actuals_bio)):
        if response[i] is None:
            response[i] = label2idx["O"]
        prediction.append(response[i])
    return prediction, actuals_bio, pads


def print_responses_symbolic_am(fname_review, fname_response, processed, collate=True, is_numbers=False, is_inline=False, is_cot=False, 
                    symbols=False, is_zero_shot=False):
    
    suffix = "" if is_cot else "AM/"
    reviews = [json.loads(l) for l in open(suffix + fname_review, "r", encoding="utf-8").readlines()]
    replies = [json.loads(l) for l in open(suffix + fname_response, "r", encoding="utf-8").readlines()]
    reviews.sort(key=lambda x:x["index"])
    replies.sort(key=lambda x:x["index"])
    if collate:
        print(len(reviews), len(replies), len(processed))

    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]

    failed = []
    num_tries = 5
    total_pads = 0
    points = 0
    perfs = []

    for i, (a_pt, pt) in enumerate(zip(processed, reviews)):
        if pt["index"] != i:
            print("WARN! {}".format(i))
            break
        text = a_pt.review
        for trial in range(num_tries):
            points += 1
            if "try" + str(trial) not in pt["model_responses"]:
                failed.append(("rev", i, trial))
            else:
                if is_numbers:
                    if is_inline:
                        preds, actuals, pads = get_response_idxs_indices_inline_symbolic_am(a_pt.review_bio, pt["model_responses"]["try" + str(trial)], text,
                                                                                is_zero_shot=is_zero_shot)
                    else:
                        preds, actuals, pads = get_response_idxs_indices_symbolic_am(a_pt.review_bio, pt["model_responses"]["try" + str(trial)], text)
                elif is_cot:
                    preds, actuals, pads = get_response_idxs_indices_inline_cot_symbolic_am(a_pt.review_bio, pt["model_responses"]["try" + str(trial)], text,
                                                                                symbols=symbols, is_zero_shot=is_zero_shot)
                else:
                    preds, actuals, pads = get_response_idxs_new_symbolic_am(a_pt.review_bio, pt["model_responses"]["try" + str(trial)], text)
                tries[trial] += preds
                gtruths[trial] += actuals
                perfs.append(("rev", i, trial, f1_score(np.array(actuals), np.array(preds), labels=[0,1,2], average="micro")*100.))

    for i, (a_pt, pt) in enumerate(zip(processed, replies)):
        text = a_pt.reply
        if pt["index"] != i:
            print("WARN! {}".format(i))
            break
        for trial in range(num_tries):
            points += 1
            if "try" + str(trial) not in pt["model_responses"]:
                failed.append(("rep", i, trial))
            else:
                if is_numbers:
                    if is_inline:
                        preds, actuals, pads = get_response_idxs_indices_inline_symbolic_am(a_pt.reply_bio, pt["model_responses"]["try" + str(trial)], text, is_zero_shot=is_zero_shot)
                    else:
                        preds, actuals, pads = get_response_idxs_indices_symbolic_am(a_pt.reply_bio, pt["model_responses"]["try" + str(trial)], text)
                elif is_cot:
                    preds, actuals, pads = get_response_idxs_indices_inline_cot_symbolic_am(a_pt.reply_bio, pt["model_responses"]["try" + str(trial)], text,
                                                                                symbols=symbols, is_zero_shot=is_zero_shot)
                else:
                    preds, actuals, pads = get_response_idxs_new_symbolic_am(a_pt.reply_bio, pt["model_responses"]["try" + str(trial)], text)
                tries[trial] += preds
                gtruths[trial] += actuals
                perfs.append(("rep", i, trial, f1_score(np.array(actuals), np.array(preds), labels=[0,1,2], average="micro")*100.))

    raw_preds = {"predictions": tries, "actuals": gtruths}

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(gtuth), np.array(tri), labels=[0,1,2], average="micro")*100.
        f1_avg_tries[trial] = round(f1, 2)

    perfs.sort(key=lambda x:x[-1])
    if collate:
        print("perfs", perfs[:5])

    arr = f1_avg_tries #[round(f*100./points, 2) for f in f1_avg_tries]
    _arr = np.array(arr)
    if collate:
        print(len(failed))
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))
    return round(_arr.mean(), 2), round(_arr.std(), 2), raw_preds
    

def get_response_idxs_amr(actuals_bio, responses):
    sentences = [l.strip() for l in responses.replace("-", "").replace("Response:", "").strip().split("\n")[1:]]
    response = [label2idx[r] if r in label2idx else None for r in sentences]
    pads = 0
    # Handle predictions that are too short
    if len(actuals_bio) > len(response):
        pads += 1
        diff = len(actuals_bio) - len(response)
        for i in range(diff):
            response.append(label2idx["O"])
    prediction = []
    for i in range(len(actuals_bio)):
        if response[i] is None:
            response[i] = label2idx["O"]
        prediction.append(response[i])
    return prediction, actuals_bio, pads


def print_responses_amr(fname_review, fname_response, processed, collate=True):
    
    reviews = [json.loads(l) for l in open("AM/" + fname_review, "r", encoding="utf-8").readlines()]
    replies = [json.loads(l) for l in open("AM/" + fname_response, "r", encoding="utf-8").readlines()]
    #reviews = collator("AM/" + fname_review, processed, returnit=True, log=collate)
    #replies = collator("AM/" + fname_response, processed, returnit=True, log=collate)
    #reviews.sort(key=lambda x:x["index"])
    #replies.sort(key=lambda x:x["index"])
    #if collate:
    #    print(len(reviews), len(replies), len(processed))

    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]

    failed = []
    num_tries = 5
    total_pads = 0
    points = 0
    perfs = []

    for i, (a_pt, pt) in enumerate(zip(processed, reviews)):
        if pt["index"] != i:
            print("WARN! {}".format(i))
            break
        for trial in range(num_tries):
            points += 1
            if "try" + str(trial) not in pt["model_responses"]:
                failed.append(("rev", i, trial))
            else:
                preds, actuals, pads = get_response_idxs_amr(a_pt.review_bio, pt["model_responses"]["try" + str(trial)])
                tries[trial] += preds
                gtruths[trial] += actuals
                perfs.append(("rev", i, trial, f1_score(np.array(actuals), np.array(preds), labels=[0,1,2], average="micro")*100.))

    for i, (a_pt, pt) in enumerate(zip(processed, replies)):
        if pt["index"] != i:
            print("WARN! {}".format(i))
            break
        for trial in range(num_tries):
            points += 1
            if "try" + str(trial) not in pt["model_responses"]:
                failed.append(("rep", i, trial))
            else:
                preds, actuals, pads = get_response_idxs_amr(a_pt.reply_bio, pt["model_responses"]["try" + str(trial)])
                tries[trial] += preds
                gtruths[trial] += actuals
                perfs.append(("rep", i, trial, f1_score(np.array(actuals), np.array(preds), labels=[0,1,2], average="micro")*100.))

    raw_preds = {"predictions": tries, "actuals": gtruths}

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(gtuth), np.array(tri), labels=[0,1,2], average="micro")*100.
        f1_avg_tries[trial] = round(f1, 2)

    perfs.sort(key=lambda x:x[-1])
    if collate:
        print("perfs", perfs[:5])

    arr = f1_avg_tries #[round(f*100./points, 2) for f in f1_avg_tries]
    _arr = np.array(arr)
    if collate:
        print(len(failed))
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))
    return round(_arr.mean(), 2), round(_arr.std(), 2), raw_preds

