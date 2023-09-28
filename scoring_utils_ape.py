# Code for printing concrete and symbolic scoring for APE.
# Each main method ("print_responses_etc") calls in the parsing functions defined here,
# and returns an array of responses and a dictionary with each prediction for statistical analysis.


from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
import json
import re

from loading_code_rrv2 import *

def fixer_zero(s):
    _s = s.replace("- Argument: ", "")
    _s = fixer(_s)
    return _s


fixer = lambda x: x.replace("- ", "").replace("<sep>", "").replace("|START|", "").strip()


def get_response_idxs_ape_concrete(data_p, model_p, is_zero_shot=False):

    d_review = [fixer(p) for p in data_p.review]
    d_reply = [fixer(p) for p in data_p.reply]
    
    arg_rebs = {}
    this_arg_index = None
    this_reb_index = None
    currently_reading = None
    
    arg_count_zero_shot = 0
    reb_count_zero_shot = 0
    
    def preprocess_zero_shot(txt):
        if re.findall(r'line \d* from the response that is a', txt.strip()) != []:
            return ":".join(txt.split(":")[1:])
        if re.findall(r'line \d* from the review that is a', txt.strip()) != []:
            return ":".join(txt.split(":")[1:])
        return txt
        
    for sentence in model_p.split("\n"):
        if "pairs of argument and rebuttal" in sentence.lower() or sentence == "":
            continue
        if not is_zero_shot:
            if re.search("argument \d", sentence.lower()) is not None:
                this_arg_index = fixer(sentence).replace(":", "").strip().split(" ")[-1]
                this_index = this_arg_index
                currently_reading = "argument"
                continue
            if re.search("rebuttal \d", sentence.lower()) is not None:
                this_reb_index = fixer(sentence).replace(":", "").strip().split(" ")[-1]
                this_index = this_reb_index
                currently_reading = "rebuttal"
                continue
        else:
            if sentence.lower().strip().startswith("- argument"):
                this_arg_index = arg_count_zero_shot
                arg_count_zero_shot += 1
                this_index = str(this_arg_index)
                currently_reading = "argument"
                continue
            if sentence.lower().strip().startswith("- rebuttal"):
                this_reb_index = reb_count_zero_shot
                reb_count_zero_shot += 1
                this_index = str(this_reb_index)
                currently_reading = "rebuttal"
                continue
        if currently_reading is None:
            continue
        if currently_reading == "argument" and this_arg_index is None:
            continue
        if currently_reading == "rebuttal" and this_reb_index is None:
            continue

        if currently_reading + "_" + this_index not in arg_rebs:
            arg_rebs[currently_reading + "_" + this_index] = []
            
        if not is_zero_shot:
            arg_rebs[currently_reading + "_" + this_index].append(fixer(sentence))
        else:
            arg_rebs[currently_reading + "_" + this_index].append(fixer_zero(preprocess_zero_shot(sentence)))
    
    flattened_arg_rebs = {}
    for k,v in arg_rebs.items():
        head, number = k.split("_")
        if number not in flattened_arg_rebs:
            flattened_arg_rebs[number] = {}
        flattened_arg_rebs[number][head] = v
        
    table = [[0 for _ in d_reply] for _ in d_review]

    for k,v in flattened_arg_rebs.items():
        if "argument" in v and "rebuttal" in v:
            idxs = [d_review.index(a) for a in v["argument"] if a in d_review]
            jdxs = [d_reply.index(a) for a in v["rebuttal"] if a in d_reply]
            for i in idxs:
                for j in jdxs:
                    table[i][j] = 1
    return table


def print_responses_ape_concrete(fname, all_processed, collate=True, is_zero_shot=False):
    all_responses = [json.loads(l) for l in open("APE/" + fname, "r", encoding="utf-8").readlines()]
    #if collate:
    #    print(len(all_processed), len(all_responses))
    #all_responses = collator("APE/" + fname, all_processed, returnit=True, log=collate)
    all_responses.sort(key=lambda k:k["index"])
    
    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]
    failed = []
    num_tries = 5
    switches, points, total_pads = 0, 0, 0
    perfs = []

    for i, (processed, response) in enumerate(zip(all_processed, all_responses)):
        for trial in range(num_tries):
            if "try{}".format(trial) not in response["model_responses"]:
                failed.append(i)
                continue
            responses = get_response_idxs_ape_concrete(processed,
                                          response["model_responses"]["try{}".format(trial)],
                                          is_zero_shot=is_zero_shot)
            xtries, xgtruths = [], []
            for resp, tag_row in zip(responses, processed.tags):
                tries[trial] += resp
                gtruths[trial] += list(tag_row.numpy())
            _f1 = f1_score(np.array(xgtruths), np.array(xtries))*100.
            perfs.append((i, trial, _f1))

    perfs.sort(key=lambda x: x[-1])
    if collate:
        print(perfs[:5])

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(gtuth), np.array(tri))*100. #, labels=[0,1], average="micro")
        f1_avg_tries[trial] = round(f1, 2)

    if collate:
        print(points)
    arr = f1_avg_tries #[round(f*100./points, 2) for f in f1_avg_tries]
    _arr = np.array(arr)
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))
    return round(_arr.mean(), 2), round(_arr.std(), 2)


def get_response_idxs_full_ape_symbolic(data_p, model_p, is_zero_shot=False):
    d_review = [fixer(p) for p in data_p.review]
    d_reply = [fixer(p) for p in data_p.reply]
    table = [[0 for _ in d_reply] for _ in d_review]
    
    response = model_p.replace("|end of answer|", "").replace("|start of answer|", "").strip()

    i = 0
    for row in response.split("\n"):
        if i > len(table) - 1:
            break
        if all([e.isnumeric() for e in row.strip().split(" ")]):
            j = 0
            for _e in row.strip().split(" "):
                if j > len(table[0]) - 1:
                    break
                e = _e.replace(",", "").replace("]", "").replace("[", "")
                if e.isnumeric():
                    if int(e) != 0:
                        table[i][j] = 1
                    else:
                        table[i][j] = int(e)
                    j += 1
            i += 1

    return table

def get_response_idxs_zero_shot_ape_symbolic(data_p, model_p, is_zero_shot=False):
    d_review = [fixer(p) for p in data_p.review]
    d_reply = [fixer(p) for p in data_p.reply]
    
    arg_rebs = {}
    this_arg_index = None
    this_reb_index = None
    currently_reading = None
    
    arg_count_zero_shot = 0
    reb_count_zero_shot = 0
    this_index = 0
    
    def this_fixer(txt, currently_reading):
        try:
            if re.search(r'line\s\d*', txt) is not None:
                x = txt.split("line")[1]
                return [int(x.strip().split(" ")[0].strip())]
            if re.search(r'line[|s]\s\d*\-\d*', txt) is not None:
                x = txt.split("lines")[1]
                x = txt.split("lines")[1]
                rng = x.strip().split(" ")[0].strip()
                a, b = rng.split("-")
                return [i for i in range(int(a.strip()), int(b.strip()) + 1)]                
        except:
            pass
        
        # Hasn't found anything, so we'll fall back to the indices from the response
        if txt.strip().startswith("-"):
            txt = txt.strip()[1:]            
        arr = d_review if "argument" else d_reply
        if txt in arr:
            return [arr.index(txt)]
        if fixer(txt) in arr:
            return [arr.index(fixer(txt))]
        return []

    for sentence in model_p.split("\n"):
        if "pairs of argument and rebuttal" in sentence.lower() or sentence == "":
            continue
        if re.search("- argument \d", sentence.lower()) is not None:
            this_arg_index = fixer(sentence).replace(":", "").strip().split(" ")[-1]
            this_index = this_arg_index
            currently_reading = "argument"
            continue
        if re.search("- rebuttal \d", sentence.lower()) is not None:
            this_reb_index = fixer(sentence).replace(":", "").strip().split(" ")[-1]
            this_index = this_reb_index
            currently_reading = "rebuttal"
            continue
        if currently_reading is None:
            continue
        if currently_reading == "argument" and this_arg_index is None:
            continue
        if currently_reading == "rebuttal" and this_reb_index is None:
            continue
        if currently_reading + "_" + this_index not in arg_rebs:
            arg_rebs[currently_reading + "_" + this_index] = []
        arg_rebs[currently_reading + "_" + this_index] += this_fixer(sentence, currently_reading)
    
    flattened_arg_rebs = {}
    for k,v in arg_rebs.items():
        head, number = k.split("_")
        if number not in flattened_arg_rebs:
            flattened_arg_rebs[number] = {}
        flattened_arg_rebs[number][head] = v
        
    table = [[0 for _ in d_reply] for _ in d_review]

    for key in flattened_arg_rebs.keys():
        v = flattened_arg_rebs[key]
        if "argument" in v and "rebuttal" in v:
            idxs = v["argument"] #[d_review.index(a) for a in v["argument"] if a in d_review]
            jdxs = v["rebuttal"] #[d_reply.index(a) for a in v["rebuttal"] if a in d_reply]
            for i in idxs:
                if i >= len(table):
                    continue
                for j in jdxs:
                    if j >= len(table[0]):
                        continue
                    table[i][j] = 1
    return table


def get_response_idxs_numbers_zero_shot_ape_symbolic(data_p, model_p, is_zero_shot=False):
    table = [[0 for _ in range(len(data_p.reply))] for __ in range(len(data_p.review))]
    model_response = model_p.replace("|end of answer|", "").strip()
    
    def this_fixer(txt):
        try:
            if re.search(r'line\s\d*', txt) is not None:
                x = txt.split("line")[1]
                return [int(x.strip().split(" ")[0].strip())]
            if re.search(r'line[|s]\s\d*\-\d*', txt) is not None:
                x = txt.split("lines")[1]
                x = txt.split("lines")[1]
                rng = x.strip().split(" ")[0].strip()
                a, b = rng.split("-")
                return [i for i in range(int(a.strip()), int(b.strip()) + 1)]                
        except:
            return []
        return []

    resps = {}
    for r in model_response.split("\n"):
        if r == "" or not r.startswith("- argument"):
            continue
        try:
            arg, reb = r.split(":")
            arg0, arg1 = arg.split("(")
            argument_index = int(arg0.replace("- argument", "").strip())
        except:
            continue
        rebuttal = this_fixer(reb) #[int(i.strip()) for i in reb.split(" ") if i.isnumeric()]
        argument = this_fixer(arg1) #[int(i.strip()) for i in arg1.replace(")", "").split(" ") if i.isnumeric()]

        resps[str(argument_index)] = {"argument": argument, "rebuttal": rebuttal}
    print(resps)
    for key in resps.keys():
        v = resps[key]
        if "argument" in v and "rebuttal" in v:
            idxs = v["argument"] #[d_review.index(a) for a in v["argument"] if a in d_review]
            jdxs = v["rebuttal"] #[d_reply.index(a) for a in v["rebuttal"] if a in d_reply]
            for i in idxs:
                if i >= len(table):
                    continue
                for j in jdxs:
                    if j >= len(table[0]):
                        continue
                    table[i][j] = 1
    return table



def get_response_idxs_ape_symbolic(data_p, model_p, is_zero_shot=False):
    table = [[0 for _ in range(len(data_p.reply))] for __ in range(len(data_p.review))]
    model_response = model_p.replace("|end of answer|", "").strip()

    resps = {}
    for r in model_response.split("\n"):
        if r == "" or not r.startswith("- argument"):
            continue
        try:
            arg, reb = r.split(":")
            arg0, arg1 = arg.split("(")
            argument_index = int(arg0.replace("- argument", "").strip())
        except:
            continue
        rebuttal = [int(i.strip()) for i in reb.split(" ") if i.isnumeric()]
        argument = [int(i.strip()) for i in arg1.replace(")", "").split(" ") if i.isnumeric()]

        resps[str(argument_index)] = {"argument": argument, "rebuttal": rebuttal}

    for key in resps.keys():
        v = resps[key]
        if "argument" in v and "rebuttal" in v:
            idxs = v["argument"] #[d_review.index(a) for a in v["argument"] if a in d_review]
            jdxs = v["rebuttal"] #[d_reply.index(a) for a in v["rebuttal"] if a in d_reply]
            for i in idxs:
                if i >= len(table):
                    continue
                for j in jdxs:
                    if j >= len(table[0]):
                        continue
                    table[i][j] = 1
    return table


def print_responses_ape_symbolic(fname, all_processed, collate=True, is_zero_shot=False, is_numbers=False, is_full=False):
    all_responses = [json.loads(l) for l in open("APE/" +fname, "r", encoding="utf-8").readlines()]
    #if collate:
    #    print(len(all_processed), len(all_responses))
    #all_responses = collator("APE/" +fname, all_processed, returnit=True, log=collate)
    all_responses.sort(key=lambda k:k["index"])
    
    tries = [[], [], [], [], []]
    gtruths = [[], [], [], [], []]
    f1_avg_tries = [0, 0, 0, 0, 0]
    failed = []
    num_tries = 5
    switches, points, total_pads = 0, 0, 0
    perfs = []

    for i, (processed, response) in enumerate(zip(all_processed, all_responses)):
        for trial in range(num_tries):
            if "try{}".format(trial) not in response["model_responses"]:
                failed.append(i)
                continue
            if is_zero_shot:
                if is_numbers:
                    responses = get_response_idxs_numbers_zero_shot_ape_symbolic(processed,
                                                  response["model_responses"]["try{}".format(trial)],
                                                  is_zero_shot=is_zero_shot)                                    
                else:
                    responses = get_response_idxs_zero_shot_ape_symbolic(processed,
                                                  response["model_responses"]["try{}".format(trial)],
                                                  is_zero_shot=is_zero_shot)                
            else:
                if is_full:
                    responses = get_response_idxs_full_ape_symbolic(processed,
                                                       response["model_responses"]["try{}".format(trial)],
                                                       is_zero_shot=False)
                else:
                    responses = get_response_idxs_ape_symbolic(processed,
                                                  response["model_responses"]["try{}".format(trial)],
                                                  is_zero_shot=is_zero_shot)
            xtries, xgtruths = [], []
            for resp, tag_row in zip(responses, processed.tags):
                tries[trial] += resp
                gtruths[trial] += list(tag_row.numpy())
            _f1 = f1_score(np.array(xgtruths), np.array(xtries))*100.
            perfs.append((i, trial, _f1))

    perfs.sort(key=lambda x: x[-1])
    if collate:
        print(perfs[:5])

    for trial in range(num_tries):
        tri = tries[trial]
        gtuth = gtruths[trial]
        f1 = f1_score(np.array(gtuth), np.array(tri))*100. #, labels=[0,1], average="micro")
        f1_avg_tries[trial] = round(f1, 2)

    if collate:
        print(points)
    arr = f1_avg_tries #[round(f*100./points, 2) for f in f1_avg_tries]
    _arr = np.array(arr)
    print(arr, round(_arr.mean(), 2), "+/-", round(_arr.std(), 2))
    return round(_arr.mean(), 2), round(_arr.std(), 2)


