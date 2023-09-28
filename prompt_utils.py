# Utilities for prompting. We don't use many of these and are here just for completeness.

import json

def collator(fname, processed, writeto=False, returnit=False, log=True, return_redo=False):
    # Due to the failure-prone nature of our API we load the code, sort it, and ensure that no datapoints are missing.
    lines = [json.loads(l) for l in open(fname, "r", encoding="utf-8").readlines()]
    lines.sort(key=lambda x:x["index"])
    if log:
        print("lines from file", len(lines))
    lines.sort(key=lambda x:x["index"])
    seen = {}
    ptr = 0
    for i in range(len(processed)):
        seen[lines[i]["index"]] = lines[i]

    keys = set([l["index"] for l in lines])
    targets = set([i for i in range(len(processed))])
    missing = list(targets - keys)
    missing.sort()
    if log:
        print("missing indices!!:", missing)
    
    #print(len(lines))
    new_lines = []
    seen = {}
    for l in lines:
        if l["model_responses"] == {}:
            continue
        else:
            if l["index"] not in seen:
                seen[l["index"]] = []
                new_lines.append(None)
            seen[l["index"]].append(l)
            #new_lines.append(l)

    keys = [k for k in seen.keys()]
    sorted(keys)
    if log:
        print("number of unique keys", len(keys))
    _keys = set(keys)
    missing = list(targets - _keys)
    if log:
        print("Points with empty response: ", missing, "redo these ones!")
        if return_redo:
            return missing

    lows = {}
    for i in range(len(keys)):
        new_lines[i] = seen[i][-1]
        for j in range(5):
            if "try" + str(j) not in new_lines[i]["model_responses"]:
                if new_lines[i]["index"] not in lows:
                    lows[new_lines[i]["index"]] = []
                lows[new_lines[i]["index"]].append("{}".format(j))

    if log:
        if lows != []:
            print("Entries with missing trial: ")
            for k,z in lows.items():
                print("{}: {}".format(k, ",".join(z)))

    ix = 0

    for i in range(len(new_lines)):
        if ix != new_lines[i]["index"]:
            print(ix)
            ix = new_lines[i]["index"]
        ix += 1
    new_lines.sort(key=lambda x:x["index"])
    if log:
        print("Nones in new_lines", any([x is None for x in new_lines]))
        print("new_lines/seen:", len(new_lines), len(seen))
    if writeto:
        with open(fname, "w", encoding="utf-8") as f:
            [f.write(json.dumps(l) + "\n") for l in new_lines]
    if returnit:
        return new_lines


def load_and_show(fname, prompt=False, ix = 0, trial = 0, returnit=False):
    reviews = [json.loads(l) for l in open(fname).readlines()]
    if prompt:
        print(reviews[ix]["prompt"])
        if returnit:
            return reviews[ix]["prompt"]
    else:
        print(reviews[ix]["model_responses"]["try" + str(trial)])
        if returnit:
            return reviews[ix]["model_responses"]["try" + str(trial)]


def get_arr(arr, point):
    resp = []
    for i, p in enumerate(arr):
        if p != 0:
            resp.append(point.reply[i] + "\n")
    return resp


def extract_spans_from_point_new(point):
    tag = point.tags
    samples_review, samples_response = [], []
    samples_review_dic, samples_response_dic = {}, {}
    
    tmp_arr = []
    last_j, last_i = -1, -1
    state = False
    for i in range(tag.shape[0]):
        if sum(tag[i]) == 0:
            continue
        for j in range(tag.shape[1]):
            if j < last_j and last_j != -1:
                continue
            if tag[i][j] != 0:
                #print(i, j, last_j)
                if not state:
                    last_i = i
                tmp_arr.append(point.review[i] + "\n")
                state = True
                last_j = j
                break
            else:
                if state:
                    samples_review.append(tmp_arr)
                    samples_review_dic[last_i] = tmp_arr
                    samples_response.append(get_arr(tag[last_i], point))
                    samples_response_dic[last_i] = get_arr(tag[last_i], point)
                    tmp_arr = []
                    last_j = -1
                    last_i = -1
                    state = False

    if tmp_arr != []:
        samples_review.append(tmp_arr)
        samples_response.append(get_arr(tag[last_i], point))
        samples_review_dic[last_i] = tmp_arr
        samples_response_dic[last_i] = get_arr(tag[last_i], point)
        
    return samples_review_dic, samples_response_dic


def load_to_example(processed, arr_review, arr_response):
    # Load a set of AM responses and use these instead of test points.
    points = []
    
    clean_response = lambda r: [l.strip() for l in r.replace("-", "").replace("Response:", "").strip().split("\n")[1:]]
    
    def map_to_old_strings(pt, txt, is_review=False):
        actuals_bio = pt.review_bio if is_review else pt.reply_bio
        actuals_txt = pt.review if is_review else pt.reply
        response = [label2idx[r] if r in label2idx else None for r in txt]
        return_arr = []
        if len(actuals_bio) > len(response):
            diff = len(actuals_bio) - len(response)
            for i in range(diff):
                response.append(label2idx["O"])
        for i in range(len(actuals_bio)):
            if response[i] is None or response[i] == "O":
                continue
            return_arr.append(actuals_txt[i])
        return return_arr
    
    points = []
    for i in range(len(processed)):
        rev = arr_review[i]
        resp = arr_response[i]
        if rev["index"] != resp["index"]:
            print("WARNING: indices don't match at", i)
        rev_bio = clean_response(rev["model_responses"]["try0"])
        resp_bio = clean_response(resp["model_responses"]["try0"])
        pt = {"review": map_to_old_strings(processed[i], rev_bio, is_review=True),
              "reply": map_to_old_strings(processed[i], resp_bio, is_review=False)}
        points.append(pt)
    return points
