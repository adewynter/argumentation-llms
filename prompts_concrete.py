from prompt_utils import extract_spans_from_point_new

def build_prompt_pair(pt, ex):
    '''
    Test prompt to see if the model was able to do both review and rebuttal at the same time.
    Spoiler alert: it wasn't.
    '''
    prompt = "Here are two passages. One is a critique, and the other is a rebuttal.\n"
    prompt += "Return the lines from the text that correspond to each argument, as a BIO tag."
    prompt += "\n"
    prompt += "For example:\n"
    prompt += "|start of critique|\n"
    prompt += "\n".join(ex.review)
    prompt += "|end of critique|\n"
    prompt += "|start of rebuttal|\n"
    prompt += "\n".join(ex.reply)
    prompt += "\n"
    prompt += "|end of rebuttal|\nResponse:"
    prompt += "|start of critique|\n"
    prompt += ",".join([idx2labels[l] for l in ex.review_bio])
    prompt += "|end of critique|\n"
    prompt += "|start of rebuttal|\n"
    prompt += ",".join([idx2labels[l] for l in ex.reply_bio])
    prompt += "\n"
    prompt += "|end of rebuttal|\nResponse:"
    prompt += "Passages begin:\n"
    prompt += "|start of critique|\n"
    prompt += "\n".join(pt.review)
    prompt += "\n"
    prompt += "|end of critique|\n"
    prompt += "\n"
    prompt += "|start of rebuttal|\n"
    prompt += "\n".join(pt.reply)
    prompt += "|end of rebuttal|"
    return prompt

###################
# AM
###################
def build_prompt_single(pt, exs, is_reply):
    '''
    Concrete non-CoT prompt for AM
    '''
    def build_response(bio, text):
        arr = []
        for l,s in zip(bio, text):
            if l == 1:
                arr.append("|START| " + s.replace("<sep>", "").strip())
            if l == 2:
                arr.append(s.replace("<sep>", "").strip())
        return arr
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "Return every line from the passage that is part of an argument, and ONLY these lines.\n"
    prompt += "Mark the beginning of every argument with |START|.\n"
    prompt += "\n"
    if exs is not None:
        prompt += "\nFor example:\n"
        for ex in exs:
            text = ex.reply if is_reply else ex.review
            bio = ex.reply_bio if is_reply else ex.review_bio
            prompt += "|passage start|\n - "
            prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in text])
            prompt += "\n"
            prompt += "|passage end|\n"
            prompt += "Response:\n - "
            prompt += "\n - ".join(build_response(bio, text))
            prompt += "\n"
    prompt += "\n"
    prompt += "Passage:\n"
    prompt += "|passage start|\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in query_text])
    prompt += "\n"
    prompt += "|passage end|\n"
    return prompt


def build_prompt_single_cot(pt, exs, is_reply):
    '''
    Concrete CoT prompt for AM
    '''
    
    def build_response_cot(bio, text):
        res = "Let's think step-by-step and read it line-by-line.\n"
        is_first = True
        for i, (l,s) in enumerate(zip(bio, text)):
            if i == 0:
                res += "We read: \"{}\": ".format(s)
            else:
                res += "Next we read: \"{}\": ".format(s)
            if l == 0:
                res += "This sentence is not related to an argument, so we skip it.\n"
            if l == 1:
                res += "Since the sentence is starting a new argument, it should be prepended by |START|.\n".format(s)
            if l == 2:
                res += "This follows the previous sentence. So we are still reading an argument. We add it to the last argument and do not prepend it by |START|.\n"        
        res += "So the answer should be:\n"
        return res
        
    
    def build_response(bio, text):
        arr = []
        for l,s in zip(bio, text):
            if l == 1:
                arr.append("|START| " + s.replace("<sep>", "").strip())
            if l == 2:
                arr.append(s.replace("<sep>", "").strip())
        return arr
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "Return every line from the passage that is part of an argument, and ONLY these lines.\n"
    prompt += "Mark the beginning of every argument with |START|.\n"
    prompt += "\n"
    if exs is not None:
        prompt += "\nFor example:\n"
        for ex in exs:
            text = ex.reply if is_reply else ex.review
            bio = ex.reply_bio if is_reply else ex.review_bio
            prompt += "|passage start|\n - "
            prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in text])
            prompt += "\n"
            prompt += "|passage end|\n"
            prompt += "Response:\n"
            prompt += build_response_cot(bio, text) + "\n -"
            prompt += "\n - ".join(build_response(bio, text))
            prompt += "\n"
    prompt += "\n"
    prompt += "Passage:\n"
    prompt += "|passage start|\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in query_text])
    prompt += "\n"
    prompt += "|passage end|\n"
    # For COT:
    prompt += "Response:\n"
    prompt += "Let's think step-by-step and read it line-by-line.\n"
    prompt += "We read: \""
    return prompt


###################
# APE
###################
def build_prompt_pair_ape(pt, exs=None):
    '''
    Concrete prompt for APE
    '''    
    def get_responses(_pt):
        revs, resps = extract_spans_from_point_new(_pt)
        tmp_revs_array = [(k, v) for k, v in revs.items()]
        tmp_revs_array.sort(key=lambda x: x[0])
        resp_text_to_return = ""
        for i in range(len(tmp_revs_array)):
            idx, txt = tmp_revs_array[i]
            txt_review = "\n  - ".join([l.replace("<sep>", "").strip() for l in txt])
            txt_resp = "\n  - ".join([l.replace("<sep>", "").strip() for l in resps[idx]])
            resp_text_to_return += "- argument {}".format(i)
            resp_text_to_return += "\n - "
            resp_text_to_return += txt_review
            resp_text_to_return += "\n"
            resp_text_to_return += "- rebuttal {}".format(i)
            resp_text_to_return += "\n - "
            resp_text_to_return += txt_resp
            resp_text_to_return += "\n"
        return resp_text_to_return

    prompt = "Here is a |review| and a |response|.\n"
    prompt += "Return all pairs of argument and rebuttal from the text.\n"
    prompt += "Arguments must come from the |review|.\n"
    prompt += "Rebuttals must come from the |response|.\n"
    prompt += "\n"
    prompt += "Example:\n"
    if exs is not None:
        for ex in exs:
            prompt += "|start of review|"
            prompt += "\n - "
            prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in ex.review])
            prompt += "|end of review|\n"
            prompt += "|start of response|"
            prompt += "\n - "
            prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in ex.reply])
            prompt += "\n"
            prompt += "|end of response|\nAnswer:\n"
            prompt += "|start of answer|\n"
            prompt += get_responses(ex)
            prompt += "|end of answer|\n"
            prompt += "\n"
    prompt += "Passages begin:\n"
    prompt += "|start of review|"
    prompt += "\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in pt.review])
    prompt += "|end of review|\n"
    prompt += "|start of response|"
    prompt += "\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in pt.reply])
    prompt += "\n"
    prompt += "|end of response|\nAnswer:\n"
    prompt += "|start of answer|"
    return prompt

