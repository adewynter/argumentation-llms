from prompt_utils import extract_spans_from_point_new

###################
# AM
###################
def build_prompt_symbolic(pt, exs, is_reply):
    '''
    Symbolic non-COT AM prompt with BIO tags as return signature.
    '''
    def build_response(bio, text):
        arr = []
        for l,s in zip(bio, text):
            if l == 1:
                arr.append("B")
            if l == 2:
                arr.append("I")
            if l == 0:
                arr.append("O")
        return arr
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Mark the beginning of every argument with B.\n"
    prompt += "- Mark the inside of every argument with I.\n"
    prompt += "- Mark everything else with O.\n"
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
    prompt += "Response:\n - "
    return prompt


def build_prompt_symbolic_w_numbers(pt, exs, is_reply):
    '''
    Symbolic non-COT AM prompt with line indices as return signature.
    '''
    def build_response(bio, text):
        arr = []
        for i, (l,s) in enumerate(zip(bio, text)):
            if l == 1:
                arr.append("("+ str(i) + ")")
            if l == 2:
                arr.append(str(i))
            if l == 0:
                continue
        return arr
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Group and return the index of lines corresponding to an argument.\n"
    prompt += "- Put the index of the beginning of the argument in parenthesis, and then list below the remaining indices, if any.\n"
    prompt += "- Do not return anything else.\n"
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
    prompt += "Response:\n - "
    return prompt


def build_prompt_symbolic_w_numbers_in_lines(pt, exs, is_reply):
    '''
    Symbolic non-COT AM prompt with line indices as return signature, and inline indices.
    '''
    def build_response(bio, text):
        arr = []
        for i, (l,s) in enumerate(zip(bio, text)):
            if l == 1:
                arr.append("("+ str(i) + ")")
            if l == 2:
                arr.append(str(i))
            if l == 0:
                continue
                #arr.append("O")
        return arr
    query_text = pt.reply if is_reply else pt.review

    def conv(i, s):
        return "{}) {}".format(i, s.replace("<sep>", "").strip())

    prompt = "Here is a passage.\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Group and return the index of lines corresponding to an argument.\n"
    prompt += "- Put the index of the beginning of the argument in parenthesis, and then list below the remaining indices, if any.\n"
    prompt += "- Do not return anything else.\n"
    prompt += "\n"
    if exs is not None:
        prompt += "\nFor example:\n"
        for ex in exs:
            text = ex.reply if is_reply else ex.review
            bio = ex.reply_bio if is_reply else ex.review_bio
            prompt += "|passage start|\n - "
            prompt += "\n - ".join([conv(i, l) for i, l in enumerate(text)])
            prompt += "\n"
            prompt += "|passage end|\n"
            prompt += "Response:\n - "
            prompt += "\n - ".join(build_response(bio, text))
            prompt += "\n"
    else:
        prompt += "Return your answer in the form:\n"
        prompt += "|start of answer|\n"
        prompt += "- (line number where argument 0 starts)"
        prompt += "- remaining line number belonging to argument 0"
        prompt += "- next remaining line number belonging to argument 0"
        prompt += "\n  - ...\n"
        prompt += "- (line number where argument 1 starts)"
        prompt += "- remaining line number belonging to argument 1"
        prompt += "- next remaining line number belonging to argument 1"
        prompt += "\n  - ...etc\n"
        prompt += "|end of answer|\n"
    prompt += "\n"
    prompt += "Passage:\n"
    prompt += "|passage start|\n - "
    prompt += "\n - ".join([conv(i, l) for i, l in enumerate(query_text)])
    prompt += "\n"
    prompt += "|passage end|\n"
    prompt += "Response:\n - "
    return prompt


def build_prompt_symbolic_cot(pt, exs, is_reply):
    '''
    Symbolic COT AM prompt with BIO tags as return signature.
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
                res += "This sentence is not related to an argument, so we mark it as \"O\".\n"
            if l == 1:
                res += "Since the sentence is starting a new argument, it should be marked as \"B\".\n".format(s)
            if l == 2:
                res += "This follows the previous sentence. So we are still reading an argument. We mark it as \"I\".\n"        
        res += "So the answer should be:\n"
        return res

    def build_response(bio, text):
        arr = []
        for l,s in zip(bio, text):
            if l == 1:
                arr.append("B")
            if l == 2:
                arr.append("I")
            if l == 0:
                arr.append("O")
        return arr
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Mark the beginning of every argument with B.\n"
    prompt += "- Mark the inside of every argument with I.\n"
    prompt += "- Mark everything else with O.\n"
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
            prompt += build_response_cot(bio, text) + "\n - "
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


def build_prompt_symbolic_w_numbers_in_lines_cot(pt, exs, is_reply):
    '''
    Symbolic COT AM prompt with line indices as return signature, and indices inline.
    '''
    def build_response_cot(bio, text):
        res = "Let's think step-by-step and read it line-by-line.\n"
        is_first = True
        for i, (l,s) in enumerate(zip(bio, text)):
            if i == 0:
                res += "We read: \"{}) {}\" at index {}: ".format(i, s, i)
            else:
                res += "Next we read: \"{}) {}\" at index {}: ".format(i, s, i)
            if l == 0:
                res += "This sentence is not related to an argument, so we skip it.\n"
            if l == 1:
                res += "Since the sentence is starting a new argument, and the index is {}, it should be marked as \"({})\".\n".format(s, i, i)
            if l == 2:
                res += "This follows the previous sentence. The index is {}. So we are still reading an argument. We mark it as \"{}\".\n".format(i, i)
        res += "So the answer should be:\n"
        return res
    
    def build_response(bio, text):
        arr = []
        for i, (l,s) in enumerate(zip(bio, text)):
            if l == 1:
                arr.append("("+ str(i) + ")")
            if l == 2:
                arr.append(str(i))
            if l == 0:
                continue
                #arr.append("O")
        return arr
    query_text = pt.reply if is_reply else pt.review

    def conv(i, s):
        return "{}) {}".format(i, s.replace("<sep>", "").strip())

    
    prompt = "Here is a passage.\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Group and return the index of lines corresponding to an argument.\n"
    prompt += "- Put the index of the beginning of the argument in parenthesis, and then list below the remaining indices, if any.\n"
    prompt += "- Do not return anything else.\n"
    prompt += "\n"
    if exs is not None:
        prompt += "\nFor example:\n"
        for ex in exs:
            text = ex.reply if is_reply else ex.review
            bio = ex.reply_bio if is_reply else ex.review_bio
            prompt += "|passage start|\n - "
            prompt += "\n - ".join([conv(i, l) for i, l in enumerate(text)])
            prompt += "\n"
            prompt += "|passage end|\n"
            prompt += "Response:\n"
            prompt += build_response_cot(bio, text) + "\n - "
            prompt += "\n - ".join(build_response(bio, text))
    else:
        prompt += "Return your answer in the form:\n"
        prompt += "|start of answer|\n"
        prompt += "- (line number where argument 0 starts)"
        prompt += "- remaining line number belonging to argument 0"
        prompt += "- next remaining line number belonging to argument 0"
        prompt += "\n  - ...\n"
        prompt += "- (line number where argument 1 starts)"
        prompt += "- remaining line number belonging to argument 1"
        prompt += "- next remaining line number belonging to argument 1"
        prompt += "\n  - ...etc\n"
        prompt += "|end of answer|\n"
        prompt += "Let's think step-by-step and read it line-by-line.\n"
    prompt += "\n"
    prompt += "Passage:\n"
    prompt += "|passage start|\n - "
    prompt += "\n - ".join([conv(i, l) for i, l in enumerate(query_text)])
    prompt += "\n"
    prompt += "|passage end|\n"
    prompt += "Response:"
    prompt += "Let's think step-by-step and read it line-by-line.\n"
    prompt += "We read: \""
    return prompt


def build_prompt_symbolic_w_numbers_cot(pt, exs, is_reply):
    '''
    Symbolic COT AM prompt with line indices as return signature.
    '''
    def build_response_cot(bio, text):
        res = "Let's think step-by-step and read it line-by-line.\n"
        is_first = True
        for i, (l,s) in enumerate(zip(bio, text)):
            if i == 0:
                res += "We read: \"{}\" at index {}: ".format(s, i)
            else:
                res += "Next we read: \"{}\" at index {}: ".format(s, i)
            if l == 0:
                res += "This sentence is not related to an argument, so we skip it.\n"
            if l == 1:
                res += "Since the sentence is starting a new argument, and the index is {}, it should be marked as \"({})\".\n".format(s, i, i)
            if l == 2:
                res += "This follows the previous sentence. The index is {}. So we are still reading an argument. We mark it as \"{}\".\n".format(i, i)
        res += "So the answer should be:\n"
        return res
    
    def build_response(bio, text):
        arr = []
        for i, (l,s) in enumerate(zip(bio, text)):
            if l == 1:
                arr.append("("+ str(i) + ")")
            if l == 2:
                arr.append(str(i))
            if l == 0:
                continue
                #arr.append("O")
        return arr
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Group and return the index of lines corresponding to an argument.\n"
    prompt += "- Put the index of the beginning of the argument in parenthesis, and then list below the remaining indices, if any.\n"
    prompt += "- Do not return anything else.\n"
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
            prompt += build_response_cot(bio, text) + "\n - "
            prompt += "\n - ".join(build_response(bio, text))
    else:
        prompt += "Let's think step-by-step and read it line-by-line.\n"
    prompt += "\n"
    prompt += "Passage:\n"
    prompt += "|passage start|\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in query_text])
    prompt += "\n"
    prompt += "|passage end|\n"
    prompt += "Response:"
    prompt += "Let's think step-by-step and read it line-by-line.\n"
    prompt += "We read: \""
    return prompt


###################
# APE
###################
def build_prompt_pair_symbolic_ape(pt, exs=None):
    '''
    Symbolic non-COT APE prompt with line indices as return signature.
    '''
    def get_responses(_pt):
        # Symbolic version
        revs, resps = extract_spans_from_point_new(_pt)
        tmp_revs_array = [(k, v) for k, v in revs.items()]
        tmp_revs_array.sort(key=lambda x: x[0])

        resp_text_to_return = ""
        for i in range(len(tmp_revs_array)):
            idx, _ = tmp_revs_array[i]
            txt_review = " ".join([str(_pt.review.index(l.strip())) for l in revs[idx]])
            txt_resp = " ".join([str(_pt.reply.index(l.strip())) for l in resps[idx]])
            resp_text_to_return += "- argument {} ".format(i)
            resp_text_to_return += "({}) : ".format(txt_review)
            resp_text_to_return += txt_resp
            resp_text_to_return += "\n"
        return resp_text_to_return

    def get_str(s, b):
        if b != 0:
            return s.replace("<sep>", "").strip()
        else:
            return "-"
    
    prompt = "Here is a |review| and a |response|.\n"
    prompt += "Return the indices of |argument|.\n"
    prompt += "Return the indices of its associated |rebuttal|.\n"
    prompt += "Arguments must come from the |review|.\n"
    prompt += "Rebuttals must come from the |response|.\n"
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
    prompt += "|start of answer|"
    return prompt


def build_prompt_pair_symbolic_pt2(pt, exs=None):
    '''
    Symbolic non-COT APE prompt with binary matrix as return signature (unused).
    '''
    def get_responses(_pt):
        rows = []
        for i in range(len(_pt.tags)):
            rows.append(" ".join([str(x.item()) for x in _pt.tags[i]]))
        return "\n".join(rows)

    
    prompt = "Here is a |review| and a |response|.\n"
    prompt += "Build a table with the overlapping |argument| and |rebuttal| pairs.\n"
    prompt += "Every |argument| is from the |review|.\n"
    prompt += "Every |rebuttal| is from the |response|.\n"
    prompt += "The rows correspond to each sentence in the |review|.\n"
    prompt += "The columns correspond to each sentence in the |response|.\n"
    prompt += "Whenever an |argument| overlaps with a |rebuttal|, add a 1. Otherwise, add a 0 in the table.\n"
    if exs is not None:
        prompt += "Example:\n"
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
            prompt += "\n|end of answer|\n"
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
    prompt += "|start of answer|"
    return prompt


def build_prompt_pair_symbolic_full(pt, exs=None):
    '''
    Symbolic non-COT APE prompt with binary matrix as return signature.
    '''
    def get_responses(_pt):
        # Symbolic version
        tags = _pt.tags
        string = ""
        for row in tags:
            string += " ".join([str(i.item()) for i in row]) + "\n"
        return string

    def get_str(s, b):
        if b != 0:
            return s.replace("<sep>", "").strip()
        else:
            return "-"
    
    prompt = "Here is a |review| and a |response|.\n"
    prompt += "All arguments are in the |review|.\n"
    prompt += "All rebuttals are in the |response|.\n"
    prompt += "Do the following:\n"
    prompt += "- Construct a matrix of the argument-rebuttal pairs, in terms of their indices.\n"
    prompt += "  - The matrix must be of size (number of lines in the |review|, number of lines in the |response|)\n"
    prompt += "  - The rows correspond to each sentence in the |review|.\n"
    prompt += "  - The columns correspond to each sentence in the |response|.\n"
    prompt += "- Whenever a |rebuttal| refutes an |argument|, mark as a 1 the (row, column) entry corresponding to the (line number from the review, line number from the response) for that argument and rebuttal pair.\n"
    prompt += "- All other entries in the matrix must be zero.\n"
    if exs is not None:
        prompt += "For example:\n"
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
    else:
        prompt += "Return your answer in the form of a (0, 1)-valued matrix.\n"
        prompt += "If you have m lines in the |review| and n lines in the |response|, the matrix must be m by n.\n"
        prompt += "All entries are zero unless the rebuttal in column j is a rebuttal to argument in row i:\n"
        prompt += "|start of answer|\n"
        prompt += "... n columns ... \n"
        prompt += "... (i, j) is 1 if the ith argument matches the rebuttal in the jth rebuttal.\n"
        prompt += "... m rows ...\n"
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
    prompt += "|start of answer|"
    return prompt


def build_prompt_pair_symbolic_w_numbers(pt, exs=None):
    '''
    Symbolic non-COT APE prompt with line indices as return signature, and inline indices.
    '''
    def get_responses(_pt):
        # Symbolic version
        revs, resps = extract_spans_from_point_new(_pt)
        tmp_revs_array = [(k, v) for k, v in revs.items()]
        tmp_revs_array.sort(key=lambda x: x[0])

        resp_text_to_return = ""
        for i in range(len(tmp_revs_array)):
            idx, _ = tmp_revs_array[i]
            txt_review = " ".join([str(_pt.review.index(l.strip())) for l in revs[idx]])
            txt_resp = " ".join([str(_pt.reply.index(l.strip())) for l in resps[idx]])
            resp_text_to_return += "- argument {} ".format(i)
            resp_text_to_return += "({}) : ".format(txt_review)
            resp_text_to_return += txt_resp
            resp_text_to_return += "\n"
        return resp_text_to_return

    def get_str(s, b):
        if b != 0:
            return s.replace("<sep>", "").strip()
        else:
            return "-"
        
    def conv(i, s):
        return "{}) {}".format(i, s.replace("<sep>", "").strip())
    
    prompt = "Here is a |review| and a |response|.\n"
    prompt += "Return the indices of |argument|.\n"
    prompt += "Return the indices of its associated |rebuttal|.\n"
    prompt += "Arguments must come from the |review|.\n"
    prompt += "Rebuttals must come from the |response|.\n"
    prompt += "Example:\n"
    if exs is not None:
        for ex in exs:
            prompt += "|start of review|"
            prompt += "\n - "
            prompt += "\n - ".join([conv(i, l) for i, l in enumerate(ex.review)])
            prompt += "|end of review|\n"
            prompt += "|start of response|"
            prompt += "\n - "
            prompt += "\n - ".join([conv(i, l) for i, l in enumerate(ex.reply)])
            prompt += "\n"
            prompt += "|end of response|\nAnswer:\n"
            prompt += "|start of answer|\n"
            prompt += get_responses(ex)
            prompt += "|end of answer|\n"
    prompt += "\n"
    prompt += "Passages begin:\n"
    prompt += "|start of review|"
    prompt += "\n - "
    prompt += "\n - ".join([conv(i, l) for i, l in enumerate(pt.review)])
    prompt += "|end of review|\n"
    prompt += "|start of response|"
    prompt += "\n - "
    prompt += "\n - ".join([conv(i, l) for i, l in enumerate(pt.reply)])
    prompt += "\n"
    prompt += "|start of answer|"
    return prompt



def build_prompt_pair_x_symbolic(pt, exs=None, offsets=None):
    """
    Simulate hybrid (first AM, then APE). Not in the paper due to extremely low performance.
    """
    def get_responses(_pt, offset_array_revs=None, offset_array_resps=None):
        # Symbolic version
        revs, resps = extract_spans_from_point_new(_pt)
        tmp_revs_array = [(k, v) for k, v in revs.items()]
        tmp_revs_array.sort(key=lambda x: x[0])
        resp_text_to_return = ""
        
        if offset_array_revs is None:
            offset_array_revs = {}
            counter = 0
            for idx, (rev, bio) in enumerate(zip(_pt.review, _pt.review_bio)):
                offset_array_revs[rev.strip()] = (idx, counter)
                if bio != 0:
                    counter += 1

        if offset_array_resps is None:
            offset_array_resps = {}
            counter = 0
            for idx, (rev, bio) in enumerate(zip(_pt.reply, _pt.reply_bio)):
                offset_array_resps[rev.strip()] = (idx, counter)
                if bio != 0:
                    counter += 1
                
        def get_offset(pt, rev=True):
            if rev:
                return str(offset_array_revs[pt.strip()][-1])
            else:
                return str(offset_array_resps[pt.strip()][-1])

        for i in range(len(tmp_revs_array)):
            idx, _ = tmp_revs_array[i]
            txt_review = " ".join([get_offset(l, rev=True) for l in revs[idx]])
            txt_resp = " ".join([get_offset(l, rev=False) for l in resps[idx]])
            resp_text_to_return += "- argument {} ".format(i)
            resp_text_to_return += "({}) : ".format(txt_review)
            resp_text_to_return += txt_resp
            resp_text_to_return += "\n"
        return resp_text_to_return

    prompt = "Here is a |review| and a |response|.\n"
    prompt += "Return the indices of |argument|.\n"
    prompt += "Return the indices of its associated |rebuttal|.\n"
    prompt += "Arguments must come from the |review|.\n"
    prompt += "Rebuttals must come from the |response|.\n"
    prompt += "Example:\n"
    if exs is not None:
        for ex in exs:
            prompt += "|start of review|"
            prompt += "\n - "
            prompt += "\n - ".join([l.replace("<sep>", "").strip() for l, b in zip(ex.review, ex.review_bio) if b != 0])
            prompt += "|end of review|\n"
            prompt += "|start of response|"
            prompt += "\n - "
            prompt += "\n - ".join([l.replace("<sep>", "").strip() for l, b in zip(ex.reply, ex.reply_bio) if b != 0])
            prompt += "\n|end of response|\nAnswer:\n"
            prompt += "|start of answer|\n"
            prompt += get_responses(ex)
            prompt += "|end of answer|\n"
    prompt += "\n"
    prompt += "Passages begin:\n"
    prompt += "|start of review|"
    prompt += "\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in pt["review"]])
    prompt += "\n|end of review|\n"
    prompt += "|start of response|"
    prompt += "\n - "
    prompt += "\n - ".join([l.replace("<sep>", "").strip() for l in pt["reply"]])
    prompt += "\n|end of response|\n"
    prompt += "|start of answer|"
    return prompt