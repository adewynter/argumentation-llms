def build_prompt_with_amr(pt_tuple, exs, is_reply, include_context):
    '''
    AMR (symbolic, AM) prompt. 
    - The boolean `include_context` is to return a prompt with the context (sentences) plus the graph,
    or only the graph. 
    - `pt_tuple` must be a tuple of the processed point instance and the corresponding AMR graph.
    '''
    # Point is always the first, graph is the second
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
    
    def extract(tup, txt):
        p, g = tup
        # Symbolic
        response = [l.replace("<sep>", "").strip() for l in txt]
        # AMR with sentence
        if include_context:
            response = g
        else:
            response = ["# "+ "\n".join(_g.split("\n")[1:]) for _g in g]
        return response
    
    pt, pt_amr = pt_tuple
    query_text = pt.reply if is_reply else pt.review
    
    prompt = "Here is a passage.\n"
    prompt += "Every line is prepended with \"#\".\n"
    prompt += "For every line in the passage:\n"
    prompt += "- Mark the beginning of every argument with B.\n"
    prompt += "- Mark the inside of every argument with I.\n"
    prompt += "- Mark everything else with O.\n"
    if exs is not None:
        prompt += "\nFor example:\n"
        for ex_tuple in exs:
            ex = ex_tuple[0]
            text = ex.reply if is_reply else ex.review
            bio = ex.reply_bio if is_reply else ex.review_bio
            prompt += "|passage start|\n - "
            prompt += "\n - ".join(extract(ex_tuple, text))
            prompt += "\n"
            prompt += "|passage end|\n"
            prompt += "Response:\n - "
            prompt += "\n - ".join(build_response(bio, text))
            prompt += "\n"
    prompt += "\n"
    prompt += "Passage:\n"
    prompt += "|passage start|\n"
    prompt += "\n - ".join(extract(pt_tuple, query_text))
    prompt += "\n"
    prompt += "|passage end|\n"
    prompt += "Response:\n"
    return prompt
