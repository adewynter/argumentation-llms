# Can GPT-3/4 Reason in Argumentation?

This is the repository for the paper ["I'd Like to Have an Argument, Please": Argumentative Reasoning in Large Language Models
](https://arxiv.org/abs/2309.16938v2), by Adrian de Wynter and Tangming Yuan. It contains everything to reproduce the results from the paper.

In the paper performed _abstraction evaluation_ (as opposed to standard prompt phrasing) to test the LLMs' reasoning capabilities.
With that we showed:
1. Scoring-wise, the LLMs match or surpass the SOTA in AM and APE.
2. Under certain I/O abstractions LLMs perform well, even beating chain-of-thought--we call this _symbolic prompting_.
3. HOWEVER, statistical analysis on the LLMs outputs when subject to small, yet still human-readable, alterations in the I/O representations (e.g., asking for BIO tags as opposed to line numbers) showed that the models are **not** performing reasoning.

All of that suggests that LLM applications to some tasks, such as data labelling and paper reviewing, must be done with care.

## Rundown

1. Use `Data_Collection.ipynb` to collect data and load the specific prompts. You will need the RR-v2 dataset from [here](https://github.com/LiyingCheng95/MLMC/tree/main/data/rr-submission-v2). Other dependencies (like your own LLMClient class) are noted in the notebook.
2. Use `Evaluation.ipynb` to parse the results. We'll upload the prompts and outputs when I figure out how to push 3Gb worth of data to Github without it complaining.

## Citation

If you find this work useful, please consider citing our paper:
_Note_: I will update this with the COMMA citation once it is live.
```
@misc{dewynter2024id,
      title={"I'd Like to Have an Argument, Please": Argumentative Reasoning in Large Language Models}, 
      author={Adrian de Wynter and Tangming Yuan},
      year={2024},
      eprint={2309.16938},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2309.16938v2}
}
```

## Licence

MIT licence
