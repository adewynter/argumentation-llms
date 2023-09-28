# Code taken and modified from https://github.com/TianyuTerry/MLMC
import torch

START_TAG = 'START'
STOP_TAG = 'STOP'
PAD_TAG = 'PAD'

sentiment2id = {'negative': 3, 'neutral': 4, 'positive': 5}
label2idx = {'O': 0, 'B': 1, 'I': 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5}
idx2labels = ['O', 'B', 'I', START_TAG, STOP_TAG, PAD_TAG]
iobes_label2idx = {'O': 0, 'B': 1, 'I': 2, START_TAG: 3, STOP_TAG: 4, PAD_TAG: 5, 'E': 6, 'S': 7}
iobes_idx2labels = ['O', 'B', 'I', START_TAG, STOP_TAG, PAD_TAG, 'E', 'S']
semi_label2idx = {'O': 0, 'A': 1, START_TAG: 2, STOP_TAG: 3, PAD_TAG: 5}
semi_idx2labels = ['O', 'A', START_TAG, STOP_TAG, PAD_TAG]

B_PREF= "B"
I_PREF = "I"
S_PREF = "S"
E_PREF = "E"
O = "O"

def get_spans(tags):
    """
    for spans
    """
    tags = tags.strip().split('<tag>')
    length = len(tags)
    spans = []
    start = -1
    for i in range(length):
        if tags[i].endswith('B'):
            if start != -1:
                spans.append([start, i - 1])
            start = i
        elif tags[i].endswith('O'):
            if start != -1:
                spans.append([start, i - 1])
                start = -1
    if start != -1:
        spans.append([start, length - 1])
    return spans


class Instance(object):
    
    def __init__(self, sentence_pack, args, is_train):
        self.id = sentence_pack['id']
        self.sentence = sentence_pack['sentence']
        self.last_review = sentence_pack['split_idx']
        self.sents = self.sentence.strip().split(' <sentsep> ')
        self.review = self.sents[:self.last_review+1]
        self.reply = self.sents[self.last_review+1:]
        self.sen_length = len(self.sents)
        self.review_length = self.last_review + 1
        self.reply_length = self.sen_length - self.last_review - 1
        self.review_bert_tokens = []
        self.reply_bert_tokens = []
        self.review_num_tokens = []
        self.reply_num_tokens = []
        self.length = len(self.sents)
        if is_train:
            self.tags = torch.full((self.review_length, self.reply_length), -1, dtype=torch.long)
        else:
            self.tags = torch.zeros(self.review_length, self.reply_length).long()
        
        review_bio_list = [O] * self.review_length
        reply_bio_list = [O] * self.reply_length

        for triple in sentence_pack['triples']:
            aspect = triple['target_tags']
            opinion = triple['opinion_tags']
            aspect_span = get_spans(aspect)
            opinion_span = get_spans(opinion)

            for l, r in aspect_span:
                for i in range(l, r+1):
                    if i == l:
                        review_bio_list[i] = 'B'
                    else:
                        review_bio_list[i] = 'I'

            for l, r in opinion_span:
                for i in range(l, r+1):
                    if i == l:
                        reply_bio_list[i-self.review_length] = 'B'
                    else:
                        reply_bio_list[i-self.review_length] = 'I'

            for al, ar in aspect_span:
                for pl, pr in opinion_span:
                    for i in range(al, ar+1):
                        for j in range(pl, pr+1):
                            self.tags[i][j-self.review_length] = 1
        encoding_scheme = 'BIO'
        if encoding_scheme == 'BIO' or not is_train:
            review_bio_list = [label2idx[label] for label in review_bio_list]
            reply_bio_list = [label2idx[label] for label in reply_bio_list]
            self.review_bio = torch.LongTensor(review_bio_list)
            self.reply_bio = torch.LongTensor(reply_bio_list)
        elif encoding_scheme == 'IOBES' and is_train:
            review_bio_list = [iobes_label2idx[label] for label in convert_bio_to_iobes(review_bio_list)]
            reply_bio_list = [iobes_label2idx[label] for label in convert_bio_to_iobes(reply_bio_list)]
            self.review_bio = review_bio_list
            self.reply_bio = reply_bio_list


def load_data_instances(sentence_packs, num_instances, is_train):
    instances = list()
    if num_instances != -1:
        for sentence_pack in sentence_packs[num_instances]:
            instances.append(Instance(sentence_pack, None, is_train))
    else:
        for sentence_pack in sentence_packs:
            instances.append(Instance(sentence_pack, None, is_train))
    return instances
