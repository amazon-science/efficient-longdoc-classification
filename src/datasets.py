import numpy as np
import spacy
import pytextrank
import random
import torch
from torch.utils.data import Dataset

class TruncatedDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]

        return {
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(self.labels[index])
        }

class TruncatedPlusTextRankDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def apply_textrank(self, text):
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        doc = nlp(text)
        num_phrases = len(list(doc._.phrases))
        num_sents = len(list(doc.sents))
        tr = doc._.textrank
        running_length = 0
        key_sents_idx = []
        key_sents = []
        for sentence in tr.summary(limit_phrases=num_phrases, limit_sentences=num_sents, preserve_order=False):
            if running_length <= (self.max_len - 2):
                sentence_str = str(sentence)
                sentence_tokens = self.tokenizer.tokenize(sentence_str)
                running_length += len(sentence_tokens)
                key_sents.append(sentence_str)
                key_sents_idx.append(sentence.sent.start)

        reorder_idx = list(np.argsort(key_sents_idx))
        selected_text = ''
        for idx in reorder_idx:
            selected_text += key_sents[idx] + ' '
        return selected_text

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        if inputs.get("overflowing_tokens"):
            # select key sentences if text is longer than max length
            selected_text = self.apply_textrank(text)

            second_inputs = self.tokenizer.encode_plus(
                text=selected_text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True
            )
        else:
            second_inputs = inputs

        ids = (inputs['input_ids'], second_inputs['input_ids'])
        mask = (inputs['attention_mask'], second_inputs['attention_mask'])
        token_type_ids = (inputs["token_type_ids"], second_inputs["token_type_ids"])

        return {
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(self.labels[index])
        }

class TruncatedPlusRandomDataset(Dataset):
    def __init__(self, text, labels, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def select_random_sents(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        sents = list(doc.sents)
        running_length = 0
        sent_idxs = list(range(len(sents)))
        selected_idx = []
        while running_length <= (self.max_len - 2) and sent_idxs:
            idx = random.choice(sent_idxs)
            sent_idxs.remove(idx)
            sentence = str(sents[idx])
            sentence_tokens = self.tokenizer.tokenize(sentence)
            running_length += len(sentence_tokens)
            selected_idx.append(idx)

        reorder_idx = sorted(selected_idx)
        selected_text = ''
        for idx in reorder_idx:
            selected_text += str(sents[idx]) + ' '
        return selected_text

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        if inputs.get("overflowing_tokens"):
            # select random sentences if text is longer than max length
            selected_text = self.select_random_sents(text)
            second_inputs = self.tokenizer.encode_plus(
                text=selected_text,
                text_pair=None,
                add_special_tokens=True,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_token_type_ids=True,
                return_overflowing_tokens=True
            )
        else:
            second_inputs = inputs

        ids = (inputs['input_ids'], second_inputs['input_ids'])
        mask = (inputs['attention_mask'], second_inputs['attention_mask'])
        token_type_ids = (inputs["token_type_ids"], second_inputs["token_type_ids"])

        return {
            'ids': torch.tensor(ids),
            'mask': torch.tensor(mask),
            'token_type_ids': torch.tensor(token_type_ids),
            'labels': torch.tensor(self.labels[index])
        }

class ChunkDataset(Dataset):
    def __init__(self, text, labels, tokenizer, chunk_len=200, overlap_len=50):
        self.tokenizer = tokenizer
        self.text = text
        self.labels = labels
        self.overlap_len = overlap_len
        self.chunk_len = chunk_len

    def __len__(self):
        return len(self.labels)

    def chunk_tokenizer(self, tokenized_data, targets):
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []

        previous_input_ids = tokenized_data["input_ids"]
        previous_attention_mask = tokenized_data["attention_mask"]
        previous_token_type_ids = tokenized_data["token_type_ids"]
        remain = tokenized_data.get("overflowing_tokens")

        input_ids_list.append(torch.tensor(previous_input_ids, dtype=torch.long))
        attention_mask_list.append(torch.tensor(previous_attention_mask, dtype=torch.long))
        token_type_ids_list.append(torch.tensor(previous_token_type_ids, dtype=torch.long))
        targets_list.append(torch.tensor(targets, dtype=torch.long))

        if remain:  # if there is any overflowing tokens
            # remain = torch.tensor(remain, dtype=torch.long)
            idxs = range(len(remain) + self.chunk_len)
            idxs = idxs[(self.chunk_len - self.overlap_len - 2)
                        ::(self.chunk_len - self.overlap_len - 2)]
            input_ids_first_overlap = previous_input_ids[-(
                    self.overlap_len + 1):-1]
            start_token = [101]
            end_token = [102]

            for i, idx in enumerate(idxs):
                if i == 0:
                    input_ids = input_ids_first_overlap + remain[:idx]
                elif i == len(idxs):
                    input_ids = remain[idx:]
                elif previous_idx >= len(remain):
                    break
                else:
                    input_ids = remain[(previous_idx - self.overlap_len):idx]

                previous_idx = idx

                nb_token = len(input_ids) + 2
                attention_mask = np.ones(self.chunk_len)
                attention_mask[nb_token:self.chunk_len] = 0
                token_type_ids = np.zeros(self.chunk_len)
                input_ids = start_token + input_ids + end_token
                if self.chunk_len - nb_token > 0:
                    padding = np.zeros(self.chunk_len - nb_token)
                    input_ids = np.concatenate([input_ids, padding])

                input_ids_list.append(torch.tensor(input_ids, dtype=torch.long))
                attention_mask_list.append(torch.tensor(attention_mask, dtype=torch.long))
                token_type_ids_list.append(torch.tensor(token_type_ids, dtype=torch.long))
                targets_list.append(torch.tensor(targets, dtype=torch.long))

        return ({
            'ids': input_ids_list,
            'mask': attention_mask_list,
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })

    def __getitem__(self, index):
        text = " ".join(str(self.text[index]).split())
        targets = self.labels[index]

        data = self.tokenizer.encode_plus(
            text=text,
            text_pair=None,
            add_special_tokens=True,
            max_length=self.chunk_len,
            truncation=True,
            pad_to_max_length=True,
            return_token_type_ids=True,
            return_overflowing_tokens=True
        )

        chunk_token = self.chunk_tokenizer(data, targets)
        return chunk_token