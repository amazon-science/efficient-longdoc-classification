import torch
from transformers import LongformerModel, BertModel
import torch.nn.functional as F
import numpy as np

class BERTPlus(torch.nn.Module):
    def __init__(self, dropout_rate, num_labels):
        super(BERTPlus, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(768*2, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, truncated_output = self.bert(ids[:, 0,:], attention_mask=mask[:, 0,:], token_type_ids=token_type_ids[:, 0,:], return_dict=False)
        _, additional_text_output = self.bert(ids[:, 1,:], attention_mask=mask[:, 1,:], token_type_ids=token_type_ids[:, 1,:], return_dict=False)
        concat_output = torch.cat((truncated_output, additional_text_output), dim=1) # batch_size, 768*2
        drop_output = self.dropout(concat_output) # batch_size, 768*2
        logits = self.classifier(drop_output) # batch_size, num_labels
        return logits

class BERTClass(torch.nn.Module):
    def __init__(self, dropout_rate, num_labels):
        super(BERTClass, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, ids, mask, token_type_ids):
        _, bert_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        drop_output = self.dropout(bert_output)
        logits = self.classifier(drop_output)
        return logits

class LongformerClass(torch.nn.Module):
    def __init__(self, num_labels):
        super(LongformerClass, self).__init__()
        self.longformer = LongformerModel.from_pretrained('allenai/longformer-base-4096', add_pooling_layer=False,
                                                  gradient_checkpointing=True)
        self.classifier = LongformerClassificationHead(hidden_size=768, hidden_dropout_prob=0.1, num_labels=num_labels)

    def forward(self, ids, mask, token_type_ids):
        # Initialize global attention on CLS token
        global_attention_mask = torch.zeros_like(ids)
        global_attention_mask[:, 0] = 1
        sequence_output, _ = self.longformer(ids, attention_mask=mask, global_attention_mask=global_attention_mask,
                            token_type_ids=token_type_ids, return_dict=False)
        logits = self.classifier(sequence_output)
        return logits

class LongformerClassificationHead(torch.nn.Module):
    # This class is from https://huggingface.co/transformers/_modules/transformers/models/longformer
    # /modeling_longformer.html#LongformerForSequenceClassification
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_labels): # config from transformers.LongformerConfig.from_pretrained('allenai/longformer-base-4096')
        super().__init__()
        self.dense = torch.nn.Linear(hidden_size, hidden_size)
        self.dropout = torch.nn.Dropout(hidden_dropout_prob)
        self.out_proj = torch.nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]  # take <s> token (equiv. to [CLS])
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output

class ToBERTModel(torch.nn.Module):
    def __init__(self, num_labels, device):
        super(ToBERTModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.trans = torch.nn.TransformerEncoderLayer(d_model=768, nhead=2)
        self.fc = torch.nn.Linear(768, 30)
        self.classifier = torch.nn.Linear(30, num_labels)
        self.device = device

    def forward(self, ids, mask, token_type_ids, length):
        _, pooled_out = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)

        chunks_emb = pooled_out.split_with_sizes(length)
        batch_emb_pad = torch.nn.utils.rnn.pad_sequence(
            chunks_emb, padding_value=0, batch_first=True)
        batch_emb = batch_emb_pad.transpose(0, 1)  # (B,L,D) -> (L,B,D)
        padding_mask = np.zeros([batch_emb.shape[1], batch_emb.shape[0]]) # Batch size, Sequence length
        for idx in range(len(padding_mask)):
            padding_mask[idx][length[idx]:] = 1 # padding key = 1 ignored

        padding_mask = torch.tensor(padding_mask).to(self.device, dtype=torch.bool)
        trans_output = self.trans(batch_emb, src_key_padding_mask=padding_mask)
        mean_pool = torch.mean(trans_output, dim=0) # Batch size, 768
        fc_output = self.fc(mean_pool)
        relu_output = F.relu(fc_output)
        logits = self.classifier(relu_output)

        return logits