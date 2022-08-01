import warnings
import time
import os
import argparse
import numpy as np
import logging
import glob
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import cuda
from torch.utils.data import DataLoader
from transformers import LongformerTokenizer, BertTokenizer
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import dataloader
import models
import datasets

def chunk_collate_fn(batches):
    """
    Create batches for ChunkDataset
    """
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]

def create_dataloader(dataset_class, text_set, label_set, tokenizer, max_length, batch_size, num_workers):
    """
    Create appropriate dataloaders for the given data
    :param dataset_class: Dataset to use as defined in datasets.py
    :param text_set: dict of lists of texts for train/dev/test splits, keys=['train', 'dev', 'test']
    :param label_set: dict of lists of labels for train/dev/test splits, keys=['train', 'dev', 'test']
    :param tokenizer: tokenizer of choice e.g. LongformerTokenizer, BertTokenizer
    :param max_length: maximum length of sequence e.g. 512
    :param batch_size: batch size for dataloaders
    :param num_workers: number of workers for dataloaders
    :return: set of dataloaders for train/dev/test splits, keys=['train', 'dev', 'test']
    """
    dataloaders = {}

    if 'train' in text_set.keys():
        split = 'train'
        dataset = dataset_class(text_set[split], label_set[split], tokenizer, max_length)
        if isinstance(dataset, datasets.ChunkDataset):
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                            pin_memory=True, collate_fn=chunk_collate_fn)
        else:
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                        pin_memory=True)

    for split in ['dev', 'test']:
        dataset = dataset_class(text_set[split], label_set[split], tokenizer, max_length)
        if isinstance(dataset, datasets.ChunkDataset):
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                            pin_memory=True, collate_fn=chunk_collate_fn)
        else:
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                        pin_memory=True)

    return dataloaders

def get_long_texts_and_labels(text_dict, label_dict, tokenizer, max_length=512):
    """
    Find texts that have more than a given max token length and their labels
    :param text_dict: dict of lists of texts for train/dev/test splits, keys=['train', 'dev', 'test']
    :param label_dict: dict of lists of labels for train/dev/test splits, keys=['train', 'dev', 'test']
    :param tokenizer: tokenizer of choice e.g. LongformerTokenizer, BertTokenizer
    :param max_length: maximum length of sequence e.g. 512
    :return: dicts of lists of texts with more than the max token length and their labels
    """
    long_text_set = {'dev': [], 'test': []}
    long_label_set = {'dev': [], 'test': []}
    for split in ['dev', 'test']:
        long_text_idx = []
        for idx, text in enumerate(text_dict[split]):
            if len(tokenizer.tokenize(text)) > (max_length - 2):
                long_text_idx.append(idx)
        long_text_set[split] = [text_dict[split][i] for i in long_text_idx]
        long_label_set[split] = [label_dict[split][i] for i in long_text_idx]
    return long_text_set, long_label_set

class Classification(pl.LightningModule):
    """
    Pytorch Lightning module to train all models
    """
    def __init__(self, model, lr, scheduler, label_type, chunk, num_labels, dataset_size, epochs, batch_size):
        super().__init__()
        self.model = model
        self.lr = lr
        self.scheduler = scheduler
        self.label_type = label_type
        self.chunk = chunk
        self.num_labels = num_labels
        self.dataset_size = dataset_size
        self.epochs = epochs
        self.batch_size = batch_size
        if self.label_type == 'binary_class':
            self.eval_metric = torchmetrics.Accuracy(num_classes=self.num_labels)
        elif self.label_type == 'multi_label':
            self.eval_metric = torchmetrics.F1(num_classes=self.num_labels, average='micro')
        elif self.label_type == 'multi_class':
            self.eval_metric = torchmetrics.Accuracy(num_classes=self.num_labels, multiclass=True)

    def training_step(self, batch, batch_idx):
        start = time.time()
        metrics = {}
        if self.chunk:
            ids = [data['ids'] for data in batch]
            mask = [data['mask'] for data in batch]
            token_type_ids = [data['token_type_ids'] for data in batch]
            targets = [data['targets'][0] for data in batch]
            length = [data['len'] for data in batch]

            ids = torch.cat(ids)
            mask = torch.cat(mask)
            token_type_ids = torch.cat(token_type_ids)
            targets = torch.stack(targets)
            length = torch.cat(length)
            length = [x.item() for x in length]

            ids = ids.to(self.device)
            mask = mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            y = targets.to(self.device)

            y_hat = self.model(ids, mask, token_type_ids, length)
        else:
            ids = batch['ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            y = batch['labels'].to(self.device)

            y_hat = self.model(ids, mask, token_type_ids)

        if self.label_type == 'multi_label' or self.label_type == 'binary_class':
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float())  # sigmoid + binary cross entropy loss
            preds = torch.sigmoid(y_hat)

        elif self.label_type == 'multi_class':
            loss = F.cross_entropy(y_hat, y)  # softmax + cross entropy loss
            preds = torch.softmax(y_hat, dim=-1)

        metrics['loss'] = loss

        self.log('train_eval_metric', self.eval_metric(preds, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('losses', {'train_loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log('train_time', time.time() - start, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return metrics

    def validation_step(self, batch, batch_idx, prefix='val_'):
        start = time.time()
        metrics = {}
        if self.chunk:
            ids = [data['ids'] for data in batch]
            mask = [data['mask'] for data in batch]
            token_type_ids = [data['token_type_ids'] for data in batch]
            targets = [data['targets'][0] for data in batch]
            length = [data['len'] for data in batch]

            ids = torch.cat(ids)
            mask = torch.cat(mask)
            token_type_ids = torch.cat(token_type_ids)
            targets = torch.stack(targets)
            length = torch.cat(length)
            length = [x.item() for x in length]

            ids = ids.to(self.device)
            mask = mask.to(self.device)
            token_type_ids = token_type_ids.to(self.device)
            y = targets.to(self.device)

            y_hat = self.model(ids, mask, token_type_ids, length)

        else:
            ids = batch['ids'].to(self.device)
            mask = batch['mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            y = batch['labels'].to(self.device)

            y_hat = self.model(ids, mask, token_type_ids)

        if self.label_type == 'multi_label' or self.label_type == 'binary_class':
            loss = F.binary_cross_entropy_with_logits(y_hat, y.float())  # sigmoid + loss
            preds = torch.sigmoid(y_hat)

        elif self.label_type == 'multi_class':
            loss = F.cross_entropy(y_hat, y)  # softmax + loss
            preds = torch.softmax(y_hat, dim=-1)

        metrics[prefix + 'loss'] = loss
        metrics['preds'] = preds
        metrics['y'] = y

        self.log(prefix + 'eval_metric', self.eval_metric(preds, y), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(prefix + 'loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('losses', {prefix + 'loss': loss}, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log(prefix + 'time', time.time() - start, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return metrics

    def validation_epoch_end(self, outputs, prefix='val_'):
        labels = []
        predictions = []
        for output in outputs:
            for out_labels in output['y'].detach().cpu():
                labels.append(out_labels)
            for out_predictions in output['preds'].detach().cpu():
                predictions.append(out_predictions)

        labels = torch.stack(labels).int()
        predictions = torch.stack(predictions)

        y_pred = predictions.numpy()
        y_true = labels.numpy()

        if self.label_type == 'multi_label' or self.label_type == 'binary_class':
            y_pred_labels = np.where(y_pred > 0.5, 1, 0)

        elif self.label_type == 'multi_class':
            y_pred_labels = np.argmax(y_pred, axis=1)

        logging.info("Epoch: {}".format(self.current_epoch))

        logging.info(
            prefix + 'accuracy: {}'.format(accuracy_score(y_true, y_pred_labels)))

        if self.label_type == 'binary_class':
            average_type = 'macro'
            logging.info(prefix + average_type + '_precision: {}'.format(precision_score(y_true, y_pred_labels, average=average_type)))
            logging.info(
                prefix + average_type + '_recall: {}'.format(recall_score(y_true, y_pred_labels, average=average_type)))
            logging.info(
                prefix + average_type + '_f1: {}'.format(f1_score(y_true, y_pred_labels, average=average_type)))

        else:
            for average_type in ['micro', 'macro', 'weighted']:
                logging.info(prefix + average_type + '_precision: {}'.format(precision_score(y_true, y_pred_labels, average=average_type)))
                logging.info(
                    prefix + average_type + '_recall: {}'.format(recall_score(y_true, y_pred_labels, average=average_type)))
                logging.info(
                    prefix + average_type + '_f1: {}'.format(f1_score(y_true, y_pred_labels, average=average_type)))


    def test_step(self, batch, batch_idx):
        metrics = self.validation_step(batch, batch_idx, 'test_')
        return metrics

    def test_epoch_end(self, outputs):
        self.validation_epoch_end(outputs, prefix="test_")

    def configure_optimizers(self):
        opt = {}
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        opt['optimizer'] = optimizer
        if not self.scheduler:
            return opt
        else:
            num_steps = self.dataset_size * self.epochs / self.batch_size
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=num_steps * 0.1, num_training_steps=num_steps
            )
            opt['lr_scheduler'] = scheduler
            return opt

if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)  # ignore future warnings
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True,
                        help="Model name: {bert, bertplustextrank, bertplusrandom, longformer, tobert}")
    parser.add_argument('--data', type=str, required=True,
                        help="Dataset name: {eurlex, hyperpartisan, books, 20news}")
    parser.add_argument('--batch_size', type=int, required=True, help="Batch size")
    parser.add_argument('--lr', type=float, required=False, help="Learning rate e.g. 0.005, 5e-05")
    parser.add_argument('--epochs', type=int, required=True, help="Number of epochs")
    parser.add_argument("--scheduler", action='store_true', help="Use a warmup scheduler with warmup steps of 0.1 of "
                                                                 "the total training steps")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of data loader workers")
    parser.add_argument('--model_dir', type=str, default='./ckpts/', help="Path to save the best model")
    parser.add_argument("--seed", type=int, default=3456, help="Random seed")
    parser.add_argument("--inverted", action='store_true', help="Use the Inverted EURLEX dataset")
    parser.add_argument("--pairs", action='store_true', help="Use the Paired Book Summary dataset")
    parser.add_argument("--eval", action='store_true', help="Evaluate only, do not train")
    parser.add_argument("--ckpt", type=str, help="Path to a saved ckpt for continued training or evaluation"
                                                 "e.g. bert_hyperpartisan_b8_e20_s3456_lr3e-05--epoch=17.ckpt")

    args = parser.parse_args()

    device = 'cuda' if cuda.is_available() else 'cpu'
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED
    seed_everything(args.seed, workers=True)
    dropout_rate = 0.1
    chunk = True if args.model_name.lower() == 'tobert' else False

    if args.data.lower() == 'eurlex':
        label_type = 'multi_label'
        text_set, label_set, num_labels = dataloader.prepare_eurlex_data(inverted=args.inverted)

    elif args.data.lower() == 'books':
        label_type = 'multi_label'
        text_set, label_set, num_labels = dataloader.prepare_book_summaries(pairs=args.pairs)

    elif args.data.lower() == '20news':
        label_type = 'multi_class'
        text_set, label_set, num_labels = dataloader.prepare_20news_data()

    elif args.data.lower() == 'hyperpartisan':
        label_type = 'binary_class'
        text_set, label_set, num_labels = dataloader.prepare_hyperpartisan_data()

    else:
        raise Exception("Data not found: {}".format(args.data))

    dataset_size = len(label_set['train']) # to calculate the num of steps for warm up scheduler

    if args.model_name.lower() == 'longformer':
        tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096', do_lower_case=True)
        max_length = 4096
    else:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        max_length = 512

    if args.model_name.lower() == 'bert':
        model = models.BERTClass(dropout_rate, num_labels)
        dataset_class = datasets.TruncatedDataset

    elif args.model_name.lower() == 'bertplustextrank':
        model = models.BERTPlus(dropout_rate, num_labels)
        dataset_class = datasets.TruncatedPlusTextRankDataset

    elif args.model_name.lower() == 'bertplusrandom':
        model = models.BERTPlus(dropout_rate, num_labels)
        dataset_class = datasets.TruncatedPlusRandomDataset

    elif args.model_name.lower() == 'longformer':
        model = models.LongformerClass(num_labels)
        dataset_class = datasets.TruncatedDataset

    elif args.model_name.lower() == 'tobert':
        max_length = 200 # divide documents into chunks up to 200 tokens
        model = models.ToBERTModel(num_labels, device)
        dataset_class = datasets.ChunkDataset

    else:
        raise Exception("Model not found: {}".format(args.model_name))

    dataloaders = create_dataloader(dataset_class, text_set, label_set, tokenizer, max_length, args.batch_size,
                                    args.num_workers)

    long_text_set, long_label_set = get_long_texts_and_labels(text_set, label_set, tokenizer)
    long_dataloaders = create_dataloader(dataset_class, long_text_set, long_label_set, tokenizer, max_length,
                                         args.batch_size, args.num_workers)

    model.to(device)

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
        print(f'Model directory created: {args.model_dir}')

    task = Classification(model, args.lr, args.scheduler, label_type, chunk, num_labels, dataset_size, args.epochs,
                          args.batch_size)

    inverted_str = '_inverted' if args.inverted else ''
    pairs_str = '_pairs' if args.pairs else ''
    scheduler_str = '_warmup' if args.scheduler else ''

    output_model_name = args.model_name + '_' + args.data + inverted_str + pairs_str + '_b' + str(args.batch_size) + \
                        '_e' + str(args.epochs) + '_s' + str(args.seed) + '_lr' + str(args.lr) + scheduler_str

    logging.basicConfig(filename=output_model_name + '.log', level=logging.DEBUG)
    logger = TensorBoardLogger('tb_logs', name=output_model_name)

    for arg in vars(args):
        logging.info("{}: {}".format(arg, getattr(args, arg)))

    if not args.eval: # train mode
        ckpt_config = ModelCheckpoint(
            monitor="val_eval_metric_epoch",
            verbose=False,
            save_top_k=1,
            save_weights_only=False,
            mode='max',
            every_n_val_epochs=1,
            dirpath=args.model_dir,
            filename=output_model_name + "--{epoch}"
        )
        if args.ckpt:
            trainer = pl.Trainer(logger=logger,
                                 callbacks=ckpt_config,
                                 gpus=1,
                                 deterministic=True,
                                 log_gpu_memory='min_max',
                                 num_sanity_val_steps=0,
                                 max_epochs=args.epochs,
                                 resume_from_checkpoint=args.model_dir + args.ckpt)

        else:
            trainer = pl.Trainer(logger=logger,
                                 callbacks=ckpt_config,
                                 gpus=1,
                                 deterministic=True,
                                 log_gpu_memory='min_max',
                                 num_sanity_val_steps=0,
                                 max_epochs=args.epochs)

        print("Training: {}".format(output_model_name))
        trainer.fit(model=task, train_dataloader=dataloaders['train'], val_dataloaders=dataloaders['dev'])

        for _ckpt in range(len(trainer.checkpoint_callbacks)):
            logging.info("Testing")
            paths = trainer.checkpoint_callbacks[_ckpt]
            ckpt_path = trainer.checkpoint_callbacks[_ckpt].best_model_path
            logging.info("Checkpoint path: {}".format(ckpt_path))
            metrics = trainer.test(test_dataloaders=dataloaders['test'], ckpt_path=ckpt_path)
            for metric in metrics:
                for key in metric:
                    logging.info("{}: {}".format(key, metric[key]))

            for split in ['dev', 'test']:
                logging.info("Evaluating on long documents in the {} set only".format(split))
                metrics = trainer.test(test_dataloaders=long_dataloaders[split], ckpt_path=ckpt_path)
                for metric in metrics:
                    for key in metric:
                        logging.info("long_{}_{}: {}".format(split, key, metric[key]))

    else: # eval mode
        if args.ckpt:
            ckpt_paths = glob.glob(args.model_dir + args.ckpt)
        else:
            ckpt_paths = glob.glob(args.model_dir + output_model_name + '*.ckpt')

        logging.info("Evaluating: {}".format(output_model_name))

        for ckpt_path in ckpt_paths:
            logging.info("Checkpoint path: {}".format(ckpt_path))
            task.load_from_checkpoint(ckpt_path, model=model, lr=args.lr, scheduler=args.scheduler, label_type=label_type,
                                      num_labels=num_labels)

            trainer = pl.Trainer(gpus=1)

            for split in ['dev', 'test']:
                logging.info("Evaluating on all documents in the {} set".format(split))
                metrics = trainer.test(model= task, test_dataloaders=dataloaders[split])
                for metric in metrics:
                    for key in metric:
                        logging.info("all_{}_{}: {}".format(split, key, metric[key]))

            for split in ['dev', 'test']:
                logging.info("Evaluating on long documents in the {} set only".format(split))
                metrics = trainer.test(model= task, test_dataloaders=long_dataloaders[split])
                for metric in metrics:
                    for key in metric:
                        logging.info("long_{}_{}: {}".format(split, key, metric[key]))