import numpy as np
import pandas as pd
import os
import torch
from datahandler import DataHandler
from configuration import Config
from helper import *
from preprocessing import Preprocessing
import advertools as adv
from parsivar import Normalizer, Tokenizer
from dataset import CustomDataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

def train(epoch, model, loader, optimizer, loss_function, max_grad_norm, device):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    y_predict, y_true = [], []
    model.train()
    #with tqdm(loader, unit='batch') as tepoch:
    for _, data in enumerate(tqdm(loader), 0):
        # for _, data in tqdm(enumerate(loader, 0)):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.long)
        
        outputs = model(ids, mask)
        #outputs = torch.tensor(outputs)
        
        loss = loss_function(outputs.logits, targets)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.logits, dim=1)
        n_correct += calcuate_accu(big_idx, targets)
        for y_pred in big_idx.cpu().numpy():
             y_predict.append(y_pred)
        for y_target in targets.cpu().numpy():
             y_true.append(y_target)
        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)
        if _ % 1000 == 0:
            loss_step = tr_loss / nb_tr_steps
            accu_step = (n_correct * 100) / nb_tr_examples
            print(f"\nTraining Loss per 1000 steps: {loss_step}")
            f1 = f1_score(y_true, y_predict, average='macro')
            print(f"Training F1-Score per 1000 steps: {f1}")
            # print(f"Training Accuracy per 1000 steps: {accu_step}")
            print("--------------------------------------------------")
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_loss = tr_loss / nb_tr_steps
    epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f"Training Loss Epoch {epoch}: {epoch_loss}")
    f1 = f1_score(y_true, y_predict, average='macro')
    print(f"Training F1-Score {epoch}: {f1}")
    # print(f"Training Accuracy Epoch {epoch}: {epoch_accu}")

def valid(model, loader, device, dataset_type):
    nb_tr_steps = 0
    nb_tr_examples = 0
    n_correct = 0
    y_predict = []
    y_true = []
    model.eval()
    with torch.no_grad():
        for _, data in enumerate(tqdm(loader), 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.long)
            outputs = model(ids, mask)
            # try:
            big_val, big_idx = torch.max(outputs.logits, dim=1)
            # except:
            #     print("\n##########################################")
            #     print(f"Data at: {_}")
            #     print(outputs)
            #     print(outputs.data)
            #     print(outputs.data.shape)
            #     print("##########################################")
            #     continue
            n_correct += calcuate_accu(big_idx, targets)
            for y_pred in big_idx.cpu().numpy():
                y_predict.append(y_pred)
            for y_target in targets.cpu().numpy():
                y_true.append(y_target)
            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            if _ % 1000 == 0:
                accu_step = (n_correct * 100) / nb_tr_examples
                f1 = f1_score(y_true, y_predict, average='macro')
                print(f"\n{dataset_type} F1-Score {epoch}: {f1}")
                # print(f"{dataset_type} Accuracy per 1000 steps: {accu_step}")
                print("-------------------------------------")
    # epoch_accu = (n_correct * 100) / nb_tr_examples
    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct * 100) / nb_tr_examples}')
    epoch_accu = (n_correct * 100) / nb_tr_examples
    f1 = f1_score(y_true, y_predict, average='macro')
    print(f"{dataset_type} F1-Score {epoch}: {f1}")
    # print(f"Training Accuracy Epoch {epoch}: {epoch_accu}")
    print("========================================================")
    return y_true, y_predict

def evaluation_method(y_true, y_pred, IDS_TO_LABELS):
    f1 = f1_score(y_true, y_pred, average='macro')
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')

    clf_report = classification_report(y_true, y_pred)
    return {
        "y-true": [IDS_TO_LABELS[y] for y in y_true],
        "y-pred": [IDS_TO_LABELS[y] for y in y_pred],
        "f1": f1, "accuracy": acc,
        "precision": pre, "recall": rec,
        "clf-report": clf_report
    }


if __name__ == "__main__":
    CONFIG = Config().get_re_args()
    if CONFIG.cuda:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        DEVICE = 'cpu'

    TRAIN = DataHandler.load_csv(CONFIG.train_csv)
    VAL = DataHandler.load_csv(CONFIG.dev_csv)
    TEST = DataHandler.load_csv(CONFIG.test_csv)

    print("Train-set Information")
    train_info = helper(TRAIN)
    print("\n----------------------------------\n")
    print("Validation-set Information")
    val_info = helper(VAL)
    print("\n----------------------------------\n")
    print("Test-set Information")
    test_info = helper(TEST)
    print("-------------------------------------------------------------------------------------")

    LABELS_TO_IDS = {label: ids for ids, label in enumerate(list(set(TRAIN['label'].tolist())))}
    IDS_TO_LABELS = {ids: label for label, ids in LABELS_TO_IDS.items()}
    print("LABELS_TO_IDS:", LABELS_TO_IDS)
    print("IDS_TO_LABELS:", IDS_TO_LABELS)
    print("-------------------------------------------------------------------------------------")

    ##Preprocessing
    normalizer = Normalizer()
    tokenizer = Tokenizer()
    stopwords = sorted(adv.stopwords['persian'])
    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~1234567890؟۱۲۳۴۵۶۷۸۹۰'''

    Preprocess = Preprocessing(normalizer, tokenizer, stopwords, punc)

    TRAIN['premise'] = [Preprocess.preprocessor(text) for text in TRAIN['premise']]
    TRAIN['hypothesis'] = [Preprocess.preprocessor(text) for text in TRAIN['hypothesis']]

    VAL['premise'] = [Preprocess.preprocessor(text) for text in VAL['premise']]
    VAL['hypothesis'] = [Preprocess.preprocessor(text) for text in VAL['hypothesis']]

    TEST['premise'] = [Preprocess.preprocessor(text) for text in TEST['premise']]
    TEST['hypothesis'] = [Preprocess.preprocessor(text) for text in TEST['hypothesis']]
    
    #Tokenization & DataLoader
    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG.pretrained_model)
    MAX_LEN = CONFIG.max_len

    TRAIN_DATASET = CustomDataset(TRAIN, TOKENIZER, MAX_LEN, LABELS_TO_IDS)
    VAL_DATASET = CustomDataset(VAL, TOKENIZER, MAX_LEN, LABELS_TO_IDS)
    TEST_DATASET = CustomDataset(TEST, TOKENIZER, MAX_LEN, LABELS_TO_IDS)

    TRAIN_PARAMS = {'batch_size': CONFIG.train_batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    VALID_PARAMS = {'batch_size': CONFIG.valid_batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    TRAIN_LOADER = DataLoader(TRAIN_DATASET, **TRAIN_PARAMS)
    VAL_LOADER = DataLoader(VAL_DATASET, **VALID_PARAMS)
    TEST_LOADER = DataLoader(TEST_DATASET, **VALID_PARAMS)

    print(len(TRAIN_LOADER))
    print(len(VAL_LOADER))
    print(len(TEST_LOADER))
    print("-------------------------------------------------------------------------------------")

    #Finetuning
    MODEL = BertForSequenceClassification.from_pretrained(CONFIG.pretrained_model, num_labels=len(LABELS_TO_IDS))
    MODEL.to(DEVICE)

    LOSS_FUNCTION = torch.nn.CrossEntropyLoss()
    OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=CONFIG.learning_rate)

    for epoch in range(CONFIG.epochs):
        print(f"Training epoch: {epoch + 1}")
        train(epoch, MODEL, TRAIN_LOADER, OPTIMIZER, LOSS_FUNCTION, CONFIG.max_grad_norm, DEVICE)
        print("=================================================================")

    print("*********************************DEV******************************")
    y_val_true, y_val_pred = valid(MODEL, VAL_LOADER, DEVICE, "Validation")
    print("*********************************TEST******************************")
    y_test_true, y_test_pred = valid(MODEL, TEST_LOADER, DEVICE, "Test")

    results = {
        "val": evaluation_method(y_val_true, y_val_pred, IDS_TO_LABELS),
        "test": evaluation_method(y_test_true, y_test_pred, IDS_TO_LABELS)
    }
    print(f"VAL, F1-Score: {results['val']['f1']}, Accuracy: {results['val']['accuracy']}")
    print(f"TEST, F1-Score: {results['test']['f1']}, Accuracy: {results['test']['accuracy']}")

    DataHandler.write_json(data=results,
                            path=os.path.join(CONFIG.prediction_path, "re-parsbert.json")
                            )
    print(f"Saved pretrained file into:{CONFIG.pretrained_new}")
    TOKENIZER.save_vocabulary(CONFIG.pretrained_new)
    torch.save(MODEL.state_dict(), os.path.join(CONFIG.pretrained_new, "pytorch_model_re.pth"))
    labels_dict = {
        "ids-to-labels": IDS_TO_LABELS,
        "labels-to-ids": LABELS_TO_IDS
    }
    DataHandler.write_json(data=labels_dict,
                        path=os.path.join(CONFIG.pretrained_new, "labels.json"))
    print('All files saved')

