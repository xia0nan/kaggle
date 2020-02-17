import re
import string

import torch
from transformers import BertModel, BertTokenizer


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)
    
    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)


# define model
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # pre-trained BERT model by HuggingFace
        pretrained_weights = config.model_name
        # from official documentation: https://github.com/huggingface/transformers#quick-tour
        self.base_model = BertModel.from_pretrained(pretrained_weights)
        # add dense layer to the end
        self.fc1 = torch.nn.Linear(768, 1)
        
    def forward(self, ids, masks):
        x = self.base_model(ids, attention_mask=masks)[1]
        x = self.fc1(x)
        return x


def bert_encode(text, max_len=512):
    """ BERT encoder for text """
    # tokenize text using BERT tokenizer
    text = tokenizer.tokenize(text)
    # remove 2 tokens for start and end token
    text = text[:max_len-2]
    # add start and end token
    input_sequence = ["[CLS]"] + text + ["[SEP]"]
    # convert token to token_id
    tokens = tokenizer.convert_tokens_to_ids(input_sequence)
    # padding to max_len
    tokens += [0] * (max_len - len(input_sequence))
    # triangle masking
    pad_masks = [1] * len(input_sequence) + [0] * (max_len - len(input_sequence))
    return tokens, pad_masks


def clean_text(text):
    """ Basic text cleaning """
    text = text.lower()
    text = re.sub(r'[^a-z0-9] \n', '', text)
    return text


class Dataset(torch.utils.data.Dataset):
    """build training pytorch dataset """
    def __init__(self, train_tokens, train_pad_masks, targets):
        
        super(Dataset, self).__init__()
        self.train_tokens = train_tokens
        self.train_pad_masks = train_pad_masks
        self.targets = targets
        
    def __getitem__(self, index):
        
        tokens = self.train_tokens[index]
        masks = self.train_pad_masks[index]
        target = self.targets[index]
        
        return (tokens, masks), target
    
    def __len__(self,):
        
        return len(self.train_tokens)

def process():
    
    # configuration
    config = Config(
        testing=False,
        model_name="bert-base-uncased",
        max_lr=1e-5,
        epochs=2,
        bs=12,
        discriminative=False,
        max_seq_len=64,
        sample_size=6000,
        path_to_dataset = '/kaggle/input/nlp-getting-started/'
    )
    
    # read train set and test set
    test = pd.read_csv(os.path.join(path_to_dataset, 'test.csv'))
    train = pd.read_csv(os.path.join(path_to_dataset, 'train.csv'))
    
    # load BERT Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Torch define device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # load model and put on GPU
    model = Model()
    model.to(device)
    
    # Use first 6000 for training, rest for validation
    train_text = train.text[:config.sample_size]
    val_text = train.text[config.sample_size:]
    # clean text
    train_text = train_text.apply(clean_text)
    val_text = val_text.apply(clean_text)
    
    # encode text and get mask
    train_tokens = []
    train_pad_masks = []
    for text in train_text:
        tokens, masks = bert_encode(text)
        train_tokens.append(tokens)
        train_pad_masks.append(masks)
        
    train_tokens = np.array(train_tokens)
    train_pad_masks = np.array(train_pad_masks)
    
    # same for validation
    val_tokens = []
    val_pad_masks = []
    for text in val_text:
        tokens, masks = bert_encode(text)
        val_tokens.append(tokens)
        val_pad_masks.append(masks)
        
    val_tokens = np.array(val_tokens)
    val_pad_masks = np.array(val_pad_masks)
    
    # build training dataset
    train_dataset = Dataset(
                        train_tokens=train_tokens,
                        train_pad_masks=train_pad_masks,
                        targets=train.target[:config.sample_size]
    )
    
    # define hyperparameters
    batch_size = 12
    EPOCHS = 2
    
    # build training dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=config.bs, shuffle=True)
    
    # define loss and optimizer
    criterion = torch.nn.BCEWithLogitsLoss()
    # Use Adam Optimizer with learning rate of 0.00001
    opt = torch.optim.Adam(model.parameters(), lr=config.max_lr)
    
    # train model
    model.train()
    y_preds = []

    # train 2 epochs
    for epoch in range(config.epochs):

        for i, ((tokens, masks), target) in enumerate(train_dataloader):

            y_pred = model(
                        tokens.long().to(device), 
                        masks.long().to(device)
                    )
            loss = criterion(y_pred, target[:, None].float().to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
            print("Step:", i)
        print('\rEpoch: %d/%d, %f%% loss: %0.2f'% (epoch+1, EPOCHS, (i+1)/len(train_dataloader)*100, loss.item()), end='')
        print()
        
    # build validation dataset and dataloader
    val_dataset = Dataset(
                        train_tokens=val_tokens,
                        train_pad_masks=val_pad_masks,
                        targets=train.target[6000:].reset_index(drop=True)
    )
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=3, shuffle=False)
    
    # define accuracy metric
    def accuracy(y_actual, y_pred):
        y_ = y_pred > 0
        return np.sum(y_actual == y_).astype('int') / y_actual.shape[0]
    
    # evaluate model on val dataset
    model.eval()
    avg_acc = 0
    for i, ((tokens, masks), target) in enumerate(val_dataloader):

        y_pred = model(
                    tokens.long().to(device), 
                    masks.long().to(device), 
                )
        loss = criterion(y_pred,  target[:, None].float().to(device))
        acc = accuracy(target.cpu().numpy(), y_pred.detach().cpu().numpy().squeeze())
        avg_acc += acc
        print('\r%0.2f%% loss: %0.2f, accuracy %0.2f'% (i/len(val_dataloader)*100, loss.item(), acc), end='')
    print('\nAverage accuracy: ', avg_acc / len(val_dataloader))
    
    # define test dataset
    class TestDataset(torch.utils.data.Dataset):
        
        def __init__(self, test_tokens, test_pad_masks):
            
            super(TestDataset, self).__init__()
            self.test_tokens = test_tokens
            self.test_pad_masks = test_pad_masks
            
        def __getitem__(self, index):
            
            tokens = self.test_tokens[index]
            masks = self.test_pad_masks[index]
            
            return (tokens, masks)
        
        def __len__(self,):
            
            return len(self.test_tokens)
        
    # encode test text and get mask
    test_tokens = []
    test_pad_masks = []
    for text in test.text:
        tokens, masks = bert_encode(text)
        test_tokens.append(tokens)
        test_pad_masks.append(masks)
        
    test_tokens = np.array(test_tokens)
    test_pad_masks = np.array(test_pad_masks)
    
    # encode test text and get mask
    test_tokens = []
    test_pad_masks = []
    for text in test.text:
        tokens, masks = bert_encode(text)
        test_tokens.append(tokens)
        test_pad_masks.append(masks)
        
    test_tokens = np.array(test_tokens)
    test_pad_masks = np.array(test_pad_masks)
    
    # build test dataset and dataloader
    test_dataset = TestDataset(
        test_tokens=test_tokens,
        test_pad_masks=test_pad_masks
    )
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=3, shuffle=False)
    
    # get result from test dataset
    model.eval()
    y_preds = []
    for (tokens, masks) in test_dataloader:

        y_pred = model(
                    tokens.long().to(device), 
                    masks.long().to(device), 
                )
        y_preds += y_pred.detach().cpu().numpy().squeeze().tolist()
        
    # get submission dataframe
    submission_df = pd.read_csv(os.path.join(path_to_dataset, 'sample_submission.csv'))
    submission_df['target'] = (np.array(y_preds) > 0).astype('int')
    print(submission_df.target.value_counts())
    print(submission_df.head())
    
    # output csv
    submission_df.to_csv('submission.csv', index=False)
    
if __name__ == "__main__":
    process()