import torch
import pickle

# to load dictionary corpus data
with open("./data/voca2idex.pickle", "rb") as fr:
    voca2idex = pickle.load(fr)
with open("./data/idex2voca.pickle", "rb") as fr:
    idex2voca = pickle.load(fr)
with open("./data/tag2idex.pickle", "rb") as fr:
    tag2idex = pickle.load(fr)
with open("./data/idex2tag.pickle", "rb") as fr:
    idex2tag = pickle.load(fr)

# to load train data  
with open("./data/content_input_20220531.pickle", "rb") as fr:
    content_input = pickle.load(fr)
with open("./data/tag_input_20220531.pickle", "rb") as fr:
    tag_input = pickle.load(fr)
with open("./data/word_id_input_20220531.pickle", "rb") as fr:
    word_id_input = pickle.load(fr)
    
# configuration parameters
VOCA_SIZE = len(voca2idex)
print(VOCA_SIZE)      


input_x = torch.tensor(content_input)
input_y = torch.tensor(tag_input)

from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

batch_size=32
total_dataset = TensorDataset(input_x, input_y)

num_train = int(len(total_dataset)*0.8)
num_validation = int(len(total_dataset)*0.1)
num_test = len(total_dataset)-num_train-num_validation

train_dataset, validation_dataset, test_dataset = random_split(total_dataset, [num_train, num_validation, num_test])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


print(len(total_dataset))
print(len(train_dataset))
print(len(validation_dataset))
print(len(test_dataset))
print('sanity check: ', len(total_dataset)==len(train_dataset)+len(validation_dataset)+len(test_dataset))

del total_dataset, num_train, num_validation, num_test, train_dataset, validation_dataset, test_dataset

max_length = seq_length = 512
num_tag = 7

from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder
import math

# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx= int(voca2idex['<pad>']))
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

    
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = max_length):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])



class BERT(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 dim_feedforward: int = 512,
                 num_labels=num_tag,
                 max_length=max_length,
                 dropout: float = 0.1):
        super(BERT, self).__init__()
        self.src_tok_emb = TokenEmbedding(vocab_size=src_vocab_size, emb_size=emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.classifier = nn.Linear(emb_size, num_labels)
        self.num_labels = num_labels
        self.max_length = max_length
        if dropout:
            self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                src: Tensor,
                src_mask: Tensor,
                src_padding_mask: Tensor):
        # print(f'src.size(): {src.size()}')
        # print(f'src_tok_emb(src): {self.src_tok_emb(src)}')
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        # print('transformer src: ', src)
        # print('transformer src_mask: ', src_mask)
        # print('transformer src_mask.size(): ', src_mask.size())
        # print('transformer src_padding_mask: ', src_padding_mask)
        # print('transformer src_padding_mask.size(): ', src_padding_mask.size()) 
        # print('transformer src_emb: ', src_emb)
        # print('transformer src_emb.size(): ', src_emb.size())
        output = self.transformer_encoder(src=src_emb, mask=src_mask, src_key_padding_mask=src_padding_mask) 
            # [batch_size, max_length, emb_size]
        # print('transformer_encoder output.size(): ', output.size())
        output = self.dropout(output)
        output = self.classifier(output) #[batch_size, max_length*num_label]
        # print('classifier output.size(): ', output.size())
        output = output.view([-1, self.num_labels, self.max_length]) # [batch_size, num_label, max_length]
        return output 

def generate_square_subsequent_mask(size):
    mask = (torch.triu(torch.ones((size, size), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src):
    src_seq_len = src.shape[1] # src:  [batch_size, seq_length]
    src_padding_mask = (src == voca2idex['<pad>'])
    return src_padding_mask

SRC_VOCAB_SIZE = VOCA_SIZE #21062
EMB_SIZE = 512 # hidden # 512
NHEAD = 8 # 8
FFN_HID_DIM = 512  # 512
BATCH_SIZE = batch_size
NUM_ENCODER_LAYERS = 3

DEVICE = "cuda"
model = BERT(NUM_ENCODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, FFN_HID_DIM).to(DEVICE)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    
def train_epoch(model, optimizer, data_loader, max_length=max_length, DEVICE='cuda'):
    import gc
    from tqdm import tqdm
    model.train()
    loss_fn = torch.nn.CrossEntropyLoss()
    losses = 0
    num_correct = []
    num_test = 0
    torch.autograd.set_detect_anomaly(True)
    for src, tgt in data_loader:
        src = src.type(torch.LongTensor).to(DEVICE) #[batch_size, seq_length]
        tgt = tgt.type(torch.LongTensor).to(DEVICE) #[batch_size, seq_length]
        src_padding_mask = create_mask(src)
        logits = model(src, None, src_padding_mask)
            #torch.Size([batch_size, , num_tag, src_seq_length])
        optimizer.zero_grad()
        loss = loss_fn(logits, tgt)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
        optimizer.step()
        losses += loss.item()
        
        # accuracy
        _, pred_word = torch.max(logits, dim=1)
        num_correct.append(int((torch.flatten(tgt)==torch.flatten(pred_word)).sum()))
        num_test += int(torch.flatten(pred_word).size(0))
        
    accuracy = sum(num_correct) / num_test
    
    # garbage collector
    gc.collect()
    torch.cuda.empty_cache()
 
    return losses / len(data_loader), accuracy


def test_epoch(model, optimizer, data_loader, max_length=max_length, DEVICE='cuda'):
    import gc
    import numpy as np
    with torch.no_grad():
        model.eval()
        loss_fn = torch.nn.CrossEntropyLoss()
        losses = 0
        num_correct = []
        num_test = 0
        y_probability = np.array([])
        y_pred = np.array([])
        y_true = np.array([])
        for src, tgt in data_loader:
            src = src.type(torch.LongTensor).to(DEVICE) #[batch_size, seq_length]
            tgt = tgt.type(torch.LongTensor).to(DEVICE) #[batch_size, seq_length]
            src_padding_mask = create_mask(src)
            logits = model(src, None, src_padding_mask)
                #torch.Size([batch_size, , num_tag, src_seq_length])
            
            loss = loss_fn(logits, tgt)
            losses += loss.item()
            
            # accuracy
            _, pred_word = torch.max(logits, dim=1)
            num_correct.append(int((torch.flatten(tgt)==torch.flatten(pred_word)).sum()))
            num_test += int(torch.flatten(pred_word).size(0))
            
            y_probability = np.append(y_probability, logits.cpu().detach().numpy())
            y_pred = np.append(y_pred, pred_word.cpu().detach().numpy())
            y_true = np.append(y_true, tgt.cpu().detach().numpy())
            
    accuracy = sum(num_correct) / num_test
    # garbage collector
    gc.collect()
    torch.cuda.empty_cache()
    
    return losses / len(data_loader), accuracy,  y_pred, y_true

def slack_alarm(message):
    """
    message : string
    """
    import os
    from slack import WebClient
    from slack.errors import SlackApiError

    SLACK_API_TOKEN = 'xoxb-3456243383942-3465240022692-nQxw8PlFwhcywqhYlzO3jqmX'
    client = WebClient(token=SLACK_API_TOKEN)

    try:
        response = client.chat_postMessage(channel='#deep-learning',text=message)
        assert response["message"]["text"] == message

        #filepath="./tmp.txt"
        #response = client.files_upload(channels='#random', file=filepath)
        #assert response["file"]  # the uploaded file
    except SlackApiError as e:
        # You will get a SlackApiError if "ok" is False
        assert e.response["ok"] is False
        assert e.response["error"]  # str like 'invalid_auth', 'channel_not_found'
        print(f"Got an error: {e.response['error']}")
        
def train_all_sample(model, optimizer, max_length, train_loader, validation_loader, test_loader, device, total_num_epoch, running_num_epoch, tf_board_directory, model_save_directory, best_model_save_directory=None, slack_message=False):
    '''
    Parameters:
        train_loader
        validation_loader
        test_loader
        total_num_epoch : the total number of epoch that have run
        running_num_epoch : the number of epoch that run in this time
        tf_board_directory
        model_save_directory : save directory to save the final model
        best_model_save_directory : [optional] save directory to save the best model (best validation accuracy)
        slack_message : [boolian] slack message notification (current epoch)
    '''
    #################################################################################
    # load modules and set parameters
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(tf_board_directory)
        ## logdir=./python/run/GAT_Net/run_02

    if (total_num_epoch < 0 or running_num_epoch <= 0):
        import sys
        sys.exit("Check the number of epoch. It is incorrect")

    # optimizer = torch.optim.Adam(model.parameters(), lr=3e-6, weight_decay=0.95)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.7, patience=50, min_lr=1e-8)

    #################################################################################
    #running code
    import time
    from tqdm import tqdm
    import numpy as np
    total_time_start = time.time() # to measure time
    best_validation_accuracy = None
    for epoch in tqdm(range(1, running_num_epoch+1)):
        epoch_time_start = time.time() # to measure time
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss, train_accuracy = train_epoch(model, optimizer, train_loader, max_length=max_length, DEVICE=device)
        _, validation_accuracy, _, _ = test_epoch(model, optimizer, validation_loader, max_length=max_length, DEVICE=device)
        #scheduler.step(validation_accuracy)

        # to save the metrics and model
        if best_validation_accuracy is None or validation_accuracy >= best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            if not(best_model_save_directory is None):            
                torch.save({
                'epoch': total_num_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                }, best_model_save_directory)
        _, test_accuracy, _, _ = test_epoch(model, optimizer, test_loader, max_length=max_length, DEVICE=device)
        total_num_epoch = total_num_epoch + 1
        epoch_time_end = time.time() # to measure time    
        writer.add_scalar('loss in train', loss, total_num_epoch) #tensorboard
        writer.add_scalar('train accuracy', train_accuracy, total_num_epoch) #tensorboard
        writer.add_scalar('validation accuracy', validation_accuracy, total_num_epoch) #tensorboard    
        writer.add_scalar('test accuracy', test_accuracy, total_num_epoch) #tensorboard
        writer.add_scalar('learning rate', lr, total_num_epoch) #tensorboard
        # print(f'ToTal Epoch: {total_num_epoch:03d}, LR: {lr:7f}, Loss: {loss:.7f}, '
        #       f'Val MAE: {validation_error:.7f}, Test MAE: {test_error:.7f}, Time: {epoch_time_end - epoch_time_start}')

        # to send message
        if slack_message:
            slack_alarm('[Life3 JupyterNotebook] : ' + str(epoch) + ' / ' + str(running_num_epoch) + '\ntrain accuracy: ' + str(train_accuracy) + '\nvalidation_accuracy: ' + str(validation_accuracy) + '\ntest_accuracy: ' + str(test_accuracy))
            
    total_time_finish = time.time() # to measure time
    print(f'Done. Total running Time: {total_time_finish - total_time_start}')
    writer.close() #tensorboard : if close() is not declared, the writer does not save any valeus.

    # model save
    torch.save({
            'epoch': total_num_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, model_save_directory)
    print('total number of epoches : ', total_num_epoch)
    print("-------------------------done------------------------------")
    
save_directory = 'run_04'
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0015334791245047435)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.8, patience=10, min_lr=1e-4)

train_all_sample(model=model,
                optimizer=optimizer,
                max_length=max_length,
                train_loader=train_loader, 
                validation_loader=validation_loader, 
                test_loader=test_loader, 
                device='cuda', 
                total_num_epoch=0, 
                running_num_epoch=200, 
                tf_board_directory = 'tfboard/'+save_directory, 
                model_save_directory='model/'+save_directory, 
                best_model_save_directory='model/'+save_directory+'_best',
                slack_message=True)
del save_directory
# slack_alarm('The program execution on JupyterNotebook is done')
