import torch
import pickle
with open("./data/content_input_20220527.pickle", "rb") as fr:
    content_input = pickle.load(fr)
with open("./data/tag_input_20220527.pickle", "rb") as fr:
    tag_input = pickle.load(fr)
with open("./data/word_id_input_20220527.pickle", "rb") as fr:
    word_id_input = pickle.load(fr)
    
input_x = torch.tensor(content_input)
input_y = torch.tensor(tag_input)

from torch.utils.data import TensorDataset 
from torch.utils.data import DataLoader
from torch.utils.data import random_split

batch_size=8
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

import torch
from transformers import BertModel
pretrained_model = BertModel.from_pretrained('skt/kobert-base-v1')

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

class KoBERTNER(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768, #768
                 num_labels=29,
                 max_length=512, 
                 dr_rate=None,
                 params=None):
        super(KoBERTNER, self).__init__()
        self.bert = bert # Load model body
        self.dr_rate = dr_rate
        self.max_length = max_length
        # Set up token classification head
        self.num_labels = num_labels # num_tags
        self.classifier = nn.Linear(hidden_size, self.max_length*self.num_labels)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
            
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask) #[batch_size, hidden_dim]
            # ouitput[0] : the attention weights
            # output[1] : the output from the model
        # Apply classifier to encoder representation
        output = self.dropout(output[1])
        output = self.classifier(output) #[batch_size, max_length*num_label]
        output = output.view([-1, self.num_labels, self.max_length]) # [batch_size, num_label, max_length]
        return output

DEVICE = "cuda"
model = KoBERTNER(pretrained_model, dr_rate=0.3).to(DEVICE)
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
        src = src.type(torch.LongTensor).to(DEVICE)
        tgt = tgt.type(torch.LongTensor).to(DEVICE)
        logits = model(input_ids= src, attention_mask= torch.tensor([[1]*max_length]*batch_size).to(DEVICE))
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
            src = src.type(torch.LongTensor).to(DEVICE)
            tgt = tgt.type(torch.LongTensor).to(DEVICE)
            logits = model(input_ids= src, attention_mask= torch.tensor([[1]*max_length]*batch_size).to(DEVICE))
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


print("example run")
loss_fn = torch.nn.CrossEntropyLoss()
for src, tgt in train_loader:
    src = src.type(torch.LongTensor).to(DEVICE)
    tgt = tgt.type(torch.LongTensor).to(DEVICE)
    print(src)
    print(tgt)
    logits = model(input_ids= src, attention_mask= torch.tensor([[1]*max_length]*batch_size).to(DEVICE))
#     logits = model(token_ids= src, valid_length= [0]*len(src), segment_ids= tgt)
    print('logits: ')
    print(logits)
    loss = loss_fn(logits, tgt)
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    break
    
# # example
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
# train_epoch(model, optimizer=optimizer, data_loader=train_loader, max_length=max_length, DEVICE='cpu') 
# test_epoch(model, optimizer=optimizer, data_loader=train_loader, max_length=max_length)    
        
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
    
save_directory = 'run_02'
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
