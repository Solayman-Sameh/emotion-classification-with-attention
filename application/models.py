"""Padding-aware classifiers; this file is also shipped in the application."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AdditiveAttention(nn.Module):
    def __init__(self,dim,projection):
        super().__init__(); self.project=nn.Linear(dim,projection); self.score=nn.Linear(projection,1,bias=False)
    def forward(self,states,mask):
        if (~mask.any(1)).any(): raise ValueError('All-padding attention is invalid')
        # Float32 softmax avoids fp16 overflow; weights remain actual probabilities.
        scores=self.score(torch.tanh(self.project(states))).squeeze(-1).float()
        scores=scores.masked_fill(~mask,float('-inf'))
        weights=torch.softmax(scores,dim=1).masked_fill(~mask,0)
        return torch.bmm(weights.to(states.dtype).unsqueeze(1),states).squeeze(1),weights

class RecurrentClassifier(nn.Module):
    def __init__(self,kind,vocab_size,embedding_dim=50,hidden=128,dropout=.3,attention_dim=128,embedding=None):
        super().__init__(); self.kind=kind
        self.embedding=nn.Embedding(vocab_size,embedding_dim,padding_idx=0)
        if embedding is not None:
            with torch.no_grad(): self.embedding.weight.copy_(torch.as_tensor(embedding))
        with torch.no_grad(): self.embedding.weight[0].zero_()
        bidir=kind=='bilstm_attention'; cls=nn.GRU if kind=='gru' else nn.LSTM
        self.rnn=cls(embedding_dim,hidden,batch_first=True,bidirectional=bidir)
        self.attention=AdditiveAttention(hidden*2,attention_dim) if bidir else None
        self.drop=nn.Dropout(dropout); self.head=nn.Linear(hidden*(2 if bidir else 1),6)
    def forward(self,ids):
        mask=ids.ne(0); lengths=mask.sum(1)
        if (lengths==0).any(): raise ValueError('Empty/all-padding sequences rejected')
        # Right-padding is an explicit input invariant.
        if ((~mask).cummax(1).values & mask).any(): raise ValueError('Only right-padding is supported')
        packed=pack_padded_sequence(self.embedding(ids),lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_out,hidden=self.rnn(packed)
        if isinstance(hidden,tuple): hidden=hidden[0]
        if self.attention:
            states,_=pad_packed_sequence(packed_out,batch_first=True,total_length=ids.shape[1])
            context,weights=self.attention(states,mask)
        else: context,weights=hidden[-1],None
        return self.head(self.drop(context)),weights

def load_recurrent(path,device='cpu'):
    checkpoint=torch.load(path,map_location=device,weights_only=True)
    model=RecurrentClassifier(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model,checkpoint
