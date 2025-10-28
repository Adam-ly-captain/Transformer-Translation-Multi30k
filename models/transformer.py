from .base import BaseModel, BaseComponent
from utils.activation import get_activation_module
import torch.nn as nn
import torch
import math


class FeedForwardNetwork(BaseComponent):
    def __init__(self, activation=nn.ReLU(), dim_feedforward=[2048, 512, 256], dropout=0.1):
        super(FeedForwardNetwork, self).__init__()

        layers = []
        in_dim = dim_feedforward[0]
        for i, out_dim in enumerate(dim_feedforward[1:]):
            layers.append(nn.Linear(in_features=in_dim, out_features=out_dim))
            if i < len(dim_feedforward) - 2:
                layers.append(activation)
                layers.append(nn.Dropout(p=dropout))
            in_dim = out_dim
            
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.mlp(x)
    
    
class SelfAttention(BaseComponent):
    def __init__(self, word_dim, nheads, masked=False):
        super(SelfAttention, self).__init__()
        
        self.masked = masked
        self.word_dim = word_dim
        self.nheads = nheads
        self.d_k = word_dim // nheads
        self.q_linear = nn.Linear(self.d_k, self.d_k)
        self.k_linear = nn.Linear(self.d_k, self.d_k)
        self.v_linear = nn.Linear(self.d_k, self.d_k)

    def forward(self, q, k, v, padding_mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)
        atten = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.word_dim // self.nheads, dtype=torch.float32))
        if self.masked:
            seq_len = q.size(1)
            mask = torch.triu(torch.ones((seq_len, seq_len), device=q.device), diagonal=1).bool()
            atten = atten.masked_fill(mask, float('-inf'))
        
        if padding_mask is not None:
            padding_mask = padding_mask.unsqueeze(1).repeat(1, q.shape[1], 1)
            atten = atten.masked_fill(padding_mask, float('-inf'))
            
        atten = torch.softmax(atten, dim=-1)
        atten = torch.matmul(atten, v)
        return atten
    

class MultiHeadAttention(BaseComponent):
    def __init__(self, word_dim, nheads, masked=False):
        super(MultiHeadAttention, self).__init__()
        
        self.masked = masked
        self.word_dim = word_dim
        self.nheads = nheads
        self.d_k = word_dim // nheads
        self.self_attention_heads = nn.ModuleList(
            [SelfAttention(word_dim=word_dim, nheads=nheads, masked=masked) for _ in range(nheads)]
        )
        self.head_project = nn.Linear(self.word_dim, self.word_dim)

    def forward(self, q, k, v, padding_mask=None):
        q = q.view(q.size(0), q.size(1), self.nheads, self.d_k)
        k = k.view(k.size(0), k.size(1), self.nheads, self.d_k)
        v = v.view(v.size(0), v.size(1), self.nheads, self.d_k)

        atten = self.self_attention_heads[0](q[:, :, 0, :], k[:, :, 0, :], v[:, :, 0, :], padding_mask=padding_mask)
        for i, head in enumerate(self.self_attention_heads[1:]):
            atten = torch.cat((atten, head(q[:, :, i + 1, :], k[:, :, i + 1, :], v[:, :, i + 1, :], padding_mask=padding_mask)), dim=-1)

        atten = self.head_project(atten)
        return atten


class TransformerEncoder(BaseComponent):
    def __init__(self, nheads, activation=nn.ReLU(), dim_feedforward=[128, 512, 256], dropout=0.1, embedding_dim=128):
        super(TransformerEncoder, self).__init__()

        assert embedding_dim % nheads == 0, "Embedding dimension must be divisible by number of heads."
        assert dim_feedforward[0] == embedding_dim, "First dimension of feedforward network must match embedding dimension."
        
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=True)
        self.feed_forward = FeedForwardNetwork(activation=activation, dim_feedforward=dim_feedforward, dropout=dropout)
        self.multi_head_attention = MultiHeadAttention(word_dim=embedding_dim, nheads=nheads, masked=False)
        
    def forward(self, x, padding_mask=None):
        q = x
        k = x
        v = x
        atten = self.multi_head_attention(q, k, v, padding_mask=padding_mask)
        atten = self.add_norm(x, atten)
        feed_forward_out = self.feed_forward(atten)
        return self.add_norm(atten, feed_forward_out)
    
    def add_norm(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)


class TransformerDecoder(BaseComponent):
    def __init__(self, nheads, activation=nn.ReLU(), dim_feedforward=[2048, 512, 256], dropout=0.1, embedding_dim=128):
        super(TransformerDecoder, self).__init__()

        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim, elementwise_affine=True)
        self.feed_forward = FeedForwardNetwork(activation=activation, dim_feedforward=dim_feedforward, dropout=dropout)
        self.multi_head_attention_masked = MultiHeadAttention(word_dim=embedding_dim, nheads=nheads, masked=True)
        self.multi_head_attention = MultiHeadAttention(word_dim=embedding_dim, nheads=nheads, masked=False)

    def forward(self, x, encoder_output, tgt_padding_mask=None, src_padding_mask=None):
        q = x
        k = x
        v = x
        masked_atten = self.multi_head_attention_masked(q, k, v, padding_mask=tgt_padding_mask)
        masked_atten = self.add_norm(x, masked_atten)
        
        q = masked_atten
        k = encoder_output
        v = encoder_output
        atten = self.multi_head_attention(q, k, v, padding_mask=src_padding_mask)
        atten = self.add_norm(masked_atten, atten)
        feed_forward_out = self.feed_forward(atten)
        return self.add_norm(atten, feed_forward_out)

    def add_norm(self, x, sublayer_output):
        return self.layer_norm(x + sublayer_output)
    
    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Transformer(BaseModel):
    def __init__(
        self, 
        model_name="transformer",
        dataset_config=None,
        **kwargs
    ):
        super(Transformer, self).__init__(model_name=model_name)
        self.model_config = kwargs
        self.dataset_config = dataset_config or {}

        self.nhead = kwargs.get("nhead", 8)
        self.num_encoder_layers = kwargs.get("num_encoder_layers", 6)
        self.num_decoder_layers = kwargs.get("num_decoder_layers", 6)
        self.dim_encoder_feedforward = kwargs.get("dim_encoder_feedforward")
        self.dim_decoder_feedforward = kwargs.get("dim_decoder_feedforward")
        self.dropout = kwargs.get("dropout", 0.1)
        self.activation = get_activation_module(
            kwargs.get("activation", "relu")
        )
        self.word_dim = kwargs.get("word_dim", 128)
        self.device = kwargs.get("device", "cpu")
        
        self.src_word_embedding = nn.Embedding(
            num_embeddings=dataset_config.get("src_vocab_size"),
            embedding_dim=self.word_dim,
            padding_idx=dataset_config.get("pad_idx", 0)
        )

        self.tgt_word_embedding = nn.Embedding(
            num_embeddings=dataset_config.get("tgt_vocab_size"),
            embedding_dim=self.word_dim,
            padding_idx=dataset_config.get("pad_idx", 0)
        )
        
        self.encoders = nn.ModuleList(
            [TransformerEncoder(
                nheads=self.nhead, activation=self.activation, dim_feedforward=self.dim_encoder_feedforward, dropout=self.dropout, embedding_dim=self.word_dim
            ) for _ in range(self.num_encoder_layers)]
        )
        
        self.decoders = nn.ModuleList(
            [TransformerDecoder(
                nheads=self.nhead, activation=self.activation, dim_feedforward=self.dim_decoder_feedforward, dropout=self.dropout, embedding_dim=self.word_dim
            ) for _ in range(self.num_decoder_layers)]
        )
        
        self.final_linear = nn.Linear(self.dim_decoder_feedforward[-1], dataset_config.get("tgt_vocab_size"))
    
    def sinusoidal_position_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x, mode='train'):
        if mode == 'train':
            # parallel training
            src, tgt_shift, tgt = x
            src = src.to(self.device)
            tgt_shift = tgt_shift.to(self.device)
            tgt = tgt.to(self.device)
            
            src_padding_mask = src == 0
            tgt_padding_mask = tgt_shift == 0
            src_emb = self.src_word_embedding(src)
            tgt_emb = self.tgt_word_embedding(tgt_shift)
            src_pe = self.sinusoidal_position_encoding(src.size(1), self.word_dim).to(self.device)
            tgt_pe = self.sinusoidal_position_encoding(tgt_shift.size(1), self.word_dim).to(self.device)
            src_emb = src_emb + src_pe  # element-wise addition
            tgt_emb = tgt_emb + tgt_pe  # element-wise addition

            for encoder in self.encoders:
                src_emb = encoder(src_emb, src_padding_mask)

            for decoder in self.decoders:
                tgt_emb = decoder(tgt_emb, src_emb, tgt_padding_mask, src_padding_mask)

            logits = self.final_linear(tgt_emb)
            return logits
        
        elif mode == 'predict':
            return self.predict(x)

    def predict(self, x, beam_size=5):
        # auto-regressive prediction
        src, tgt_start, tgt = x
        src = src.to(self.device)
        tgt_start = tgt_start.to(self.device)
        tgt_max_len = tgt.size(1)

        src_padding_mask = src == 0
        src_emb = self.src_word_embedding(src)
        src_pe = self.sinusoidal_position_encoding(src.size(1), self.word_dim).to(self.device)
        tgt_pe = self.sinusoidal_position_encoding(tgt_max_len, self.word_dim).to(self.device)
        src_emb = src_emb + src_pe  # element-wise addition
        for encoder in self.encoders:
            src_emb = encoder(src_emb, src_padding_mask)

        batch_size = src.size(0)
        beam_candidates = [
            (
                torch.ones((batch_size, 1), dtype=torch.long, device=self.device), 
                torch.zeros(batch_size, dtype=torch.float, device=self.device)
            ) for _ in range(beam_size)
        ]

        # generated = torch.ones((batch_size, tgt_max_len + 1), dtype=torch.long, device=self.device)  # +1 for start token
        # generated[:, 0] = tgt_start[:, 0]
        for t in range(1, tgt_max_len + 1):
            temp_candidates = []
            for seq, score in beam_candidates:
                tgt_input = seq
                tgt_padding_mask = tgt_input == 0
                
                tgt_emb = self.tgt_word_embedding(tgt_input)
                tgt_emb = tgt_emb + tgt_pe[:t, :]  # element-wise addition

                for decoder in self.decoders:
                    tgt_emb = decoder(tgt_emb, src_emb, tgt_padding_mask, src_padding_mask)

                logits = self.final_linear(tgt_emb)
                prob = torch.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(prob[:, -1, :], beam_size)
                # print(topk_indices.shape)
                for i in range(beam_size):
                    next_seq = torch.cat((seq, topk_indices[:, i].unsqueeze(1)), dim=-1)
                    next_score = score + topk_probs[:, i]
                    temp_candidates.append((next_seq, next_score))

            beam_candidates = self.beam_search_sorted(batch_size, temp_candidates, beam_size=beam_size)

            # if (next_token == self.dataset_config.get("eos_idx", 2)).all():
            #     break

        return self.beam_search_sorted(batch_size, beam_candidates, beam_size=1)[0][0], logits

    def beam_search_sorted(self, batch_size, beam_candidates, beam_size=5):
        # Sort the beam candidates by their scores
        new_candidates = [(torch.tensor([], dtype=torch.long).to(self.device), torch.tensor([], dtype=torch.float).to(self.device)) for _ in range(beam_size)]
        for i in range(batch_size):
            sample_scores = []
            for seq, score in beam_candidates:
                sample_scores.append(score[i])

            _, indices = torch.topk(torch.tensor(sample_scores), beam_size)
            for beam_idx, idx in enumerate(indices):
                new_candidates[beam_idx] = (
                    torch.cat((new_candidates[beam_idx][0], beam_candidates[idx][0][i].unsqueeze(0)), dim=0),
                    torch.cat((new_candidates[beam_idx][1], beam_candidates[idx][1][i].unsqueeze(0)), dim=0)
                )
                
        return new_candidates