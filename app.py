import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import os
import unicodedata
from collections import Counter

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Constants for special tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
special_tokens = ['<unk>', '<pad>', '<sos>', '<eos>']

# Model parameters
input_dim = 46880  # Source vocabulary size (English)
output_dim = 50000  # Target vocabulary size (Gujarati)
hid_dim = 128
enc_layers = 2
dec_layers = 2
enc_heads = 4
dec_heads = 4
enc_pf_dim = 256
dec_pf_dim = 256
enc_dropout = 0.1
dec_dropout = 0.1

# Define the CustomTokenizer class
class CustomTokenizer:
    def __init__(self, texts=None, max_vocab_size=50000, language='en'):
        self.max_vocab_size = max_vocab_size
        self.language = language
        self.word2idx = {'<unk>': UNK_IDX, '<pad>': PAD_IDX, '<sos>': SOS_IDX, '<eos>': EOS_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(special_tokens)
        
        if texts is not None:
            # Build vocabulary
            word_freq = Counter()
            for text in texts:
                # Apply language-specific normalization
                if language == 'gu':
                    text = unicodedata.normalize('NFKD', text)
                else:
                    text = text.lower().strip()
                words = text.split()
                word_freq.update(words)
            
            # Add most common words to vocabulary
            for word, freq in word_freq.most_common(max_vocab_size - len(special_tokens)):
                if word not in self.word2idx:
                    self.word2idx[word] = self.vocab_size
                    self.idx2word[self.vocab_size] = word
                    self.vocab_size += 1
    
    def encode(self, text):
        text = unicodedata.normalize('NFKD', text) if self.language == 'gu' else text.lower().strip()
        words = text.split()
        return [SOS_IDX] + [self.word2idx.get(word, UNK_IDX) for word in words] + [EOS_IDX]
    
    def decode(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices if idx not in [PAD_IDX, SOS_IDX, EOS_IDX]])

# Define the model classes
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, attn_variant, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.attn_variant = attn_variant
        self.device = device
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        if attn_variant == 'multiplicative':
            self.W = nn.Linear(self.head_dim, self.head_dim)
        elif attn_variant == 'additive':
            self.Wa = nn.Linear(self.head_dim, self.head_dim)
            self.Ua = nn.Linear(self.head_dim, self.head_dim)
            self.V = nn.Linear(self.head_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        if self.attn_variant == 'multiplicative':
            K_transformed = self.W(K)
            energy = torch.matmul(Q, K_transformed.transpose(-2, -1)) / self.scale
        elif self.attn_variant == 'general':
            energy = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        else:  # additive
            Q_transformed = self.Wa(Q)
            K_transformed = self.Ua(K)
            energy = torch.tanh(Q_transformed.unsqueeze(-2) + K_transformed.unsqueeze(-3))
            energy = self.V(energy).squeeze(-1)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(500, hid_dim)
        self.layers = nn.ModuleList([
            EncoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
            for _ in range(n_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, attn_variant, device)
        self.positionwise_feedforward = nn.Sequential(
            nn.Linear(hid_dim, pf_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(pf_dim, hid_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, attn_variant, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(500, hid_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(hid_dim, n_heads, pf_dim, dropout, attn_variant, device)
            for _ in range(n_layers)
        ])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
        
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# Load model and tokenizers
def load_model_and_tokenizers(model_path, src_tokenizer_path, trg_tokenizer_path):
    # Load model
    encoder = torch.load(f"{model_path}_encoder.pt")
    decoder = torch.load(f"{model_path}_decoder.pt")
    model = Seq2SeqTransformer(encoder, decoder, PAD_IDX, PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(f"{model_path}.pt"))
    model.eval()

    if not os.path.exists(src_tokenizer_path) or not os.path.exists(trg_tokenizer_path):
        # Load dataset and create tokenizers
        dataset = load_dataset("opus100", "en-fr")
        src_texts = [example['translation']['en'] for example in dataset['train']]
        trg_texts = [example['translation']['fr'] for example in dataset['train']]
        
        src_tokenizer = CustomTokenizer(src_texts, max_vocab_size=input_dim, language='en')
        trg_tokenizer = CustomTokenizer(trg_texts, max_vocab_size=output_dim, language='fr')
        
        # Save tokenizers
        torch.save({'word2idx': src_tokenizer.word2idx, 'idx2word': src_tokenizer.idx2word}, src_tokenizer_path)
        torch.save({'word2idx': trg_tokenizer.word2idx, 'idx2word': trg_tokenizer.idx2word}, trg_tokenizer_path)
    else:
        src_tokenizer = CustomTokenizer(torch.load(src_tokenizer_path)['word2idx'], torch.load(src_tokenizer_path)['idx2word'], special_tokens)
        trg_tokenizer = CustomTokenizer(torch.load(trg_tokenizer_path)['word2idx'], torch.load(trg_tokenizer_path)['idx2word'], special_tokens)

    return model, src_tokenizer, trg_tokenizer

def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, device, max_length=128):
    model.eval()

    # Tokenize and encode the source sentence
    src_tokens = torch.tensor([src_tokenizer.encode(sentence)]).to(device)

    # Initialize target sequence with <sos>
    trg_tokens = torch.tensor([[SOS_IDX]]).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Get model prediction
            output, _ = model(src_tokens, trg_tokens)

            # Get the next token prediction
            pred_token = output.argmax(2)[:, -1].item()

            # Add predicted token to target sequence
            trg_tokens = torch.cat([trg_tokens, torch.tensor([[pred_token]]).to(device)], dim=1)

            # Stop if <eos> is predicted
            if pred_token == EOS_IDX:
                break

    # Convert tokens back to text
    translated_text = trg_tokenizer.decode(trg_tokens.squeeze().cpu().numpy())
    return translated_text

# Initialize tokenizers and model
model_path = "en-fr-transformer-multiplicative"
src_tokenizer_path = "src_tokenizer.pth"
trg_tokenizer_path = "trg_tokenizer.pth"
model, src_tokenizer, trg_tokenizer = load_model_and_tokenizers(model_path, src_tokenizer_path, trg_tokenizer_path)

# Streamlit app
st.title("English to French Translator")
st.write("Translate English sentences to French using a Transformer model.")

input_text = st.text_input("Enter an English sentence:")
if st.button("Translate"):
    if input_text:
        translation = translate_sentence(model, input_text, src_tokenizer, trg_tokenizer, device)
        st.write("Translation:")
        st.write(translation)
    else:
        st.write("Please enter a sentence to translate.")
