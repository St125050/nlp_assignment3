import streamlit as st
import torch
import pandas as pd
from collections import Counter
import unicodedata
from datasets import load_dataset
from datasets.utils import DownloadConfig
from huggingface_hub.utils import HfHubHTTPError

# Define constants for special tokens
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3

# Define the CustomTokenizer class
class CustomTokenizer:
    def __init__(self, texts, max_vocab_size=50000, language='en'):
        self.max_vocab_size = max_vocab_size
        self.language = language
        self.word2idx = {'<unk>': UNK_IDX, '<pad>': PAD_IDX, '<sos>': SOS_IDX, '<eos>': EOS_IDX}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)

        word_freq = Counter()
        for text in texts:
            words = text.lower().split() if language == 'en' else unicodedata.normalize('NFKC', text).split()
            word_freq.update(words)

        for word, _ in word_freq.most_common(max_vocab_size - len(self.word2idx)):
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

    def encode(self, text):
        words = text.lower().split() if self.language == 'en' else unicodedata.normalize('NFKC', text).split()
        return [SOS_IDX] + [self.word2idx.get(word, UNK_IDX) for word in words] + [EOS_IDX]

    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<unk>') for idx in indices if idx not in {PAD_IDX, SOS_IDX, EOS_IDX}])

# Load the dataset
try:
    dataset = load_dataset('wmt14', 'fr-en', split='train[:1%]', download_config=DownloadConfig(delete_extracted=True))
    # Extract English and French sentences
    en_texts = [example['en'] for example in dataset]
    fr_texts = [example['fr'] for example in dataset]
except HfHubHTTPError as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Dummy Encoder, Decoder, Seq2SeqTransformer (replace these with your actual model implementations)
class Encoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Decoder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class Seq2SeqTransformer(torch.nn.Module):
    def __init__(self, encoder, decoder, pad_idx, eos_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.device = device
    def forward(self, src, trg):
        return torch.randn(src.size(0), trg.size(1), 10).to(self.device)  # Dummy output for testing

# Load the trained model
def load_model(model_path, attn_variant, device):
    enc = Encoder()
    dec = Decoder()
    model = Seq2SeqTransformer(enc, dec, PAD_IDX, EOS_IDX, device).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Translation function
def translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, device, max_length=128):
    model.eval()
    src_tokens = torch.tensor([src_tokenizer.encode(sentence)]).to(device)
    trg_tokens = torch.tensor([[SOS_IDX]]).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            output = model(src_tokens, trg_tokens)
            pred_token = output.argmax(2)[:, -1].item()
            trg_tokens = torch.cat([trg_tokens, torch.tensor([[pred_token]]).to(device)], dim=1)
            if pred_token == EOS_IDX:
                break
    translated_text = trg_tokenizer.decode(trg_tokens.squeeze().cpu().numpy())
    return translated_text

# Streamlit app
st.title("English to French Translator")

# Load models and tokenizers
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
src_tokenizer = CustomTokenizer(en_texts, language='en')
trg_tokenizer = CustomTokenizer(fr_texts, language='fr')

general_model_path = 'en-fr-transformer-general.pt'
multiplicative_model_path = 'en-de-transformer-multiplicative.pt'

general_model = load_model(general_model_path, 'general', device)
multiplicative_model = load_model(multiplicative_model_path, 'multiplicative', device)

# User input
input_text = st.text_area("Enter English sentences (one per line):")
model_choice = st.selectbox("Choose the Attention Model", ("Multiplicative", "General"))

if st.button("Translate"):
    if input_text:
        sentences = input_text.split('\n')
        results = {
            "English Sentence": [],
            "Translation": []
        }
        
        model = general_model if model_choice == "General" else multiplicative_model
        
        for sentence in sentences:
            results["English Sentence"].append(sentence)
            results["Translation"].append(translate_sentence(model, sentence, src_tokenizer, trg_tokenizer, device))
        
        results_df = pd.DataFrame(results)
        st.write("Translation Results:")
        st.dataframe(results_df)
    else:
        st.write("Please enter some sentences to translate.")
