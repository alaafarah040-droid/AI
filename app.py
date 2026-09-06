import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tokenizers import Tokenizer
import gradio as gr

# --- 1. Model Configuration ---
MAX_LEN = 128
N_BLOCKS = 6
EMBED_DIM = 256
N_HEADS = 8
FF_DIM = 1024
DROPOUT = 0.0

# الأسماء المطبقة في مستودعك الخاص
TOKENIZER_PATH = "assistant_bpe_tokenizer.json"
WEIGHTS_PATH = "assistant_gpt_weights.weights.h5"

# --- 2. Load Tokenizer ---
if not os.path.exists(TOKENIZER_PATH):
    raise FileNotFoundError(f"لم يتم العثور على ملف Tokenizer: {TOKENIZER_PATH}")

tok = Tokenizer.from_file(TOKENIZER_PATH)
VOCAB_SIZE = tok.get_vocab_size()

SPECIALS = ["<PAD>", "<UNK>", "<|user|>", "<|assistant|>", "<EOS>"]
PAD_ID, UNK_ID, USER_ID, ASST_ID, EOS_ID = [tok.token_to_id(s) for s in SPECIALS]

# --- 3. Build Model Architecture ---
def build_transformer_block(x, i):
    # Pre-LN Causal Attention Block
    ln_attn = layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_{i}")(x)
    attn_out = layers.MultiHeadAttention(
        num_heads=N_HEADS, 
        key_dim=EMBED_DIM // N_HEADS, 
        name=f"attn_{i}"
    )(ln_attn, ln_attn, use_causal_mask=True)
    x = layers.Add(name=f"res_attn_{i}")([x, attn_out])

    # Pre-LN FFN Block
    ln_ffn = layers.LayerNormalization(epsilon=1e-6, name=f"ln_ffn_{i}")(x)
    ffn_up = layers.Dense(FF_DIM, activation="relu", name=f"ffn_up_{i}")(ln_ffn)
    ffn_down = layers.Dense(EMBED_DIM, name=f"ffn_down_{i}")(ffn_up)
    ffn_drop = layers.Dropout(DROPOUT, name=f"ffn_drop_{i}")(ffn_down)
    x = layers.Add(name=f"res_ffn_{i}")([x, ffn_drop])
    return x

def create_model():
    inputs = layers.Input(shape=(None,), dtype="int32", name="tokens")
    tok_emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM, name="tok_emb")(inputs)
    
    # Static positional embeddings setup
    pos_ids = tf.range(start=0, limit=tf.shape(inputs)[1], delta=1)
    pos_emb = layers.Embedding(MAX_LEN, EMBED_DIM, name="pos_emb")(pos_ids)
    
    x = layers.Add()([tok_emb, pos_emb])
    x = layers.Dropout(DROPOUT, name="emb_drop")(x)

    for i in range(N_BLOCKS):
        x = build_transformer_block(x, i)

    x = layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
    outputs = layers.Dense(VOCAB_SIZE, name="lm_head")(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="assistant_gpt")

model = create_model()

# Load Weights
if os.path.exists(WEIGHTS_PATH):
    model.load_weights(WEIGHTS_PATH)
    print("تم تحميل أوزان النموذج بنجاح!")
else:
    print(f"تنبيه: لم يتم العثور على ملف الأوزان {WEIGHTS_PATH}")

# --- 4. Generation Logic ---
def sample_next_token(logits, temperature=0.7, top_k=40):
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        top_k = min(top_k, logits.shape[-1])
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1:]
        logits = tf.where(indices_to_remove, -np.inf, logits)
    
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    return np.random.choice(len(probs), p=probs)

def generate_response(prompt, max_new_tokens=60, temperature=0.7, top_k=40):
    formatted_prompt = f"<|user|> {prompt.strip()} <|assistant|>"
    input_ids = tok.encode(formatted_prompt).ids

    for _ in range(max_new_tokens):
        cond_ids = input_ids[-MAX_LEN:]
        tensor_input = tf.constant([cond_ids], dtype=tf.int32)
        
        logits = model(tensor_input, training=False)
        next_token_logits = logits[0, -1, :]
        
        next_token = sample_next_token(next_token_logits, temperature=temperature, top_k=top_k)
        
        if next_token == EOS_ID:
            break
            
        input_ids.append(next_token)

    generated_text = tok.decode(input_ids)
    if "<|assistant|>" in generated_text:
        response = generated_text.split("<|assistant|>")[-1].replace("<EOS>", "").strip()
    else:
        response = generated_text
    return response

# --- 5. UI Setup (Gradio Interface) ---
def chat_fn(message, history):
    return generate_response(message)

demo = gr.ChatInterface(
    fn=chat_fn,
    title="🤖 Tiny AI Assistant",
    description="نموذج توليد نصوص صغير مدرب باستخدام Causal Transformer و Byte-level BPE Tokenizer.",
    examples=["How do I boil an egg?", "Give three tips for staying healthy.", "Hello! How are you?"]
)

if __name__ == "__main__":
    demo.launch()
