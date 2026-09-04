import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tokenizers import Tokenizer

# --- 1. الثوابت المعمارية للنموذج ---
VOCAB = 8000
MAX_LEN = 256
EMBED_DIM = 256
N_HEADS = 8
N_BLOCKS = 6
FF_DIM = 1024
DROPOUT = 0.1

SPECIALS = ["<PAD>", "<UNK>", "<|user|>", "<|assistant|>", "<EOS>"]

# --- 2. تحميل التوكنايزر ---
tok = Tokenizer.from_file("assistant_bpe_tokenizer.json")
PAD = tok.token_to_id("<PAD>")
EOS = tok.token_to_id("<EOS>")

# --- 3. بناء هيكل النموذج وتحميل الأوزان ---
def block(x, i):
    h = layers.LayerNormalization(epsilon=1e-6, name=f"ln_attn_{i}")(x)
    a = layers.MultiHeadAttention(
        num_heads=N_HEADS, key_dim=EMBED_DIM // N_HEADS, dropout=DROPOUT, name=f"attn_{i}"
    )(h, h, use_causal_mask=True)
    x = layers.Add(name=f"res_attn_{i}")([x, a])
    
    h = layers.LayerNormalization(epsilon=1e-6, name=f"ln_ffn_{i}")(x)
    f = layers.Dense(FF_DIM, activation="gelu", name=f"ffn_up_{i}")(h)
    f = layers.Dense(EMBED_DIM, name=f"ffn_down_{i}")(f)
    f = layers.Dropout(DROPOUT, name=f"ffn_drop_{i}")(f)
    return layers.Add(name=f"res_ffn_{i}")([x, f])

inp = layers.Input(shape=(MAX_LEN - 1,), dtype="int32", name="tokens")
x = layers.Embedding(VOCAB, EMBED_DIM, name="tok_emb")(inp) + layers.Embedding(MAX_LEN - 1, EMBED_DIM, name="pos_emb")(tf.range(MAX_LEN - 1))
x = layers.Dropout(DROPOUT, name="emb_drop")(x)

for i in range(N_BLOCKS):
    x = block(x, i)

x = layers.LayerNormalization(epsilon=1e-6, name="ln_final")(x)
logits = layers.Dense(VOCAB, name="lm_head")(x)

model = tf.keras.Model(inp, logits)
model.load_weights("assistant_gpt_weights.weights.h5")

# تسريع الرسم البياني للتنفيذ بواسطة tf.function
@tf.function(reduce_retracing=True)
def fast_predict(inp_tensor):
    return model(inp_tensor, training=False)

# --- 4. دالة التوليد (Inference) المحسّنة ---
def predict_response(message, history):
    context = ""
    for user_msg, bot_msg in history:
        context += f"<|user|> {user_msg} <|assistant|> {bot_msg} "
    
    context += f"<|user|> {message} <|assistant|>"
    input_ids = tok.encode(context).ids
    
    max_input_length = (MAX_LEN - 1) - 40
    if len(input_ids) > max_input_length:
        input_ids = input_ids[-max_input_length:]
        
    generated = list(input_ids)
    
    for _ in range(40):
        seq_len = len(generated)
        if seq_len >= (MAX_LEN - 1):
            break
            
        pad_len = (MAX_LEN - 1) - seq_len
        padded_input = generated + [PAD] * pad_len
        
        inp_tensor = tf.constant([padded_input], dtype=tf.int32)
        logits = fast_predict(inp_tensor)
        
        next_token_logits = logits[0, seq_len - 1, :] / 0.7
        top_k_logits, top_k_indices = tf.math.top_k(next_token_logits, k=10)
        probs = tf.nn.softmax(top_k_logits).numpy()
        
        next_token_idx = np.random.choice(len(probs), p=probs)
        next_token = top_k_indices.numpy()[next_token_idx]
        
        if next_token == EOS or next_token == PAD:
            break
            
        generated.append(int(next_token))
        
    response_ids = generated[len(input_ids):]
    res = tok.decode(response_ids, skip_special_tokens=True).strip()
    return res if res else "I see."

# --- 5. تشغيل واجهة المحادثة Gradio ---
demo = gr.ChatInterface(
    fn=predict_response,
    title="Custom 8.8M GPT Assistant",
    description="Custom Small GPT-style Transformer trained on DailyDialog.",
    examples=["Hi, how are you?", "What are you doing?", "Do you want to get some lunch?"]
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=10000)
