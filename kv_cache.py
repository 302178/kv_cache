import numpy as np
import torch
import time
import math
import torch.nn.functional as F

torch.set_printoptions(8)

k_cache = []
v_cache = []
cur_len = 0
base = 0

def gelu(x):
    # x_output = F.gelu(x)
    sqrt_two_over_pi = torch.tensor(math.sqrt(2 / math.pi), dtype=x.dtype)  # calculate PI
    x_output = 0.5 * x * (1 + torch.tanh(sqrt_two_over_pi * (x + 0.044715 * x.pow(3))))

    return x_output


def softmax(x):
    """
        Task: Use torch API to implement `softmax` function, search the specific formula by yourself
        Input: Tensor
        Output: Tensor
    """
    # print(x.shape)
    x_max = torch.max(x, dim=-1, keepdim=True)[0]
    e_x = torch.exp(x - x_max)
    softmax_result = e_x / torch.sum(e_x, dim=-1, keepdim=True)

    return softmax_result


def layer_norm(x, g_b, eps: float = 1e-5):
    """
        Task: Use torch API to implement `layernorm` function, search `layernorm` by yourself
        Input:
            x: Tensor
            g_b: dictionary that load from gpt2 weight. g-gamma and b-bias are the keys
        Output: Tensor
    """

    g, b = torch.Tensor(g_b['g']), torch.Tensor(g_b['b'])

    x_mean = x.mean(-1, keepdim=True)
    x_var = x.var(-1, unbiased=False, keepdim=True)
    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)

    return g * x_norm + b


def linear(x, w_b):  # [m, in], [in, out], [out] -> [m, out]
    """
        Task: implement linear layer
        Input:
            x: Tensor
            w_b: dictionary that load from gpt2 weight. w-weight and b-bias are the keys
        Output: Tensor
    """
    w, b = w_b['w'], w_b['b']

    w = torch.Tensor(w)
    b = torch.Tensor(b)

    return x @ w + b

def ffn(x, mlp):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: use `gelu` `linear` to implement ffn
        Notes: x --linear--> --gelu--> --linear--> output
        Input:
            x: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    w_b1, w_b2 = mlp['c_fc'], mlp['c_proj']
    w1, b1 = w_b1['w'], w_b1['b']
    w2, b2 = w_b2['w'], w_b2['b']
    w1 = torch.Tensor(w1)
    b1 = torch.Tensor(b1)
    w2 = torch.Tensor(w2)
    b2 = torch.Tensor(b2)

    x = x @ w1 + b1
    x = gelu(x)
    x = x @ w2 + b2

    return x



def attention(q, k, v, mask):  # [n_q, d_k], [n_k, d_k], [n_k, d_v], [n_q, n_k] -> [n_q, d_v]
    """
        Task: use torch API to implement attention computation according to formula(1) of the following paper
              where d_k account for the last dimension of `k`
        Paper: https://arxiv.org/abs/1706.03762
        Input:
            q: Tensor
            k: Tensor
            v: Tensor
            mask: Tensor
            mlp: dictionary that load from gpt2 weight. w_b1 and w_b2 are the params of two linear layer
        Output: Tensor
    """
    scores = torch.matmul(q, k.transpose(0, 1))
    # print(f'{scores.shape} and {mask.shape}')
    scores += mask
    d_k = q.size(-1)
    scores = scores / math.sqrt(d_k)
    attn_w = softmax(scores)
    output = torch.matmul(attn_w, v)
    return output


def mha(x, attn, n_head, floor):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
        Task: Complete the code of the multi-head attention

        Input:
            x: Tensor
            attn: dictionary that load from gpt2 weight. c_attn and c_proj are the params of two linear layer
            n_head: number of head
        Output: Tensorying multi-head attention and linear transformation, shape [n_seq, n_embd].
    """
    global base, cur_len
    c_attn, c_proj = attn['c_attn'], attn['c_proj']
    if base == 0:
        # qkv projection
        x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

        # Split into qkv
        """
            Task: Split the q,k,v matrix from the tensor x
            Notes: [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
        """
        qkv = x.chunk(3, dim=-1)  # [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
        _, k, v = qkv
        k_cache.append(k)
        v_cache.append(v)
        # print(f'last kv is {k_cache[-1].shape}, {v_cache[-1].shape}')

        # Split into heads
        qkv_heads = [qkv_part.chunk(n_head, dim=-1) for qkv_part in
                     qkv]  # 3 * [n_seq, n_embd] -> 3 * n_head * [n_seq, n_embd/n_head]
        qkv_heads = list(zip(*qkv_heads))  # [3, n_head, n_seq, n_embd/n_head]
        # print(len(qkv_heads))
        # Causal mask to hide future inputs from being attended to
        """
            Task: Construct mask matrix
            Notes: 
                | 0  -inf -inf ... -inf |
                | 0    0  -inf ... -inf |
                | 0    0    0  ... -inf |
                |...  ...  ... ...  ... | 
                | 0    0    0  ...   0  |
            Mask is a tensor whose dimension is [n_seq, n_seq]
        """
        causal_mask = torch.triu(torch.ones(x.size(0), x.size(0)))
        causal_mask.fill_diagonal_(0)
        causal_mask = causal_mask * float('-inf')
        causal_mask = torch.nan_to_num(causal_mask, nan=0.0)

        # Perform attention over each head
        out_heads = [attention(q, k, v, causal_mask) for q, k, v in qkv_heads]  # n_head * [n_seq, n_embd/n_head]
    else:
        # qkv projection
        x = linear(x, c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

        qkv = x.chunk(3, dim=-1)  # [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
        q, k, v = qkv
        k = torch.cat((k_cache[floor], k), dim=0)
        v = torch.cat((v_cache[floor], v), dim=0)
        k_cache[floor] = k
        v_cache[floor] = v  # update kv_cache

        # Split into heads
        q_heads = q.chunk(n_head, dim=-1)
        k_heads = k.chunk(n_head, dim=-1)  # n_heads * [n_seq, n_embd/n_head]
        v_heads = v.chunk(n_head, dim=-1)  # n_heads * [n_seq, n_embd/n_head]

        causal_mask = torch.zeros(x.size(0), cur_len)
        # print(causal_mask.shape)
        causal_mask = causal_mask * float('-inf')
        causal_mask = torch.nan_to_num(causal_mask, nan=0.0)  # get mask

        out_heads = []
        for i in range(n_head):
            q_head = q_heads[i]
            k_head = k_heads[i]
            v_head = v_heads[i]
            out_head = attention(q_head, k_head, v_head, causal_mask)
            out_heads.append(out_head)

    # Merge heads
    """
        Task: merge multi-heads results
        Notes: n_head * [n_seq, n_embd/n_head] --> [n_seq, n_embd]
    """
    x = torch.cat(out_heads, dim=-1)

    # Out projection
    x = linear(x, c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, block, n_head, floor):  # [n_seq, n_embd] -> [n_seq, n_embd]
    mlp, attn, ln_1, ln_2 = block['mlp'], block['attn'], block['ln_1'], block['ln_2']

    # multi-head causal self attention
    x = x + mha(layer_norm(x, ln_1), attn, n_head=n_head, floor=floor)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, ln_2), mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, params, n_head):  # [n_seq] -> [n_seq, n_vocab]
    wte, wpe, blocks, ln_f = params['wte'], params['wpe'], params['blocks'], params['ln_f']
    global base, cur_len
    # token + positional embeddings
    # print(blocks)
    if base == 0:
        cur_len = len(inputs)
        x = wte[inputs] + wpe[range(cur_len)]  # [n_seq] -> [n_seq, n_embd]
    else:
        x = wte[inputs] + wpe[cur_len]
        cur_len += 1

        # print(f'{inputs}\n')
        # print(f'next_id and next_x is {next_id}, {next_x.shape}\n')
    # print(wte[inputs])
    # print(f"the x shape is {x.shape},{wte[inputs].shape},{wpe[range(len(inputs))].shape}")
    x = torch.Tensor(x)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    # print(x.shape)
    # forward pass through n_layer transformer blocks
    floor = 0
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head, floor=floor)  # [n_seq, n_embd] -> [n_seq, n_embd]
        floor += 1
    # projection to vocab
    x = layer_norm(x, ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    from tqdm import tqdm
    next_id = None
    global base
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        # print(inputs)
        if next_id is None:
            logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
            base = 1
        else:
            logits = gpt2(next_id, params, n_head=n_head)

        next_id = np.argmax(logits[-1])  # greedy sampling
        # print(next_id)
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 5, model_size: str = "774M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    start = time.time()
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    end = time.time()
    print(f"Time taken to generate {n_tokens_to_generate} tokens: {end - start:.2f}s")

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    import fire

    fire.Fire(main)