# kv_cache
<img width="1217" alt="5f8894be1e8a819621c83125bf09643" src="https://github.com/user-attachments/assets/8dfae8a9-aeff-4f22-a41f-56265cb2f26b">

# 代码运行

+ 无KV_cache：`python NSL-gpt2.py  "Alan Turing theorized that computers would one day become"  --n_tokens_to_generate 40`

+ 添加KV_cache：`python kv_cache.py  "Alan Turing theorized that computers would one day become"  --n_tokens_to_generate 40`

如图所示，在使用774M的模型的情况下，添加KV_cache后模型推理时间从16s降至8s。

# 实现思路

思考题一：

使用pytorch提供的API补全，按照模型网络结构图和代码中给出的提示按部就班实现即可。

思考题二-KVcache：

1、添加变量声明：

+ kv_cache声明为两个全局变量，存储每一层transformer_block中已有token的K，V矩阵。
+ 标识变量base：0代表开始构建prompt的K,V。1表示KV_cache中已有缓存，可以拼接新的KV
+ cur_len：现在已有的token数量

2、generate函数修改

```python
        if next_id is None:
            logits = gpt2(inputs, params, n_head=n_head)  # model forward pass
            base = 1
        else:
            logits = gpt2(next_id, params, n_head=n_head)
```

有基础KV_cache后，只需要传入已生成的最后一个token

3、gpt2函数修改

```python
    if base == 0:
        cur_len = len(inputs)
        x = wte[inputs] + wpe[range(cur_len)]  # [n_seq] -> [n_seq, n_embd]
    else:
        x = wte[inputs] + wpe[cur_len]
        cur_len += 1
```

有基础KV_cache后，仅需要对单个token进行编码

```python
floor = 0
    for block in blocks:
        x = transformer_block(x, block, n_head=n_head, floor=floor) 
        floor += 1
```

传入transformer_blocck时记录是第几层

4、mha函数修改

```python
qkv = x.chunk(3, dim=-1)  # [n_seq, 3*n_embd] -> 3 * [n_seq, n_embd]
q, k, v = qkv
k = torch.cat((k_cache[floor], k), dim=0)
v = torch.cat((v_cache[floor], v), dim=0)
k_cache[floor] = k
v_cache[floor] = v  # update kv_cache
```

获取cache中的kv，在第一维上进行拼接，同时更新cache

```python
causal_mask = torch.zeros(x.size(0), cur_len)
# print(causal_mask.shape)
causal_mask = causal_mask * float('-inf')
causal_mask = torch.nan_to_num(causal_mask, nan=0.0)  # get mask
```

生成掩码针对传入的单个token进行，规模为(1，cur_len)，值为0。

