import torch
from torch import nn, einsum
from einops import rearrange, repeat, reduce
import torch.nn.functional as F

# from vector_quantize_pytorch import VectorQuantize
from discreteKeyValue import KeyBlock, cdist

import pdb

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) and val is not None else d

# main class

class DiscreteKeyValueBottleneck(nn.Module):
    def __init__(
        self,
        # dim,
        *,
        num_memories,
        num_classes=10,
        dim_embed=None,
        num_memory_codebooks=1,
        encoder_args=None,
        dim_memory=None,
        merge_memories="None",
        **kwargs
    ):
        super().__init__()
        model_base_class = encoder_args[0]
        model_base_kwargs = encoder_args[1]
        self.encoder = model_base_class(**model_base_kwargs)
        encoder_out_dim = self.encoder.cls_dim
        dim = encoder_out_dim
        dim_embed = default(dim_embed, dim)
        self.dim_embed = dim_embed

        self.vq = VectorQuantize(
            dim = dim * num_memory_codebooks,
            codebook_size = num_memories,
            heads=num_memory_codebooks,
            separate_codebook_per_head = True,
            # learnable_codebook=True,
            # ema_update=False,
            **kwargs
        )

        dim_memory = default(dim_memory, dim)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))
        # print()
        rand_proj = torch.empty(num_memory_codebooks, dim_embed, dim, requires_grad=True)
        nn.init.xavier_normal_(rand_proj)

        self.register_buffer('rand_proj', rand_proj)
        self.merge_memories_lst = {
            "None": dim_memory,
            "average": dim_memory,
            "concat": dim_memory * num_memory_codebooks,
            "sum": dim_memory
        }
        #
        # self.merge_memories_fn = self.merge_memories_lst[merge_memories]
        self.merge_memories = merge_memories
        self.dim_classifier = self.merge_memories_lst[merge_memories]
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.dim_classifier),
            nn.ReLU(),
            nn.Linear(self.dim_classifier, num_classes)
        )

        self.count_step = 0
        self.code_book = None

    def forward(
        self,
        x,
        return_intermediates = False,
        merge_memories="None",
        freeze_codebook=False,
        kv_softmax=False,
        **kwargs
    ):
        self.count_step += 1
        merge_memories = default(merge_memories, self.merge_memories)

        pdb.set_trace()

        if exists(self.encoder):
            x = self.encoder(x, return_latent=True, **kwargs)
        if x.ndim < 3:  # -> x: b d
            x = x[:, None]     # -> x: b n d, n=1 num_tokens

        assert x.shape[-1] == self.dim_embed, f'encoding has a dimension of {x.shape[-1]} but dim_embed (defaults to dim) is set to {self.dim_embed} on init'

        x = einsum('b n d, c d e -> b n c e', x, self.rand_proj)
        x = rearrange(x, 'b n c e -> b n (c e)')

        vq_out = self.vq(x, freeze_codebook=freeze_codebook)
        code_book = self.vq._codebook.embed

        self.code_book = default(self.code_book, code_book)

        if self.count_step % 200 == 0:
            diff = self.code_book - code_book
            print("Code book changed:", diff.max().item(), diff.min().item())
            self.code_book = code_book

        quantized, memory_indices, commit_loss = vq_out
        # quantized: k value that closest to x
        # memory_indices: index of k
        if not kv_softmax:
            if memory_indices.ndim == 2:
                memory_indices = rearrange(memory_indices, '... -> ... 1')

            memory_indices = rearrange(memory_indices, 'b n h -> b h n')

            values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
            memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])

            memories = values.gather(2, memory_indices)
        else:
            kv_score =  torch.matmul(quantized, code_book.transpose(-2, -1))

        if "mean" == merge_memories:
            memories = reduce(memories, 'b h n d -> b n d', 'mean')
        elif "concat" == merge_memories:
            memories = rearrange(memories, 'b h n d -> b n h d')
            memories = rearrange(memories, 'b n h d -> b n (h d)')

        if return_intermediates:
            return memories, vq_out
        mem_size = memories.size()
        # memories = memories.view(-1, mem_size[-1])
        out = self.classifier(memories.view(-1, mem_size[-1]))
        # out_size = mem_size
        # out_size[-1] = out.size(-1)
        # out = out.view(out_size)
        return out, x, quantized, memories

class DiscreteKey_VQ(nn.Module):
    def __init__(
        self,
        num_codebooks=1,
        num_classes=10,
        dim_embed=None,
        num_key_per_codebook=10,
        encoder_args=None,
        value_dim=None,
        merge_memories="mean",
        is_embed=False,
        repeat_hidden=False,
        filter_topk_value=-1,
        **kwargs
    ):
        super().__init__()
        model_base_class = encoder_args[0]
        model_base_kwargs = encoder_args[1]
        self.encoder = model_base_class(**model_base_kwargs)
        encoder_out_dim = self.encoder.cls_dim
        dim = encoder_out_dim
        dim_embed = default(dim_embed, dim)
        self.dim_embed = dim_embed

        self.vq = VectorQuantize(
            dim = dim * num_memory_codebooks,
            codebook_size = num_memories,
            heads=num_memory_codebooks,
            separate_codebook_per_head = True,
            # learnable_codebook=True,
            # ema_update=False,
            **kwargs
        )

        dim_memory = default(dim_memory, dim)
        self.values = nn.Parameter(torch.randn(num_memory_codebooks, num_memories, dim_memory))
        # print()
        rand_proj = torch.empty(num_memory_codebooks, dim_embed, dim, requires_grad=True)
        nn.init.xavier_normal_(rand_proj)

        self.register_buffer('rand_proj', rand_proj)
        self.merge_memories_lst = {
            "None": dim_memory,
            "average": dim_memory,
            "concat": dim_memory * num_memory_codebooks,
        }
        #
        # self.merge_memories_fn = self.merge_memories_lst[merge_memories]
        self.merge_memories = merge_memories
        self.dim_classifier = self.merge_memories_lst[merge_memories]
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.dim_classifier),
            nn.ReLU(),
            nn.Linear(self.dim_classifier, num_classes)
        )

        self.count_step = 0
        self.code_book = None

    def forward(
        self,
        x,
        return_intermediates = False,
        merge_memories="None",
        freeze_codebook=False,
        kv_softmax=False,
        **kwargs
    ):
        self.count_step += 1
        merge_memories = default(merge_memories, self.merge_memories)

        pdb.set_trace()

        if exists(self.encoder):
            x = self.encoder(x, return_latent=True, **kwargs)
        if x.ndim < 3:  # -> x: b d
            x = x[:, None]     # -> x: b n d, n=1 num_tokens

        assert x.shape[-1] == self.dim_embed, f'encoding has a dimension of {x.shape[-1]} but dim_embed (defaults to dim) is set to {self.dim_embed} on init'

        x = einsum('b n d, c d e -> b n c e', x, self.rand_proj)
        x = rearrange(x, 'b n c e -> b n (c e)')

        vq_out = self.vq(x, freeze_codebook=freeze_codebook)
        code_book = self.vq._codebook.embed

        self.code_book = default(self.code_book, code_book)

        if self.count_step % 200 == 0:
            diff = self.code_book - code_book
            print("Code book changed:", diff.max().item(), diff.min().item())
            self.code_book = code_book

        quantized, memory_indices, commit_loss = vq_out
        # quantized: k value that closest to x
        # memory_indices: index of k
        if not kv_softmax:
            if memory_indices.ndim == 2:
                memory_indices = rearrange(memory_indices, '... -> ... 1')

            memory_indices = rearrange(memory_indices, 'b n h -> b h n')

            values = repeat(self.values, 'h n d -> b h n d', b = memory_indices.shape[0])
            memory_indices = repeat(memory_indices, 'b h n -> b h n d', d = values.shape[-1])

            memories = values.gather(2, memory_indices)
        else:
            kv_score =  torch.matmul(quantized, code_book.transpose(-2, -1))

        if "mean" == merge_memories:
            memories = reduce(memories, 'b h n d -> b n d', 'mean')
        elif "concat" == merge_memories:
            memories = rearrange(memories, 'b h n d -> b n h d')
            memories = rearrange(memories, 'b n h d -> b n (h d)')

        if return_intermediates:
            return memories, vq_out
        mem_size = memories.size()
        # memories = memories.view(-1, mem_size[-1])
        out = self.classifier(memories.view(-1, mem_size[-1]))
        # out_size = mem_size
        # out_size[-1] = out.size(-1)
        # out = out.view(out_size)
        return out, x, quantized, memories


# DiscreteKey_VQ, DiscreteKey_VQ_UniformKey

class DiscreteKey_VQ_UniformKey(nn.Module):
    def __init__(
        self,
        *,
        num_codebooks=1,
        num_classes=10,
        dim_embed=None,
        num_key_per_codebook=10,
        encoder_args=None,
        value_dim=None,
        merge_memories="mean",
        is_embed=True,
        repeat_hidden=False,
        filter_topk_value=-1,
        norm_hk=False,
        allow_map_wrong=False,
        hard_sofmax=False,
        **kwargs
    ):
        super().__init__()

        self.num_codebooks = num_codebooks
        self.num_key_per_codebook = num_key_per_codebook
        self.num_classes = num_classes
        self.num_key_per_class = num_key_per_codebook//num_classes
        self.repeat_hidden = repeat_hidden
        self.norm_hk = norm_hk
        self.filter_topk_value = filter_topk_value
        self.allow_map_wrong = allow_map_wrong
        self.hard_sofmax = hard_sofmax

        model_base_class = encoder_args[0]
        model_base_kwargs = encoder_args[1]
        self.encoder = model_base_class(**model_base_kwargs)
        
        self.encoder_out_dim = self.encoder.cls_dim
        default_dim = self.encoder_out_dim
        dim_embed = default(dim_embed, default_dim)
        self.dim_embed = dim_embed
        self.is_embed = is_embed

        if not self.is_embed:
            self.num_codebooks = self.encoder_out_dim//self.dim_embed
            assert self.num_codebooks*self.dim_embed == self.encoder_out_dim
        if self.repeat_hidden:
            total_dim_embed = self.dim_embed
        else:
            total_dim_embed = self.dim_embed*self.num_codebooks
        # self.embedding_module = nn.Sequential(nn.BatchNorm1d(self.encoder_out_dim),
        #                                       nn.ReLU(),
        #                                       nn.Linear(self.encoder_out_dim, total_dim_embed))
        
        embeding_m_lst = []
        for _ in range(self.num_codebooks):
            embeding_m_lst.append(nn.Sequential(nn.BatchNorm1d(self.encoder_out_dim),
                                              nn.ReLU(),
                                              nn.Linear(self.encoder_out_dim, self.dim_embed)))
        self.embedding_module = nn.ModuleList(embeding_m_lst)
        m_lst = []
        for _ in range(self.num_codebooks):
            m_lst.append(KeyBlock(num_keys=self.num_key_per_codebook, 
                                    feature_dim=self.dim_embed, 
                                    num_classes=self.num_classes))
        self.key_module = nn.ModuleList(m_lst)

        self.value_dim = default(value_dim, default_dim)
        self.values = nn.Parameter(torch.randn(self.num_codebooks, self.num_key_per_codebook, self.value_dim))


        self.merge_memories_lst = {
            "mean": self.value_dim,
            "concat": self.value_dim * self.num_codebooks,
            "sum": self.value_dim,
        }
        self.dim_classifier = self.merge_memories_lst[merge_memories]

        self.merge_memories = merge_memories
        # self.classifier = nn.Sequential(
        #     nn.Linear(self.dim_classifier, self.dim_classifier//2),
        #     nn.BatchNorm1d(self.dim_classifier),
        #     nn.ReLU(),
        #     nn.Linear(self.dim_classifier//2, num_classes)
        # )

        k_lst = []
        for _ in range(self.num_codebooks):
            k_lst.append(nn.Linear(self.dim_embed, self.dim_classifier))
        self.key_embeding = nn.ModuleList(k_lst)

        v_lst = []
        for _ in range(self.num_codebooks):
            v_lst.append(KeyBlock(num_keys=self.num_key_per_codebook, 
                                    feature_dim=self.dim_embed, 
                                    num_classes=self.num_classes))
        self.value_module = nn.ModuleList(v_lst)

        # self.classifier = nn.Linear(self.dim_classifier, num_classes)
        self.classifier =   nn.Sequential(
                            nn.Linear(self.dim_classifier, self.dim_classifier//10),
                            nn.BatchNorm1d(self.dim_classifier//10),
                            nn.ReLU(),
                            nn.Linear(self.dim_classifier//10, num_classes))
        
        h_cls_lst = []
        v_cls_lst = []
        for _ in range(self.num_codebooks):
            # h_cls_lst.append(nn.Linear(self.dim_embed, num_classes))
            h_cls_lst.append(nn.Sequential(
                            nn.Linear(self.dim_embed, self.dim_embed//10),
                            nn.BatchNorm1d(self.dim_embed//10),
                            nn.ReLU(),
                            nn.Linear(self.dim_embed//10, num_classes)))

            v_cls_lst.append(nn.Linear(self.value_dim , num_classes))
        
        self.classifier_hidden = nn.ModuleList(h_cls_lst)
        self.classifier_value = nn.ModuleList(v_cls_lst)

        self.softmax = nn.Softmax(dim=-1)
        self.running_step = 0

    def forward(
        self,
        inputX,
        return_immediate = False,
        kv_softmax=True,
        filter_topk_value=-1,
        targets=None,
        hard_sofmax=False,
        is_attacking=True,
        robust_predict=False,
        value_as_key_centroid=False,
        key_idx_to_map1= None,
        key_idx_to_map2 = None,
        value_idx_to_map = None,
        **kwargs
    ):
        hard_sofmax = hard_sofmax or self.hard_sofmax
        # encoded_x = self.encoder(inputX, return_latent=True, **kwargs)
        encoded_x = self.encoder(inputX, return_latent=True)

        # x = einsum('b d, d t -> b t', x, self.rand_proj)
        # encoded_x = self.embedding_module(encoded_x)
        # # x = encoded_x
        # if self.repeat_hidden:
        #     x = torch.stack([encoded_x for _ in range(self.num_codebooks)])
        # else:
        #     x = rearrange(encoded_x, "b (n d) -> b n d", n=self.num_codebooks)
        #     x = rearrange(x, "b n d -> n b d")
        # if self.norm_hk:
        #     x = F.normalize(x, dim=-1)
        if self.is_embed:
            x = []
            for idx_m in range(self.num_codebooks):
                e_m = self.embedding_module[idx_m](encoded_x)
                if self.norm_hk:
                    e_m = F.normalize(x, dim=-1)
                x.append(e_m)
        else:
            x = rearrange(encoded_x, "b (n d) -> b n d", n=self.num_codebooks)
            x = rearrange(x, "b n d -> n b d")
        
        memories = []     # n_codebook, bz, dim
        memories_hard = []
        quantize_key_lst = []
        all_key_lst = []
        weight_key_lst = []
        hk_mse_loss = []
        key_label_lst = []
        ratio_lst = []
        v_score_lst = []
        key_idx_map1_lst = []
        key_idx_map2_lst = []
        key_idx_wlabel_lst = []
        kk_dis_lst = []
        out_cls_hidden = []
        out_cls_value = []
        out_cls_value_soft = []
        if key_idx_to_map1 is None:
            key_idx_to_map1 = [None]*self.num_codebooks
            key_idx_to_map2 = [None]*self.num_codebooks

        idx_m_to_be_wrong = []
        if self.training and self.allow_map_wrong and (is_attacking is False or is_attacking is None):
            idx_m_to_be_wrong = torch.randperm(self.num_codebooks)[:int(1.5*self.num_codebooks/2)]

        for idx_m, key_module_m in enumerate(self.key_module):
            hidden_m = x[idx_m]
            value_m = self.values[idx_m]
            idx_min_wlabel_true, quantize_idx, idx_min_true, quantize_key_m, distance_m, ratio = key_module_m(hidden_m, hidden_label=targets, 
            hard_map=hard_sofmax, idx_key_to_map=key_idx_to_map1[idx_m], idx_key2=key_idx_to_map2[idx_m], is_attacking=is_attacking)
            key_idx_map1_lst.append(quantize_idx)
            key_idx_map2_lst.append(idx_min_true)
            ratio_lst.append(ratio)
            selected_key_unique = torch.unique(quantize_idx)
            all_key_m = key_module_m.key
            if self.norm_hk:
                all_key_m = F.normalize(all_key_m, dim=-1)
            # if robust_predict:
            # quantize_key_m = hidden_m + (all_key_m[quantize_idx] - hidden_m).detach()
            
            # hk_mse_loss.append(F.mse_loss(hidden_m.detach(), all_key_m[quantize_idx], reduction = 'none').mean())
            # hk_mse_loss.append(F.mse_loss(hidden_m, quantize_key_m, reduction = 'none').mean())
            hk_mse_loss.append(F.mse_loss(hidden_m, all_key_m[idx_min_wlabel_true].detach(), reduction = 'none').mean())
            quantize_key_lst.append(all_key_m[idx_min_wlabel_true])
            key_idx_wlabel_lst.append(idx_min_wlabel_true)
            # quantize_key_lst.append(quantize_key_m)

            all_key_lst.append(all_key_m)
            weight_key_lst.append(key_module_m.get_weight_key())
            key_label_lst.append(key_module_m.labels)
            out_cls_hidden.append(self.classifier_hidden[idx_m](hidden_m))
            if value_as_key_centroid:
                key_projected = self.key_embeding[idx_m](quantize_key_m)
                v_idx, mem_m, _ = self.value_module[idx_m](key_projected, hidden_label=targets)
            
            elif not kv_softmax:
                # mem_m = value_m[quantize_idx]
                mem_m = quantize_key_m
            else:
                ################### L2
                # kk_disl2 = F.mse_loss(hidden_m[:, None], all_key_m.detach()[None], reduction = 'none').sum(dim=-1)   # NOTE: detach all key so it allow all keys to be active. Without detach, it only active some keys
                # kk_disl2_key = F.mse_loss(hidden_m[:, None], all_key_m[None], reduction = 'none').sum(dim=-1)
                # kk_disl2_h = F.mse_loss(hidden_m.detach()[:, None], all_key_m[None], reduction = 'none').sum(dim=-1)
                
                kk_disl2 = F.mse_loss(quantize_key_m[:, None], all_key_m.detach()[None], reduction = 'none').sum(dim=-1)   # NOTE: detach all key so it allow all keys to be active. Without detach, it only active some keys
                kk_disl2_key = F.mse_loss(quantize_key_m[:, None], all_key_m[None], reduction = 'none').sum(dim=-1)
                kk_disl2_h = F.mse_loss(quantize_key_m.detach()[:, None], all_key_m[None], reduction = 'none').sum(dim=-1)
                
                if hard_sofmax:
                    kk_disl2 = F.mse_loss(quantize_key_m[:, None], all_key_m[None].detach(), reduction = 'none').sum(dim=-1)
                mean_kkl2 = kk_disl2.mean(dim=-1)
                mean_kkl2_key = kk_disl2.mean(dim=-1)
                mean_kkl2_h = kk_disl2_h.mean(dim=-1)
                # min_dis = kk_disl2.min(dim=-1)[0]
                # min_dis_key = kk_disl2_key.min(dim=-1)[0]
                # min_dis = 0
                # min_dis_key = 0
                # kk_dis = 5 - 20*self.num_key_per_codebook*kk_disl2/kk_disl2.sum(dim=-1)[..., None]
                # kk_dis_key = 5 - 20*self.num_key_per_codebook*kk_disl2_key/kk_disl2_key.sum(dim=-1)[..., None]

                # kk_dis = 5 - 10*(kk_disl2 - min_dis[..., None])/(mean_kkl2 - min_dis)[..., None]
                # kk_dis_key = 5 - 10*(kk_disl2_key - min_dis_key[..., None])/(mean_kkl2_key - min_dis_key)[..., None]
                # kk_dis = 5 - 20*(kk_disl2 - min_dis)/(mean_kkl2 - min_dis)[..., None]
                # kk_dis_key = 5 - 20*(kk_disl2_key - min_dis_key)/(mean_kkl2_key - min_dis_key)[..., None]
                # kk_dis_h = 5 - 20*(kk_disl2_h - min_dis_key)/(mean_kkl2_h - min_dis_key)[..., None]
                # kk_dis = 5 - 20*kk_disl2/mean_kkl2[..., None]
                # kk_dis_key = 5 - 20*kk_disl2/mean_kkl2_key[..., None]
                # kk_dis_h = 5 - 20*kk_disl2_h/mean_kkl2_h[..., None]
                kk_dis = kk_disl2/mean_kkl2[..., None]
                min_kk_dis = kk_dis.min(dim=-1)[0]
                kk_dis = 5 + 20*(min_kk_dis[..., None] - kk_dis)
                
                kk_dis_key = kk_disl2_key/mean_kkl2_key[..., None]
                min_kk_dis_key = kk_dis_key.min(dim=-1)[0]
                kk_dis_key = 5 + 20*(min_kk_dis_key[..., None] - kk_dis_key)
                
                # # kk_dis_h = 5 - 20*kk_disl2_h/mean_kkl2_h[..., None]

                kk_dis_h = kk_disl2_h/mean_kkl2_h[..., None]
                min_kk_dis_h = kk_dis_h.min(dim=-1)[0]
                kk_dis_h = 5 + 20*(min_kk_dis_h[..., None] - kk_dis_h)
                # kk_dis = 5-kk_disl2
                # kk_dis_key = 5-kk_disl2_key
                # kk_dis_h = 5-kk_disl2_h
                if self.filter_topk_value > 0:
                    top_k_idx = torch.topk(kk_disl2, k=self.num_key_per_codebook - self.filter_topk_value)[1]  # k largest kk_disl2 index
                    distance_weight = torch.ones_like(kk_dis)
                    kk_dis[torch.arange(len(kk_dis))[:, None], top_k_idx] = -50
                if hard_sofmax:
                    # kk_dis = 10 - 50*kk_disl2/kk_disl2.max(dim=-1)[0][..., None]
                    # min_dis = kk_disl2.min(dim=-1)[0]
                    # max_dis = kk_disl2.max(dim=-1)[0]
                    # mean_kkl2 = kk_disl2.mean(dim=-1)
                    # # kk_dis = 10 - 1000*(kk_disl2 - min_dis[..., None])/(max_dis - min_dis)[..., None]
                    # kk_dis = 5 - 20*(kk_disl2)/mean_kkl2[..., None]

                    kk_dis = kk_disl2/mean_kkl2[..., None]
                    min_kk_dis = kk_dis.min(dim=-1)[0]
                    kk_dis = 5 + 20*(min_kk_dis[..., None] - kk_dis)
                ################### Dot product
                # sqrt_dk = torch.sqrt(torch.tensor(hidden_m.shape[-1]))
                # kk_dis = torch.matmul(hidden_m, all_key_m.transpose(-1, 0).detach())/sqrt_dk
                # kk_dis_key = torch.matmul(hidden_m, all_key_m.transpose(-1, 0))/sqrt_dk
                # kk_dis_h = torch.matmul(hidden_m.detach(), all_key_m.transpose(-1, 0))/sqrt_dk
                # if hard_sofmax:
                #     kk_dis = torch.matmul(quantize_key_m, all_key_m.transpose(-1, 0).detach())/sqrt_dk
                ###################
                v_score = nn.Softmax(dim=-1)(kk_dis)       # bz, num_key
                # v_score_key = nn.Softmax(dim=-1)(kk_dis_key)       # bz, num_key
                # v_score_h = nn.Softmax(dim=-1)(kk_dis_h)       # bz, num_key
                # v_score = kk_dis
                v_score_lst.append(v_score)
                # kk_dis_lst.append(kk_dis)    # work well, key separated
                kk_dis_lst.append(kk_dis_key)
                # v_score = kk_sim_scale
                # if self.training:
                #     mem_m = hidden_m
                #     mem_m_h = (v_score[..., None]*value_m[None]).sum(dim=-2)
                # else:
                #     mem_m = (v_score[..., None]*value_m[None].detach()).sum(dim=-2)
                #     mem_m_h = hidden_m
                # mem_m2 = (v_score[..., None].detach()*value_m[None]).sum(dim=-2)
                # mem_m_h = (v_score[..., None]*value_m[None]).sum(dim=-2) 
                # mem_m_h = mem_m.detach()/2 + value_m[quantize_idx]/2
                # if self.training:
                #     rand_w = torch.randint(50, 120, [len(hidden_m), 1]).to(hidden_m.device)/100
                # else:
                #     rand_w = torch.ones([len(hidden_m), 1]).to(hidden_m.device)
                #     if not hard_sofmax:
                #         rand_w = rand_w * 0.9
                # if idx_m in idx_m_to_be_wrong and torch.rand(1) > 0.4:
                #     # import pdb; pdb.set_trace()
                #     new_idx = torch.randint(0, self.num_key_per_codebook, size=(len(idx_min_wlabel_true),)).to(hidden_m.device)
                #     is_replace_idx = torch.randint(0, 2, size=(len(idx_min_wlabel_true),)).to(hidden_m.device)
                #     new_idx = new_idx*is_replace_idx
                #     idx_min_wlabel_true = (1-is_replace_idx)*idx_min_wlabel_true + new_idx
                
                mem_m = (v_score[..., None]*value_m[None].detach()).sum(dim=-2)
                # mem_m_h = value_m[idx_min_wlabel_true]
                mem_m_h = (v_score[..., None].detach()*value_m[None]).sum(dim=-2)
                if idx_m in idx_m_to_be_wrong and torch.rand(1) > 0.4:
                    mem_m_h = (v_score[..., None]*value_m[None]).detach().sum(dim=-2)
                # out_cls_value.append(self.classifier_value[idx_m]((v_score[..., None]*value_m[None]).sum(dim=-2)))
                out_cls_value.append(self.classifier_value[idx_m](value_m[idx_min_wlabel_true]))
                out_cls_value_soft.append(self.classifier_value[idx_m]((v_score[..., None]*value_m[None]).sum(dim=-2)))
                # mem_m_h = hidden_m
                # mem_m_h = (v_score[..., None].detach()*value_m[None]).sum(dim=-2)   # testing with multiple head
                # mem_m_h = (1-rand_w)*mem_m2 + rand_w*value_m[idx_min_wlabel_true].detach()
                # mem_m_h = value_m[idx_min_wlabel_true]
                # mem_m = torch.matmul(v_score, value_m)      # (bz, num_key, 1)*(1, num_key, dim)=bz, num_key,dim
                # mem_m = torch.matmul(v_score, all_key_m)  
                # if v_score[0].max() < 0.6:
                #     import pdb; pdb.set_trace()
            memories.append(mem_m)
            memories_hard.append(mem_m_h)
        # if (self.running_step % 100==0 and self.training) or self.running_step % 500 == 0:
        # if self.running_step>10000 and self.running_step % 200 == 0 and (is_attacking is None or is_attacking is False):
        #     import pdb; pdb.set_trace()
        # if self.running_step>10000 and self.running_step % 2000 == 0:
        #     import pdb; pdb.set_trace()
        # if is_attacking is None or is_attacking == False:
        #     import pdb ; pdb.set_trace()
        # if self.running_step > 5000:
        #     if not is_attacking:
        #         import pdb; pdb.set_trace()
        #         self.running_step = 0
        # if self.running_step > 50:
        #     self.running_step = 0
        #     import pdb; pdb.set_trace()
        
        self.running_step += 1
        
        
        if "mean" == self.merge_memories:
            memories = sum(memories)/self.num_codebooks # n_codebook, bz, dim
            memories_hard = sum(memories_hard)/self.num_codebooks
            # memories = reduce(memories, 'n b d -> b d', 'mean')
        elif "sum" == self.merge_memories:
            memories = sum(memories) # n_codebook, bz, dim
            memories_hard = sum(memories_hard)
            # memories = reduce(memories, 'n b d -> b d', 'mean')
        elif "concat" == self.merge_memories:
            # memories = F.normalize(memories, dim=-1)
            memories = torch.cat(memories, dim=-1)
            memories_hard = torch.cat(memories_hard, dim=-1)
            # memories = rearrange(memories, 'n b d -> b n d')
            # memories = rearrange(memories, 'b n d -> b (n d)')

        
        out = self.classifier(memories)
        out_h = self.classifier(memories_hard)

        if return_immediate:
            return {
                'logit': out,
                'logit_h': out_h,
                'logit_cls_h': out_cls_hidden,
                'out_cls_value': out_cls_value,
                "out_cls_value_soft": out_cls_value_soft,
                'hidden_feature': x,
                'quantized_key': quantize_key_lst,
                'weight_key': weight_key_lst,
                'value': memories,
                "all_value_lst": self.values,
                'all_key_lst': all_key_lst,
                'hk_mse_loss': hk_mse_loss,
                "key_label": key_label_lst,
                # 'cls_weight': self.classifier.weight,
                "ratio_lst": ratio_lst,
                "v_score_lst": v_score_lst,
                "kk_dis_lst": kk_dis_lst,
                "key_idx_map2_lst": key_idx_map2_lst,
                "key_idx_map1_lst": key_idx_map1_lst,
                "key_idx_wlabel_lst": key_idx_wlabel_lst
            }

        return out

# for n, p in model.named_parameters(): print(n, p.grad.max() if p.grad is not None else None)
# clean_output.max(dim=-1)

