from math import ceil
from typing import List

import numpy as np
import sarathi.metrics.cuda_timer
import torch

from vidur.profiling.common.cuda_timer import CudaTimer
import sarathi.metrics.cuda_timer

# monkey patching the CudaTimer class to use the sarathi implementation
sarathi.metrics.cuda_timer.CudaTimer = CudaTimer

# Also patch where it is likely used/imported in Sarathi
try:
    import sarathi.model_executor.attention.base_attention_wrapper
    sarathi.model_executor.attention.base_attention_wrapper.CudaTimer = CudaTimer
except ImportError:
    pass

# Monkeypatch fix for Sarathi FlashinferAttentionWrapper missing get_instance
try:
    from sarathi.model_executor.attention.flashinfer_attention_wrapper import (
        FlashinferAttentionWrapper,
    )

    if not hasattr(FlashinferAttentionWrapper, "get_instance"):

        class DummyCacheConfig:
            def __init__(self, block_size, num_gpu_blocks):
                self.block_size = block_size
                self.num_gpu_blocks = num_gpu_blocks
                self.num_cpu_blocks = 0 

        class LazyFlashinferWrapperProxy:
            def __init__(self):
                self._backend = None

            def init(self, model_config, parallel_config, block_size, device, max_num_blocks):
                # Wrap block_size and max_num_blocks in a dummy config
                cache_config = DummyCacheConfig(block_size, max_num_blocks)
                
                self._backend = FlashinferAttentionWrapper(
                    model_config, parallel_config, cache_config, device
                )

            def get_cache_block(self, num_blocks, dtype, device):
                # We need to manually call set_cache on the backend because
                # Sarathi's get_cache_block typically returns the cache but
                # doesn't automatically register it if it's external.
                # However, FlashinferAttentionWrapper.forward expects self.gpu_cache to be set.
                
                # Fix: The backend's get_cache_block likely only takes num_blocks.
                # It probably uses the dtype and device stored during __init__.
                cache = self._backend.get_cache_block(num_blocks)
                
                # We will create a list of references to the same cache block
                # to simulate all layers having memory allocated (or just satisfy the index access).
                num_layers = self._backend.num_layers
                fake_layer_caches = [cache for _ in range(num_layers)]
                
                # Directly set the gpu_cache attribute if set_cache doesn't exist
                if hasattr(self._backend, "set_cache"):
                     self._backend.set_cache(fake_layer_caches)
                else:
                     self._backend.gpu_cache = fake_layer_caches
                     
                return cache

            def __getattr__(self, name):
                if self._backend is None:
                    raise RuntimeError("FlashinferAttentionWrapper not initialized! Call init() first.")
                return getattr(self._backend, name)

        _proxy_instance = LazyFlashinferWrapperProxy()

        def get_instance_patched(cls):
            return _proxy_instance

        FlashinferAttentionWrapper.get_instance = classmethod(get_instance_patched)

    # Monkeypatch MetricsStore to ensure singleton exists
    # We create a robust Mock that ignores all calls/attributes to avoid dependency issues
    from sarathi.metrics.metrics_store import MetricsStore

    if not hasattr(MetricsStore, "_instance"):
        class BlackHoleMock:
            def __init__(self, *args, **kwargs): pass
            def __getattr__(self, _): return self
            def __call__(self, *args, **kwargs): return self
            def __enter__(self): return self
            def __exit__(self, *args): pass
            
        MetricsStore._instance = BlackHoleMock()

except ImportError:
    pass

from sarathi.config import ParallelConfig
from sarathi.model_executor.attention import (
    AttentionBackend,
    get_attention_wrapper,
    set_attention_backend,
)

from vidur.profiling.attention.attention_input import AttentionInput
from vidur.profiling.attention.sequence_proxy import SequenceMetadataProxy
from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.common.timer_stats_store import TimerStatsStore

WARMUP_STEPS = 2
ACTIVE_STEPS = 5


class AttentionWrapper:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        max_num_blocks: int,
        max_model_len: int,
        block_size: int,
        attention_backend: AttentionBackend,
        dtype: torch.dtype,
    ):
        self.time_stats_store = TimerStatsStore(profile_method="kineto")

        self._model_config = model_config
        self._parallel_config = parallel_config
        self._dtype = dtype
        self._device = torch.device("cuda")

        self._max_model_len = max_model_len
        self._n_worker_q_heads = self._model_config.get_num_q_heads(
            self._parallel_config
        )
        self._n_worker_kv_heads = self._model_config.get_num_kv_heads(
            self._parallel_config
        )
        self._head_dim = self._model_config.get_head_size()

        self._block_size = block_size

        self._attention_backend = attention_backend
        set_attention_backend(attention_backend)
        get_attention_wrapper().init(
            self._model_config,
            self._parallel_config,
            self._block_size,
            self._device,
            max_num_blocks, # Passed explicitly
        )
        self._max_blocks_per_sequence = ceil(max_model_len / self._block_size)
        # We create (big) KV tensors and reuse them
        self.max_num_blocks = max_num_blocks
        self.kv_cache = get_attention_wrapper().get_cache_block(
            self.max_num_blocks, dtype=self._dtype, device=self._device
        )

    def _get_input_tensors(
        self,
        attention_input: AttentionInput,
    ):
        num_tokens_per_seq = (
            attention_input.prefill_chunk_size if attention_input.is_prefill else 1
        )
        batch_size = attention_input.batch_size
        query = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_q_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        key = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        value = torch.randn(
            batch_size * num_tokens_per_seq,
            self._n_worker_kv_heads * self._head_dim,
            dtype=self._dtype,
            device=self._device,
        )
        # Create SequenceMetadataProxy objects corresponding to AttentionInput
        seq_metadata_list: List[SequenceMetadataProxy] = []
        for _ in range(attention_input.batch_size):
            num_blocks = ceil(
                (num_tokens_per_seq + attention_input.kv_cache_size) / self._block_size
            )
            # TODO(nitinkedia7): Investigate why high=max_num_blocks fails with a CUDA illegal memory access
            seq_metadata = SequenceMetadataProxy(
                is_prompt=attention_input.is_prefill,
                total_len=num_tokens_per_seq + attention_input.kv_cache_size,
                processed_len=attention_input.kv_cache_size,
                block_table=np.random.default_rng()
                .integers(low=0, high=self.max_num_blocks - 1, size=num_blocks)
                .tolist(),
            )
            seq_metadata_list.append(seq_metadata)
        return seq_metadata_list, query, key, value, self.kv_cache

    @torch.inference_mode()
    def profile(
        self,
        attention_input: AttentionInput,
    ):
        # batch size is always 1 for prefill and can be different for decode
        assert attention_input.is_valid(self._max_model_len)

        seq_metadata_list, query, key, value, kv_cache = self._get_input_tensors(
            attention_input,
        )
        get_attention_wrapper().begin_forward(seq_metadata_list)

        for _ in range(WARMUP_STEPS):
            get_attention_wrapper().forward(query, key, value, kv_cache)
        torch.cuda.synchronize()

        self.time_stats_store.clear_stats()

        for _ in range(ACTIVE_STEPS):
            get_attention_wrapper().forward(query, key, value, kv_cache)
        torch.cuda.synchronize()

        get_attention_wrapper().end_forward()

        return {
            "time_stats": self.time_stats_store.get_stats(),
            "n_embd": self._model_config.embedding_dim,
            "n_q_head": self._model_config.num_q_heads,
            "n_kv_head": self._model_config.num_kv_heads,
            "block_size": self._block_size,
            "num_tensor_parallel_workers": self._parallel_config.tensor_parallel_size,
            "max_model_len": self._max_model_len,
            "batch_size": attention_input.batch_size,
            "prefill_chunk_size": attention_input.prefill_chunk_size,
            "kv_cache_size": attention_input.kv_cache_size,
            "is_prefill": attention_input.is_prefill,
            "attention_backend": self._attention_backend,
        }
