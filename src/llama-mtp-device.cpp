#include "llama-context.h"
#include "llama-model.h"
#include "llama-mtp-device.h"

#include <algorithm>
#include <atomic>
#include <cstring>

static std::atomic<uint64_t> mtp_gpu_counts[4] = {};

ggml_tensor llama_mtp_device_rows(const ggml_tensor * tensor, int64_t first, int64_t count) {
    GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    GGML_ASSERT(first >= 0 && count > 0 && first + count <= tensor->ne[1]);
    ggml_tensor view = *tensor;
    view.ne[1] = count;
    view.ne[2] = view.ne[3] = 1;
    view.nb[2] = view.nb[1] * count;
    view.nb[3] = view.nb[2];
    view.data = (char *) tensor->data + first * tensor->nb[1];
    view.view_src = nullptr;
    view.view_offs = 0;
    view.op = GGML_OP_NONE;
    return view;
}

void llama_mtp_count_device_input(size_t bytes) {
    mtp_gpu_counts[1].fetch_add(bytes, std::memory_order_relaxed);
}

void llama_mtp_gpu_stats(uint64_t * counts, bool reset) {
    for (int i = 0; i < 4; ++i) {
        counts[i] = reset ? mtp_gpu_counts[i].exchange(0) : mtp_gpu_counts[i].load();
    }
}

bool llama_context::enable_mtp_device(llama_context & draft) {
    if (model.arch != LLM_ARCH_QWEN35 || draft.model.arch != LLM_ARCH_QWEN35 ||
        cparams.n_seq_max != 1 || draft.cparams.n_seq_max != 1 ||
        cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || draft.cparams.ctx_type != LLAMA_CONTEXT_TYPE_MTP ||
        cparams.pipeline_parallel || draft.cparams.pipeline_parallel ||
        !cparams.embeddings_nextn || cparams.embeddings_nextn_masked ||
        cparams.pooling_type != LLAMA_POOLING_TYPE_NONE ||
        model.hparams.n_embd_out() != draft.model.hparams.n_embd_out()) {
        return false;
    }
    for (const auto & backend : backends) {
        auto * dev = ggml_backend_get_device(backend.get());
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU ||
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0) {
            continue;
        }
        for (const auto & draft_backend : draft.backends) {
            if (ggml_backend_get_device(draft_backend.get()) == dev) {
                mtp_device_output_enabled = true;
                mtp_device_output_backend = backend.get();
                draft.mtp_device_input_backend = draft_backend.get();
                draft.synchronize();
                draft.gf_res_prev->reset();
                init_mtp_tokens(draft);
                return true;
            }
        }
    }
    return false;
}

bool llama_context::capture_mtp_hidden(ggml_tensor * tensor, ggml_backend_t backend, int64_t first, int64_t count) {
    // Small verification batches avoid device-copy setup overhead.
    if (!mtp_device_output_enabled || balloc->get_n_tokens() <= 32) {
        return false;
    }
    const auto n_embd = model.hparams.n_embd_out();
    if (backend != mtp_device_output_backend || tensor->type != GGML_TYPE_F32 ||
        tensor->ne[0] != n_embd || tensor->ne[1] != count || !ggml_is_contiguous(tensor)) {
        read_mtp_hidden(0, mtp_hidden_rows);
        mtp_device_output_enabled = false;
        mtp_hidden_rows = 0;
        LLAMA_LOG_WARN("%s: incompatible output; using host handoff\n", __func__);
        return false;
    }
    if (!mtp_hidden) {
        ggml_init_params params = { ggml_tensor_overhead(), nullptr, true };
        mtp_hidden_ctx.reset(ggml_init(params));
        if (mtp_hidden_ctx) {
            mtp_hidden = ggml_new_tensor_2d(mtp_hidden_ctx.get(), GGML_TYPE_F32, n_embd, cparams.n_batch + 1);
            ggml_set_name(mtp_hidden, "mtp_hidden_resident");
            mtp_hidden_buf.reset(ggml_backend_alloc_ctx_tensors(mtp_hidden_ctx.get(), backend));
        }
        if (!mtp_hidden_buf) {
            mtp_hidden = nullptr;
            mtp_hidden_ctx.reset();
            mtp_device_output_enabled = false;
            LLAMA_LOG_WARN("%s: allocation failed; using host handoff\n", __func__);
            return false;
        }
        LLAMA_LOG_WARN("%s: resident buffer %.3f MiB\n", __func__, ggml_backend_buffer_get_size(mtp_hidden_buf.get()) / 1048576.0);
    }
    GGML_ASSERT(first == mtp_hidden_rows);
    auto dst = llama_mtp_device_rows(mtp_hidden, first + 1, count);
    ggml_backend_tensor_copy_async(backend, backend, tensor, &dst);
    mtp_hidden_rows += count;
    mtp_gpu_counts[0].fetch_add(count * n_embd * sizeof(float), std::memory_order_relaxed);
    return true;
}

void llama_context::read_mtp_hidden(int64_t first, int64_t count) {
    if (count <= 0 || mtp_hidden_rows == 0) {
        return;
    }
    GGML_ASSERT(first >= 0 && first + count <= mtp_hidden_rows);
    auto beg = mtp_hidden_host_valid.begin() + first;
    auto end = beg + count;
    if (std::all_of(beg, end, [](uint8_t ready) { return ready != 0; })) {
        return;
    }
    const size_t row_bytes = model.hparams.n_embd_out() * sizeof(float);
    ggml_backend_synchronize(mtp_device_output_backend);
    ggml_backend_tensor_get(mtp_hidden, embd_nextn.data + first * model.hparams.n_embd_out(),
            (first + 1) * row_bytes, count * row_bytes);
    std::fill(beg, end, 1);
    mtp_gpu_counts[2].fetch_add(count * row_bytes, std::memory_order_relaxed);
}

bool llama_context::mtp_handoff_ready(const llama_context & draft, const llama_batch & batch) const {
    if (!mtp_device_output_enabled || !mtp_hidden || !draft.mtp_device_input_backend ||
        mtp_hidden_rows != batch.n_tokens || batch.n_tokens <= 0 || !batch.token ||
        !batch.pos || !batch.n_seq_id || !batch.seq_id || !output_swaps.empty()) {
        return false;
    }
    if (ggml_backend_get_device(mtp_device_output_backend) != ggml_backend_get_device(draft.mtp_device_input_backend)) {
        return false;
    }
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        if (batch.n_seq_id[i] != 1 || batch.seq_id[i][0] != 0 || batch.pos[i] != batch.pos[0] + i) {
            return false;
        }
    }
    return true;
}

int llama_context::decode_mtp_device(llama_context & target, const llama_batch & batch, const float * pending) {
    if (!target.mtp_handoff_ready(*this, batch)) {
        return -1;
    }
    target.synchronize();
    const size_t row_bytes = model.hparams.n_embd_out() * sizeof(float);
    ggml_backend_tensor_set(target.mtp_hidden, pending, 0, row_bytes);
    mtp_gpu_counts[3].fetch_add(row_bytes, std::memory_order_relaxed);
    mtp_input_hidden = target.mtp_hidden;
    mtp_input_tokens = target.mtp_tokens_ready(*this, batch) ? target.mtp_token_cache : nullptr;
    struct input_guard {
        ggml_tensor * & hidden;
        ggml_tensor * & tokens;
        ~input_guard() { hidden = nullptr; tokens = nullptr; }
    } guard { mtp_input_hidden, mtp_input_tokens };
    return llama_decode(this, batch);
}

bool llama_mtp_enable_gpu_handoff(llama_context * target, llama_context * draft) {
    return target->enable_mtp_device(*draft);
}

bool llama_mtp_gpu_handoff_ready(llama_context * target, llama_context * draft, const llama_batch & batch) {
    return target->mtp_handoff_ready(*draft, batch);
}

int32_t llama_decode_mtp_gpu(llama_context * draft, llama_context * target, llama_batch batch, const float * pending) {
    return draft->decode_mtp_device(*target, batch, pending);
}

void llama_context::copy_mtp_hidden_rows(int32_t first, int32_t count, float * dst) {
    GGML_ASSERT(!cparams.embeddings_nextn_masked && first >= 0 && count > 0 && dst);
    GGML_ASSERT(((int64_t) first + count) * model.hparams.n_embd_out() <= (int64_t) embd_nextn.size);
    synchronize();
    read_mtp_hidden(first, count);
    std::memcpy(dst, embd_nextn.data + (size_t) first * model.hparams.n_embd_out(),
            (size_t) count * model.hparams.n_embd_out() * sizeof(float));
}

void llama_mtp_copy_hidden_rows(llama_context * target, int32_t first, int32_t count, float * dst) {
    target->copy_mtp_hidden_rows(first, count, dst);
}

void llama_context::init_mtp_tokens(const llama_context & draft) {
    const char * enabled = getenv("LLAMA_MTP_TOKEN_REUSE");
    if ((enabled && atoi(enabled) == 0) || &model != &draft.model || !loras->empty() || !draft.loras->empty() ||
            model.hparams.n_embd_inp() != model.hparams.n_embd || model.hparams.f_embedding_scale != 0.0f ||
            cparams.n_batch != draft.cparams.n_batch || cparams.n_ubatch != draft.cparams.n_ubatch) {
        return;
    }
    const auto * mtp_weight = model.layers[model.hparams.n_layer()].nextn.embed_tokens;
    if (mtp_weight && mtp_weight != model.tok_embd) {
        return;
    }
    ggml_init_params params = { 2*ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) { return; }
    auto * scratch = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, model.hparams.n_embd, cparams.n_ubatch);
    auto * cache = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, model.hparams.n_embd, cparams.n_batch);
    ggml_set_name(scratch, "mtp_token_scratch");
    ggml_set_name(cache, "mtp_token_cache");
    ggml_backend_buffer_ptr buf { ggml_backend_alloc_ctx_tensors(ctx.get(), mtp_device_output_backend) };
    if (!buf) {
        LLAMA_LOG_WARN("%s: allocation failed; using separate token embeddings\n", __func__);
        return;
    }
    synchronize();
    gf_res_prev->reset();
    mtp_token_ctx = std::move(ctx);
    mtp_token_buf = std::move(buf);
    mtp_token_scratch = scratch;
    mtp_token_cache = cache;
    LLAMA_LOG_WARN("%s: shared token table, resident buffers %.3f MiB\n", __func__,
            ggml_backend_buffer_get_size(mtp_token_buf.get())/1048576.0);
}

void llama_context::capture_mtp_tokens(ggml_tensor * tensor, int64_t first, int64_t count) {
    // The fixed scratch address preserves graph reuse across microbatches.
    GGML_ASSERT(tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor));
    GGML_ASSERT(tensor->ne[0] == model.hparams.n_embd && tensor->ne[1] == count);
    GGML_ASSERT(first == mtp_token_rows);
    auto dst = llama_mtp_device_rows(mtp_token_cache, first, count);
    ggml_backend_tensor_copy_async(mtp_device_output_backend, mtp_device_output_backend, tensor, &dst);
    mtp_token_rows += count;
}

bool llama_context::mtp_tokens_ready(const llama_context & draft, const llama_batch & batch) const {
    if (!mtp_token_cache || batch.n_tokens <= 32 || mtp_token_rows != batch.n_tokens ||
            !loras->empty() || !draft.loras->empty() || !batch.token || !batch.pos ||
            mtp_token_ids.size() != (size_t) batch.n_tokens || mtp_token_pos.size() != (size_t) batch.n_tokens) {
        return false;
    }
    const size_t n = batch.n_tokens;
    return std::equal(mtp_token_ids.begin(), mtp_token_ids.end(), batch.token) &&
           std::equal(mtp_token_pos.begin(), mtp_token_pos.end(), batch.pos) &&
           n <= cparams.n_batch;
}
