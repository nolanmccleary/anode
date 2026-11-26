#include <mlx/mlx.h>
#include <cstdlib>
#include <cstring>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mc = mlx::core;
using mc::array;

struct Llama {
    int vocab, d, ff, h, kv, layers, hd;
    float eps, scale;
    std::map<std::string, array> w;
    std::optional<array> ones;
    std::vector<std::optional<array>> kc, vc;
};

static array make_f32(const float* data, const long long* shape, int ndim) {
    mc::Shape sh;
    size_t n = 1;
    for (int i = 0; i < ndim; i++) {
        sh.push_back((int)shape[i]);
        n *= (size_t)shape[i];
    }
    float* buf = (float*)malloc(n * sizeof(float));
    memcpy(buf, data, n * sizeof(float));
    return array((void*)buf, sh, mc::float32, [](void* p) { free(p); });
}

static array make_ids(const int* ids, int n) {
    int* buf = (int*)malloc((size_t)n * sizeof(int));
    memcpy(buf, ids, (size_t)n * sizeof(int));
    return array((void*)buf, mc::Shape{n}, mc::int32, [](void* p) { free(p); });
}

static const array& wl(Llama* L, int l, const char* suf) {
    std::string s = "model.layers." + std::to_string(l) + "." + suf + ".weight";
    return L->w.at(s);
}

static array rms(Llama* L, const array& x) {
    return mc::fast::rms_norm(x, L->ones, L->eps);
}

static array layer(Llama* L, int l, array x, bool prefill) {
    array nm = rms(L, x);
    array q = mc::matmul(nm, wl(L, l, "self_attn.q_proj"));
    array k = mc::matmul(nm, wl(L, l, "self_attn.k_proj"));
    array v = mc::matmul(nm, wl(L, l, "self_attn.v_proj"));
    if (L->kc[l]) {
        k = mc::concatenate({*L->kc[l], k}, 0);
        v = mc::concatenate({*L->vc[l], v}, 0);
    }
    L->kc[l] = k;
    L->vc[l] = v;
    int s = x.shape(0), t = k.shape(0);
    array qh = mc::reshape(
        mc::transpose(mc::reshape(q, {s, L->h, L->hd}), {1, 0, 2}),
        {1, L->h, s, L->hd});
    array kh = mc::reshape(
        mc::transpose(mc::reshape(k, {t, L->kv, L->hd}), {1, 0, 2}),
        {1, L->kv, t, L->hd});
    array vh = mc::reshape(
        mc::transpose(mc::reshape(v, {t, L->kv, L->hd}), {1, 0, 2}),
        {1, L->kv, t, L->hd});
    array o = mc::fast::scaled_dot_product_attention(
        qh, kh, vh, L->scale, prefill ? "causal" : "", std::nullopt);
    array o0 = mc::reshape(o, {L->h, s, L->hd});
    array attn = mc::reshape(mc::transpose(o0, {1, 0, 2}), {s, L->h * L->hd});
    x = mc::add(x, mc::matmul(attn, wl(L, l, "self_attn.o_proj")));
    array nm2 = rms(L, x);
    array gate = mc::matmul(nm2, wl(L, l, "mlp.gate_proj"));
    array up = mc::matmul(nm2, wl(L, l, "mlp.up_proj"));
    array hh = mc::matmul(
        mc::multiply(mc::multiply(gate, mc::sigmoid(gate)), up),
        wl(L, l, "mlp.down_proj"));
    x = mc::add(x, hh);
    return x;
}

static void copy_out(array a, float* out) {
    mc::eval({a});
    memcpy(out, a.data<float>(), (size_t)a.size() * sizeof(float));
}

extern "C" void* mlxllama_new(int vocab, int d, int ff, int h, int kv, int layers, float eps, float scale) {
    Llama* L = new Llama();
    L->vocab = vocab; L->d = d; L->ff = ff; L->h = h; L->kv = kv;
    L->layers = layers; L->hd = d / h; L->eps = eps; L->scale = scale;
    float* b = (float*)malloc((size_t)d * sizeof(float));
    for (int i = 0; i < d; i++) b[i] = 1.0f;
    L->ones = array((void*)b, mc::Shape{d}, mc::float32, [](void* p) { free(p); });
    L->kc.resize(layers);
    L->vc.resize(layers);
    return L;
}

extern "C" void mlxllama_set_weight(void* hp, const char* name, const float* data, const long long* shape, int ndim) {
    Llama* L = (Llama*)hp;
    L->w.insert_or_assign(std::string(name), make_f32(data, shape, ndim));
}

extern "C" void mlxllama_prefill(void* hp, const int* ids, int n, float* out_last) {
    Llama* L = (Llama*)hp;
    for (int l = 0; l < L->layers; l++) {
        L->kc[l] = std::nullopt;
        L->vc[l] = std::nullopt;
    }
    array x = mc::take(L->w.at("model.embed_tokens.weight"), make_ids(ids, n), 0);
    for (int l = 0; l < L->layers; l++) x = layer(L, l, x, true);
    array logits = mc::matmul(rms(L, x), mc::transpose(L->w.at("model.embed_tokens.weight")));
    std::vector<array> keep{logits};
    for (int l = 0; l < L->layers; l++) { keep.push_back(*L->kc[l]); keep.push_back(*L->vc[l]); }
    mc::eval(keep);
    copy_out(mc::take(logits, n - 1, 0), out_last);
}

extern "C" void mlxllama_decode(void* hp, int tok, float* out) {
    Llama* L = (Llama*)hp;
    int one = tok;
    array x = mc::take(L->w.at("model.embed_tokens.weight"), make_ids(&one, 1), 0);
    for (int l = 0; l < L->layers; l++) x = layer(L, l, x, false);
    array logits = mc::matmul(rms(L, x), mc::transpose(L->w.at("model.embed_tokens.weight")));
    std::vector<array> keep{logits};
    for (int l = 0; l < L->layers; l++) { keep.push_back(*L->kc[l]); keep.push_back(*L->vc[l]); }
    mc::eval(keep);
    copy_out(mc::reshape(logits, {L->vocab}), out);
}

extern "C" void mlxllama_free(void* hp) {
    delete (Llama*)hp;
}

extern "C" void mlxop_matmul(const float* a, const float* b, float* out, int m, int k, int n) {
    array A((void*)a, mc::Shape{m, k}, mc::float32, [](void*) {});
    array B((void*)b, mc::Shape{k, n}, mc::float32, [](void*) {});
    array C = mc::matmul(A, B);
    mc::eval({C});
    memcpy(out, C.data<float>(), (size_t)m * (size_t)n * sizeof(float));
}

extern "C" void mlxop_sdpa(const float* q, const float* kk, const float* vv, const float* mask, float* out, const int* p) {
    int kv = p[0], grp = p[1], s = p[2], ctx = p[3], hd = p[4];
    int heads = kv * grp;
    float scale;
    memcpy(&scale, &p[23], 4);
    int qn = s * heads * hd, kn = ctx * kv * hd;
    array Qf((void*)q, mc::Shape{qn}, mc::float32, [](void*) {});
    array Kf((void*)kk, mc::Shape{kn}, mc::float32, [](void*) {});
    array Vf((void*)vv, mc::Shape{kn}, mc::float32, [](void*) {});
    array Q = mc::as_strided(Qf, {1, heads, s, hd}, {0, (long)p[7], (long)p[8], (long)p[9]}, 0);
    array K = mc::as_strided(Kf, {1, kv, ctx, hd}, {0, (long)p[11], (long)p[12], (long)p[13]}, 0);
    array V = mc::as_strided(Vf, {1, kv, ctx, hd}, {0, (long)p[15], (long)p[16], (long)p[17]}, 0);
    array M0((void*)mask, mc::Shape{s, ctx}, mc::float32, [](void*) {});
    array M = mc::reshape(M0, {1, 1, s, ctx});
    array O = mc::fast::scaled_dot_product_attention(Q, K, V, scale, "", M);
    mc::eval({O});
    memcpy(out, O.data<float>(), (size_t)heads * (size_t)s * (size_t)hd * sizeof(float));
}
