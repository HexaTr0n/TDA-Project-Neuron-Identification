/* Vietoris-Rips H0 and H1 barcodes for small Euclidean clouds.
 * A test tool for machines without stablebear. Standard mod-2 column reduction,
 * truncated at the enclosing radius, parallel over clouds.
 * Build: gcc -O3 -march=native -fopenmp -shared -fPIC rips.c -o librips.so
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef struct { double d; int a, b; } Edge;
typedef struct { int e0, e1, e2; } Tri;   /* edge ranks, e0 is the largest */
static int cmp_edge(const void *x, const void *y) {
    const Edge *p = x, *q = y;
    if (p->d < q->d) return -1;
    if (p->d > q->d) return 1;
    if (p->a != q->a) return p->a - q->a;
    return p->b - q->b;
}
static int cmp_tri(const void *x, const void *y) {
    const Tri *p = x, *q = y;
    if (p->e0 != q->e0) return p->e0 - q->e0;
    if (p->e1 != q->e1) return p->e1 - q->e1;
    return p->e2 - q->e2;
}
static int find(int *u, int x) { while (u[x] != x) { u[x] = u[u[x]]; x = u[x]; } return x; }
static int top_bit(const uint64_t *c, int w) {
    for (int i = w - 1; i >= 0; i--) if (c[i]) return 64 * i + 63 - __builtin_clzll(c[i]);
    return -1;
}
/* Writes n-1 H0 bar lengths to h0, and up to cap H1 (birth, death) pairs. Returns H1 count. */
static int rips_core(const double *D, int n, double *h0, double *b1, double *d1, int cap) {
    int ne = n * (n - 1) / 2, k = 0;
    if (n < 2) return 0;
    double enclosing = INFINITY;
    for (int i = 0; i < n; i++) {
        double mx = 0;
        for (int j = 0; j < n; j++) if (D[i*n+j] > mx) mx = D[i*n+j];
        if (mx < enclosing) enclosing = mx;
    }
    Edge *E = malloc(sizeof(Edge) * ne);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) { E[k].d = D[i*n+j]; E[k].a = i; E[k].b = j; k++; }
    qsort(E, ne, sizeof(Edge), cmp_edge);
    int *u = malloc(sizeof(int) * n), *R = malloc(sizeof(int) * n * n), nh0 = 0;
    for (int i = 0; i < n; i++) u[i] = i;
    for (int e = 0; e < ne; e++) {
        R[E[e].a*n+E[e].b] = R[E[e].b*n+E[e].a] = e;
        int a = find(u, E[e].a), b = find(u, E[e].b);
        if (a != b) { u[a] = b; h0[nh0++] = E[e].d; }
    }
    int ntrunc = ne;
    while (ntrunc > 0 && E[ntrunc-1].d > enclosing) ntrunc--;
    int nt = 0, tcap = 1024;
    Tri *T = malloc(sizeof(Tri) * tcap);
    for (int a = 0; a < n; a++) for (int b = a + 1; b < n; b++) for (int c = b + 1; c < n; c++) {
        int x = R[a*n+b], y = R[a*n+c], z = R[b*n+c], t;
        if (x >= ntrunc || y >= ntrunc || z >= ntrunc) continue;
        if (x < y) { t = x; x = y; y = t; }
        if (x < z) { t = x; x = z; z = t; }
        if (y < z) { t = y; y = z; z = t; }
        if (nt == tcap) { tcap *= 2; T = realloc(T, sizeof(Tri) * tcap); }
        T[nt].e0 = x; T[nt].e1 = y; T[nt].e2 = z; nt++;
    }
    qsort(T, nt, sizeof(Tri), cmp_tri);
    int w = (ne + 63) / 64;
    uint64_t **col_of_low = calloc(ne, sizeof(uint64_t *));
    uint64_t *c = malloc(sizeof(uint64_t) * w);
    int nb = 0;
    for (int t = 0; t < nt; t++) {
        memset(c, 0, sizeof(uint64_t) * w);
        c[T[t].e0 / 64] ^= 1ULL << (T[t].e0 % 64);
        c[T[t].e1 / 64] ^= 1ULL << (T[t].e1 % 64);
        c[T[t].e2 / 64] ^= 1ULL << (T[t].e2 % 64);
        int low = top_bit(c, w);
        while (low >= 0 && col_of_low[low]) {
            uint64_t *o = col_of_low[low];
            for (int i = 0; i < w; i++) c[i] ^= o[i];
            low = top_bit(c, w);
        }
        if (low >= 0) {
            col_of_low[low] = malloc(sizeof(uint64_t) * w);
            memcpy(col_of_low[low], c, sizeof(uint64_t) * w);
            double birth = E[low].d, death = E[T[t].e0].d;
            if (death > birth && nb < cap) { b1[nb] = birth; d1[nb] = death; nb++; }
        }
    }
    for (int e = 0; e < ne; e++) free(col_of_low[e]);
    free(col_of_low); free(c); free(T); free(R); free(u); free(E);
    return nb;
}
static int rips_one(const double *X, int n, int dim, double *h0, double *b1, double *d1, int cap) {
    if (n < 2) return 0;
    double *D = malloc(sizeof(double) * n * n);
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
        double s = 0;
        for (int c = 0; c < dim; c++) { double t = X[i*dim+c] - X[j*dim+c]; s += t * t; }
        D[i*n+j] = sqrt(s);
    }
    int nb = rips_core(D, n, h0, b1, d1, cap);
    free(D);
    return nb;
}
/* Same, from a symmetric n x n dissimilarity matrix. */
int rips_matrix(const double *D, int n, double *h0, double *b1, double *d1, int cap) {
    return rips_core(D, n, h0, b1, d1, cap);
}
typedef struct { double v; int i; } Px;
static int cmp_px(const void *x, const void *y) {
    const Px *p = x, *q = y;
    if (p->v < q->v) return -1;
    if (p->v > q->v) return 1;
    return p->i - q->i;
}
/* Sublevel-set H0 of an h x w image, 4-connected. Writes bar lengths, returns the count.
 * The bar of the global minimum never dies and is left out. */
static int cubical_one(const double *img, int h, int w, double *bars) {
    int n = h * w, nb = 0;
    Px *o = malloc(sizeof(Px) * n);
    int *u = malloc(sizeof(int) * n);
    double *birth = malloc(sizeof(double) * n);
    char *on = calloc(n, 1);
    for (int i = 0; i < n; i++) { o[i].v = img[i]; o[i].i = i; u[i] = i; }
    qsort(o, n, sizeof(Px), cmp_px);
    for (int k = 0; k < n; k++) {
        int p = o[k].i, r = p / w, c = p % w;
        double v = o[k].v;
        on[p] = 1; birth[p] = v;
        int nbr[4] = { r > 0 ? p - w : -1, r < h - 1 ? p + w : -1, c > 0 ? p - 1 : -1, c < w - 1 ? p + 1 : -1 };
        for (int d = 0; d < 4; d++) {
            int q = nbr[d];
            if (q < 0 || !on[q]) continue;
            int a = find(u, p), b = find(u, q);
            if (a == b) continue;
            if (birth[a] > birth[b]) { int t = a; a = b; b = t; }   /* a is the elder */
            if (v > birth[b]) bars[nb++] = v - birth[b];
            u[b] = a;
        }
    }
    free(o); free(u); free(birth); free(on);
    return nb;
}
void cubical_batch(const double *imgs, int N, int h, int w, double *bars, int cap, int *nb) {
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < N; i++) {
        double *tmp = malloc(sizeof(double) * h * w);
        nb[i] = cubical_one(imgs + (int64_t)i * h * w, h, w, tmp);
        for (int j = 0; j < nb[i] && j < cap; j++) bars[(int64_t)i * cap + j] = tmp[j];
        if (nb[i] > cap) nb[i] = cap;
        free(tmp);
    }
}
/* Clouds concatenated in X. offsets has N+1 entries in points. h0 is laid out like X
 * by point (entry offsets[i] onward holds n_i-1 lengths). b1, d1 are N x cap. */
void rips_batch(const double *X, const int64_t *offsets, int N, int dim,
                double *h0, double *b1, double *d1, int cap, int *nh1) {
#pragma omp parallel for schedule(dynamic, 16)
    for (int i = 0; i < N; i++) {
        int64_t o = offsets[i];
        nh1[i] = rips_one(X + o * dim, (int)(offsets[i+1] - o), dim, h0 + o,
                          b1 + (int64_t)i * cap, d1 + (int64_t)i * cap, cap);
    }
}

/* Time-ordered matching between peak diagrams. Events (t, m) as in spike distance, but an
 * unmatched event costs its mass m, and a match costs q*|dt|*min(m, m') + |m - m'|.
 * All clouds concatenated in t and m, with offsets, N+1 entries. Fills the N x N matrix D. */
void ordered_match(const double *t, const double *m, const int64_t *off, int N, double q, double *D) {
#pragma omp parallel for schedule(dynamic, 4)
    for (int i = 0; i < N; i++) {
        int na = (int)(off[i+1] - off[i]);
        const double *ta = t + off[i], *ma = m + off[i];
        for (int j = i + 1; j < N; j++) {
            int nb = (int)(off[j+1] - off[j]);
            const double *tb = t + off[j], *mb = m + off[j];
            double *prev = malloc(sizeof(double) * (nb + 1)), *cur = malloc(sizeof(double) * (nb + 1));
            prev[0] = 0;
            for (int b = 1; b <= nb; b++) prev[b] = prev[b-1] + mb[b-1];
            for (int a = 1; a <= na; a++) {
                cur[0] = prev[0] + ma[a-1];
                for (int b = 1; b <= nb; b++) {
                    double lo = ma[a-1] < mb[b-1] ? ma[a-1] : mb[b-1];
                    double c = prev[b-1] + q * fabs(ta[a-1] - tb[b-1]) * lo + fabs(ma[a-1] - mb[b-1]);
                    double x = prev[b] + ma[a-1], y = cur[b-1] + mb[b-1];
                    if (x < c) c = x;
                    if (y < c) c = y;
                    cur[b] = c;
                }
                double *tmp = prev; prev = cur; cur = tmp;
            }
            D[(int64_t)i*N+j] = D[(int64_t)j*N+i] = prev[nb];
            free(prev); free(cur);
        }
    }
}
