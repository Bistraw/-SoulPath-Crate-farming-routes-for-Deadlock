"""Evaluación de la población en GPU con CUDA (CuPy RawKernel).

Paraleliza el decodificador `guiado` en dos niveles:
    - un bloque CUDA por agente (los N agentes se decodifican a la vez);
    - dentro del bloque, un hilo por caja candidata: cada hilo calcula su mejor posición de
      inserción y su valor almas/tiempo, y una reducción en memoria compartida elige la caja
      que se agrega en ese paso.
El resultado es idéntico al de `_insercion_voraz` en CPU (float64, mismos desempates:
primera posición mínima y caja de menor índice entre valores iguales).

Requiere una GPU NVIDIA y CuPy:  pip install "cupy-cuda12x[ctk]"
(el extra [ctk] trae NVRTC de CUDA 12, que todavía compila para GPUs Pascal como la GTX 10xx;
el toolkit de CUDA 13 ya no las soporta).
"""
import numpy as np

try:
    import cupy as cp
except ImportError:  # el resto del proyecto funciona sin GPU
    cp = None

HILOS = 256

_FUENTE = r"""
#define INFINITY __longlong_as_double(0x7ff0000000000000LL)
extern "C" __global__
void guiado(const double* __restrict__ C,      // (n+1) x (n+1), fila = origen
            const double* __restrict__ S,      // almas por nodo, S[0] = 0
            const double* __restrict__ sesgo,  // N x n, e^(gamma (X - 0.5))
            const unsigned char* __restrict__ alcanzable,  // n+1
            int n, double tmax, int regreso,
            int* __restrict__ rutas,           // N x (n+2): secuencia con el 0 inicial (y final)
            int* __restrict__ largos)          // N: nodos en la ruta (sin contar los 0)
{
    const int agente = blockIdx.x;
    const int tid = threadIdx.x;
    const int m = n + 1;
    const double EPS = 1e-12;

    extern __shared__ unsigned char mem[];
    double* red_val = (double*)mem;                         // blockDim
    double* red_cos = red_val + blockDim.x;                 // blockDim
    int* red_j = (int*)(red_cos + blockDim.x);              // blockDim
    int* red_p = red_j + blockDim.x;                        // blockDim
    int* seq = red_p + blockDim.x;                          // n + 2
    unsigned char* libre = (unsigned char*)(seq + n + 2);   // n + 1

    __shared__ int L;          // longitud de seq
    __shared__ double t;       // tiempo acumulado
    __shared__ int sigue;

    for (int j = tid; j < m; j += blockDim.x) libre[j] = alcanzable[j];
    if (tid == 0) {
        seq[0] = 0;
        if (regreso) { seq[1] = 0; L = 2; } else { L = 1; }
        t = 0.0;
        sigue = 1;
    }
    __syncthreads();

    const double* mi_sesgo = sesgo + (size_t)agente * n;

    while (sigue) {
        // 1) cada hilo evalúa sus cajas candidatas
        double mejor_val = -INFINITY, mejor_cos = 0.0;
        int mejor_j = 0x7fffffff, mejor_p = 0;
        for (int j = 1 + tid; j < m; j += blockDim.x) {
            if (!libre[j]) continue;
            double dmin = INFINITY;
            int pmin = 0;
            for (int p = 0; p < L - 1; ++p) {
                int a = seq[p], b = seq[p + 1];
                double d = C[a * m + j] + C[j * m + b] - C[a * m + b];
                if (d < dmin) { dmin = d; pmin = p; }
            }
            if (!regreso) {                           // ruta abierta: también al final
                double d = C[seq[L - 1] * m + j];
                if (d < dmin) { dmin = d; pmin = L - 1; }
            }
            if (t + dmin <= tmax) {
                double v = S[j] / (dmin + EPS) * mi_sesgo[j - 1];
                if (v > mejor_val) {                  // j crece: empate -> se queda el menor
                    mejor_val = v; mejor_cos = dmin; mejor_j = j; mejor_p = pmin;
                }
            }
        }
        red_val[tid] = mejor_val; red_cos[tid] = mejor_cos;
        red_j[tid] = mejor_j; red_p[tid] = mejor_p;
        __syncthreads();

        // 2) reducción argmax (desempate por índice de caja menor)
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                double v2 = red_val[tid + s];
                int j2 = red_j[tid + s];
                if (v2 > red_val[tid] || (v2 == red_val[tid] && j2 < red_j[tid])) {
                    red_val[tid] = v2; red_cos[tid] = red_cos[tid + s];
                    red_j[tid] = j2; red_p[tid] = red_p[tid + s];
                }
            }
            __syncthreads();
        }

        // 3) el hilo 0 inserta la caja ganadora
        if (tid == 0) {
            if (red_j[0] == 0x7fffffff) {
                sigue = 0;
            } else {
                int j = red_j[0], pos = red_p[0] + 1;
                for (int k = L; k > pos; --k) seq[k] = seq[k - 1];
                seq[pos] = j;
                L += 1;
                t += red_cos[0];
                libre[j] = 0;
            }
        }
        __syncthreads();
    }

    int* salida = rutas + (size_t)agente * (n + 2);
    for (int k = tid; k < L; k += blockDim.x) salida[k] = seq[k];
    if (tid == 0) largos[agente] = L - 1 - (regreso ? 1 : 0);
}
"""


def disponible():
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


class EvaluadorCUDA:
    """Copia la instancia a la GPU una vez y decodifica poblaciones completas con `guiado`."""

    def __init__(self, inst, gamma):
        if not disponible():
            raise RuntimeError('No hay GPU CUDA o falta CuPy (pip install "cupy-cuda12x[ctk]")')
        self.inst, self.gamma, self.n = inst, gamma, inst.n
        self.C = cp.asarray(np.ascontiguousarray(inst.C, dtype=np.float64))
        self.S = cp.asarray(inst.S.astype(np.float64))
        self.alcanzable = cp.asarray(inst.alcanzable.astype(np.uint8))
        self.kernel = cp.RawKernel(_FUENTE, "guiado")
        n = self.n
        self.mem = HILOS * (2 * 8 + 2 * 4) + (n + 2) * 4 + (n + 1)
        self.nombre = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()

    def __call__(self, X):
        N, n = X.shape
        # mismo cálculo que el decodificador en CPU, para que los resultados coincidan
        sesgo = cp.asarray(np.exp(self.gamma * (X - 0.5)))
        rutas = cp.zeros((N, n + 2), dtype=cp.int32)
        largos = cp.zeros(N, dtype=cp.int32)
        self.kernel((N,), (HILOS,),
                    (self.C, self.S, sesgo, self.alcanzable, np.int32(n), np.float64(self.inst.tmax),
                     np.int32(self.inst.regreso), rutas, largos),
                    shared_mem=self.mem)
        rutas, largos = cp.asnumpy(rutas), cp.asnumpy(largos)
        return [rutas[i, 1:1 + largos[i]].tolist() for i in range(N)]
