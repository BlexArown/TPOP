#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void init_data(double *A, double *b, double *c, int m, int n) {
    for (int i = 0; i < m * n; i++) A[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < n; i++) b[i] = (double)rand() / RAND_MAX;
    for (int i = 0; i < m; i++) c[i] = 0.0;
}

void multiply_row_strip(double *A, double *b, double *c, int m, int n, int k) {
    #pragma omp parallel for schedule(static, k)
    for (int i = 0; i < m; i++) {
        double sum = 0.0;
        int row_offset = i * n;
        for (int j = 0; j < n; j++) {
            sum += A[row_offset + j] * b[j];
        }
        c[i] = sum;
    }
}

void multiply_col_strip(double *A, double *b, double *c, int m, int n, int k) {
    #pragma omp parallel
    {
        double *local_c = (double*)calloc(m, sizeof(double));

        #pragma omp for schedule(static, k)
        for (int j = 0; j < n; j++) {
            double bj = b[j];
            for (int i = 0; i < m; i++) {
                local_c[i] += A[i * n + j] * bj;
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < m; i++) {
                c[i] += local_c[i];
            }
        }
        free(local_c);
    }
}

void multiply_block(double *A, double *b, double *c, int m, int n, int p, int q) {
    #pragma omp parallel
    {
        double *local_c = (double*)calloc(m, sizeof(double));

        #pragma omp for collapse(2) schedule(dynamic)
        for (int bi = 0; bi < m; bi += p) {
            for (int bj = 0; bj < n; bj += q) {
                int i_max = (bi + p > m) ? m : bi + p;
                int j_max = (bj + q > n) ? n : bj + q;

                for (int i = bi; i < i_max; i++) {
                    double sum = 0.0;
                    int row_offset = i * n;
                    for (int j = bj; j < j_max; j++) {
                        sum += A[row_offset + j] * b[j];
                    }
                    local_c[i] += sum;
                }
            }
        }

        #pragma omp critical
        {
            for (int i = 0; i < m; i++) {
                c[i] += local_c[i];
            }
        }
        free(local_c);
    }
}


int main(int argc, char* argv[]) {
    int m = 40000;
    int n = 40000;
    int k = 64;
    int p = 128, q = 128;

    int threads = 1;
    int algo = 1;

    if (argc > 1) threads = atoi(argv[1]);
    if (argc > 2) algo = atoi(argv[2]);

    omp_set_num_threads(threads);

    double *A = (double*)malloc(sizeof(double) * m * n);
    double *b = (double*)malloc(sizeof(double) * n);
    double *c = (double*)malloc(sizeof(double) * m);

    if (!A || !b || !c) {
        printf("Ошибка выделения памяти!\n");
        return 1;
    }

    init_data(A, b, c, m, n);

    double start_time = omp_get_wtime();

    if (algo == 1) {
        multiply_row_strip(A, b, c, m, n, k);
    } else if (algo == 2) {
        multiply_col_strip(A, b, c, m, n, k);
    } else if (algo == 3) {
        multiply_block(A, b, c, m, n, p, q);
    }

    double end_time = omp_get_wtime();
    double exec_time = end_time - start_time;

    printf("THREADS:%d TIME:%f\n", threads, exec_time);

    free(A);
    free(b);
    free(c);
    return 0;
}
