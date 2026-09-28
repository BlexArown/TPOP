#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

double f(double x) {
    double sin_2x = sin(2.0 * x);
    return x / (sin_2x * sin_2x * sin_2x);
}

double simpson_integral(double a, double b, int n) {
    double h = (b - a) / n;
    double sum = f(a) + f(b);
    double sum_odd = 0.0;
    double sum_even = 0.0;

    #pragma omp parallel for reduction(+:sum_odd, sum_even)
    for (int i = 1; i < n; i++) {
        double x = a + i * h;
        if (i % 2 == 0) {
            sum_even += f(x);
        } else {
            sum_odd += f(x);
        }
    }

    sum += 4.0 * sum_odd + 2.0 * sum_even;
    return (h / 3.0) * sum;
}

int main(int argc, char* argv[]) {
    double a = 0.1;
    double b = 0.5;
    double eps = 1E-6;

    int threads = 1;
    if (argc > 1) {
        threads = atoi(argv[1]);
    } else {
        threads = omp_get_max_threads();
    }
    omp_set_num_threads(threads);

    int n = 2000000; 
    double I_n = 0.0, I_2n = 0.0;
    double error = 0.0;
    
    double start_time = omp_get_wtime();

    I_n = simpson_integral(a, b, n);

    do {
        n *= 2;
        I_2n = simpson_integral(a, b, n);
        error = fabs(I_2n - I_n) / 15.0;
        I_n = I_2n;
    } while (error > eps);

    double end_time = omp_get_wtime();
    double exec_time = end_time - start_time;

    printf("THREADS:%d TIME:%f\n", threads, exec_time);

    return 0;
}
