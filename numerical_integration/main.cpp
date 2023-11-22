#include <iostream>
#include <vector>
#include <iomanip>

using namespace std;

// given function
double Function(double x) {
	return x * exp(x);
}

// rectangle method
double RectanInt(double (*func) (double), double left, double right, int n) {
	double step = (right - left) / n;
	double ans = 0;
	// calculating the sum
	for (int i = 1; i <= n; i++) {
		ans += step * func(left + step * (i - 0.5));
	}
	return ans;
}

double TrapezoidInt(double (*func) (double), double left, double right, int n) {
	double step = (right - left) / n;
	double ans = step / 2 * (func(left) + func(right));
	// calculating the sum
	for (int i = 1; i < n; i++) {
		ans += step * func(left + step * i);
	}
	return ans;
}

// Simpson method
double SimpsonInt(double (*func) (double), double left, double right, int n) {
	double step = (right - left) / (2 * n);
	double ans = step / 3 * (func(left) + 4 * func(left + (2 * n - 1) * step) + func(right));
	// calculating the sum
	for (int i = 1; i < n; i++) {
		ans += step / 3 * (4 * func(left + step * (2 * i - 1)) + 2 * func(left + step * 2 * i));
	}
	return ans;
}

// main cycle for each method
float CalculateQuad(double (*method) (double (double), double, double, int),
					double (*func) (double), double left, double right,
					int n, double eps, int k, bool output_sol) {
	double sum_prev = 0, sum_cur = 0;
	int mult = 1;
	while (true) {
		sum_prev = sum_cur;
		sum_cur = method(func, left, right, mult * n);
		mult *= 2;
		if (abs(sum_prev - sum_cur) / ((1 << k) - 1) < eps) {
			if (output_sol) {
				cout << "Value of Quadrature on the last iteration: " 
					 << fixed << setprecision(13) << sum_cur << endl;
				double rich_clar = sum_prev + (sum_prev - sum_cur) / ((1 << k) - 1);
				cout << "Richardson clarification: " << rich_clar << endl;
			}
			return sum_cur;
		}
	}
}

int main() {
	double left = 0, right = 1;
	int n = 8;
	double eps = 1e-6;
	cout << "Integral of x * exp(x) from 0 to 1, actual answer is 1" << endl;
	cout << "Rectangle method" << endl;
	double rectan_sum = CalculateQuad(&RectanInt, &Function, left, right, n, eps, 2, true);
	cout << "Trapezoid method" << endl;
	double trap_sum = CalculateQuad(&TrapezoidInt, &Function, left, right, n, eps, 2, true);
	cout << "Simpson method" << endl;
	double simp_sum = CalculateQuad(&SimpsonInt, &Function, left, right, n, eps, 4, true);
	return 0;
}