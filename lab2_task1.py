import pandas as pd
from numpy import linspace, vectorize

class iteration_method:
    def __init__(self, func, func_derivative, left_dots, right_dots, eps=10**(-8)):
        self.function = func
        self.function_derivative = func_derivative
        self.A = left_dots
        self.B = right_dots
        self.M1 = []
        self.m1 = []
        self.alpha = []
        self.q = []
        self.epsilon = []

        for i in range(len(self.A)):
            x_range = linspace(self.A[i], self.B[i], 1000)
            self.M1.append(vectorize(func_derivative)(x_range).max())
            self.m1.append(vectorize(func_derivative)(x_range).min())
            self.alpha.append(2 / (self.M1[i] + self.m1[i]))
            self.q.append(abs(self.M1[i] - self.m1[i]) / abs(self.M1[i] + self.m1[i]))
            self.epsilon.append((1 - self.q[i]) / self.q[i] * eps)
            self.result = []
            self.t_result = []

    def fit(self):
        for i in range(len(self.A)):
            t = 1
            x = (self.A[i] + self.B[i]) / 2
            x_k = x - self.alpha[i] * self.function(x)
            while (abs(x_k - x) > self.epsilon[i]):
                x = x_k
                x_k = x - self.alpha[i] * self.function(x)
                t += 1

            self.result.append(x_k)
            self.t_result.append(t)
        return

    def display_result(self):
        if (len(self.result) != 0):
            result_df = pd.DataFrame(index=['{0} корень'.format(i + 1) for i in range(len(self.A))])
            result_df['a'] = self.A
            result_df['b'] = self.B
            result_df['M1'] = self.M1
            result_df['m1'] = self.m1
            result_df['alpha'] = self.alpha
            result_df['q'] = self.q
            result_df['Значение корня'] = self.result
            result_df['Количество итераций'] = self.t_result
            return result_df
        else:
            return 'method needs to be fitted first try <name of iteration_method variable>.fit()'

