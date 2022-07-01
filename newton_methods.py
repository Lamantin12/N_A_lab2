import pandas as pd

class newton_method:
    def __init__(self, func, func_derivative, left_dots, right_dots):
        self.function = func
        self.function_derivative = func_derivative
        self.method = ''
        self.A = left_dots
        self.B = right_dots
        self.result = []
        self.result_t = []
        self.epsilon = 0

    def fit(self, method, eps=10**(-12), **kwargs):
        self.epsilon = eps
        self.method = method

        if self.method == 'standard':
            for i in range(len(self.A)):
                center = (self.A[i] + self.B[i]) / 2
                res = self.standard_newton(center)
                self.result.append(res[0])
                self.result_t.append(res[1])
        elif self.method == 'simple':
            for i in range(len(self.A)):
                center = (self.A[i] + self.B[i]) / 2
                res = self.simple_newton(center)
                self.result.append(res[0])
                self.result_t.append(res[1])
        elif self.method == 'standard_with_residue':
            center = (self.A[0] + self.B[0]) / 2
            self.result = self.standard_newton_residual(center, kwargs['stop'])
        elif self.method == 'simple_with_residue':
            center = (self.A[0] + self.B[0]) / 2
            self.result = self.simple_newton_residual(center,kwargs['stop'])
        elif self.method == 'multiple':
            center = (self.A[0] + self.B[0]) / 2
            for i in range(1, kwargs['multiple_edge']):
                res = self.newton_for_nultiple(center,i)
                self.result.append(res[0])
                self.result_t.append(res[1])
        else:
            print('Cannot fit model, you entered wrong method, try standard/simple/multiple/standard_with_residue/simple_with_residue')
            return

    def standard_newton(self, x, endpoint=1000):
        t = 1
        x_k = x - (self.function(x)) / (self.function_derivative(x))
        while (abs(x_k - x) > self.epsilon) and (t < endpoint):
            x = x_k
            x_k = x - (self.function(x)) / (self.function_derivative(x))
            t += 1
        return [x_k, t]

    def simple_newton(self, x, endpoint=1000):
        t = 1
        df = self.function_derivative(x)
        x_k = x - (self.function(x)) / self.function_derivative(x)
        while (abs(x_k - x) > self.epsilon) and (t < endpoint):
            x = x_k
            x_k = x - (self.function(x)) / df
            t += 1
        return [x_k, t]

    def standard_newton_residual(self, x, endpoint=11):
        r = []
        t = 1
        x_k = x - (self.function(x)) / (self.function_derivative(x))
        r.append(abs(self.function(x_k)))
        while t < endpoint:
            x = x_k
            x_k = x - (self.function(x)) / (self.function_derivative(x))
            t += 1
            r.append(abs(self.function(x_k)))
        return r

    def simple_newton_residual(self, x, endpoint=11):
        t = 1
        r = []
        df = self.function_derivative(x)
        x_k = x - (self.function(x)) / df
        r.append(abs(self.function(x_k)))
        while t < endpoint:
            x = x_k
            x_k = x - self.function(x) / df
            t += 1
            r.append(abs(self.function(x_k)))
        return r

    def newton_for_nultiple(self, x, m):
        x_k = x - m * self.function(x) / self.function_derivative(x)
        t = 1
        while (abs(x_k - x) > self.epsilon):
            x = x_k
            x_k = x - m * self.function(x) / self.function_derivative(x)
            t += 1
        return [x_k, t]

    def display_result(self):
        if len(self.result) != 0:
            result_df = pd.DataFrame(index=['{0} корень'.format(i + 1) for i in range(len(self.A))])
            result_df['a'] = self.A
            result_df['b'] = self.B
            result_df['Значение корня'] = self.result
            result_df['Количество итераций'] = self.result_t
            return result_df
        else:
            return 'method needs to be fitted first try <name of newton_method variable>.fit()'
