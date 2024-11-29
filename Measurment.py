import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
from scipy import optimize
from scipy import signal
import scipy.ndimage
import scipy.signal
import os
import sympy
import sympy.plotting
from sympy.plotting.plot import List2DSeries

def lorentzian(params, x, y):
        ampl, freq, wid, alpha, var = params
        return y - (ampl * wid**2) / ((x - freq - alpha*y)**2 + wid**2) - var

def equation(y, x, params):
    ampl, freq, wid, alpha, var = params
    return y - (ampl * wid**2) / ((x - freq - alpha*y)**2 + wid**2) - var

class Measurment:
    def __init__(self, filename : str):
        self.filename = filename
        try:
            os.mkdir(self.filename)
        except: 
            a = 0
        curves = {}

        v = 0
        freq = []
        x = []
        y = []
        with open(self.filename + ".csv", 'r') as file:
            reader = csv.reader(file, delimiter="\t")
            curve = Lorentz(0)
            for row in reader:
                if row[0] == "ZI parameter 1": continue
                current_v = float(row[1].replace(',', '.'))

                if current_v != v:
                    
                    curve = Lorentz(10*v, freq=freq, x=x, y=y)
                    curves.update({10*v : curve})
                    freq = []
                    x = []
                    y = []   
                v = current_v

                freq.append(float(row[0].replace(',', '.')))
                x.append(float(row[2].replace(',', '.')))
                y.append(float(row[3].replace(',', '.')))
            
            curve = Lorentz(10*v, freq=freq, x=x, y=y)
            curves.update({10*v : curve})
        
        curves = dict(sorted(curves.items()))
        self.curves = curves
        self.vs = curves.keys() 
        self.cs = curves.values()

    def plot_curve(self, v):
        self.curves[v].plot()

    def plot_curves(self):
        for curve in self.curves.values():
            curve.plot()

    def save_curves(self, filename = "Curves"):
        for curve in self.curves.values():
            curve.save(filename)

    #TODO: подумать, как убрать повторение кода
    def save_params(self):
        self.save_ampls()
        self.save_ampls_square()
        self.save_freq()
        self.save_q()

    def save_ampls_square(self):
        plt.grid()
        for curve in self.curves.values():
            plt.title("Amplitude^2")
            plt.plot(curve.v, curve.resonant_amplitude, 'o', color='red')
            plt.savefig(f"{self.filename}\\Amplitude^2.png")
        plt.close()

    def save_ampls(self):
        plt.grid()
        for curve in self.curves.values():
            plt.title("Amplitude")
            plt.plot(curve.v, np.sqrt(curve.resonant_amplitude), 'o', color='red')
            plt.savefig(f"{self.filename}\\Amplitude.png")
        plt.close()

    def save_ampls(self):
        plt.grid()
        for curve in self.curves.values():
            plt.title("Amplitude")
            plt.plot(curve.v, np.sqrt(curve.resonant_amplitude), 'o', color='red')
            plt.savefig(f"{self.filename}\\Amplitude.png")
        plt.close()

    def save_freq(self):
        plt.grid()
        for curve in self.curves.values():
            plt.title("Frequency")
            plt.plot(curve.v, curve.resonant_freq, 'o', color='red')
            plt.savefig(f"{self.filename}\\Frequency.png")
        plt.close()

    def save_q(self):
        plt.ylim(0, 5000)
        plt.grid()
        for curve in self.curves.values():
            plt.title("Q-factor")
            plt.plot(curve.v, curve.q, 'o', color='red')
            plt.savefig(f"{self.filename}\\Q-factor.png")
        plt.close()

    def save_disp(self):
        plt.grid()
        for curve in self.curves.values():
            plt.title("Variance")
            plt.plot(curve.v, curve.variance, 'o', color='red')
            plt.savefig(f"{self.filename}\\Variance.png")
        plt.close()

    filename = str
    curves = dict 
    vs = []
    cs = []

    pass

class Lorentz:
    def __init__(self, v=[], freq=[], x=[], y=[]):
        if freq == []: return
        self.v = v
        self.freq = freq
        self.x = x
        self.y = y
        self.calculate()

    def find_r(self):
        if self.freq == []: return
        step = self.freq[1] - self.freq[0]
        for i in range(len(self.freq)):
            if i == len(self.freq)-1:
                self.r = len(self.freq)
                return
            if self.freq[i+1] - self.freq[i] > step:
                self.r = i
                return
            
    # def lorentzian(self, x, amp, freq, wid, var):
    #     return amp*wid**2/((x-freq)**2+wid**2) + var

    def remove_slope(self, x, y):
        if x == []: return y
        bx = x[:80] + x[self.r-80:self.r]
        by = y[:80] + y[self.r-80:self.r]
        ly = np.polyfit(bx, by, 1)
        ly = [ly[0]*a + ly[1] for a in x]

        # plt.plot(x, y)
        # plt.plot(bx, by)
        # plt.plot(x, ly)
        # plt.show()
        # plt.clf()

        y = [xx[1] - ly[xx[0]] for xx in enumerate(y)]
        return y
    
    def remove_zero(self, f, list):
        #TODO: сделать через кружок
        if len(list) == 0:
            return []
        a = 50

        left_x = np.mean(list[:a])   
        right_x = np.mean(list[self.r-a:self.r])

        return [(xi - (left_x + right_x)/2) for xi in list]

    def filter_noise(self, x):
        n = 5  # the larger n is, the smoother curve will be
        b = [1.0 / n] * n
        a = 1
        y = signal.lfilter(b, a, x)

        return y, x - y #filtered signal and noise

    def calc_variance(self):
        x_comp_no_noise, x_noise = self.filter_noise(self.x)
        y_comp_no_noise, y_noise = self.filter_noise(self.y)
        x_noise_mean_square = np.mean([x**2 for x in x_noise])#np.max(x_noise)
        y_noise_mean_square = np.mean([y**2 for y in y_noise])#np.max(y_noise)
        return x_noise_mean_square + y_noise_mean_square

    # def calc_fit(self, f):
    #     # Solve for y values using fsolve   
    #     y_values = []
    #     for xx in f:
    #         # Use fsolve to find the root for y for each x
    #         y_solution = optimize.fsolve(equation, 1e-8, args=(xx, self.params))  # Starting guess for y is 1
    #         y_values.append(y_solution[0])  # Store the solution for y
    #     return y_values

    def error(self, params, x, y):
        ampl, freq, wid, alpha, var = params
        error_values = np.abs(lorentzian(params, x, y))
        return np.sum(error_values)

    def fit(self):
        if self.x == []:
            return
        bounds = [(0, max(self.ampl_square)), 
                (min(self.freq), max(self.freq)), 
                (0, 5000), 
                (-1e+11, 1e+11), 
                (0, np.mean(self.ampl_square))]
        
        result = optimize.differential_evolution(self.error, 
                                                bounds, 
                                                args=(np.array(self.freq), 
                                                np.array(self.ampl_square)), 
                                                maxiter=1000, 
                                                popsize=20)

        self.params = result.x

        self.resonant_amplitude = self.params[0] - self.params[4]
        self.resonant_freq = self.params[1]
        self.width = self.params[2]
        self.alpha = self.params[3]
        self.variance = self.params[4]
        self.q = self.params[1]/self.params[2]/2
        # self.ampl_fit = self.calc_fit(self.freq)
        # self.ampl_fit = [self.lorentzian(f, self.resonant_amplitude,
        #                             self.resonant_freq,
        #                             self.width, 
        #                             self.variance) for f in self.freq]
        
    def calculate(self):
        self.find_r()
        self.x = self.remove_zero(self.freq, self.x)
        self.y = self.remove_zero(self.freq, self.y)
        self.ampl_square = [x**2 + y**2 for x, y in zip(self.x, self.y)]
        self.ampl = [np.sqrt(x**2 + y**2) for x, y in zip(self.x, self.y)]
        self.fit()

    #TODO: мб сохранение в файл
    def plot(self):
        # figure, axis = plt.subplots(1, 3)
        # figure.suptitle(f"V = {self.v}")
        # axis[0].plot(self.freq, self.x)
        # axis[0].set_title("X")
        # axis[1].plot(self.freq, self.y)
        # axis[1].set_title("Y")
        # axis[2].set_title("Amplitude square")
        # axis[2].plot(self.freq, self.ampl_square)
        # axis[2].plot(self.freq, self.ampl_fit)
        # plt.plot(self.freq, self.ampl_square)
        # plt.plot(self.freq, self.ampl_fit)
        # plt.title(f"{self.alpha*1e-10} 10^10")
        # plt.show()
        
        ampl1, freq, wid, alpha, var = self.params
        x, y = sympy.symbols('x y')
        p1 = sympy.plot_implicit(sympy.Eq(y - (ampl1 * wid**2) / ((x - freq - alpha*y)**2 + wid**2) - var, 0), (x, min(self.freq), max(self.freq)), (y, 0, max(self.ampl_square)), show=False, title=f"V = {self.v}")
        p1.append(List2DSeries(self.freq, self.ampl_square))
        p1
        p1.show()
        pass
    
    def save(self, filename = "Curves"):
        figure, axis = plt.subplots(1, 3)
        figure.suptitle(f"V = {self.v}")
        axis[0].plot(self.freq, self.x)
        axis[0].set_title("X")
        axis[1].plot(self.freq, self.y)
        axis[1].set_title("Y")
        axis[2].set_title("Amplitude square")
        axis[2].plot(self.freq, self.ampl_square)
        axis[2].plot(self.freq, self.ampl_fit)
        figure.savefig(f"{filename}\\V = {self.v}.png")
        plt.close()

    r = 0
    v = 0
    freq = []
    x = []
    y = []
    ampl = []
    ampl_square = []
    ampl_fit = []
    params = []
    resonant_amplitude = 0
    resonant_freq = 0
    width = 0
    q = 0
    alpha = 0
    variance = 0

    pass


'''
def remove_zero_and_noise(freq, x): 
    if len(freq) == 0: 
        return []
    
    #TODO: проверить как работает
    #TODO: сделать через кружок
    left_x = np.mean(x[50:])   
    right_x = np.mean(x[:50])

    x = [(xi - (left_x + right_x)/2) for xi in x]

    return x




#Read out of file and write to list
list = [] # [частота, напряжение, х-, у-компонента]
with open('measurment.csv', 'r') as f:
    reader = csv.reader(f, delimiter="\t")
    v = 0
    freq = []
    x = []
    y = []
    for row in reader:
        current_v = float(row[1].replace(',', '.'))
        if current_v != v:
            y = remove_zero_and_noise(freq, y, v)
            x = remove_zero_and_noise(freq, x, v)

            list.append([freq, v, [(x1**2 + y1**2) for x1, y1 in zip(x, y)]]) # делаем (x^2 + y^2) [частота, напряга, амплитуда]
            freq = []
            x = []
            y = []
        v = current_v

        freq.append(float(row[0].replace(',', '.')))
        x.append(float(row[2].replace(',', '.')))
        y.append(float(row[3].replace(',', '.')))

    y = remove_zero_and_noise(freq, y, v)
    x = remove_zero_and_noise(freq, x, v)

    list.append([freq, v, [(x1**2 + y1**2) for x1, y1 in zip(x, y)]]) # [частота, напряга, амплитуда]
list.pop(0)
'''



# def lorentzian(x, amp, freq, wid, var):
#     return amp*wid**2/((x-freq)**2+wid**2) + var

# params = [] # [амплитуда, частота, ширина]
# #Fitting
# for m in list:
#     freq = m[0]
#     v = m[1]
#     x = m[2]
#     popt, pcov_lorentzian = scipy.optimize.curve_fit(lorentzian, freq, x, p0=[max(x), freq[x.index(max(x))], 1000]) #TODO:автоматическое определение параметров    
#     params.append(popt) 

# #Записывает все картинки в файл Pictures вместе с аппроксимацией
# for m, p in zip(list, params):
#     v = m[1]
#     plt.plot(m[0], lorentzian(m[0], p[0], p[1], p[2]), 'o', color='blue')
#     plt.plot(m[0], m[2], 'o', color='orange')
#     plt.title(f"V = {10*v}")
#     plt.savefig(f"Pictures\\V = {10*v}-savgol.png")
#     plt.clf()

# #Рисуем параметры
# for m, p in zip(list, params):
#     plt.plot(m[1], p[0], 'o')
# plt.savefig("Pictures\\Amplitude-savgol.png")
# plt.clf()

# for m, p in zip(list, params):
#     plt.plot(m[1], p[1], 'o')
# plt.savefig("Pictures\\Frequency-savgol.png")
# plt.clf()

# for m, p in zip(list, params):
#     plt.plot(m[1], p[1]/p[2], 'o')
# plt.savefig("Pictures\\Q-factor-savgol.png")
# plt.clf()
