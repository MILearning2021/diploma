#https://www.kaggle.com/jdarcy/introducing-ssa-for-time-series-decomposition

#The general procedure in SSA is as follows:

#1.Embed the time series by forming a Hankel matrix of lagged window (length K) vectors.
#2.Decompose the embedded time series via Singular Value Decomposition
#3.Eigentripple Grouping is the process of identifying eigenvalue-eigenvector pairs as trend, seasonal and noise
#4.Reconstruct the time series from the eigenvalue-eigenvector pairs identified as trend and seasonal. 

#This is done through a process called diagonal averaging.

# стандартные библиотеки
import numpy as np
import pandas as pd

# графика
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 50

from cycler import cycler
cols = plt.get_cmap('tab10').colors
plt.rcParams['axes.prop_cycle'] = cycler(color = cols)

class SSA(object):
    """ 
        Класс реализации методики "анализа сингулярного спектра" для анализа временных рядов.
    """
    
    def __init__(self, L, ts):
        """ 
            Создает экземпляр класса. 
            Аргументы: 
                L - окно анализа, int
                ts - временной ряд, np.array     
        """
        self.ts = ts # временной ряд
        self.L = L # окно анализа
        self.N = self.ts.shape[0] # количество наблюдений
        self.T = np.arange(0, self.N) # метки ряда 
      
    def trajectory_matrix(self):
        """ Строит и возвращает матрицу траекторий. """

        K = self.N - self.L + 1 # количество колонок в "матрице траекторий"  
        self.X = np.column_stack([self.ts[i : i + self.L] for i in range(0, K)]) # "матрица траекторий"
         
        return self.X
    
    def SVD(self):
        """ Сингулярное разложение матрицы траекторий. """

        self.d = np.linalg.matrix_rank(self.X) # внутренняя размерность матрицы
        self.U, self.Sigma, self.V = np.linalg.svd(self.X) # сингулярное разложение матрицы
        # V = self.V.T обратно транспонируем составляющую V матрицы, 
        # т.к. метод SVD numpy возвращает транспонированную матрицу V
        self.Xi = np.array([self.Sigma[i] * np.outer(self.U[:, i], self.V.T[:, i]) for i in range(0, self.d)])
            
        return self.Xi

    def pict_contribution_of_components(self, num_of_components = None):
        """ Рисует графики. """
        
        def contribution_of_components(Sigma):
            """ Нормированный вклад каждой из компонент, вклад компонент накопительным итогом. """
        
            sigma_sumsq = (Sigma**2).sum()
        
            return Sigma**2 / sigma_sumsq * 100, (Sigma**2).cumsum() / sigma_sumsq * 100
        
        fig, ax = plt.subplots(1, 2, figsize=(14, 3))
        _norm, _cumulative_total = contribution_of_components(self.Sigma)
        
        if num_of_components:
            lim1 = min(num_of_components, _norm.shape[0])
            lim2 = min(num_of_components, _cumulative_total.shape[0])
        else:
            lim1 = _norm.shape[0]
            lim2 = _cumulative_total.shape[0]
        
        # Sigma - собственные числа матрицы, у наиболее значимого компонента наибольшее собственное число 
        
        # нормированный вклад каждой из компонент
        ax[0].plot(_norm) 
        ax[0].set_xlim(0, lim1)
        ax[0].set_title("Относительный вклад компонент $\mathbf{X}_i$")
        ax[0].set_xlabel("$i$")
        ax[0].set_ylabel("Вклад (%)")
        
        # суммарный вклад компонент накопительным итогом
        ax[1].plot(_cumulative_total) 
        ax[1].set_xlim(0, lim2)
        ax[1].set_title("Накопительный вклад компонент $\mathbf{X}_i$")
        ax[1].set_xlabel("$i$")
        ax[1].set_ylabel("Вклад (%)")
    
    def pict_components(self, num_of_components = None):
        """ Рисует графики компонент. """
 
        def X_to_TS(X_i):
            """ Averages the anti-diagonals of the given elementary matrix, X_i, and returns a time series. """
        
            # Reverse the column ordering of X_i
            X_rev = X_i[::-1]
            
            # Full credit to Mark Tolonen at https://stackoverflow.com/a/6313414 for this one:
            return np.array([X_rev.diagonal(i).mean() for i in range(-X_i.shape[0] + 1, X_i.shape[1])])
        
        if num_of_components:
            n = min(num_of_components, self.d) # In case of noiseless time series with d < num_of_components.
        else:
            n = self.d

        # Fiddle with colour cycle - need more colours!
        plt.figure(figsize = (14, 8), dpi = 150)
        fig = plt.subplot()#yticks = [])
        color_cycle = cycler(color = plt.get_cmap('tab20').colors)
        fig.axes.set_prop_cycle(color_cycle)

        # Convert elementary matrices straight to a time series - no need to construct any Hankel matrices.
        self.Xi_ts = [] # хранит компоненты в виде временного ряда
               
        for ind in range(n):
            self.Xi_ts.append(X_to_TS(self.Xi[ind]))
            fig.axes.plot(self.T, self.Xi_ts[ind], lw = 2)
            
        self.Xi_ts = np.array(self.Xi_ts) # пусть хранит в виде массива np.array
        
        fig.axes.plot(self.ts, alpha = 1, lw = .5)
        fig.axes.grid()
        fig.set_xlabel("$t$")
        fig.set_ylabel(r"$\tilde{X}_i(t)$")
        legend = [r"$\tilde{X}_{%s}$" %i for i in range(n)] + ["$X$"]
        fig.set_title(f"Первые {n} компонент временного ряда")
        fig.legend(legend, loc = (1.05, 0.1))

if __name__ == "__main__":
    pass