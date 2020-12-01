import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.stats import poisson, norm
import random as rand
import matplotlib as mp
import matplotlib.pyplot as plt


@dataclass
class Parameters:
    e_b: float
    # tao: float
    number_of_blue: int = 1
    number_of_green: int = 1
    total_n: int = number_of_blue + number_of_green
    h_b: float = 1
    h_g: float = 1
    w_min: float = 0.0
    # Alternatively, ref_dist can be 'normal on normal'
    ref_distribution: str = "Poisson" 
    value_distribution: str = "vh vl"
    vh : float = 2
    vl : float = 0
    alpha: float = 0.5
    alpha_b: float = 0.7
    alpha_g: float = 0.5
    
    vh_freq: float = 0.5
    b_v_freq: float = vh_freq
    g_v_freq: float = vh_freq
    
    value_mean: float = vh_freq * vh + (1-vh_freq) * vl
    value_variance: float = (vh ** 2) * vh_freq + (vl ** 2) * (1-vh_freq) - (value_mean ** 2)
    
    b_value_mean: float = value_mean
    g_value_mean: float = value_mean
    
    b_value_variance: float = value_variance
    g_value_variance: float = value_variance
    
    b_value_sigma : float = b_value_variance ** (0.5)
    g_value_sigma : float = g_value_variance ** (0.5)
    
    prob_b : float = number_of_blue / total_n
    prob_b_h : float = prob_b * b_v_freq
    prob_b_l : float = prob_b * (1 - b_v_freq)
    prob_g : float = number_of_green / total_n
    prob_g_h : float = prob_g * (g_v_freq)
    prob_g_l : float = prob_g * (1 - g_v_freq)
    
    # Should be equilibrium employed g_v_freq
    # gh_earning : float = (1 - e_b)*(h_g)*(alpha_g*g_v_freq)
    
    r: float = 1.0
    
    def calculate_threshold(self):
        p = self
        e_b = p.e_b
        e_g = 1 - e_b
        # figure somehting out here
        # self.alpha_b = self.alpha
        # self.alpha_g = 0.5 + (self.alpha_g - 0.5 + p.tao * e_g)/(1+p.tao)
        # print('alpha b is ')
        # print(self.alpha_b)
        
        if self.ref_distribution == "Poisson":
            b_h_lambda = 1/(p.b_v_freq * p.number_of_blue) * ( (e_b * p.h_b * ((p.b_v_freq * p.alpha_b) + (1-p.alpha_b)*(1-p.b_v_freq))) + (1-p.h_g)*e_g * ((p.g_v_freq* p.alpha_g) + (1-p.g_v_freq) * (1-p.alpha_g)))
            b_l_lambda = 1/((1-p.b_v_freq) * p.number_of_blue) * ( (e_b * p.h_b * (((1-p.b_v_freq) * p.alpha_b) + (1-p.alpha_b)*(p.b_v_freq))) + (1-p.h_g)*e_g * (((1-p.g_v_freq)* p.alpha_g) + (p.g_v_freq * (1-p.alpha_g))))
            
            
            g_h_lambda = 1/(p.g_v_freq * p.number_of_green) * ( (e_g * p.h_g * ((p.g_v_freq * p.alpha_g) + (1-p.alpha_g)*(1-p.g_v_freq))) + (1-p.h_b)*e_b * ((p.b_v_freq* p.alpha_b) + (1-p.b_v_freq) * (1-p.alpha_b)))
            g_l_lambda = 1/((1-p.g_v_freq) * p.number_of_green) * ( (e_g * p.h_g * (((1-p.g_v_freq) * p.alpha_g) + (1-p.alpha_g)*(p.g_v_freq))) + (1-p.h_b)*e_b * (((1-p.b_v_freq)* p.alpha_b) + (p.b_v_freq * (1-p.alpha_b))))
            
            
            p_b_h_zero = poisson.pmf(0, b_h_lambda)
            p_b_l_zero = poisson.pmf(0, b_l_lambda)
            
            p_g_h_zero = poisson.pmf(0, g_h_lambda)
            p_g_l_zero = poisson.pmf(0, g_l_lambda)
        
        prob_b = p.number_of_blue / p.total_n
        prob_b_h = prob_b * p.b_v_freq
        prob_b_l = prob_b * (1-p.b_v_freq)
        prob_g = p.number_of_green / p.total_n
        prob_g_h = prob_g * p.g_v_freq
        prob_g_l = prob_g * (1 - p.g_v_freq)
        
        l_h_s = p.w_min - 1
        r_h_s = p.w_min
        
     
        while abs(l_h_s - r_h_s) != 0:
            l_h_s = r_h_s 
            r_h_s = (
                (
                    (   (p_b_h_zero*prob_b_h + p_g_h_zero*prob_g_h)* p.vh + p.vl*(prob_b_l + prob_g_l) )
                )
            /
                (
                    # Denominator
                    (p_b_h_zero * prob_b_h+ p_g_h_zero* prob_g_h) + prob_b_l + prob_g_l
                )
            )
        threshold = max(r_h_s, p.w_min)
        # threshold = 0.1541919477612662
        self.threshold = threshold
        return (threshold)
    
    def hire_continuous(self) -> float:
        p = self
        e_b = self.e_b
        e_g = 1 - e_b
        
        assert self.threshold > self.w_min, "Make sure to calculate threshold before hiring"
        assert e_b <= 1, "Something wrong with the logic"
        
        if self.ref_distribution == "Poisson":
            b_h_lambda = 1/(p.b_v_freq * p.number_of_blue) * ( (e_b * p.h_b * ((p.b_v_freq * p.alpha_b) + (1-p.alpha_b)*(1-p.b_v_freq))) + (1-p.h_g)*e_g * ((p.g_v_freq* p.alpha_g) + (1-p.g_v_freq) * (1-p.alpha_g)))
            b_l_lambda = 1/((1-p.b_v_freq) * p.number_of_blue) * ( (e_b * p.h_b * (((1-p.b_v_freq) * p.alpha_b) + (1-p.alpha_b)*(p.b_v_freq))) + (1-p.h_g)*e_g * (((1-p.g_v_freq)* p.alpha_g) + (p.g_v_freq * (1-p.alpha_g))))
            
            
            g_h_lambda = 1/(p.g_v_freq * p.number_of_green) * ( (e_g * p.h_g * ((p.g_v_freq * p.alpha_g) + (1-p.alpha_g)*(1-p.g_v_freq))) + (1-p.h_b)*e_b * ((p.b_v_freq* p.alpha_b) + (1-p.b_v_freq) * (1-p.alpha_b)))
            g_l_lambda = 1/((1-p.g_v_freq) * p.number_of_green) * ( (e_g * p.h_g * (((1-p.g_v_freq) * p.alpha_g) + (1-p.alpha_g)*(p.g_v_freq))) + (1-p.h_b)*e_b * (((1-p.b_v_freq)* p.alpha_b) + (p.b_v_freq * (1-p.alpha_b))))            
            
            p_b_h_zero = poisson.pmf(0, b_h_lambda)
            p_b_l_zero = poisson.pmf(0, b_l_lambda)
            
            p_g_h_zero = poisson.pmf(0, g_h_lambda)
            p_g_l_zero = poisson.pmf(0, g_l_lambda)

        # prob hired from pool - same for both men and women
        
        # b_p_h_pool = (
        #     (1 -
        #          ((1 - p_b_h_zero) * p.prob_b_h * p.number_of_blue + (1 - p_g_h_zero) * p.prob_g_h * p.number_of_green)
        #          )
        # /
        #     (
        #     (p_b_h_zero * p.prob_b_h + p.prob_b_l)*p.number_of_blue + (p_g_h_zero * p.prob_g_h + p.prob_g_l)*p.number_of_green
        #     )
        # )
        
        # Commented out p.prob_b_h because it is given if (1-p_g_h_zero)
        b_p_h_pool = (
            (1 -
                 ((1 - p_b_h_zero)*p.b_v_freq * p.number_of_blue + (1 - p_g_h_zero)*p.g_v_freq * p.number_of_green)
                 )
        /
            (
            (p_b_h_zero*p.b_v_freq + p_b_l_zero*(1-p.b_v_freq) + (1-p_b_l_zero)*(1-p.b_v_freq))*p.number_of_blue + (p_g_h_zero*p.g_v_freq + p_g_l_zero*(1-p.g_v_freq) + (1-p_g_l_zero)*(1-p.g_v_freq))*p.number_of_green
            )
        )
                
        
        # print(f'{b_p_h_pool} is bph pool')
        
        if b_p_h_pool > 1:
            print('uh oh probability greater than one check logic')
        
        b_p_h_pool = b_p_h_pool if b_p_h_pool < 1 else 1
        
                     
        
        # g_p_h_pool = b_p_h_pool
        
        # prob hired given blue received a referral
        # Prob v > threshold * 1 + Prob v< threshold and hired from pool
        b_p_h_r = p.prob_b_h
        g_p_h_r = p.prob_g_h
        
        
        b_p_h_not_r = b_p_h_pool
        # e_b_next = ((p_b_h_zero * p.prob_b_h + p.prob_b_l) * b_p_h_not_r + (1-p_b_h_zero)*b_p_h_r)*p.number_of_blue
        # e_g_next = ((p_g_h_zero * p.prob_g_h + p.prob_g_l) * b_p_h_pool + (1-p_g_h_zero)* g_p_h_r) * p.number_of_green
        
    
        
        
        e_b_next = ((p_b_h_zero * p.b_v_freq + p_b_l_zero*(1-p.b_v_freq) + (1-p_b_l_zero)*(1-p.b_v_freq)) * b_p_h_not_r + (1-p_b_h_zero)*p.b_v_freq) * p.number_of_blue
        e_g_next = ((p_g_h_zero*p.g_v_freq + p_g_l_zero*(1-p.g_v_freq) + (1-p_g_l_zero)*(1-p.g_v_freq)) * b_p_h_pool + (1-p_g_h_zero)*p.g_v_freq) * p.number_of_green
        # print(e_g_next + e_b_next)
        return (e_b_next)

def run_periods(periods = 15, e_b = 0.8, n= 2.0, alpha_b= 1, alpha_g= 1, 
                      h_b= 1, h_g= 1 ):
    e_b = 0.8
    tao = 0.5
    periods = periods

        
    for period in range(periods):
        p = Parameters(e_b = e_b, number_of_blue= n, number_of_green=n, alpha_b= alpha_b, alpha_g= alpha_g, 
                      h_b= h_b, h_g= h_g)
        if period == 0:
            print(f'The parameters for p are {p}')
        print(f'the male employment rate in period {period} is {e_b}')
        p.calculate_threshold()

        print(f'the skill threshold for this period is {p.threshold} ')
        # if e_b <= 0.5:
            # print(f'It took {period} generations for the male employment rate to equal the female employment rate')
            # break
        e_b = p.hire_continuous()
        

def run_period(e_b, n: float = 2.0, alpha_b: float = 0.8, alpha_g: float = 0.8, 
                      h_b: float = 0.8, h_g: float = 0.8):
    
    p = Parameters(e_b = e_b, number_of_blue= n, number_of_green=n, alpha_b= alpha_b, alpha_g= alpha_g, 
                      h_b= h_b, h_g= h_g )
    
    
    p.calculate_threshold()

    e_b_new = p.hire_continuous()
    return e_b_new
    
def find_steady_state(e_b_0: float, n: float, alpha_b: float, alpha_g: float, 
                      h_b: float, h_g: float, max_iterations: int = 1000, return_iterations: bool = False):
    iteration = 0
    e_b = e_b_0
    e_b_new = 0
    if return_iterations:
        while abs(e_b - e_b_new) > 0.000001 and iteration < max_iterations:
            iteration +=1
            e_b = e_b_new if e_b_new else e_b
            e_b_new= run_period(e_b = e_b, n = n, alpha_b=alpha_b, alpha_g=alpha_g, h_b = h_b, h_g = h_g)
    
    else:
        while e_b != e_b_new and iteration < max_iterations:
            iteration +=1
            e_b = e_b_new if e_b_new else e_b
            e_b_new= run_period(e_b = e_b, n = n, alpha_b=alpha_b, alpha_g=alpha_g, h_b = h_b, h_g = h_g)
        
    
    if iteration == max_iterations:
        # print('max iteration reached')
        if return_iterations:
            return iteration
        else:
            return e_b
    
    else:
        # print(f'reached in iteration # {iteration}')
        if return_iterations:
            return iteration
        else:
            return e_b
    
    
def plot_e_b():
    n_array = np.linspace(0.6, 3, num = 10)
    a_b_array = np.linspace(0.5, 1, num = 6)
    a_g_array = np.linspace(0.5, 1, num = 6)
    h_g_array = np.linspace(0.5, 1, num = 6)
    h_b_array = np.linspace(0.5, 1, num = 6)

    e_b_array_dict = {
    'n, a_b':
    [n_array, list(map(lambda a_b: [a_b, np.array([find_steady_state(e_b_0 = 0.8, n=n, alpha_b = a_b, 
                                            alpha_g = 0.5, h_b = 1, h_g = 1) for n in n_array])], a_b_array))]
    ,
    'n, h_b, h_g = 0.5':
        [n_array, list(map(lambda h_b: [h_b, np.array([find_steady_state(e_b_0 = 0.8, n=n, alpha_b = 1, 
                                            alpha_g = 0.5, h_b = h_b, h_g = 0.5) for n in n_array])], h_b_array))]
    ,
    'h_b, n = 2 and alpha':
        [h_b_array, list(map(lambda a: [a, np.array([find_steady_state(e_b_0 = 0.8, n=2.0, alpha_b = a, 
                                            alpha_g = a, h_b = h_b, h_g = 0.5) for h_b in h_b_array])], a_b_array))]
    ,
    'h_b, alpha_b, alpha_g = 0.5 and h_g =0.5 and n = 2':
        [h_b_array, list(map(lambda a: [a, np.array([find_steady_state(e_b_0 = 0.8, n=2.0, alpha_b = a, 
                                            alpha_g = 0.5, h_b = h_b, h_g = 0.5) for h_b in h_b_array])], a_b_array))]
    }

    for key in e_b_array_dict:
        print(e_b_array_dict[key][1])

    # print(e_b_arrays_n)
    fig, axs = plt.subplots(len(e_b_array_dict),1, figsize = (25,25))

    for j,key in enumerate(e_b_array_dict):
        
        key_list = key.split(',')
        e_b_arrays = e_b_array_dict[key][1]
        x_array = e_b_array_dict[key][0]
        
        for i in range(len(a_b_array)):
            axs[j].plot(x_array, e_b_arrays[i][1], label=f"{key_list[1]} = {e_b_arrays[i][0]}, {key_list[2] if len(key_list) > 2 else ''}")
            axs[j].legend()
            axs[j].set_ylim(0.4,0.8)
            axs[j].set_xlabel(f'{key_list[0]}')
            axs[j].set_ylabel('e_b')
            axs[j].set_title(f'e_b as a function of {key_list[0]}')
    plt.show()


def plot_iterations():
    n_array = np.linspace(0.6, 3, num = 10)
    a_b_array = np.linspace(0.5, 1, num = 6)
    a_g_array = np.linspace(0.5, 1, num = 6)
    h_g_array = np.linspace(0.5, 1, num = 6)
    h_b_array = np.linspace(0.5, 1, num = 6)
    
    iteration_no_array_dict = {
            'n, a_b':
        [n_array, list(map(lambda a_b: [a_b, np.array([find_steady_state(e_b_0 = 0.8, n=n, alpha_b = a_b, 
                                                alpha_g = 0.5, h_b = 1, h_g = 1, return_iterations=True) for n in n_array])], a_b_array))]
        ,
        'n, h_b, h_g = 0.5 and alpha_b = 1':
            [n_array, list(map(lambda h_b: [h_b, np.array([find_steady_state(e_b_0 = 0.8, n=n, alpha_b = 1, 
                                                alpha_g = 0.5, h_b = h_b, h_g = 0.5, return_iterations=True) for n in n_array])], h_b_array))]
        ,
        'h_b, n = 2 and alpha':
            [h_b_array, list(map(lambda a: [a, np.array([find_steady_state(e_b_0 = 0.8, n=2.0, alpha_b = a, 
                                                alpha_g = a, h_b = h_b, h_g = 0.5, return_iterations=True) for h_b in h_b_array])], a_b_array))]
        ,
        'h_b, alpha_b, alpha_g = 0.5 and h_g =0.5 and n = 2':
            [h_b_array, list(map(lambda a: [a, np.array([find_steady_state(e_b_0 = 0.8, n=2.0, alpha_b = a, 
                                                alpha_g = 0.5, h_b = h_b, h_g = 0.5, return_iterations=True) for h_b in h_b_array])], a_b_array))]
    }                       

    fig, axs = plt.subplots(len(iteration_no_array_dict),1, figsize = (25,25))

    for j,key in enumerate(iteration_no_array_dict):
        
        key_list = key.split(',')
        iteration_no_arrays = iteration_no_array_dict[key][1]
        x_array = iteration_no_array_dict[key][0]
        
        for i in range(len(a_b_array)):
            axs[j].plot(x_array, iteration_no_arrays[i][1], label=f"{key_list[1]} = {iteration_no_arrays[i][0]}, {key_list[2] if len(key_list) > 2 else ''}")
            axs[j].legend()
            axs[j].set_ylim(0,20)
            axs[j].set_xlabel(f'{key_list[0]}')
            axs[j].set_ylabel('iteration_no')
            axs[j].set_title(f'iteration_no as a function of {key_list[0]}')
    plt.show()

# print(find_steady_state(e_b_0 = 0.8, n=2.0, alpha_b = 1.0, 
                                                # alpha_g = 0.5, h_b = 1, h_g = 1))
# run_periods(periods=15, e_b = 0.8, n=2.0, alpha_b = 1.0, 
#                                                 alpha_g = 0.5, h_b = 1, h_g = 1)

# print(run_periods())