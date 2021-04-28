import numpy as np
from dataclasses import dataclass, field
from typing import List
from scipy.stats import poisson, norm
import random as rand
import matplotlib as mp
import matplotlib.pyplot as plt

vh_freq = 0.4
@dataclass
class Parameters:
    e_b: float
    number_of_blue: int = 1
    number_of_green: int = 1
    total_n: int = number_of_blue + number_of_green
    h_b: float = 1
    h_g: float = 0.5
    w_min: float = 0.0
    # Alternatively, ref_dist can be 'normal on normal'
    ref_distribution: str = "Poisson" 
    value_distribution: str = "vh vl"
    vh : float = 2
    vl : float = 0
    alpha: float = 0.9
    alpha_b: float = alpha
    alpha_g: float = alpha
    
    # 0.2549378627974277
    vh_freq: float = vh_freq
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
    
    
    
    r: float = 1.0
    
    def calculate_threshold(self):
        p = self
        e_b = p.e_b
        e_g = 1 - e_b
        if self.ref_distribution == "Poisson":
            self.b_lambda = 1/(p.number_of_blue) * ( (e_b * p.h_b) + (1-p.h_g)*e_g)            
            
            self.g_lambda = 1/(p.number_of_green) * ( e_g * p.h_g + (1-p.h_b)*e_b )
            
            self.p_b_zero = poisson.pmf(0, self.b_lambda)
            
            self.p_g_zero = poisson.pmf(0, self.g_lambda)

        
        prob_b = p.number_of_blue / p.total_n
        prob_b_h = prob_b * p.b_v_freq
        prob_g = p.number_of_green / p.total_n
        prob_g_h = prob_g * p.g_v_freq
        l_h_s = p.w_min - 1
        r_h_s = p.w_min
     
        while abs(l_h_s - r_h_s) != 0:
            l_h_s = r_h_s 
            r_h_s = (
                            (
                                (   (self.p_b_zero* prob_b_h + self.p_g_zero * prob_g_h) * p.vh + p.vl*(1-prob_b_h + 1 - prob_g_h) )
                            )
                        /
                        (
                                # Denominator
                            (
                                self.p_b_zero * prob_b_h + self.p_g_zero * prob_g_h + 1-prob_b_h + 1 - prob_g_h
                            )
                        )
                    )
        threshold = max(r_h_s, p.w_min)
        # threshold = 0.1541919477612662
        self.threshold = threshold
        return (threshold)
    
    def hire_continuous(self) -> float:
        p = self
        e_b = p.e_b
        e_g = 1 - e_b

        if self.ref_distribution == "Poisson":
            self.b_lambda = 1/(p.number_of_blue) * ( (e_b * p.h_b) + (1-p.h_g)*e_g)            
            
            self.g_lambda = 1/(p.number_of_green) * ( e_g * p.h_g + (1-p.h_b)*e_b )
            
            self.p_b_zero = poisson.pmf(0, self.b_lambda)
            
            self.p_g_zero = poisson.pmf(0, self.g_lambda)
        
        assert self.threshold > self.w_min, "Make sure to calculate threshold before hiring"
        assert e_b <= 1, "Something wrong with the logic"
        

        # prob hired from pool - same for both men and women
        
        # Commented out p.prob_b_h because it is given if (1-p_g_h_zero)
        b_p_h_pool = (
            (1 -
                 ((1 - self.p_b_zero)*p.b_v_freq * p.number_of_blue + (1 - self.p_g_zero)*p.g_v_freq * p.number_of_green)
                 )
        /
            (
            (self.p_b_zero + (1-p.b_v_freq)*(1-self.p_b_zero))*p.number_of_blue + (self.p_g_zero + (1-self.p_g_zero)*(1-p.g_v_freq))*p.number_of_green
            )
        )
                
        
        # print(f'{b_p_h_pool} is bph pool')
        
        if b_p_h_pool > 1:
            print('uh oh probability greater than one check logic')
        
        b_p_h_pool = b_p_h_pool if b_p_h_pool < 1 else 1
        
        
        non_norm_ebh_next = (p.vh_freq * (1 - self.p_b_zero) + b_p_h_pool * self.p_b_zero *p.b_v_freq) * p.number_of_blue
        non_norm_egh_next = (p.vh_freq * (1 - self.p_g_zero) + b_p_h_pool * self.p_g_zero *p.b_v_freq) * p.number_of_green
        
        e_b_next = (p.vh_freq * (1 - self.p_b_zero) + b_p_h_pool * (1- (1-self.p_b_zero)*p.b_v_freq)) * p.number_of_blue
        e_g_next = (p.vh_freq * (1 - self.p_g_zero) + b_p_h_pool * (1- (1-self.p_g_zero)*p.b_v_freq)) * p.number_of_green
        
        ebh_next = non_norm_ebh_next/e_b_next
        egh_next = non_norm_egh_next/e_g_next
        
        # print(e_b_next, e_g_next)
        # Check logic by employment summing to 1 
        # print(e_g_next + e_b_next)
        return e_b_next, ebh_next, egh_next

def run_period(e_b: float, n: float = 2.0, alpha_b: float = 0.8, alpha_g: float = 0.8, 
    n_b : float = None , n_g : float = None, h_b: float = 0.8, h_g: float = 0.8):
    
    n_b = n_b if n_b else n
    n_g = n_g if n_g else n

    p = Parameters(e_b = e_b, number_of_blue= n_b, number_of_green=n_g, alpha_b= alpha_b, alpha_g= alpha_g, 
                      h_b= h_b, h_g= h_g)
    
    
    p.calculate_threshold()

    e_b, ebh_next, egh_next = p.hire_continuous()
    
    return e_b, ebh_next, egh_next

def find_steady_state(e_b_0: float, alpha_b: float, alpha_g: float, h_b: float,  h_g: float,
    n_b : float = None, n_g : float = None, max_iterations: int = 1000, return_iterations: bool = False):
    iteration = 0
    e_b = e_b_0
    e_b_new = 0

    n_b = n_b if n_b else n
    n_g = n_g if n_g else n

    if return_iterations:

        while abs(e_b - e_b_new) != 0.0 and iteration < max_iterations:
            iteration +=1
            e_b = e_b_new if e_b_new else e_b
            e_b_new, ebh_new, egh_new = run_period(e_b = e_b, n_b = n_b, n_g = n_g, alpha_b=alpha_b, alpha_g=alpha_g, h_b = h_b, h_g = h_g)
    
    else:
        
        while e_b!= e_b_new and iteration < max_iterations:
            iteration +=1
            e_b = e_b_new if e_b_new else e_b
            e_b_new, ebh_new, egh_new = run_period(e_b = e_b, n_b = n_b, n_g = n_g, alpha_b=alpha_b, alpha_g=alpha_g, h_b = h_b, h_g = h_g)
        
    if iteration == max_iterations and e_b != e_b_new:
        print('max iteration reached')
        if return_iterations:
            return iteration
        else:
            return e_b_new, ebh_new, egh_new
    
    else:
        print(f'reached in iteration # {iteration}')
        if return_iterations:
            return iteration
        else:
            p = Parameters(e_b = e_b, number_of_blue= n_b, number_of_green=n_g, alpha_b= alpha_b, alpha_g= alpha_g, 
                      h_b= h_b, h_g= h_g)
            p.calculate_threshold()
            print('lambda b')
            print(p.b_lambda)
            print(p.b_lambda*poisson.pmf(0, p.b_lambda))           
            return e_b_new, ebh_new, egh_new


def run_periods(periods = 15, e_b = 0.8, e_b_h = 0.5, e_g_h = 0.5, n= 2.0, n_b : float = None, n_g : float = None, alpha_b= 1, alpha_g= 1, 
                      h_b= 1, h_g= 1 ):
    e_b = e_b
    e_b_h = e_b_h
    e_g_h = e_g_h
    periods = periods

    n_b = n_b if n_b else n
    n_g = n_g if n_g else n

        
    for period in range(periods):
        p = Parameters(e_b = e_b, number_of_blue = n_b, number_of_green = n_g, alpha_b= alpha_b, alpha_g= alpha_g, 
                      h_b= h_b, h_g= h_g)
        if period == 0:
            print(f'The parameters for p are {p}')
        print(f'the male employment rate in period {period} is {e_b}')
        print(f'the high skilled male employment is {e_b_h}')
        print(f'the high skilled female emp is {e_g_h}')
        p.calculate_threshold()

        # print(f'the skill threshold for this period is {p.threshold} ')
        # if e_b <= 0.5:
            # print(f'It took {period} generations for the male employment rate to equal the female employment rate')
            # break
        e_b, e_b_h, e_g_h = p.hire_continuous()

print(find_steady_state(e_b_0 = 1.0, n_b = 1.0, n_g = 1.0, h_b = 1, h_g = 1, alpha_b = 1, alpha_g = 1, return_iterations=False))
# run_periods(e_b=0.8, h_b = 1, h_g = 1)
