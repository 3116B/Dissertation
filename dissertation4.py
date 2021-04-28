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
    e_g: float
    e_b_h: float # defined as proportion of e_b that is h (prob h given e_b)
    e_b_m: float
    e_g_h: float
    e_g_m: float
    # tao: float
    number_of_blue: int = 1
    number_of_green: int = 1
    total_n: int = number_of_blue + number_of_green
    h_b: float = 1
    h_g: float = 1
    w_min: float = 0.0
    # Alternatively, ref_dist can be 'normal on normal'
    ref_distribution: str = "Poisson" 
    value_distribution: str = "vh vm vl"
    vh : float = 1
    vm: float = 0.5
    vl : float = 0.0
    alpha: float = 0.5
    alpha_b: float = alpha
    alpha_g: float = alpha
    
    # 0.2549378627974277
    vh_freq: float = 1/3
    vm_freq: float = 1/3
    b_vh_freq: float = vh_freq
    g_vh_freq: float = vh_freq
    b_vm_freq: float = vm_freq
    g_vm_freq: float = vm_freq
    
    value_mean: float = vh_freq * vh + vm_freq * vm + (1- vh_freq - vm_freq) * vl
    value_variance: float = (vh ** 2) * vh_freq + (vm ** 2)* vm_freq + (vl ** 2) * (1-vh_freq-vm_freq) - (value_mean ** 2)
    
    b_value_mean: float = value_mean
    g_value_mean: float = value_mean
    
    b_value_variance: float = value_variance
    g_value_variance: float = value_variance
    
    b_value_sigma : float = b_value_variance ** (0.5)
    g_value_sigma : float = g_value_variance ** (0.5)
    
    prob_b : float = number_of_blue / total_n
    prob_b_h : float = prob_b * b_vh_freq
    prob_b_m : float = prob_b * b_vm_freq
    prob_b_l : float = prob_b * (1 - b_vh_freq - b_vm_freq)
    prob_g : float = number_of_green / total_n
    prob_g_h : float = prob_g * (g_vh_freq)
    prob_g_m : float = prob_g * g_vm_freq
    prob_g_l : float = prob_g * (1 - g_vh_freq - g_vm_freq)
    
    # Should be equilibrium employed g_vh_freq
    # gh_earning : float = (1 - e_b)*(h_g)*(alpha_g*g_vh_freq)
    
    r: float = 1.0
    
    def calculate_threshold(self):
        self.b_vl_freq = 1 - self.b_vm_freq -self.b_vh_freq
        self.g_vl_freq = 1 - self.g_vm_freq -self.g_vh_freq
        p = self
        e_b = p.e_b
        e_b_h = p.e_b_h
        e_b_m = p.e_b_m
        e_b_l = 1 - e_b_h - e_b_m
        e_g_h = p.e_g_h
        e_g_m = p.e_g_m
        e_g_l = 1 - e_g_h - e_g_m
        e_g = p.e_g
        # figure somehting out here
        # self.alpha_b = self.alpha
        # self.alpha_g = 0.5 + (self.alpha_g - 0.5 + p.tao * e_g)/(1+p.tao)
        # print('alpha b is ')
        # print(self.alpha_b)
        
        if self.ref_distribution == "Poisson":
            b_h_lambda = 1/(p.b_vh_freq * p.number_of_blue) * ( 
                (e_b * p.h_b * ((e_b_h * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_m))) + 
                (1-p.h_g)*e_g * ((e_g_h * p.alpha_g) + 0.5*(e_g_l + e_g_m) * (1-p.alpha_g)))
            )
            
            b_m_lambda =  1/(p.b_vm_freq * p.number_of_blue) * ( 
                (e_b * p.h_b * ((e_b_m * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_h))) + 
                (1-p.h_g)*e_g * ((e_g_m * p.alpha_g) + 0.5*(e_g_l + e_g_h) * (1-p.alpha_g)))
            )
            
            b_l_lambda =  1/(p.b_vl_freq * p.number_of_blue) * ( 
                (e_b * p.h_b * ((e_b_l * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_m + e_b_h))) + 
                (1-p.h_g)*e_g * ((e_g_l * p.alpha_g) + 0.5*(e_g_m + e_g_h) * (1-p.alpha_g)))
            )            
            
            g_h_lambda = 1/(p.g_vh_freq * p.number_of_green) * ( 
                (e_b * (1-p.h_b) * ((e_b_h * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_m))) + 
                (p.h_g)*e_g * ((e_g_h * p.alpha_g) + 0.5*(e_g_l + e_g_m) * (1-p.alpha_g)))
            )
            
            g_m_lambda =  1/(p.g_vm_freq * p.number_of_green) * ( 
                (e_b * (1-p.h_b) * ((e_b_m * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_h))) + 
                (p.h_g)*e_g * ((e_g_m * p.alpha_g) + 0.5*(e_g_l + e_g_h) * (1-p.alpha_g)))
            )
            
            g_l_lambda =  1/(p.g_vl_freq * p.number_of_green) * ( 
                (e_b * (1-p.h_b) * ((e_b_l * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_m + e_b_h))) + 
                (p.h_g)*e_g * ((e_g_l * p.alpha_g) + 0.5*(e_g_m + e_g_h) * (1-p.alpha_g)))
            )
            
            
            p_b_h_zero = poisson.pmf(0, b_h_lambda)
            p_b_m_zero = poisson.pmf(0, b_m_lambda)
            p_b_l_zero = poisson.pmf(0, b_l_lambda)
            
            p_g_h_zero = poisson.pmf(0, g_h_lambda)
            p_g_m_zero = poisson.pmf(0, g_m_lambda)
            p_g_l_zero = poisson.pmf(0, g_l_lambda)

        
        prob_b = p.prob_b
        prob_b_h = p.prob_b_h
        prob_b_m = p.prob_b_m
        prob_b_l = prob_b * (1 - p.b_vh_freq - p.b_vm_freq)
        prob_g = p.prob_g
        prob_g_h = prob_g * (p.g_vh_freq)
        prob_g_m = prob_g * p.g_vm_freq
        prob_g_l = prob_g * (1 - p.g_vh_freq - p.g_vm_freq)
        
        l_h_s = p.w_min - 1
        r_h_s = p.w_min
        
     
        while abs(l_h_s - r_h_s) != 0:
            l_h_s = r_h_s
            m_hired = 1 if p.vm >= l_h_s else 0 
            r_h_s = (
                            (
                                (   
                                    (p_b_h_zero*prob_b_h + p_g_h_zero*prob_g_h)* p.vh +
                                    ((p_b_m_zero + (1-p_b_m_zero)*m_hired) * prob_b_m + (p_g_m_zero + (1-p_g_m_zero)*m_hired) * prob_g_m) * p.vm + 
                                    p.vl*(prob_b_l + prob_g_l) )
                            )
                        /
                        (
                                # Denominator
                            (
                                p_b_h_zero * prob_b_h + p_g_h_zero * prob_g_h + p_g_m_zero + 
                                (p_b_m_zero + (1-p_b_m_zero)*m_hired) * prob_b_m + (p_g_m_zero + (1-p_g_m_zero)*m_hired) * prob_g_m + 
                                prob_b_l + prob_g_l
                            )
                        )
                    )
        self.v_tilda = r_h_s
        self.m_hired = 1 if p.vm >= self.v_tilda else 0 
        threshold = max(self.v_tilda, self.w_min)
        # threshold = 0.1541919477612662
        self.threshold = threshold
        return (threshold)
    
    def hire_continuous(self) -> float:
        
        self.b_vl_freq = 1 - self.b_vm_freq -self.b_vh_freq
        self.g_vl_freq = 1 - self.g_vm_freq -self.g_vh_freq
        p = self
        e_b = p.e_b
        e_b_h = p.e_b_h
        e_b_m = p.e_b_m
        e_b_l = 1 - e_b_h - e_b_m
        e_g_h = p.e_g_h
        e_g_m = p.e_g_m
        e_g_l = 1 - e_g_h - e_g_m
        e_g = p.e_g

        assert self.threshold >= self.w_min, "Make sure to calculate threshold before hiring"
        assert e_b <= 1, "Something wrong with the logic"
        
        # assert e_b_h + e_b_m + e_b_l == 1, f"{e_b_h, e_b_m, e_b_l}"
        # assert e_g_h + e_g_m + e_g_l == 1
        if self.ref_distribution == "Poisson":
            b_h_lambda = 1/(p.b_vh_freq * p.number_of_blue) * ( 
                (e_b * p.h_b * ((e_b_h * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_m))) + 
                (1-p.h_g)*e_g * ((e_g_h * p.alpha_g) + 0.5*(e_g_l + e_g_m)*(1-p.alpha_g)))
            )
            
            b_m_lambda =  1/(p.b_vm_freq * p.number_of_blue) * ( 
                (e_b * p.h_b * ((e_b_m * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_h))) + 
                (1-p.h_g)*e_g * ((e_g_m * p.alpha_g) + 0.5*(e_g_l + e_g_h) * (1-p.alpha_g)))
            )
            
            b_l_lambda =  1/(p.b_vl_freq * p.number_of_blue) * ( 
                (e_b * p.h_b * ((e_b_l * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_m + e_b_h))) + 
                (1-p.h_g)*e_g * ((e_g_l * p.alpha_g) + 0.5*(e_g_m + e_g_h) * (1-p.alpha_g)))
            )            
            
            g_h_lambda = 1/(p.g_vh_freq * p.number_of_green) * ( 
                (e_b * (1-p.h_b) * ((e_b_h * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_m))) + 
                (p.h_g)*e_g * ((e_g_h * p.alpha_g) + 0.5*(e_g_l + e_g_m) * (1-p.alpha_g)))
            )
            
            g_m_lambda =  1/(p.g_vm_freq * p.number_of_green) * ( 
                (e_b * (1-p.h_b) * ((e_b_m * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_l + e_b_h))) + 
                (p.h_g)*e_g * ((e_g_m * p.alpha_g) + 0.5*(e_g_l + e_g_h) * (1-p.alpha_g)))
            )
            
            g_l_lambda =  1/(p.g_vl_freq * p.number_of_green) * ( 
                (e_b * (1-p.h_b) * ((e_b_l * p.alpha_b) + (1-p.alpha_b)*(0.5 * (e_b_m + e_b_h))) + 
                (p.h_g)*e_g * ((e_g_l * p.alpha_g) + 0.5*(e_g_m + e_g_h) * (1-p.alpha_g)))
            )
            
            
            p_b_h_zero = poisson.pmf(0, b_h_lambda)
            p_b_m_zero = poisson.pmf(0, b_m_lambda)
            p_b_l_zero = poisson.pmf(0, b_l_lambda)
            
            p_g_h_zero = poisson.pmf(0, g_h_lambda)
            p_g_m_zero = poisson.pmf(0, g_m_lambda)
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
                (
                    ((1 - p_b_h_zero)*p.b_vh_freq + (1-p_b_m_zero)*p.b_vm_freq* self.m_hired) * p.number_of_blue + 
                    ((1 - p_g_h_zero)*p.g_vh_freq + (1-p_g_m_zero)*p.g_vm_freq* self.m_hired) * p.number_of_green
                     
                )
            )
        /
            (
            (1-(1-p_b_h_zero)*p.b_vh_freq - (1-p_b_m_zero)*p.b_vm_freq * self.m_hired)*p.number_of_blue +
            (1-(1-p_g_h_zero)*p.g_vh_freq - (1-p_g_m_zero)*p.g_vm_freq* self.m_hired)*p.number_of_green
            )
        )

        a =(
                    ((1 - p_b_h_zero)*p.b_vh_freq + (1-p_b_m_zero)*p.b_vm_freq* self.m_hired) * p.number_of_blue + 
                    ((1 - p_g_h_zero)*p.g_vh_freq + (1-p_g_m_zero)*p.g_vm_freq* self.m_hired) * p.number_of_green
                     
                )

                
        assert b_p_h_pool >= 0, f"whoopsy doopsy {b_p_h_pool, self.m_hired, a}"
        
        if b_p_h_pool > 1:
            print('uh oh probability greater than one check logic')
        
        b_p_h_pool = b_p_h_pool if self.v_tilda >= p.w_min else 0
        
        # print(f'{b_p_h_pool} is bph pool')
                     
        
        # g_p_h_pool = b_p_h_pool
        
        # prob hired given blue received a referral
        # Prob v > threshold * 1 + Prob v< threshold and hired from pool
        b_p_h_r = p.prob_b_h
        g_p_h_r = p.prob_g_h
        
        
        # This is an intersection of sets, we want probability given, thus converted in next step
        non_norm_ebh_next = (p_b_h_zero* b_p_h_pool + (1-p_b_h_zero)) * p.number_of_blue * p.b_vh_freq
        non_norm_egh_next = (p_g_h_zero * b_p_h_pool + (1-p_g_h_zero)) * p.number_of_green * p.g_vh_freq

        non_norm_ebm_next = (p_b_m_zero* b_p_h_pool + (1-p_b_m_zero)* (1 - (1- b_p_h_pool) *(1-p.m_hired))) * p.number_of_blue * p.b_vm_freq
        non_norm_egm_next = (p_g_m_zero* b_p_h_pool + (1-p_g_m_zero)* (1 - (1- b_p_h_pool) *(1-p.m_hired))) * p.number_of_green * p.g_vm_freq
        
        non_norm_ebl_next = b_p_h_pool * p.number_of_blue * p.b_vl_freq
        

        theta_b_h, theta_g_h = p.b_vh_freq*(1-p_b_h_zero), p.g_vh_freq*(1-p_g_h_zero)
        theta_b_m, theta_g_m = p.b_vm_freq*(1- p_b_m_zero), p.g_vm_freq* (1- p_g_m_zero)
        
        e_b_next = ((1 - theta_b_h - (p.m_hired)* theta_b_m) * b_p_h_pool + theta_b_h + theta_b_m * p.m_hired) * p.number_of_blue
        e_g_next = ((1 - theta_g_h - (p.m_hired)* theta_g_m) * b_p_h_pool + theta_g_h + theta_g_m * p.m_hired) * p.number_of_green

        # print((theta_b-theta_g)*(1-b_p_h_not_r))
        
        ebh_next = non_norm_ebh_next/e_b_next
        egh_next = non_norm_egh_next/e_g_next

        ebm_next = non_norm_ebm_next/e_b_next
        egm_next = non_norm_egm_next/e_g_next
                
        return {'e_b' : e_b_next, 'e_b_h': ebh_next, 'e_b_m': ebm_next, 'e_g_m': egm_next, 'e_g_h': egh_next, 'e_g': e_g_next}

def run_periods(periods = 15, e_b = 0.8, e_g= 0.2, e_b_h = 0.5, e_g_h = 0.5, e_b_m = 0.2, e_g_m = 0.2,
                 n= 2.0, alpha_b= 1, alpha_g= 1, h_b= 1, h_g= 1, verbose = True ):
        
    for period in range(periods):
        p = Parameters(e_b = e_b, e_g = e_g, e_b_h = e_b_h, e_g_h = e_g_h, e_b_m = e_b_m, e_g_m = e_g_m,
                        number_of_blue= n, number_of_green=n, alpha_b= alpha_b, alpha_g= alpha_g, 
                        h_b= h_b, h_g= h_g)
        p.calculate_threshold()

        if period == 0 and verbose:
            print(f'The parameters for p are {p}')
       
        # print(f'the skill threshold for this period is {p.threshold} ')
        # if e_b <= 0.5:
            # print(f'It took {period} generations for the male employment rate to equal the female employment rate')
            # break
        
        future_emp_dict = p.hire_continuous()
        e_b, e_g, e_b_h, e_g_h = future_emp_dict['e_b'], future_emp_dict['e_g'], future_emp_dict['e_b_h'], future_emp_dict['e_g_h']
        e_b_m, e_g_m = future_emp_dict['e_b_m'], future_emp_dict['e_g_m']

        print (e_b, e_g, e_b_h, e_g_h, e_b_m, e_g_m, p.m_hired)
        
# vh_freq doesnt work, watch out
def run_period(e_b, e_g, e_b_h: float = 0.5, e_g_h: float = 0.5, e_b_m: float = 0.2, e_g_m: float = 0.2,
                n: float = 2.0, alpha_b: float = 0.8, alpha_g: float = 0.8, 
                      h_b: float = 0.8, h_g: float = 0.8):
    
    p = Parameters(e_b = e_b, e_g = e_g, e_b_h = e_b_h, e_g_h = e_g_h, e_b_m = e_b_m, e_g_m = e_g_m,
                        number_of_blue= n, number_of_green=n, alpha_b= alpha_b, alpha_g= alpha_g, 
                        h_b= h_b, h_g= h_g)
    
    
    p.calculate_threshold()

    future_emp_dict = p.hire_continuous()
    e_b, e_g, e_b_h, e_g_h = future_emp_dict['e_b'], future_emp_dict['e_g'], future_emp_dict['e_b_h'], future_emp_dict['e_g_h']
    e_b_m, e_g_m = future_emp_dict['e_b_m'], future_emp_dict['e_g_m']
    return (e_b, e_g, e_b_h, e_g_h, e_b_m, e_g_m)

    # vh_freq doesn't work cuz dataclass, will need to fix
def find_steady_state(e_b_0: float, n: float, alpha_b: float, alpha_g: float, h_b: float,  h_g: float,
                       e_g_0: float = 0.5, e_b_h_0: float = 0.5, e_g_h_0: float = 0.5,
                       e_b_m_0: float = 0.2, e_g_m_0 : float = 0.2,  vh_freq = 0.4,
                       max_iterations: int = 1000, return_iterations: bool = False):
    iteration = 0
    e_b = e_b_0
    e_g = e_g_0
    e_b_h = e_b_h_0
    e_g_h = e_g_h_0
    e_b_m = e_b_m_0
    e_g_m = e_g_m_0



    e_b_new = 0
    e_g_new = 0
    ebh_new = 0
    egh_new = 0
    ebm_new = 0
    egm_new = 0

    if return_iterations:

        while (abs(e_b - e_b_new) != 0.0 or abs(e_b_h - ebh_new) != 0.0 or abs(e_g_h - egh_new) != 0.0) and iteration < max_iterations:
            iteration +=1
            e_b = e_b_new if e_b_new else e_b
            e_g = e_g_new if e_g_new else e_g
            e_b_h = ebh_new if ebh_new else e_b_h
            e_g_h = egh_new if egh_new else e_g_h
            e_b_m = ebm_new if ebm_new else e_b_m
            e_g_m = egm_new if egm_new else e_g_m
            e_b_new, e_g_new, ebh_new, egh_new, ebm_new, egm_new = run_period(e_b = e_b, e_g = e_g, e_b_h = e_b_h, e_g_h = e_g_h, 
                                e_b_m = e_b_m, e_g_m = e_g_m, n = n, alpha_b=alpha_b, alpha_g=alpha_g, h_b = h_b, h_g = h_g)

    else:
        
        while (e_b != e_b_new or e_b_h != ebh_new or e_g_h != egh_new) and iteration < max_iterations:
            iteration +=1
            e_b = e_b_new if e_b_new else e_b
            e_g = e_g_new if e_g_new else e_g            
            e_b_h = ebh_new if ebh_new else e_b_h
            e_g_h = egh_new if egh_new else e_g_h
            e_b_m = ebm_new if ebm_new else e_b_m
            e_g_m = egm_new if egm_new else e_g_m
            e_b_new, e_g_new, ebh_new, egh_new, ebm_new, egm_new = run_period(e_b = e_b, e_g = e_g, e_b_h = e_b_h, e_g_h = e_g_h, 
                                e_b_m = e_b_m, e_g_m = e_g_m, n = n, alpha_b=alpha_b, alpha_g=alpha_g, h_b = h_b, h_g = h_g)
                
    if iteration == max_iterations:
        print('max iteration reached')
        if return_iterations:
            return iteration
        else:
            return (e_b_new, e_g_new, ebh_new, egh_new, ebm_new, egh_new)
    
    else:
        print(f'reached in iteration # {iteration}')
        if return_iterations:
            return iteration
        else:
            p = Parameters(e_b = e_b, e_g=e_g, e_b_h = e_b_h, e_g_h = e_g_h, e_b_m = e_b_m, e_g_m = e_g_m,
            number_of_blue= n, number_of_green=n, alpha_b= alpha_b, alpha_g= alpha_g, 
                      h_b= h_b, h_g= h_g )
            # print('lambda bh')
            # print(p.vh_freq, n, e_b_new, ebh_new, alpha_b, alpha_g, h_b, h_g)
            # lambda_bh_n = (1/(p.vh_freq * n) * ( (e_b_new * h_b * ((ebh_new * alpha_b) + (1-alpha_b)*(1-ebh_new))) + (1-h_g)*(1-e_b_new) * ((egh_new * alpha_g) + (1-egh_new) * (1-alpha_g))))
            # print(lambda_bh_n)

            # print('lambda bl')
            # print(1/((1-p.vh_freq) * n) * ((e_b_new * h_b * (((1-ebh_new) * alpha_b) + (1-alpha_b)*(ebh_new))) + (1-h_g)*(1-e_b_new) * (egh_new * (1-alpha_g) + (1-egh_new) * (alpha_g))))
            # print('lambda gh')
            # print(1/((p.vh_freq) * n) * (((1-e_b_new) * h_g * ((egh_new * alpha_b) + (1-alpha_b)*(egh_new))) + (1-h_b)*(e_b_new) * (ebh_new * alpha_b + (1-ebh_new) * (1-alpha_g))))
            return (e_b_new, e_g_new, ebh_new, egh_new, ebm_new, egm_new)

# print(find_steady_state(e_b_0 = 0.5, e_g_0=0.5, e_b_h_0=0.2, e_g_h_0 = 0.2, e_b_m_0=0.1, e_g_m_0=0.1,
    #  n=2, alpha_b = 1.0, alpha_g = 1.0, h_b = 1.0, h_g = 1.0, return_iterations=False))
e_b = 0.8
e_g = 1 - e_b
run_periods(periods = 6, e_b = e_b, e_g =e_g, e_b_h = 2/3, e_g_h = 6/9, e_b_m = 3/9, e_g_m = 3/9, n=1.0,
            alpha_b = 1, alpha_g = 1, h_b = 1, h_g = 1, verbose = False)

# run_periods()

# plot_e_b()

