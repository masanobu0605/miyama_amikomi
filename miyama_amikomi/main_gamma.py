import pandas as pd
from scipy import optimize
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

today = dt.date.today()
today_str = today.strftime("%Y-%m-%d")

# 最大荷重，最小荷重，初期確度，物性値，寸法値の読み込み
gamma_init = 45
gamma_max = 45 * (np.pi/180) #実際に解析に用いるgammaの最大値
gamma_min = 24 * (np.pi/180) #実際に解析に用いるgammaの最小値
E = 3000
v = 0.3
m = 0.1
a = 0.215
h = 0.215
hh = h/2
x0 = 0.3728
I = np.pi/4 * (a/2) * (h/2)**3 #OK!

gamma_count = 100
gammalist = np.linspace(gamma_max,gamma_min,gamma_count)
dgamma = abs(gammalist[1] - gammalist[2])
F_guess = 10

# 関数の準備
def loop1(gamma):
    def xb(gamma):
        # ok
        aaa = (a/2 * hh) / (x0 * h/2) * np.exp(abs(np.pi/2 - 2 * gamma)) * a / (2 * np.tan(gamma))
        return aaa
    
    def T(F,gamma):
        # ok
        T = F * np.cos(gamma)
        return T

    def beta(F,gamma) :
        # ok
        return np.sqrt(T(F,gamma) / (E*I) )

    def qv(F,gamma):
        # ok
        LLL = np.exp(beta(F,gamma) * x0) + np.exp(-beta(F,gamma) * x0)
        MMM = np.exp(beta(F,gamma) * (x0 - xb(gamma))) + np.exp(beta(F,gamma) * (-x0 + xb(gamma)))
        qvqv = (T(F,gamma)**2 * h * LLL) / ( xb(gamma) * (2*x0 - xb(gamma))*LLL*T(F,gamma)  +  2*E*I *(MMM-LLL))
        return qvqv

    def MF(F,gamma):
        # ok
        aaa = m * qv(F,gamma) * xb(gamma)**2 / (np.cos(gamma))
        return aaa

    def MA(F,gamma):
        # ok
        return qv(F,gamma)*xb(gamma)*(2*x0-xb(gamma))/2 - T(F,gamma) * h/2
    
    
    #! MD関係
    def D1(F,gamma):
        # ok
        return (MA(F,gamma) - E*I*qv(F,gamma)/T(F,gamma))/(2*T(F,gamma))

    def D2(F,gamma):
        # ok
        return (MA(F,gamma) - E*I*qv(F,gamma)/T(F,gamma))/(2*T(F,gamma))

    def D3(F,gamma):
        # ok
        aaa = MA(F,gamma)/(2*T(F,gamma)) + E*I*qv(F,gamma)*(np.exp(-beta(F,gamma) * xb(gamma)) - 1) / (2 * T(F,gamma)**2)
        return aaa

    def D4(F,gamma):
        # ok
        aaa = MA(F,gamma)/(2*T(F,gamma)) + E*I*qv(F,gamma)*(np.exp(beta(F,gamma) * xb(gamma)) - 1) / (2 * T(F,gamma)**2)
        return aaa

    def EDTl(F,gamma):
        # ok
        aaa = qv(F,gamma)**2 * xb(gamma)**3 / (3*T(F,gamma)**2) 
        bbb = 0.5 * D2(F,gamma)**2 * (1 - np.exp(-2*beta(F,gamma)*xb(gamma)))*beta(F,gamma) 
        ccc = 0.5 * D1(F,gamma)**2 * (-1 + np.exp(2*beta(F,gamma)*xb(gamma)))*beta(F,gamma) 
        ddd = -2 * D1(F,gamma) * D2(F,gamma) *xb(gamma) * beta(F,gamma)**2 
        fff = (2*D1(F,gamma)*qv(F,gamma)*(1 + np.e**(beta(F,gamma)*xb(gamma)) * (-1 + beta(F,gamma)*xb(gamma)))) / (T(F,gamma) * beta(F,gamma)) 
        ggg = -(2*D2(F,gamma)*np.e**(-beta(F,gamma)*xb(gamma))*qv(F,gamma)*(-1 + np.e**(beta(F,gamma)*xb(gamma)) - beta(F,gamma)*xb(gamma)))/ (T(F,gamma) * beta(F,gamma)) 
        return aaa + bbb + ccc + ddd + fff + ggg 

    def EDTr(F,gamma):
        # ok
        aaa = 4 *D3(F,gamma) * (np.e**(beta(F,gamma)*x0) - np.e**(beta(F,gamma)*xb(gamma))) * qv(F,gamma) * T(F,gamma) * xb(gamma) 
        bbb = 2 * qv(F,gamma)**2 * (x0 - xb(gamma)) * xb(gamma)**2 
        ccc = D4(F,gamma)**2 * (-np.e**(-2*beta(F,gamma)*x0) + np.e**(-2*beta(F,gamma)*xb(gamma))) * T(F,gamma)**2 * beta(F,gamma) 
        ddd = D3(F,gamma)**2 * (np.e**(2*beta(F,gamma)*x0) - np.e**(2*beta(F,gamma)*xb(gamma))) * T(F,gamma)**2 * beta(F,gamma) 
        
        eee = np.e ** (-beta(F,gamma) * x0) * qv(F,gamma) *xb(gamma)
        fff = -np.e ** (-beta(F,gamma) * xb(gamma)) * qv(F,gamma) * xb(gamma)
        ggg = D3(F,gamma) * T(F,gamma) *(-x0 + xb(gamma)) * beta(F,gamma)**2
        hhh =4 * D4(F,gamma) * T(F,gamma) * (eee + fff + ggg)
        
        jjj  = (aaa + bbb + ccc + ddd + hhh)/(2*T(F,gamma)**2)
        return jjj

    def EDMl(F,gamma):
        # ok
        aaa = qv(F,gamma)**2 * xb(gamma)/T(F,gamma)**2 
        bbb = 2 * np.e**(-beta(F,gamma)*xb(gamma)) * (-1 + np.e**(beta(F,gamma)*xb(gamma))) * (D2(F,gamma) + D1(F,gamma)*np.e**(beta(F,gamma)*xb(gamma))) * qv(F,gamma) * beta(F,gamma)/T(F,gamma)
        ccc = 0.5 * beta(F,gamma)**3 * (D2(F,gamma)**2 * (1-np.e**(-2 * beta(F,gamma)*xb(gamma))) + D1(F,gamma)**2 * (-1 + np.e**(2 * beta(F,gamma)*xb(gamma))) + 4*D1(F,gamma)*D2(F,gamma)*xb(gamma)*beta(F,gamma)) 
        ddd = aaa + bbb + ccc
        return ddd

    def EDMr(F,gamma):
        # ok
        aaa = D4(F,gamma)**2 * (-np.e**(-2*beta(F,gamma)*x0) + np.e**(-2 * beta(F,gamma) * xb(gamma))) 
        bbb = D3(F,gamma)**2 * (np.e**(2*beta(F,gamma)*x0) - np.e**(2 * beta(F,gamma) * xb(gamma))) 
        ccc = 4 * D3(F,gamma) * D4(F,gamma) * (x0 - xb(gamma)) * beta(F,gamma) 
        ddd = (aaa + bbb + ccc) * (0.5 * beta(F,gamma)**3)
        return ddd

    def MD(F,gamma):
        # beam_deformation
        # ok
        d_EDT = 2*F*((EDTl(F,gamma - dgamma) - EDTl(F,gamma))/(dgamma) + (EDTr(F,gamma- dgamma) - EDTr(F,gamma))/(dgamma))
        d_EDM = 2*E*I*( (EDMl(F,gamma- dgamma) - EDMl(F,gamma))/(dgamma) + (EDMr(F,gamma- dgamma) - EDMr(F,gamma))/(dgamma))
        aaa = 0.5 * (d_EDM + d_EDT)
        return aaa
    # !MD関係↑
    
    
    def R(gamma):
        # ok
        aaa = a/(2 * np.sin(2 * gamma))
        RRR = 2*aaa**2 / h
        return RRR

    def alpha(gamma):
        # ok
        return abs(np.arcsin(xb(gamma)/R(gamma)))

    def MR(F,gamma):
        # ok
        aaa = 2 * qv(F,gamma) * R(gamma)**2 * np.sin(gamma)
        bbb = 0.5 * np.log((1 + np.sin(alpha(gamma)))/(1 - np.sin(alpha(gamma)))) - np.sin(alpha(gamma))
        ccc = aaa * bbb
        return ccc

    # ↓計算を行う関数
    def define_F(F):
        # ok
        WWF = (MF(F,gamma) + MD(F,gamma) + MR(F,gamma))/(2*x0 * np.sin(gamma))
        return WWF-F
    result_F = optimize.fsolve(define_F,F_guess)
    result_F = result_F
    return result_F,MF(result_F,gamma)/(2*x0 * np.sin(gamma)),MD(result_F,gamma)/(2*x0 * np.sin(gamma)),MR(result_F,gamma)/(2*x0 * np.sin(gamma)),T(result_F,gamma)

# 解析実行+データ整理
F = np.zeros(gamma_count)
F_MF = np.zeros(gamma_count)
F_MD= np.zeros(gamma_count)
F_MR = np.zeros(gamma_count)
T = np.zeros(gamma_count)
MF_and_MD = np.zeros(gamma_count)
d_gamma = np.zeros(gamma_count)
for i in range(gamma_count):
    F[i],F_MF[i],F_MD[i],F_MR[i],T[i] = loop1(gammalist[i])
    F[i] = round(F[i],5)
    T[i] = round(T[i],5)
    F_MF[i] = round(F_MF[i],5)
    F_MD[i] = round(F_MD[i],5)
    F_MR[i] = round(F_MR[i],5)
    MF_and_MD[i] = F_MF[i]+F_MD[i]
    gammalist[i] = round(gammalist[i]*180/np.pi,5)
    d_gamma[i] = round(gamma_init - gammalist[i],5)




# フィルタリング
# for i in range(gamma_count):
#     if F[i] == F_guess:
#         F[i] = 0.5*(F[i-1] + F[i+1])
#         MF[i] = 0.5*(MF[i-1] + MF[i+1])
#         MD[i] = 0.5*(MD[i-1] + MD[i+1])
#         MR[i] = 0.5*(MR[i-1] + MR[i+1])
#         MF_and_MD[i] = 0.5*(MF_and_MD[i-1] + MF_and_MD[i+1])

# データシート入力
datasheet = np.ndarray((gamma_count,8))
datasheet[:,0] = gammalist
datasheet[:,1] = d_gamma
datasheet[:,2] = F
datasheet[:,3] = T
datasheet[:,4] = F_MF
datasheet[:,5] = F_MD
datasheet[:,6] = MF_and_MD
datasheet[:,7] = F_MR
datasheet_pd = pd.DataFrame(datasheet,columns=["gammalist [deg]","d_gamma [deg]","F [N]"," T[N]","F_MF [N]","F_MD [N]","F_MF+F_MD [N]","F_MR [N]"],)
datasheet_pd.to_csv('amikomi_' +today_str + '.csv')
# データ描画
def fig():
    
    fig, ax2 = plt.subplots()
    x = d_gamma
    y1 = F
    y2 = F_MF
    y3 = F_MD
    y4 = F_MR
    ax2.plot(x, y1,linestyle = 'solid' ,color = 'black',label = "F")
    ax2.plot(x, y2,linestyle = 'solid' ,color = 'red',label = "F_MF")
    ax2.plot(x, y3,linestyle = 'solid' ,color = 'blue',label = "F_MD")
    ax2.plot(x, y4,linestyle = 'solid' ,color = 'green',label = "F_MR")
    ax2.set_title("三山さん修論図",fontname = 'MS Gothic')
    ax2.set_xlabel("d_gamma[deg]",fontname = 'MS Gothic')
    ax2.set_ylabel("F[N]")
    plt.legend()
    plt.savefig("F-d_gamma_" + today_str + ".png")
    # plt.show()
fig()