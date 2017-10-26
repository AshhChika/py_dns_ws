# coding: UTF-8
"""
射撃法+ヌメロフ法による時間に依存しない一次元シュレディンガー方程式の解法
深い井戸型ポテンシャル
1 Sept. 2017
"""
import numpy as np
import matplotlib.pyplot as plt

delta_x = 0.05

xa = 1 # 井戸の境界。 x = ±xaに井戸の端がある。
eps_E = 0.005 # 収束条件
nn = 5
xL0, xR0  = - nn * xa, nn * xa
Nx =  int((xR0 - xL0)/delta_x)
delta_x = 1.0*(xR0 - xL0)/(Nx)
i_match = int((xa - xL0) / delta_x)  #uLとuRの一致具合をcheckする位置のindex。井戸の境界に選んでいる。

nL = i_match
nR = Nx - nL

print(xL0, xR0, i_match, delta_x)
print(nL, nR)

uL = np.zeros([nL],float)
uR = np.zeros([nR],float)

E = np.pi**2/4

print("E= ",E)
print("xL0,xR0, i_match, delta_x = ",xL0,xR0, i_match, delta_x)
print("Nx, nL,nR = ",Nx, nL,nR)


def V(x): # 井戸型ポテンシャルの設定
    if np.abs(x) >  xa :
        v = 1000.0
    else :
        v = 0

    return v

# 境界条件・初期条件セット
def set_condition():
    uL[0]  = 0
    uL[1] =1e-6

    uR[0] = 0
    uR[1] =1e-6
#
set_condition()


def setk2 (E): # for E<0
    for i in range(Nx+1):
        xxL = xL0 + i*delta_x
        xxR = xR0 -  i*delta_x
        k2L[i] = E-V(xxL)
        k2R[i] = E-V(xxR)


def Numerov (N,delta_x,k2,u):  # ヌメロフ法による発展
    b = (delta_x**2)/12.0

    for i in range(1,N-1):
        u[i+1] = (2*u[i]*(1-5*b*k2[i])-(1+b*k2[i-1])*u[i-1])/(1+b*k2[i+1])


xL = np.zeros([Nx])
xR = np.zeros([Nx])

for i in range (Nx):
    xL[i] = xL0 + i*delta_x
    xR[i] = xR0 - i*delta_x

k2L = np.zeros([Nx+1])
k2R = np.zeros([Nx+1])

setk2(E)



def E_eval():
    uLdash = (uL[-1]-uL[-2])/delta_x
    uRdash = (uR[-2]-uR[-1])/delta_x
    logderi_L=  uLdash/uL[-1]
    logderi_R=  uRdash/uR[-1]
    return (logderi_L- logderi_R)/(logderi_L+logderi_R)





# ポテンシャル関数のプロット
XXX= np.linspace(xL0,xR0, Nx)
POT = np.zeros([Nx])
for i in range(Nx):
    POT[i] = V(xL0 + i*delta_x)
plt.xlabel('X (Bohr)') # ｘ軸のラベル
plt.ylabel('V (X) (Ry)') # y軸のラベル
plt.hlines([E], xL0,xR0, linestyles = "dashed")  #Energy
plt.plot(XXX,POT,'-',color = 'blue')
plt.show()
#
#k^2(x)のプロット
XXX= np.linspace(xL0,xR0, Nx+1)
plt.plot(XXX, k2L,'-')
plt.show()
#

def normarize_func(u):
    factor = ((xR0-xL0)/Nx)*(np.sum(u[1:-2]**2))
    return factor
def plot_eigenfunc(color_name):
    uuu = np.concatenate([uL[0:nL-2],uR[::-1]],axis = 0)
    XX = np.linspace(xL0,xR0, len(uuu))

    factor = np.sqrt(normarize_func(uuu))

    plt.plot(XX,uuu/factor,'-',color = color_name,label = 'Psi')
    plt.plot(XX,(uuu/factor)**2,'-',color = 'red',label = '| Psi |^2')

    plt.xlabel('X (Bohr)') # ｘ軸のラベル
    plt.ylabel('') # y軸のラベル
    plt.legend(loc = 'upper right')
    plt.show()


# 解の探索

# 境界条件1 (偶関数)
EEmin = 0.1
EEmax = 20
delta_EE = 0.01

NE = int((EEmax-EEmin)/delta_EE)
Elis = []
Solved_Eigenvalu = []
check_Elis= []
for i in range(NE+1):
    EE = EEmin+i*(EEmax-EEmin)/NE


    set_condition_even()
    setk2(EE)

    Numerov (nL,delta_x,k2L,uL)
    Numerov (nR,delta_x,k2R,uR)

    a1= E_eval()

    if a1 :
        Elis.append(EE)
        check_Elis.append(a1)
        if np.abs(a1) <= eps_E :  #解を見つけた場合のプロット
            print("Eigen_value = ", EE)
            Solved_Eigenvalu.append(EE)
            plot_eigenfunc("blue")

plt.plot(Elis, check_Elis, 'o',markersize = 3, color = 'blue',linewidth = 1)
plt.grid(True) #グラフの枠を作成
plt.xlim(EEmin, EEmax) # 描くxの範囲を[xmin,xmax]にする
plt.ylim(-10, 10) # 描くyの範囲を[ymin,ymax]にする
plt.hlines([0], EEmin,EEmax, linestyles = "dashed")  # y = y1とy2に破線を描く
plt.xlabel('Energy (Ry)') # ｘ軸のラベル
plt.ylabel('Delta_E_function') # y軸のラベル
plt.show()

# 境界条件2 (奇関数)
EEmin = 0.1
EEmax = 20
delta_EE = 0.01

NE = int((EEmax-EEmin)/delta_EE)
Elis = []
Solved_Eigenvalu = []
check_Elis= []
for i in range(NE+1):
    EE = EEmin+i*(EEmax-EEmin)/NE

    nL = i_match
    nR = Nx-nL

    uL = np.zeros([nL],float)
    uR = np.zeros([nR],float)

    set_condition_odd()
    setk2(EE)

    Numerov (nL,delta_x,k2L,uL)
    Numerov (nR,delta_x,k2R,uR)

    a1= E_eval()
    #print ("a1 = ",a1)
    if a1 :  # a1がTrueのとき
        Elis.append(EE)
        check_Elis.append(a1)
        if np.abs(a1) <= eps_E :
            print("Eigen_value = ", EE)
            Solved_Eigenvalu.append(EE)
            plot_eigenfunc("blue")


plt.plot(Elis, check_Elis, 'o',markersize = 3, color = 'red',linewidth = 1)
plt.grid(True) #グラフの枠を作成
plt.xlim(EEmin, EEmax) # 描くxの範囲を[xmin,xmax]にする
plt.ylim(-10, 10) # 描くyの範囲を[ymin,ymax]にする
plt.hlines([0], EEmin,EEmax, linestyles = "dashed")  # y = y1とy2に破線を描く
plt.xlabel('Energy (Ry)') # ｘ軸のラベル
plt.ylabel('Delta_E_function') # y軸のラベル
plt.show()
