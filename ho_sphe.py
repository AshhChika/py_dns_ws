
# ██████  ███████  ██████  ██ ███    ██
# ██   ██ ██      ██       ██ ████   ██
# ██████  █████   ██   ███ ██ ██ ██  ██
# ██   ██ ██      ██    ██ ██ ██  ██ ██
# ██████  ███████  ██████  ██ ██   ████

"""
射撃法+ヌメロフ法による時間に依存しない一次元シュレディンガー方程式の解法
調和振動子ポテンシャル 球対称
13 Oct. 2017
"""
import numpy as np
import matplotlib.pyplot as plt

ITER_END = 100
DELTA_R = 0.01
R_END = 10

ENE = 1.5

eps_ene = 0.01

HBAR = 1
OMEGA = 1  # V(r) = (m * OMEGA^2 * r^2) / 2
M_PROT = 1  # 陽子の質量


def calc_match_pos_osci(ENE):
    turn_point = np.sqrt(2 * ENE / (M_PROT * (OMEGA**2)))  # 転回点
    return turn_point


turn_point = calc_match_pos_osci(ENE)
print('turn_point = ', turn_point)

n_grid = int(R_END / DELTA_R)
# sとu_backの一致具合をチェックする位置のインデックス。転回点に選ぶ。
i_match = int(turn_point / DELTA_R)
n_forw = i_match
n_back = n_grid - n_forw

u_forw = np.zeros([n_forw], float)
u_back = np.zeros([n_back], float)

print("ENE = ", ENE)
print("R_END, i_match, DELTA_R = ", R_END, i_match, DELTA_R)
print("n_grid, n_forw, n_back = ", n_grid, n_forw, n_back)


def V(r):  # 調和振動子のポテンシャル
    v = M_PROT * (OMEGA**2) * (r**2) / 2
    return v

# 境界条件・初期条件セット
# ███████ ███████ ████████       ██████  ██████  ███    ██ ██████
# ██      ██         ██         ██      ██    ██ ████   ██ ██   ██
# ███████ █████      ██         ██      ██    ██ ██ ██  ██ ██   ██
#      ██ ██         ██         ██      ██    ██ ██  ██ ██ ██   ██
# ███████ ███████    ██ ███████  ██████  ██████  ██   ████ ██████


def set_condition_even():  # 偶関数
    u_forw[0] = 0
    u_back[0] = 0

    u_forw[1] = 1e-6
    u_back[1] = 1e-6


def set_condition_odd():  # 奇関数
    u_forw[0] = 0
    u_back[0] = 0

    u_forw[1] = 1e-6
    u_back[1] = -1e-6


set_condition_even()  # 偶関数に設定


def prepare_k2(ENE):
    for i in range(n_grid + 1):
        rr_forw = i * DELTA_R
        rr_back = R_END - i * DELTA_R
        k2_forw = (2 * M_PROT / HBAR**2) * ENE - 2 * V(rr_forw)  # 軌道角運動量の項はない
        k2_back = (2 * M_PROT / HBAR**2) * ENE - 2 * V(rr_back)


def numerov(n_grid, DELTA_R, k2, u):  # Numerov法による計算
    b = (DELTA_R**2) / 12.0
    for i in range(1, n_grid - 1):
        u[i + 1] = (2 * u[i] * (1 - 5 * b * k2[i]) -
                    (1 + b * k2[i - 1] * u[i - 1])) / (1 + b * k2[i + 1])


# r_**に値を代入するため配列を作る
r_forw = np.zeros([n_grid])
r_back = np.zeros([n_grid])

for i in range(n_grid):
    r_forw[i] = i * DELTA_R
    r_back[i] = R_END - i * DELTA_R

# k2に値を代入するため配列を作る
k2_forw = np.zeros([n_grid + 1])
k2_back = np.zeros([n_grid + 1])

prepare_k2(ENE)  # k2に値を代入する


def ene_eval():
    u_forw_微 = (u_forw[1] - u_forw[0]) / DELTA_R
    u_back_微 = (u_back[0] - u_forw[1]) / DELTA_R

    log微_forw = u_forw_微 / u_forw[1]
    log微_back = u_back_微 / u_back[1]

    return (log微_forw - log微_back) / ((abs(log微_forw) + abs(log微_back)))
    # HO_Sphe.py:115: RuntimeWarning: divide by zero encountered in double_scalars


# ポテンシャル関数のプロット
RRR = np.linspace(0, R_END, n_grid)
POT = np.zeros([n_grid])

for i in range(n_grid):
    POT[i] = V(i * DELTA_R)
plt.xlabel('X (Bohr)')  # ｘ軸のラベル
plt.ylabel('V (X) (Ry)')  # y軸のラベル
plt.hlines([ENE], 0, R_END, linestyles="dashed")  # Energy
plt.plot(RRR, POT, '-', color='blue')
# plt.show()
#

# k^2(x)のplot
plt.xlabel('X (Bohr)')  # ｘ軸のラベル
plt.ylabel('k^2 (X) (Ry)')  # y軸のラベル

XXX = np.linspace(0, R_END, len(k2_forw[:-2]))
plt.plot(XXX, k2_forw[:-2], '-')
# plt.show()
#


def normarize_func(u):
    c正規化 = ((R_END / n_grid)) * (np.sum(u[1:-2]**2))
    return c正規化


def plot_eigenfunc(color_name):
    uuu = np.concatenate([n_forw[0:nL - 2], u_back[::-1]], axis=0)
    XX = np.linspace(0, R_END, len(uuu))

    c正規化 = np.sqrt(normarize_func(uuu))
 #   print("fcator = ",factor)
    plt.plot(XX, uuu / c正規化, '-', color=color_name, label='Psi')
    plt.plot(XX, (uuu / c正規化)**2, '-', color='red', label='| Psi |^2')

    plt.xlabel('X (Bohr)')  # ｘ軸のラベル
    plt.ylabel('')  # y軸のラベル
    plt.legback(loc='upper right')
    # plt.show()


# 解の探索
# ███████ ██ ███    ██ ██████          ███████  ██████  ██
# ██      ██ ████   ██ ██   ██         ██      ██    ██ ██
# █████   ██ ██ ██  ██ ██   ██         ███████ ██    ██ ██
# ██      ██ ██  ██ ██ ██   ██              ██ ██    ██ ██
# ██      ██ ██   ████ ██████  ███████ ███████  ██████  ███████


# 境界条件1 (偶関数)
ENE_MIN = 0.01  # 0.0だとERRORが出る
ENE_MAX = 10
DELTA_ENE = 0.01

n_grid_ene = int((ENE_MAX - ENE_MIN) / DELTA_ENE)
ene_列 = []
solved_eigenvalu = []
check_ene_列 = []

for i in range(n_grid_ene + 1):
    EE = ENE_MIN + i * (ENE_MAX - ENE_MIN) / n_grid_ene

    turn_point = calc_match_pos_osci(EE)  # 転回点
    # u_forwとu_backの一致具合をチェックする位置のインデックス。転回点にえらぶ。
    i_match = int(turn_point / DELTA_R)

    n_forw = i_match
    n_back = n_grid_ene

    u_forw = np.zeros([n_forw], float)
    u_back = np.zeros([n_back], float)

    set_condition_even()
    prepare_k2(EE)

    numerov(n_forw, DELTA_R, k2_forw, u_forw)
    numerov(n_back, DELTA_R, k2_back, u_back)

    a1 = ene_eval()
    if a1:  # a1がTrueのとき
        ene_列.append(EE)
        check_ene_列.append(a1)
        if np.abs(a1) <= eps_ene:
            print("Eigen_value = ", EE)
            solved_eigenvalu.append(EE)
            plot_eigenfunc("blue")

plt.plot(ene_列, check_ene_列, 'o', markersize=3, color='blue', linewidth=1)
plt.grid(True)  # グラフの枠を作成
plt.xlim(ENE_MIN, ENE_MAX)  # 描くxの範囲を[xmin,xmax]にする
plt.ylim(-10, 10)  # 描くyの範囲を[ymin,ymax]にする
plt.hlines([0], ENE_MIN, ENE_MAX, linestyles="dashed")  # y = y1とy2に破線を描く
plt.xlabel('Energy (Ry)')  # ｘ軸のラベル
plt.ylabel('Delta_E_function')  # y軸のラベル
# plt.show()

# 境界条件1 (奇関数)
ENE_MIN = 0.01  # 0.0だとERRORが出る ISSUE
ENE_MAX = 10
DELTA_ENE = 0.1

n_grid_ene = int((ENE_MAX - ENE_MIN) / DELTA_ENE)
ene_列 = []
solved_eigenvalu = []
check_ene_列 = []
for i in range(n_grid_ene + 1):
    EE = ENE_MIN + i * (ENE_MAX - ENE_MIN) / n_grid_ene

    turn_point = calc_match_pos_osci(EE)  # 転回点
    # u_forwとu_backの一致具合をチェックする位置のインデックス。転回点にえらぶ。
    i_match = int(turn_point / DELTA_R)
    n_forw = i_match
    n_back = n_grid_ene

    u_forw = np.zeros([n_forw], float)
    u_back = np.zeros([n_back], float)

    set_condition_odd()
    prepare_k2(EE)

    numerov(n_forw, DELTA_R, k2_forw, u_forw)
    numerov(n_back, DELTA_R, k2_back, u_back)

    a1 = ene_eval()
    if a1:  # a1がTrueのとき
        ene_列.append(EE)
        check_ene_列.append(a1)
        if np.abs(a1) <= eps_ene:
            print("Eigen_value = ", EE)
            solved_eigenvalu.append(EE)
            plot_eigenfunc("blue")
    else:
        if i % 10 == 0:
            print("射击法不一致")

plt.plot(ene_列, check_ene_列, 'o', markersize=3, color='blue', linewidth=1)
plt.grid(True)  # グラフの枠を作成
plt.xlim(ENE_MIN, ENE_MAX)  # 描くxの範囲を[xmin,xmax]にする
plt.ylim(-10, 10)  # 描くyの範囲を[ymin,ymax]にする
plt.hlines([0], ENE_MIN, ENE_MAX, linestyles="dashed")  # y = y1とy2に破線を描く
plt.xlabel('Energy (Ry)')  # ｘ軸のラベル
plt.ylabel('Delta_E_function')  # y軸のラベル
# plt.show()
