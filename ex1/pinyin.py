# %%
import pypinyin
import re
import numpy as np
import sys

# %%
# 读入文件，经过处理之后写入test文件
with open("toutiao_cat_data.txt", "r", encoding="utf-8") as r:
    with open("test.txt", "w", encoding="utf-8") as w:
        print("数据处理中：")
        data = r.readlines()
        length_d = len(data)  # 代表总共需要的运行次数
        cnt = 0  # 记录目前运行次数
        last_cnt = cnt  # 根据last和cnt的间隔输出，减少输出占用的运行时间
        for line in data:  # 处理每一行
            s = []  # 记录文字
            s.append("".join(re.findall(r'([\u4e00-\u9fa5]+)', line)))  # 利用re库去除掉所有的非中文元素
            sentence = ("".join(s))  # 连接成字符串
            # print(sentence)
            pinyin_list = pypinyin.lazy_pinyin(sentence)  # 利用pypinyin标注拼音，对应形成拼音列表
            pinyin = " ".join(pinyin_list)                # 将列表连接成字符串
            # print(pinyin)
            w.write(pinyin + "\n")  # 写入到文件中保存
            w.write(sentence + "\n")
            cnt += 1
            if cnt - last_cnt >= 10000 or cnt > length_d - 100:  # 运行状态，每隔一段时间才输出，减少输出占用的运行时间
                sys.stderr.write('\rEpoch: %d/%d' % (cnt, length_d))  # 控制台动态输出err为红字，out为白字。
                sys.stderr.flush()
                last_cnt = cnt

# %%
# 计算HMM模型中的参数
with open(r"test.txt", "r", encoding="utf-8") as test:
    data = test.read()
    data = data.split('\n')  # 根据换行分割字符串

i = 0  # 记录运行次数
last_i = i  # 根据间隔输出运行状态
# 转移概率
A = {}
# 发射概率
B = {}
# 初始概率分布
pi = {}
# 因为最后有空行，所以真正训练次数是len-1
train_time = len(data) - 1
print("\n参数训练中：")
while i != train_time:
    # 把拼音分割成列表，使拼音和汉字对齐
    py = data[i].split()    # pinyin是由字符串组成的列表
    sentence = data[i + 1]  # 句子是字符串，其中每个字符是拼音所对应的汉字
    # print(py,sentence)
    # 统计汉字对应拼音的出现频率
    for j in range(len(sentence)):
        if sentence[j] not in B:
            B[sentence[j]] = {}
        # 初始化
        if py[j] not in B[sentence[j]]:
            B[sentence[j]][py[j]] = 0
        B[sentence[j]][py[j]] += 1

    # 统计每句句首汉字的出现次数
    if sentence == "":  # 有可能出现空行的情况，这样sentence[0]会报错
        i += 2
        continue
    # print(sentence[0])
    if sentence[0] not in pi:
        pi[sentence[0]] = 0
    pi[sentence[0]] += 1
    # 对于相邻的两个字a和b，统计b在a后出现的次数
    for j in range(1, len(sentence)):
        if sentence[j - 1] not in A:
            A[sentence[j - 1]] = {}
        if sentence[j] not in A[sentence[j - 1]]:
            A[sentence[j - 1]][sentence[j]] = 0
        A[sentence[j - 1]][sentence[j]] += 1
    i = i + 2

    if i - last_i >= 20000 or i > train_time - 100:  # 每隔一段时间输出，减少输出占用的运行时间
        sys.stderr.write('\rEpoch: %d/%d' % (i, train_time))  # 控制台动态输出err为红字，out为白字。
        sys.stderr.flush()
        last_i = i

# 将字典中的次数换算为概率
for i in A:
    temp = sum(A[i].values())
    for j in A[i]:
        A[i][j] = A[i][j] / temp

for i in B:
    temp = sum(B[i].values())
    for j in B[i]:
        B[i][j] = B[i][j] / temp

for i in pi:
    temp = sum(pi.values())
    pi[i] = pi[i] / temp


# %%
# A,B,Π为模型参数，O为给定的观察序列
def viterbi(pi, A, B, O) -> str:
    T = len(O)  # 隐状态长度
    # 定义δ和φ
    delta = {}
    phi = {}
    delta_t0 = {}
    phi_t0 = {}
    # 初始化T=0时的delta和phi
    for i in pi:
        if i not in B or O[0] not in B[i]:
            B[i][O[0]] = 0
            continue
        delta_t0[i] = pi[i] * B[i][O[0]]
        phi_t0[i] = ""
    delta[0] = delta_t0
    phi[0] = phi_t0
    # 动态规划求解
    for t in range(1, T):
        temp_delta = {}
        temp_phi = {}
        for i in delta[t - 1]:
            if i not in A:
                continue
            for j in A[i]:
                # 判断不在的情况
                if j not in B or O[t] not in B[j]:
                    B[j][O[t]] = 0
                if j not in temp_delta:
                    temp_delta[j] = 0
                cal_now = delta[t - 1][i] * A[i][j] * B[j][O[t]]  # 当前概率
                if cal_now > temp_delta[j]:  # 找最大值
                    temp_delta[j] = cal_now  # 更新
                    temp_phi[j] = i  # 记录路径
        # 删除字典中概率为0的项避免冗余计算
        dic0 = []
        for hz in temp_delta:
            dic0.append(hz)
        for k in dic0:
            if temp_delta[k] == 0:
                del temp_delta[k]

        delta[t] = temp_delta
        phi[t] = temp_phi
    # 找出求得的状态序列中的最大概率（值）对应的键
    for key, value in delta[T-1].items():
        if (value == max(delta[T-1].values())):
            global s
            s = key
            p = value

    state = []
    state.append(s)  # 字典中拥有最大值的键
    # 回溯得到最优状态
    for t in range(T - 2, -1, -1):
        state.append(phi[t + 1][state[-1]])
    state = ''.join(reversed(state))
    return state


# %%
# 读入文件，经过处理之后写入test文件
with open(r"./测试集.txt", 'r') as r:  # 这个文件是gbk的
    test = r.read()
test = test.split('\n')  # 文本分割
i = 0
rate = []
print("\n开始测试：")
while i != len(test):
    right = 0
    pinyin = test[i].lower().split()  # 测试集中有首字母大写的情况
    print(pinyin)
    pre = viterbi(pi, A, B, pinyin)  # 预测的到的隐状态序列
    ans = test[i + 1]
    # 计算该句的正确率
    for j in range(len(pre)):
        if pre[j] == ans[j]:
            right += 1
    rate.append(right / len(pre))
    print(pre + " " + str(rate[-1] * 100.0) + "%")
    i = i + 2

print("平均正确率" + str(np.mean(rate) * 100.0) + "%")
