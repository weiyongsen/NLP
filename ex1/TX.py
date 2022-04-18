# encoding: utf-8
import pypinyin  # 给汉字标注拼音
import re
import codecs
import json


# 预处理数据，只保留汉字语句
def ProcessingData():
    print("Start processing data......")
    input_data = codecs.open(r"./toutiao_cat_data.txt", encoding="utf-8")
    output_data = codecs.open(r"./data.txt", 'w', 'utf-8')
    for line in input_data.readlines():
        # 按照“_!_”来切分句子
        line2 = re.split(r"\_!\_", line)
        if (line2[-1] == ""):
            line2 = line2[:-1]
        length = len(line2)
        sentences = []
        # 跳过每一行开始的前三个小句，循环下标从3开始
        for i in range(3, length):
            # 只保留句中的汉字
            sentences.append("".join(re.findall(r'([\u4e00-\u9fa5]+)', line2[i])))
        s = ' '.join(sentences)
        # 标注拼音
        py = pypinyin.lazy_pinyin(s)
        py = ' '.join(py)
        output_data.write(py)
        output_data.write("\n")
        output_data.write(s)
        output_data.write("\n")
    input_data.close()
    output_data.close()


# 计算HMM模型中的初始概率、转移概率和发射概率
def init():
    txt = codecs.open(r"./data.txt", 'r', 'utf-8')
    data = txt.read()
    txt.close()
    data = data.split('\n')
    i = 0
    # 发射概率
    pinyin = {}
    # 初始概率分布
    pi = {}
    # 转移概率
    A = {}
    while (i != len(data)):
        # 跳过空行
        if (data[i] == ""):
            i = i + 1
            continue
        py = data[i].strip().split()
        # 去除空格使句子与拼音下标对齐
        sentence = data[i + 1].strip().replace(' ', '')
        # 统计汉字对应拼音的出现频率
        for j in range(len(sentence)):
            if sentence[j] not in pinyin:
                pinyin[sentence[j]] = {}
            if py[j] not in pinyin[sentence[j]]:
                pinyin[sentence[j]][py[j]] = 0
            pinyin[sentence[j]][py[j]] += 1
        # 该行按空格分割成小句
        sentence = data[i + 1].strip().split()
        for s in sentence:
            # 统计每句句首汉字的出现次数
            if s[0] not in pi:
                pi[s[0]] = 0
            pi[s[0]] += 1
            # 对于相邻的两个字a和b，统计b在a后出现的次数
            for k in range(1, len(s)):
                if s[k - 1] not in A:
                    A[s[k - 1]] = {}
                if s[k] not in A[s[k - 1]]:
                    A[s[k - 1]][s[k]] = 0
                A[s[k - 1]][s[k]] += 1
        i = i + 2

    # 计算并保存转移概率
    print("Calculate and save A......")
    for i in A:
        temp = sum(A[i].values())
        for j in A[i]:
            A[i][j] = A[i][j] / temp
    json_data = json.dumps(A, indent=4, ensure_ascii=False)
    with open('A.json', 'w', encoding='utf-8') as f:
        f.write(json_data)

    # 计算并保存发射概率
    print("Calculate and save B......")
    for i in pinyin:
        temp = sum(pinyin[i].values())
        for j in pinyin[i]:
            pinyin[i][j] = pinyin[i][j] / temp
    json_data = json.dumps(pinyin, indent=4, ensure_ascii=False)
    with open('B.json', 'w', encoding='utf-8') as f:
        f.write(json_data)

    # 计算并保存初始概率分布
    print("Calculate and save pi......")
    temp = sum(pi.values())
    for j in pi:
        pi[j] = pi[j] / temp
    json_data = json.dumps(pi, indent=4, ensure_ascii=False)
    with open('pi.json', 'w', encoding='utf-8') as f:
        f.write(json_data)
    return pi,A,pinyin

# A为状态转移矩阵，B为发射矩阵，pi为初始概率矩阵，obs为给定的观察序列
def Viterbi(pi, A, B, obs):
    T = len(obs)
    delta = {}
    phi = {}
    delta0 = {}
    phi0 = {}
    # 初始化T=0时的delta和phi
    for i in pi:
        if i not in B or obs[0] not in B[i]:
            B[i][obs[0]] = 0
            continue
        delta0[i] = pi[i] * B[i][obs[0]]
        phi0[i] = ""
    delta[0] = delta0
    phi[0] = phi0
    # 动态规划求解
    for t in range(1, T):
        temp_delta = {}
        temp_phi = {}
        # print(delta[t-1])
        for i in delta[t - 1]:
            if i not in A:
                continue
            for j in A[i]:
                if j not in B or obs[t] not in B[j]:
                    B[j][obs[t]] = 0
                if j not in temp_delta:
                    temp_delta[j] = 0
                if delta[t - 1][i] * A[i][j] * B[j][obs[t]] > temp_delta[j]:
                    temp_delta[j] = delta[t - 1][i] * A[i][j] * B[j][obs[t]]
                    temp_phi[j] = i

        # 删除字典中概率为0的项以加速计算
        list = []
        for td in temp_delta:
            list.append(td)
        for l in list:
            if temp_delta[l] == 0:
                del temp_delta[l]

        delta[t] = temp_delta
        phi[t] = temp_phi
    # 对求得的状态序列概率从小到大排序
    deltaT_sorted = sorted(zip(delta[T - 1].values(), delta[T - 1].keys()))
    # p为最大概率，phi为其对应的状态
    p, phi[T] = deltaT_sorted[-1][0], deltaT_sorted[-1][1]
    # 回溯求解对应的状态序列
    path = []
    path.append(phi[T])
    for t in range(T - 2, -1, -1):
        path.append(phi[t + 1][path[-1]])
    path = ''.join(reversed(path))
    return path


if __name__ == '__main__':
    pi,A,B=init()
    py = input("请输入测试拼音：")
    print(py.split())
    print(Viterbi(pi, A, B, py.split()))


