# %%
# 读入分词数据库
with open("中文分词词库整理/30万 中文分词词库.txt", "r") as r:
    file_read=r.readlines()
    data=[]
    for i in file_read:
        i=i.split("\t") # 查看文档格式后，使用\t分割
        data.append(i[1])
with open("中文分词词库整理/42537条伪原创词库.txt", "r") as r:
    file_read=r.readlines()
    cnt=0
    for i in file_read:
        i=i.strip().split("→")  # 查看文档格式后，使用→分割
        data.extend(i)          # 扩展添加上一个词库后的data
        cnt+=1
with open("data.txt","w") as w: # 记录下可用的分词data数据
    for i in data:
        w.write(i+"\n")

# %%
# FMM
def fmm(str):
    fenci=""    # 返回值
    cnt=7           # 初始间隔7
    length=len(str) # 句子长度
    begin=0         # 句中匹配位置初始位置
    end=begin+cnt   # 匹配位置末位置
    while begin<length:     # 跳出循环条件
        l=begin     # 在每次匹配中，起始点就是begin
        r=end if end<length else length # 末位置有可能超限，需处理
        if l==r:    # 两者相等，说明只剩一个字，不需要词库中匹配
            fenci=fenci+(str[l]+"/")
            begin+=1
            continue
        word=str[l:r]   # 取出句子中l~r-1位置的词，在词库中比对
        if word in data:    # 比对成功
            fenci+=(word+"/")   # 添加到分词中，并更新始末位置
            begin=end
            end=begin+cnt
        else:
            end-=1      # 比对失败，说明词块段要缩短
    return fenci


# %%
# BMM
def bmm(str):
    fenci=""
    cnt=7
    length=len(str)
    end=length          # BMM这里需要先初始化end
    begin=end-cnt
    while end>=0:
        l=begin if begin>=0 else 0  # 这里是初始位置可能超限，需处理
        r=end
        if l==r:    # l==r，说明词块段缩短到1，无需匹配
            fenci=(str[l]+"/")+fenci
            end-=1
            continue
        word=str[l:r]   # 从句子中取词
        if word in data:    # 匹配成功  
            fenci=(word+"/")+fenci  # 更新
            end=begin
            begin=end-cnt
        else:           # 匹配失败，初始位置加1
            begin+=1
    return fenci


# %%
# 双向算法
def BiMM(str):
    str_f=fmm(str)
    str_b=bmm(str)
    print("FMM算法结果： %s" % str_f)
    print("BMM算法结果： %s" % str_b)
    sf=str_f.strip("/").split("/")  # 使用/分割，FMM，BMM
    sb=str_b.strip("/").split("/")
    lenf=len(sf)        # 首先计算FMM，和BMM分词长度
    lenb=len(sb)
    if lenf>lenb:       # BiMM返回两者中，分词数少的那个
        return str_b
    elif lenf<lenb:
        return str_f
    else:               # 如果两者分词数相等，返回单字少的
        cnt_f=0
        cnt_b=0
        for i in sf:
            if len(i)==1:
                cnt_f+=1
        for i in sb:
            if len(i)==1:
                cnt_b+=1
        if cnt_f<cnt_b:
            return str_f
        else:           # 如果单字数相等，默认返回后向算法的
            return str_b

# %%
str="他是研究生物化学的一位科学家"      # 测试用例
print("BiMM算法结果：%s" % BiMM(str))

