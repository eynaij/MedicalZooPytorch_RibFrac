import numpy as np
from numpy import *
import scipy
import pylab
import random, math
from numpy.random import rand
from numpy import square, sqrt
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
# from result_set import KNNResultSet, RadiusNNResultSet
from sklearn.cluster import KMeans
# import  kdtree as kdtree
import  time
#from scipy.spatial import KDTree
from sklearn.neighbors import KDTree # KDTree 进行搜索
import copy


plt.style.use('seaborn')


# matplotlib显示点云函数
def Point_Show(point,point_index):

    def colormap(c, num_clusters):
        if c == -1:
            color = [1] * 3
        # surrouding object:
        else:
            color = [0] * 3
            color[c % 3] = c / num_clusters
        return color

    x = []
    y = []
    num_clusters = max(point_index) + 1
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])

    plt.scatter(x, y,color=[colormap(c,num_clusters) for c in point_index])
    plt.show()


#构建距离矩阵
def my_distance_Marix(data):
    S = np.zeros((len(data), len(data)))  # 初始化 关系矩阵 w 为 n*n的矩阵
    # step1 建立关系矩阵， 每个节点都有连线，权重为距离的倒数
    for i in range(len(data)):  # i:行
        for j in range(len(data)):  # j:列
                S[i][j] = np.linalg.norm(data[i] - data[j])  # 二范数计算两个点直接的距离，两个点之间的权重为之间距离的倒数
    return S

# @profile
def DBSCAN(data, eps, Minpts):
    """
    基于密度的点云聚类
    :param d_bbox: 点与点之间的距离矩阵
    :param eps:  最大搜索直径阈值
    :param Minpts:  最小包含其他对象数量阈值
    :return: 返回聚类结果，是一个嵌套列表,每个子列表就是这个区域的对象的序号
    """
    n = len(data)
    # 构建kd_tree
    leaf_size = 4
    tree = KDTree(data,leaf_size)
    #step1 初始化核心对象集合T,聚类个数k,聚类集合C, 未访问集合P
    T = set()    #set 集合
    k = 0        #第k类
    cluster_index = np.zeros(n,dtype=int)      #聚类集合
    unvisited = set(range(n))   #初始化未访问集合
    #step2 通过判断，通过kd_tree radius NN找出所有核心点
    nearest_idx = tree.query_radius(data, eps)  # 进行radius NN搜索,半径为epsion,所有点的最临近点储存在 nearest_idx中
    for d in range(n):
        if len(nearest_idx[d]) >= Minpts:     #临近点数 > min_sample,加入核心点
            T.add(d)    #最初得核心点
    #step3 聚类
    while len(T):     #visited core ，until all core points were visited
        unvisited_old = unvisited     #更新为访问集合
        core = list(T)[np.random.randint(0,len(T))]    #从 核心点集T 中随机选取一个 核心点core
        unvisited = unvisited - set([core])      #把核心点标记为 visited,从 unvisited 集合中剔除
        visited = []
        visited.append(core)

        while len(visited):
            new_core = visited[0]
            #kd-tree radius NN 搜索邻近
            if new_core in T:     #如果当前搜索点是核心点
                S = unvisited & set(nearest_idx[new_core])    #当前 核心对象的nearest 与 unvisited 的交集
                visited +=  (list(S))                     #对该new core所能辐射的点，再做检测
                unvisited = unvisited - S          #unvisited 剔除已 visited 的点
            visited.remove(new_core)                     #new core 已做检测，去掉new core

        cluster = unvisited_old -  unvisited    #原有的 unvisited # 和去掉了 该核心对象的密度可达对象的visited就是该类的所有对象
        T = T - cluster  #去掉该类对象里面包含的核心对象,差集
        cluster_index[list(cluster)] = k
        k += 1   #类个数
    noise_cluster = unvisited
    cluster_index[list(noise_cluster)] = -1    #噪声归类为 1
    print(cluster_index)
    print("生成的聚类个数：%d" %k)
    return cluster_index
# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    #plt.show()
    return X


if __name__ == '__main__':
    # 生成数据
    true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    true_Var = [[1, 3], [2, 2], [6, 2]]
    X = generate_X(true_Mu, true_Var)
    # begin_t =time.time()
    # cluster_index = DBSCAN(X,eps=0.5,Minpts=15)
    # dbscan_time = time.time() - begin_t
    # print("dbscan time:%f"%dbscan_time)
    # Point_Show(X,cluster_index)