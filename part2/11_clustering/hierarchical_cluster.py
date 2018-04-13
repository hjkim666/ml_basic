# SciPy에서 ward 군집 함수와 덴드로그램 함수를 import
from scipy.cluster.hierarchy import dendrogram, ward
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, y = make_blobs(random_state=0, n_samples=12)
# 데이터 배열 X 에 ward 함수를 적용
# SciPy의 ward 함수는 병합 군집을 수행할 때 생성된 거리배열을 리턴
linkage_array = ward(X)
# 클러스터 간의 거리 정보 linkage_array를 사용해 덴드로그램
dendrogram(linkage_array)

# 두 개와 세 개의 클러스터를 구분하는 커트라인을 표시
ax = plt.gca()
bounds = ax.get_xbound()
ax.plot(bounds, [7.25, 7.25], '--', c='k')
ax.plot(bounds, [4, 4], '--', c='k')

ax.text(bounds[1], 7.25, ' two\nclusters', va='center', fontdict={'size': 15})
ax.text(bounds[1], 4, ' three\nclusters', va='center', fontdict={'size': 15})
plt.xlabel("sample num")
plt.ylabel("cluster distance")
plt.show()

