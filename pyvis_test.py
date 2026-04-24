from pyvis.network import Network

# =========================================
# 네트워크 객체 생성
# =========================================
net = Network(notebook=True)


# =========================================
# 노드 추가
# =========================================
net.add_node(100, label='완제품')

net.add_node(200, label='품질')
net.add_node(210, label='품질검사')
net.add_node(220, label='중간검사')
net.add_node(230, label='입고검사')

net.add_node(400, label='납기')
net.add_node(410, label='주조 속도')
net.add_node(420, label='가공 속도')
net.add_node(430, label='포장 속도')
net.add_node(440, label='출하 속도')

net.add_node(500, label='원가')
net.add_node(510, label='환율 영향')
net.add_node(520, label='원자재 영향')
net.add_node(530, label='국내 영향')
net.add_node(540, label='국제 영향')


# =========================================
# 노드 연결
# =========================================

# 제품에 연결
net.add_edge(200, 100)
# net.add_edge(300, 100)
net.add_edge(400, 100)
net.add_edge(500, 100)

# 품질에 연결
net.add_edge(210, 200)
net.add_edge(220, 200)
net.add_edge(230, 200)

# 납기에 연결
net.add_edge(410, 400)
net.add_edge(420, 400)
net.add_edge(430, 400)
net.add_edge(440, 400)

# 원가에 연결
net.add_edge(510, 500)
net.add_edge(520, 500)
net.add_edge(530, 500)
net.add_edge(540, 500)


# =========================================
# 네트워크 시각화
# =========================================
net.show('mygraph.html')
