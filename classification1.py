import numpy as np
''  # 입력으로 공부시간을 넣었을 때 합격/불합격 여부를 판단한다.

''  # 1. 학습데이터(Training Data) 준비

# x_data : 공부시간
# t_data : 합격/불합격 여부 (0:불합격 | 1:합격)
x_data = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]).reshape(10, 1)
t_data = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1]).reshape(10, 1)

''  # 2. 임의의 직선 z = Wx + b 정의 (임의의 값으로 가중치 W, 바이어스 b 초기화)
W = np.random.rand(1, 1)  # 배열안에 배열로 된 원소가 하나. (1x1)
b = np.random.rand(1)  # 1차원 배열, 1개
print("W = ", W, ", W.shape = ", W.shape, ", b : ", b, ", b.shape = ", b.shape)


''  # 3. 손실함수 E(W,b) 정의
# 최종출력은 y = sigmoid(Wx+b) 이며, 손실함수는 cross-entropy 로 나타냄

# 임의의 직선값 z가 sigmoid의 인자로 들어가서 판별된다.
# 그리고 뒤에 과정에서 sigmoid 리턴 값과 정답과 비교한다.
# 손실함수는 실제값과의 오차를 나타내는 함수라고 보면 된다.
# 손실함수는 평균제곱 오차와 엔트로피 오차가 있는데, 평균제곱 오차는 회귀 분석에서 사용함.
# 오차를 가장 작게 만드는게 목표임
# z = Wx + b, y = sigmoid(z)
# E(W,b) = -(sigma n, i=1) {t.i log(y.i)log(1-y.i)} --> cross-entropy (공식 유도는 패스..)


def sigmoid(x):
    return 1 / (1+np.exp(-x))


def loss_func(x, t):

    delta = 1e-7  # log 무한대 발산 방지

    z = np.dot(x, W) + b  # dot() : 행렬곱
    y = sigmoid(z)

    # cross_entropy
    return -np.sum(t*np.log(y+delta) + (1-t)*np.log((1-y)+delta))


''  # 4. 수치미분 numerical_derivative 및 utility 함수 정의
# errorval - 손실함수 값을 나타냄
# predict - 미래 값을 알려줌
# sigmoid 값이 0.5 이상이면 1, 이하이면 0


def numerical_derivative(f, x):  # f는 함수, x는 W나 b가 온다.
    delta_x = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    # x의 모든 인덱스를 돌면서 계산
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        # 함수 값을 x+h에서 계산
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + delta_x
        fx1 = f(x)  # f(x+delta_x)

        # 편미분 계산
        x[idx] = tmp_val - delta_x
        fx2 = f(x)  # f(x-delta_x)
        grad[idx] = (fx1 - fx2) / (2*delta_x)

        x[idx] = tmp_val
        it.iternext()

    return grad


def error_val(x, t):
    delta = 1e-7  # log 무한대 발산 방지

    z = np.dot(x, W) + b
    y = sigmoid(z)

    # cross-entropy
    return -np.sum(t * np.log(y+delta)+(1-t)*np.log((1-y)+delta))


def predict(x):

    z = np.dot(x, W) + b
    y = sigmoid(z)

    if y > 0.5:
        result = 1
    else:
        result = 0

    return y, result


''  # 5. 학습율(learning rate) 초기화 및 손실함수가 최소가 될 때 까지 W, b 업데이트
learning_rate = 1e-3  # 발산하는 경우, ie-3 ~ ie-6으로 변경


def f(x): return loss_func(x_data, t_data)  # f(x) = loss_func(x_data, t_data)


for step in range(10001):

    W -= learning_rate * numerical_derivative(f, W)

    b -= learning_rate * numerical_derivative(f, b)
    if (step % 400 == 0):
        print("step = ", step, "error value = ", error_val(
            x_data, t_data), ", W = ", W, ", b = ", b)


(real_val, logical_val) = predict(3)
print(real_val, logical_val)
