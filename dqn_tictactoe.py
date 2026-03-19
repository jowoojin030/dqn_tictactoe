import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS

# ──────────────────────────────────────────
# 1. DQN 신경망
# ──────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 9)
        )

    def forward(self, x):
        return self.fc(x)


# ──────────────────────────────────────────
# 2. 하이퍼파라미터
# ──────────────────────────────────────────
LR          = 0.001
GAMMA       = 0.95
MEMORY_SIZE = 5000
BATCH_SIZE  = 64

device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
optimizer  = optim.Adam(policy_net.parameters(), lr=LR)
memory     = deque(maxlen=MEMORY_SIZE)


# ──────────────────────────────────────────
# 3. 게임 종료 판정
# ──────────────────────────────────────────
def check_game_over(board):
    lines = (
        [board[i:i+3] for i in range(0, 9, 3)] +   # 가로
        [board[i::3]  for i in range(3)]      +     # 세로
        [board[0::4], board[2:7:2]]                  # 대각
    )
    for line in lines:
        s = sum(line)
        if abs(s) == 3:
            return True, (1 if s > 0 else -1)
    if 0 not in board:
        return True, 0   # 무승부
    return False, None


# ──────────────────────────────────────────
# 4. 학습 스텝
# ──────────────────────────────────────────
def train_step():
    if len(memory) < BATCH_SIZE:
        return

    batch                                          = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones   = zip(*batch)

    states      = torch.FloatTensor(np.array(states)).to(device)
    actions     = torch.LongTensor(actions).view(-1, 1).to(device)
    rewards     = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones       = torch.FloatTensor(dones).to(device)

    current_q = policy_net(states).gather(1, actions)
    next_q    = target_net(next_states).max(1)[0].detach()
    target_q  = rewards + (1 - dones) * GAMMA * next_q

    loss = nn.MSELoss()(current_q, target_q.view(-1, 1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# ──────────────────────────────────────────
# 5. 자가 학습 (버그 수정 + ε-decay + 로깅)
# ──────────────────────────────────────────
def self_train(episodes=1000):
    print(f"AI 자가 학습 시작 ({episodes}판)...")
    results = []  # 'win' / 'lose' / 'draw'

    for ep in range(episodes):
        board     = np.zeros(9)
        done      = False
        ep_result = 'draw'

        while not done:
            state = board.copy()
            avail = np.where(board == 0)[0]

            # ε-greedy decay: 1.0 → 0.05
            epsilon = max(0.05, 1.0 - ep / episodes)
            if random.random() < epsilon:
                action = random.choice(avail)
            else:
                with torch.no_grad():
                    out = policy_net(torch.FloatTensor(state).to(device))
                    out[board != 0] = -999
                    action = out.argmax().item()

            # AI 착수
            board[action] = 1
            done, res = check_game_over(board)

            if done:
                # AI가 이기거나 무승부로 게임 종료
                if res == 1:
                    reward, ep_result = 1.0, 'win'
                elif res == 0:
                    reward, ep_result = 0.1, 'draw'   # 무승부에 작은 양수 보상
                else:
                    reward, ep_result = -1.0, 'lose'
            else:
                # 상대(랜덤) 착수
                opp_avail  = np.where(board == 0)[0]
                board[random.choice(opp_avail)] = -1
                done, res  = check_game_over(board)

                if res == -1:
                    reward, ep_result = -1.0, 'lose'
                elif res == 0:
                    reward, ep_result = 0.1, 'draw'
                else:
                    reward = 0.0   # 게임 계속

            memory.append((state, action, reward, board.copy(), float(done)))
            train_step()

        results.append(ep_result)

        # 200판마다 target net 동기화 + 승률 출력
        if ep > 0 and ep % 200 == 0:
            target_net.load_state_dict(policy_net.state_dict())
            recent  = results[-200:]
            win_r   = recent.count('win')  / 200 * 100
            draw_r  = recent.count('draw') / 200 * 100
            lose_r  = recent.count('lose') / 200 * 100
            print(f"[{ep:>4}/{episodes}] 승률 {win_r:5.1f}% | "
                  f"무승부 {draw_r:5.1f}% | 패배 {lose_r:5.1f}%")

    # 최종 승률
    win_r  = results.count('win')  / episodes * 100
    draw_r = results.count('draw') / episodes * 100
    print(f"\n학습 완료! 전체 승률 {win_r:.1f}% | 무승부 {draw_r:.1f}%")

    # 모델 저장
    torch.save(policy_net.state_dict(), 'dqn_tictactoe.pth')
    print("모델 저장 완료: dqn_tictactoe.pth")


# ──────────────────────────────────────────
# 6. Flask API
# ──────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route('/move', methods=['POST'])
def move():
    data  = request.get_json()
    board = np.array(data['board']).flatten()

    state_t = torch.FloatTensor(board).to(device)
    with torch.no_grad():
        q_values = policy_net(state_t)
        q_values[board != 0] = -float('inf')
        action = q_values.argmax().item()

    return jsonify({'row': int(action // 3), 'col': int(action % 3)})


# ──────────────────────────────────────────
# 7. 실행 진입점
# ──────────────────────────────────────────
if __name__ == '__main__':
    # 저장된 모델이 있으면 재학습 생략
    if os.path.exists('dqn_tictactoe.pth'):
        policy_net.load_state_dict(
            torch.load('dqn_tictactoe.pth', map_location=device)
        )
        target_net.load_state_dict(policy_net.state_dict())
        print("저장된 모델 로드 완료 — 학습 생략")
    else:
        self_train(1000)

    # Colab 환경이면 kernel port 노출, 아니면 일반 실행
    try:
        from google.colab import output as colab_output
        import threading
        threading.Thread(
            target=lambda: app.run(port=5000, host='0.0.0.0')
        ).start()
        colab_output.serve_kernel_port_as_window(5000)
    except ImportError:
        # 로컬 / 서버 환경
        app.run(port=5000, host='0.0.0.0')
