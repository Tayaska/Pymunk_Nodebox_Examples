# encoding: utf-8
u"""Q-навчання для робота з 8_6.py (сумісно з Python 2.7 і 3).

Відмінності від оригінальної стратегії:
  * справжнє правило Q-навчання: Q(s,a) += alpha*(r + gamma*max Q(s',.) - Q(s,a)),
    а не проста сума винагород;
  * epsilon-жадібний вибір дії з поступовим зменшенням epsilon;
  * багатший стан: де саме мета (зліва/спереду/справа), чи є поруч антиоб'єкт,
    чи робот біля межі арени;
  * дії: залишати, повернути ліворуч, праворуч, випадковий кут;
  * винагорода враховує наближення до мети, очки, близькість до антиоб'єкта і межу;
  * Q-таблицю можна зберегти у файл і продовжити навчання в наступному запуску.

Параметри залежать від варіанта студента (a, b, c, d), тому в кожного вони свої.
"""
from __future__ import division
import math
import random
import json

# ---------------- параметри варіанта ----------------
VARIANT = (3, 2, 1, 0)          # варіант 24: a, b, c, d (змініть на свій)
_a, _b, _c, _d = VARIANT

ALPHA = 0.05 + 0.03 * _c        # швидкість навчання (0.08)
GAMMA = 0.80 + 0.03 * _b        # дисконтування (0.86)
EPS_START = 0.30 + 0.05 * _a    # початкова ймовірність дослідження (0.45)
EPS_MIN = 0.02 + 0.01 * _d      # мінімальна ймовірність дослідження (0.02)
EPS_DECAY = 0.995               # множник зменшення epsilon за кожен крок
HALF_ANGLE = 0.8 + 0.1 * _b     # півкут огляду сенсора, рад (1.0)
SENSOR_R = 100 + 10 * _a        # дальність сенсора (130)
TURN = 0.4 + 0.1 * _c           # кут повороту дій "ліворуч/праворуч" (0.5)
TURN_COST = 0.5                 # штраф за поворот ліворуч/праворуч
SAVE_FILE = "q_table_v%d%d%d%d.json" % VARIANT

ACTIONS = ("keep", "left", "right", "random")
N_ACTIONS = len(ACTIONS)


def norm_angle(x):
    u"""Приводить кут до діапазону (-pi, pi]."""
    x = (x + math.pi) % (2 * math.pi) - math.pi
    return x


def observe(x, y, a, target, anti, center=(350, 250), arena_r=180,
            half=HALF_ANGLE, sensor_r=SENSOR_R):
    u"""Повертає стан (obj, anti_flag, border).

    obj: 0 - мету не видно, 1 - мета зліва, 2 - спереду, 3 - справа
         (сектор огляду ділиться на три частини)
    anti_flag: 1 - антиоб'єкт у секторі огляду
    border: 1 - робот ближче ніж 30 до межі арени
    """
    def rel(p):
        d = math.hypot(p[0] - x, p[1] - y)
        ang = norm_angle(math.atan2(p[1] - y, p[0] - x) - a)
        return d, ang

    obj = 0
    d, ang = rel(target)
    if d < sensor_r and abs(ang) < half:
        third = half / 3.0
        if ang > third:
            obj = 1          # кут додатний - мета ліворуч (проти годинникової)
        elif ang < -third:
            obj = 3
        else:
            obj = 2
    anti_flag = 0
    for p in anti:
        d, ang = rel(p)
        if d < sensor_r and abs(ang) < half:
            anti_flag = 1
    r = math.hypot(x - center[0], y - center[1])
    border = 1 if r > arena_r - 30 else 0
    return (obj, anti_flag, border)


class QAgent(object):
    def __init__(self, alpha=ALPHA, gamma=GAMMA, eps=EPS_START,
                 eps_min=EPS_MIN, eps_decay=EPS_DECAY, rng=None):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.rng = rng or random.Random()
        self.Q = {}                     # стан -> [Q для кожної дії]
        self.prev = None                # (стан, дія) попереднього кроку

    # --- Q-таблиця ---
    def q(self, s):
        if s not in self.Q:
            self.Q[s] = [0.0] * N_ACTIONS
        return self.Q[s]

    def best_action(self, s):
        qs = self.q(s)
        m = max(qs)
        return self.rng.choice([i for i, v in enumerate(qs) if v == m])

    def choose(self, s):
        u"""Epsilon-жадібний вибір дії."""
        if self.rng.random() < self.eps:
            act = self.rng.randrange(N_ACTIONS)
        else:
            act = self.best_action(s)
        self.eps = max(self.eps_min, self.eps * self.eps_decay)
        return act

    def update(self, s, act, reward, s_next):
        u"""Правило Q-навчання."""
        qs = self.q(s)
        target = reward + self.gamma * max(self.q(s_next))
        qs[act] += self.alpha * (target - qs[act])

    def step(self, s_next, reward):
        u"""Один крок: навчитися на попередньому переході і вибрати нову дію."""
        if self.prev is not None:
            self.update(self.prev[0], self.prev[1], reward, s_next)
        act = self.choose(s_next)
        self.prev = (s_next, act)
        return act

    # --- збереження ---
    def save(self, path=SAVE_FILE):
        data = {"eps": self.eps,
                "Q": dict((",".join(str(i) for i in s), v)
                          for s, v in self.Q.items())}
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path=SAVE_FILE):
        try:
            with open(path) as f:
                data = json.load(f)
        except (IOError, OSError, ValueError):
            return False
        self.eps = data.get("eps", self.eps)
        self.Q = dict((tuple(int(i) for i in k.split(",")), v)
                      for k, v in data["Q"].items())
        return True


def apply_action(angle, act, rng=random):
    u"""Новий напрямок робота після дії."""
    if act == 1:
        return angle + TURN
    if act == 2:
        return angle - TURN
    if act == 3:
        return 2 * math.pi * rng.random()
    return angle


def reward_fn(d_prev, d_now, dist_anti, score_delta, at_border,
              tr_prev=0.0, tr_now=0.0, act=0):
    u"""Винагорода: наближення до мети + виштовхування мети + очки - небезпека.

    tr_prev, tr_now - відстань мети від центру арени до і після кроку."""
    r = 0.05 * (d_prev - d_now)         # наближення до мети
    r += 0.15 * (tr_now - tr_prev)      # мету виштовхують до межі арени
    r += 5.0 * score_delta              # очко за мету / штраф за антиоб'єкт
    if dist_anti < 40:
        r -= 1.0                        # занадто близько до антиоб'єкта
    if act in (1, 2):
        r -= TURN_COST                  # повороти "коштують" енергії
    if at_border:
        r -= 0.5                        # довелося рятуватися від межі
    return r
