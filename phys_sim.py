# Безголова копія фізики з 8_6.py (pymunk) для налаштування стратегії
from __future__ import division, print_function
import math, random, sys
import pymunk
import rl_agent as rl

def build(seed):
    random.seed(seed)
    space = pymunk.Space()
    def cb(x, y, shape, *a):
        body = pymunk.Body(); body.position = x, y
        s = shape(body, *a); s.mass = 1; s.friction = 1
        space.add(body, s); return s
    P = ((-20,-5),(-20,5),(20,15),(20,-15))
    s0 = cb(300,300,pymunk.Poly,P); s0.score = 0
    s3 = cb(200,300,pymunk.Poly,P); s3.score = 0
    s1 = cb(300,200,pymunk.Circle,10,(0,0))
    s2 = cb(350,250,pymunk.Circle,10,(0,0))
    return space, s0, s3, s1, [s2]

def dist(a, b): return math.hypot(a[0]-b[0], a[1]-b[1])
def inC(p): return dist(p,(350,250)) < 180

def scr(s, s0, s3, p=1):
    if not inC(s.body.position):
        if dist(s.body.position, s0.body.position) < dist(s.body.position, s3.body.position):
            s0.score += p
        else:
            s3.score += p
        s.body.position = random.randint(200,400), random.randint(200,300)

def run(seed, frames, agent=None, log=None):
    space, s0, s3, s1, S2 = build(seed)
    ag = agent or rl.QAgent(rng=random.Random(seed))
    mem = {"d":None,"tr":None,"t":None,"score":0}
    b = s3.body
    acts = []
    for frame in range(frames):
        a = b.angle
        b.velocity = 100*math.cos(a), 100*math.sin(a)
        x, y = b.position
        R = dist((x,y),(350,250))
        if frame % 10 == 0:
            t = tuple(s1.body.position); anti = [tuple(s.body.position) for s in S2]
            state = rl.observe(x,y,a,t,anti)
            d_now = dist((x,y),t); tr_now = dist(t,(350,250))
            if mem["t"] is None or dist(t,mem["t"]) > 60:
                mem["d"], mem["tr"] = d_now, tr_now
            sd = s3.score - mem["score"]; mem["score"] = s3.score
            da = min([dist((x,y),p) for p in anti] or [1e9])
            act_prev = ag.prev[1] if ag.prev else 0
            reward = rl.reward_fn(mem["d"], d_now, da, sd, R>180, mem["tr"], tr_now, act_prev)
            mem["d"], mem["tr"], mem["t"] = d_now, tr_now, t
            act = ag.step(state, reward)
            b.angle = rl.apply_action(b.angle, act)
            acts.append((state, act))
        if R > 180:
            b.angle = math.atan2(250-y, 350-x)
        scr(s1, s0, s3)
        for s in S2: scr(s, s0, s3, p=-1)
        for s in [s0,s1,s3]+S2:
            s.body.velocity = s.body.velocity[0]*0.9, s.body.velocity[1]*0.9
            s.body.angular_velocity *= 0.9
        space.step(0.02)
    return s3.score, s0.score, ag, acts

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    tot = 0
    for sd in range(n):
        sc, sc0, ag, acts = run(sd, 12000)
        tot += sc
        late = acts[-200:]
        e = [a for s, a in late if s == (0,0,0)]
        print(sd, "робот", sc, "нерухомий s0", sc0,
              "state0 дії (останні 200):", {rl.ACTIONS[k]: e.count(k) for k in range(4)})
    print("середнє", tot/n)


def run_old(seed, frames):
    u"""Оригінальна strategy2 з 8_6.py у тій самій фізиці."""
    import numpy as np
    space, s0, s3, s1, S2 = build(seed)
    b = s3.body; Q = [[0,0],[0,0],[0,0]]; action = 0
    def insec(p, x, y, a):
        ang = math.atan2(p[1]-y, p[0]-x) % (2*math.pi); a %= 2*math.pi
        return dist(p,(x,y)) < 100 and a-0.5 < ang < a+0.5
    for frame in range(frames):
        a = b.angle
        b.velocity = 100*math.cos(a), 100*math.sin(a)
        x, y = b.position; R = dist((x,y),(350,250))
        if frame % 10 == 0:
            if insec(s1.body.position,x,y,a): st, rw = 1, (1 if action==0 else -1)
            elif insec(S2[0].body.position,x,y,a): st, rw = 2, (-1 if action==0 else 1)
            else: st, rw = 0, 0
            Q[st][action] += rw
            act = Q[st][action]
            if random.random() < abs(1.0/(act+0.1)): action = random.choice([0,1])
            else: action = int(np.argmax(Q[st]))
            if action: b.angle = 2*math.pi*random.random()
            if R > 180: b.angle = math.atan2(250-y, 350-x)
        scr(s1, s0, s3)
        for s in S2: scr(s, s0, s3, p=-1)
        for s in [s0,s1,s3]+S2:
            s.body.velocity = s.body.velocity[0]*0.9, s.body.velocity[1]*0.9
            s.body.angular_velocity *= 0.9
        space.step(0.02)
    return s3.score
