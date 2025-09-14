# semantic_vacuum.py
# Semantic Storytelling Vacuum — research-y simulation prototype
# deps: numpy, opencv-python

import numpy as np
import cv2
from collections import defaultdict, deque
from heapq import heappush, heappop
import random
import math
import time

# ----------------------------
# Constants
# ----------------------------
FREE, OBST, CLEANED = 0, 1, 2
DIRS4 = [(1,0),(-1,0),(0,1),(0,-1)]

# ----------------------------
# Small helpers
# ----------------------------
def inb(x,y,H,W): return 0 <= x < H and 0 <= y < W

def a_star(grid, start, goal):
    if start == goal: return [start]
    H,W = grid.shape
    sx,sy = start; gx,gy = goal
    def h(x,y): return abs(x-gx)+abs(y-gy)
    pq=[]; heappush(pq,(h(sx,sy),0,(sx,sy),None))
    came={}; g={(sx,sy):0}
    while pq:
        f, gc, node, parent = heappop(pq)
        if node in came: continue
        came[node]=parent
        if node==(gx,gy):
            path=[]; cur=node
            while cur is not None: path.append(cur); cur=came[cur]
            return path[::-1]
        x,y=node
        for dx,dy in DIRS4:
            nx,ny=x+dx,y+dy
            if not inb(nx,ny,H,W) or grid[nx,ny]==OBST: continue
            ng=gc+1
            if ng < g.get((nx,ny),1e9):
                g[(nx,ny)] = ng
                heappush(pq,(ng+h(nx,ny),ng,(nx,ny),node))
    return None

def bfs_component(mask, s):
    H,W = mask.shape
    q=deque([s]); comp=[s]; seen={s}
    while q:
        x,y=q.popleft()
        for dx,dy in DIRS4:
            nx,ny=x+dx,y+dy
            if inb(nx,ny,H,W) and mask[nx,ny] and (nx,ny) not in seen:
                seen.add((nx,ny)); q.append((nx,ny)); comp.append((nx,ny))
    return comp

def label_components(freemask):
    H,W=freemask.shape
    lab=-np.ones((H,W),np.int32); rooms=[]; idx=0
    for i in range(H):
        for j in range(W):
            if freemask[i,j] and lab[i,j]==-1:
                comp=bfs_component(freemask,(i,j))
                for x,y in comp: lab[x,y]=idx
                rooms.append(comp); idx+=1
    return lab, rooms

def boustrophedon(cells):
    byrow=defaultdict(list)
    for r,c in cells: byrow[r].append(c)
    rows=sorted(byrow.keys())
    for r in rows: byrow[r].sort()
    order=[]; flip=False
    for r in rows:
        cols = byrow[r][::-1] if flip else byrow[r]
        order += [(r,c) for c in cols]
        flip = not flip
    return order

# ----------------------------
# World generation (apartment)
# ----------------------------
def make_apartment(H=70,W=110):
    world=np.zeros((H,W),np.uint8)
    # outer walls
    world[0,:]=OBST; world[-1,:]=OBST; world[:,0]=OBST; world[:,-1]=OBST
    def box(x1,y1,x2,y2,door=None):
        world[x1:x2,y1]=OBST; world[x1:x2,y2]=OBST
        world[x1,y1:y2]=OBST; world[x2,y1:y2]=OBST
        if door:  # door = (side, pos, width)
            side,pos,width = door
            if side=='E': world[pos:pos+width,y2]=FREE
            if side=='W': world[pos:pos+width,y1]=FREE
            if side=='N': world[x1,pos:pos+width]=FREE
            if side=='S': world[x2,pos:pos+width]=FREE
    # rooms
    # living
    box(2,2,34,50, door=('E',16,2))
    # kitchen
    box(2,52,34,108, door=('W',20,2))
    # bedroom
    box(36,2,68,46, door=('E',50,2))
    # study
    box(36,48,68,108, door=('W',50,2))

    # furniture obstacles
    rng=np.random.default_rng(7)
    def put_rect(x,y,h,w): world[x:x+h,y:y+w]=OBST
    # living: couch, table
    put_rect(8,8,3,12); put_rect(20,25,4,8)
    # kitchen: counter blocks
    put_rect(6,60,3,18); put_rect(12,90,3,12)
    # bedroom: bed, wardrobe
    put_rect(45,8,5,14); put_rect(52,20,3,10)
    # study: desk, shelf
    put_rect(40,58,3,10); put_rect(55,70,3,12)

    # random clutter
    for _ in range(25):
        x=rng.integers(3,H-3); y=rng.integers(3,W-3)
        if world[x,y]==FREE: world[x:x+2,y:y+3]=OBST
    return world

# ----------------------------
# Mock perception (objects & room tags)
# ----------------------------
def semantic_rooms(world):
    free=(world==FREE).astype(np.uint8)
    labels, rooms = label_components(free)
    tags={}
    for i, comp in enumerate(rooms):
        xs=[p[0] for p in comp]; ys=[p[1] for p in comp]
        h=(max(xs)-min(xs)+1); w=(max(ys)-min(ys)+1); area=len(comp)
        ar = h/(w+1e-6)
        # simple heuristics
        if area>1200 and ar<0.9: tag="LivingRoom"
        elif area>900 and ar>=0.9: tag="Kitchen"
        elif area>700: tag="Bedroom"
        else: tag="Study"
        tags[i]=tag
    return labels, rooms, tags

# ----------------------------
# Dirt model (temporal)
# ----------------------------
class DirtForecaster:
    """
    Tracks per-room dirt time-series and predicts spikes.
    - EMA per (room, hour_of_day, day_of_week bucket)
    - seasonality via small sinusoidal weekly component (simulated)
    """
    def __init__(self, room_ids):
        self.ema = defaultdict(lambda: 0.0)   # key=(room,hour)
        self.alpha = 0.2
        self.room_ids = list(room_ids)
        # priors: kitchens evenings, living mornings/weekend, bedroom night
        self.priors = defaultdict(float)
        for r in self.room_ids:
            for h in range(24):
                base=0.05
                self.priors[(r,h)] = base
        self.bias = defaultdict(float)

    def update(self, room_id, hour, observed_dirt):
        key=(room_id, hour)
        prev=self.ema[key]
        self.ema[key] = (1-self.alpha)*prev + self.alpha*observed_dirt

    def predict(self, room_id, hour, dow):
        # EMA + heuristic priors
        seasonal = 0.0
        # simulate weekly pattern
        seasonal += 0.05*math.sin(2*math.pi*(dow/7.0) + 0.2*room_id)
        pri = self.priors[(room_id,hour)]
        ema = self.ema[(room_id,hour)]
        return max(0.0, ema + pri + seasonal)

# ----------------------------
# Planner (task + coverage)
# ----------------------------
def plan_room_sequence(start, rooms, world):
    # greedy nearest-room order by centroid
    seq=[]
    cur=start
    for comp in rooms:
        entry=min(comp, key=lambda p: abs(p[0]-cur[0])+abs(p[1]-cur[1]))
        path=a_star(world,cur,entry)
        if path: seq.append(('transit', path))
        cover=boustrophedon(comp)
        seq.append(('cover', cover))
        cur=cover[-1] if cover else cur
    return seq

# ----------------------------
# Explanations (XAI)
# ----------------------------
def explain_choice(room_tag, pred, alt_preds):
    best_alt = max(alt_preds) if alt_preds else 0.0
    margin = pred - best_alt
    if margin > 0.1:
        return f"Προτεραιότητα στην {room_tag} (προβλεπόμενη βρωμιά {pred:.2f}, σημαντικά υψηλότερη από εναλλακτικές)."
    if pred > 0.25:
        return f"Καθαρίζω την {room_tag} πριν την αναμενόμενη αιχμή (πρόληψη)."
    return f"Συνεχίζω συστηματικά στην {room_tag} λόγω γειτνίασης και χαμηλού κόστους μετάβασης."

# ----------------------------
# Visualization
# ----------------------------
def draw(world, cover, robot, labels=None, tags=None, hud_text=""):
    H,W = world.shape
    img = np.zeros((H,W,3),np.uint8)
    img[world==FREE]=(240,240,240)
    img[world==OBST]=(40,40,40)
    img[world==CLEANED]=(200,255,200)
    if cover.max()>0:
        heat = (255*(cover/cover.max())).astype(np.uint8)
        heatmap=cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        mask=cover>0
        img[mask] = (0.6*img[mask] + 0.4*heatmap[mask]).astype(np.uint8)
    rx,ry = robot
    img[rx,ry]=(0,0,255)
    vis=cv2.resize(img,(W*6,H*6),interpolation=cv2.INTER_NEAREST)
    if hud_text:
        cv2.rectangle(vis,(0,0),(vis.shape[1],30),(20,20,20),-1)
        cv2.putText(vis,hud_text,(10,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    if labels is not None and tags is not None:
        # annotate room tags (centroids)
        for rid, tag in tags.items():
            ys, xs = np.where(labels==rid)
            if len(xs)==0: continue
            cx,cy=int(xs.mean()*6), int(ys.mean()*6)
            cv2.putText(vis, tag, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
            cv2.putText(vis, tag, (cx-30, cy), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv2.LINE_AA)
    return vis

# ----------------------------
# Simulation loop
# ----------------------------
def main():
    random.seed(0); np.random.seed(0)
    world0 = make_apartment()
    labels, rooms, tags = semantic_rooms(world0)
    start=(5,5); dock=start

    # Forecaster per room
    room_ids = list(range(len(rooms)))
    forecaster = DirtForecaster(room_ids)

    # Coverage map & state
    world = world0.copy()
    cover = np.zeros_like(world, dtype=np.float32)
    robot = start
    battery = 1200
    steps = 0

    # Video
    frame0 = draw(world,cover,robot,labels,tags,"Initializing…")
    h,w,_ = frame0.shape
    out = cv2.VideoWriter("semantic_vacuum.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, (w,h))

    # Time: simulate 2 days, every hour → choose tasks
    HOURS = 2*24
    for t in range(HOURS):
        hour = t % 24
        dow  = (t // 24) % 7

        # Predict per-room dirt; add intuitive priors
        preds = {}
        for rid in room_ids:
            p = forecaster.predict(rid, hour, dow)
            # hand-crafted contextual bumps: Kitchen evenings, Living morning, Bedroom late
            tag = tags[rid]
            if tag=="Kitchen" and 18 <= hour <= 22: p += 0.25
            if tag=="LivingRoom" and 7 <= hour <= 10: p += 0.15
            if tag=="Bedroom" and (hour <= 7 or hour >= 22): p += 0.10
            preds[rid]=p

        # Choose target room by highest predicted dirt *and* reachable cheaply
        # (simple: pick max pred; tie-break by centroid distance)
        rid_best = max(room_ids, key=lambda rid: (preds[rid], -len(rooms[rid])))
        alt = [preds[r] for r in room_ids if r!=rid_best]
        reason = explain_choice(tags[rid_best], preds[rid_best], alt)

        # Plan: go to room entry, then coverage subset (budget-limited)
        comp = rooms[rid_best]
        # choose entry near current robot
        entry = min(comp, key=lambda p: abs(p[0]-robot[0])+abs(p[1]-robot[1]))
        path = a_star(world, robot, entry) or [robot]
        seq  = boustrophedon(comp)

        # Execute with budget: limit per hour cleaning steps
        hour_budget = min(80, battery)  # per hour cleaning
        executed = 0

        # transit
        for p in path[1:]:
            if battery<=0: break
            if world[p]==OBST: break
            robot=p; steps+=1; battery-=1
            if world[robot]==FREE:
                world[robot]=CLEANED; cover[robot]+=0.5
            hud=f"H{hour:02d} D{dow}  Room:{tags[rid_best]}  Step:{steps}  Batt:{battery}  Pred:{preds[rid_best]:.2f}"
            frame=draw(world,cover,robot,labels,tags,hud+" | "+reason)
            cv2.imshow("Semantic Vacuum", frame); out.write(frame)
            if cv2.waitKey(10)==27: break

        # coverage inside room (partial if budget runs out)
        for p in seq:
            if battery<=0 or executed>=hour_budget: break
            if world[p]==OBST: continue
            robot=p; steps+=1; battery-=1; executed+=1
            # observe dirt (simulated): proportional to prediction + noise
            obs = max(0.0, np.random.normal(loc=preds[rid_best], scale=0.05))
            # clean effect
            if world[robot]==FREE or world[robot]==CLEANED:
                world[robot]=CLEANED; cover[robot]+=1.0
            # update model
            forecaster.update(rid_best, hour, obs)

            cov = 100* (cover>0).sum() / (world!=OBST).sum()
            hud=f"H{hour:02d} D{dow}  Room:{tags[rid_best]}  Coverage:{cov:.1f}%  Steps:{steps}  Batt:{battery}  Obs:{obs:.2f}"
            frame=draw(world,cover,robot,labels,tags,hud+" | "+reason)
            cv2.imshow("Semantic Vacuum", frame); out.write(frame)
            if cv2.waitKey(10)==27: break

        # optionally: return to dock when low
        if battery<100:
            back = a_star(world, robot, dock)
            if back:
                for p in back[1:]:
                    if battery<=0: break
                    robot=p; steps+=1; battery-=1
                    if world[robot]==FREE:
                        world[robot]=CLEANED; cover[robot]+=0.25
                    frame=draw(world,cover,robot,labels,tags,"Returning to dock (low battery)")
                    cv2.imshow("Semantic Vacuum", frame); out.write(frame)
                    if cv2.waitKey(10)==27: break
            break  # end sim

    out.release(); cv2.destroyAllWindows()
    cov = 100* (cover>0).sum() / (world!=OBST).sum()
    print(f"Finished. Coverage={cov:.2f}%  Steps={steps}  Battery={battery}. Video: semantic_vacuum.mp4")
    print("Note: This is a research prototype with mocked perception & learning signals.")

if __name__=="__main__":
    main()
