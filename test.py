"""
TEXT MUD  v2
- BSP dungeon generation
- Room-exit based movement (A* pathfinding)
- 3-phase action system: PREP / EXEC / DELAY
- AP system
- 3 floors, factions, items
"""

import random, os, heapq
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MAP_W  = 60
MAP_H  = 30
MAP_Z  = 3
SEED   = 20260312

MIN_ROOM_W, MAX_ROOM_W = 5, 12
MIN_ROOM_H, MAX_ROOM_H = 4,  7

# ─────────────────────────────────────────
# TILES
# ─────────────────────────────────────────
T_WALL = '#'; T_FLOOR = '.'; T_CORRIDOR = ','
T_DOOR = '+'; T_STAIR_U = '<'; T_STAIR_D = '>'
PASSABLE = {T_FLOOR, T_CORRIDOR, T_DOOR, T_STAIR_U, T_STAIR_D}

@dataclass
class Tile:
    type: str       = T_WALL
    durability: int = 100

# ─────────────────────────────────────────
# PHASE TABLE  {action: (prep_turns, delay_turns, ap_cost)}
# ─────────────────────────────────────────
PHASE = {
    # movement — ap_cost = per tile; total = distance
    "move":        (0, 1, 1),   # arrive -> 1-turn delay
    "run":         (0, 1, 3),   # 2-tile burst -> 1-turn delay

    # melee
    "attack":      (0, 0, 2),   # instant
    "kk":          (0, 1, 2),   # recoil after kick
    "kk_shoot":    (0, 1, 4),   # combo recoil

    # ranged
    "shoot":       (0, 0, 3),   # pistol instant
    "shoot_sg":    (1, 1, 3),   # shotgun: aim + recoil
    "shoot_ri":    (1, 0, 3),   # rifle: aim only

    # reload / items
    "reload":      (0, 3, 0),   # 3-turn fully exposed
    "take":        (0, 1, 1),   # bend down = brief exposure
    "use_wolhon":  (1, 0, 0),   # swallow moment
    "use_heal":    (1, 1, 0),   # bandage both sides
    "use_grenade": (1, 1, 2),

    # utility
    "cover":       (0, 0, 1),
    "dodge":       (0, 0, 2),
    "rest":        (0, 4, 0),   # long rest = long exposure
    "break_wall":  (1, 2, 5),
}

# ─────────────────────────────────────────
# ITEMS
# ─────────────────────────────────────────
@dataclass
class Item:
    name: str
    itype: str
    damage: int = 0
    range_: int = 1
    ammo: int   = 0
    effect: str = ""

ITEM_DB: Dict[str, Item] = {
    "glock17":  Item("Glock 17",  "weapon_gun",   damage=18, range_=6,  ammo=15),
    "shotgun":  Item("Shotgun",   "weapon_gun",   damage=45, range_=3,  ammo=0),
    "rifle":    Item("Rifle",     "weapon_gun",   damage=35, range_=10, ammo=8),
    "knife":    Item("Knife",     "weapon_melee", damage=22),
    "bat":      Item("Iron Bat",  "weapon_melee", damage=18),
    "magazine": Item("Magazine",  "ammo",         effect="ammo+15"),
    "shells":   Item("Shells",    "ammo",         effect="ammo+6"),
    "wolhon":   Item("Wolhon",    "consumable",   effect="ap_boost"),
    "bandage":  Item("Bandage",   "consumable",   effect="heal+30"),
    "grenade":  Item("Grenade",   "consumable",   damage=60, range_=3, effect="explosion"),
}

# ─────────────────────────────────────────
# ENTITY
# ─────────────────────────────────────────
@dataclass
class Entity:
    name: str
    x: int = 0; y: int = 0; z: int = 0
    hp: int = 100; max_hp: int = 100
    ap: int = 20; max_ap: int = 20
    faction: str = "hunter"
    alive: bool  = True

    inventory: List[Item]    = field(default_factory=list)
    gun:   Optional[Item]    = None
    melee: Optional[Item]    = None
    ammo:  int               = 0
    cover: bool              = False
    status: str              = ""
    wolhon_penalty: int      = 0
    transform_timer: int     = 0

    # ── 3-phase state ──
    phase: str           = ""      # "prep" | "delay" | ""
    phase_timer: int     = 0
    queued_action: str   = ""      # action waiting to exec after prep
    queued_args: list    = field(default_factory=list)
    queued_dmg: int      = 0       # pre-calculated damage

    # ── multi-turn movement ──
    move_path: list      = field(default_factory=list)  # [(x,y), ...]
    move_delay: int      = 0       # arrival delay turns remaining

# ─────────────────────────────────────────
# ROOM
# ─────────────────────────────────────────
@dataclass
class Room:
    x: int; y: int; w: int; h: int
    def center(self): return (self.x + self.w//2, self.y + self.h//2)
    def inner(self):
        for ry in range(self.y+1, self.y+self.h-1):
            for rx in range(self.x+1, self.x+self.w-1):
                yield rx, ry
    def overlaps(self, o, m=2):
        return not (self.x+self.w+m<=o.x or o.x+o.w+m<=self.x or
                    self.y+self.h+m<=o.y or o.y+o.h+m<=self.y)
    def contains(self, x, y):
        return self.x < x < self.x+self.w-1 and self.y < y < self.y+self.h-1

# ─────────────────────────────────────────
# A* PATHFINDING
# ─────────────────────────────────────────
def astar(tiles_z, sx, sy, gx, gy, blocked=None):
    """Returns list of (x,y) from start(excl) to goal(incl)."""
    blocked = blocked or set()
    if (sx,sy)==(gx,gy): return []
    open_ = [(0, sx, sy)]
    came  = {}
    g     = {(sx,sy): 0}
    while open_:
        _, cx, cy = heapq.heappop(open_)
        if (cx,cy)==(gx,gy):
            path = []
            while (cx,cy)!=(sx,sy):
                path.append((cx,cy))
                cx,cy = came[(cx,cy)]
            return list(reversed(path))
        for dx,dy in [(0,-1),(0,1),(1,0),(-1,0)]:
            nx,ny = cx+dx, cy+dy
            if not (0<=nx<MAP_W and 0<=ny<MAP_H): continue
            if tiles_z[ny][nx].type not in PASSABLE: continue
            if (nx,ny) in blocked: continue
            ng = g[(cx,cy)] + 1
            if ng < g.get((nx,ny), 9999):
                g[(nx,ny)] = ng
                came[(nx,ny)] = (cx,cy)
                h = abs(nx-gx)+abs(ny-gy)
                heapq.heappush(open_, (ng+h, nx, ny))
    return []

# ─────────────────────────────────────────
# FIND EXIT in direction from current room
# ─────────────────────────────────────────
def find_exit(tiles_z, rooms, px, py, dx, dy):
    """
    Find the nearest corridor/door tile on the edge of the current room
    in the given direction, then follow it to the next room's entry tile.
    Returns (goal_x, goal_y) or None.
    """
    # find which room player is in
    cur_room = None
    for r in rooms:
        if r.contains(px, py):
            cur_room = r
            break

    if cur_room is None:
        # player is in corridor — just find next passable tile in direction
        nx, ny = px+dx, py+dy
        if 0<=nx<MAP_W and 0<=ny<MAP_H and tiles_z[ny][nx].type in PASSABLE:
            return nx, ny
        return None

    # scan room edge in the requested direction
    # and find the corridor opening
    candidates = []
    if dy == -1:  # north: top edge
        for x in range(cur_room.x, cur_room.x+cur_room.w):
            tx, ty = x, cur_room.y
            if tiles_z[ty][tx].type in PASSABLE:
                candidates.append((tx, ty-1))
    elif dy == 1:  # south: bottom edge
        for x in range(cur_room.x, cur_room.x+cur_room.w):
            tx, ty = x, cur_room.y+cur_room.h-1
            if tiles_z[ty][tx].type in PASSABLE:
                candidates.append((tx, ty+1))
    elif dx == 1:  # east: right edge
        for y in range(cur_room.y, cur_room.y+cur_room.h):
            tx, ty = cur_room.x+cur_room.w-1, y
            if tiles_z[ty][tx].type in PASSABLE:
                candidates.append((tx+1, ty))
    elif dx == -1: # west: left edge
        for y in range(cur_room.y, cur_room.y+cur_room.h):
            tx, ty = cur_room.x, y
            if tiles_z[ty][tx].type in PASSABLE:
                candidates.append((tx-1, ty))

    # filter valid candidates
    valid = [(x,y) for x,y in candidates
             if 0<=x<MAP_W and 0<=y<MAP_H
             and tiles_z[y][x].type in PASSABLE]

    if not valid:
        return None

    # pick closest candidate to player
    valid.sort(key=lambda p: abs(p[0]-px)+abs(p[1]-py))
    gx, gy = valid[0]

    # follow corridor until we reach the next room floor tile
    # walk up to 20 steps in direction
    cx, cy = gx, gy
    for _ in range(20):
        nx, ny = cx+dx, cy+dy
        if not (0<=nx<MAP_W and 0<=ny<MAP_H): break
        t = tiles_z[ny][nx].type
        if t == T_FLOOR:
            return nx, ny
        if t in PASSABLE:
            cx, cy = nx, ny
        else:
            break
    return gx, gy

# ─────────────────────────────────────────
# BSP MAP GENERATION
# ─────────────────────────────────────────
def place_rooms(rng, rooms, attempts=60):
    """Simple random room placement with overlap check."""
    for _ in range(attempts):
        rw = rng.randint(MIN_ROOM_W, MAX_ROOM_W)
        rh = rng.randint(MIN_ROOM_H, MAX_ROOM_H)
        rx = rng.randint(1, MAP_W - rw - 1)
        ry = rng.randint(1, MAP_H - rh - 1)
        room = Room(rx, ry, rw, rh)
        if not any(room.overlaps(r, m=2) for r in rooms):
            rooms.append(room)

def bsp_split(x, y, w, h, depth, rooms, rng):
    pass  # kept for compatibility

def carve_corridor(tiles, x1,y1,x2,y2,rng):
    if rng.random()<0.5:
        for cx in range(min(x1,x2),max(x1,x2)+1):
            if tiles[y1][cx].type==T_WALL: tiles[y1][cx]=Tile(T_CORRIDOR)
        for cy in range(min(y1,y2),max(y1,y2)+1):
            if tiles[cy][x2].type==T_WALL: tiles[cy][x2]=Tile(T_CORRIDOR)
    else:
        for cy in range(min(y1,y2),max(y1,y2)+1):
            if tiles[cy][x1].type==T_WALL: tiles[cy][x1]=Tile(T_CORRIDOR)
        for cx in range(min(x1,x2),max(x1,x2)+1):
            if tiles[y2][cx].type==T_WALL: tiles[y2][cx]=Tile(T_CORRIDOR)

def generate_floor(z, seed):
    rng = random.Random(seed + z*9999)
    tiles = [[Tile(T_WALL) for _ in range(MAP_W)] for _ in range(MAP_H)]
    rooms: List[Room] = []
    place_rooms(rng, rooms, attempts=80)
    for room in rooms:
        for rx,ry in room.inner(): tiles[ry][rx] = Tile(T_FLOOR)
    # connect rooms
    if len(rooms)>=2:
        connected=[rooms[0]]; unconn=rooms[1:]
        while unconn:
            bd=99999; ba=bb=None
            for a in connected:
                ax,ay=a.center()
                for b in unconn:
                    bx,by=b.center()
                    d=abs(ax-bx)+abs(ay-by)
                    if d<bd: bd=d;ba=a;bb=b
            ax,ay=ba.center(); bx,by=bb.center()
            carve_corridor(tiles,ax,ay,bx,by,rng)
            connected.append(bb); unconn.remove(bb)
        for _ in range(max(1,len(rooms)//3)):
            a=rng.choice(rooms); b=rng.choice(rooms)
            if a is not b:
                ax,ay=a.center(); bx,by=b.center()
                carve_corridor(tiles,ax,ay,bx,by,rng)
    # doors
    for y in range(1,MAP_H-1):
        for x in range(1,MAP_W-1):
            if tiles[y][x].type==T_CORRIDOR:
                nf=sum(1 for dy,dx in[(-1,0),(1,0),(0,-1),(0,1)]
                       if tiles[y+dy][x+dx].type==T_FLOOR)
                if nf>=2 and rng.random()<0.3:
                    tiles[y][x]=Tile(T_DOOR)
    # stairs
    if len(rooms)>=2:
        ux,uy=rooms[0].center(); tiles[uy][ux]=Tile(T_STAIR_U)
        dx,dy=rooms[-1].center(); tiles[dy][dx]=Tile(T_STAIR_D)
    return tiles, rooms

def generate_map(seed):
    floors=[]; all_rooms=[]
    for z in range(MAP_Z):
        t,r=generate_floor(z,seed)
        floors.append(t); all_rooms.append(r)
    return floors, all_rooms

# ─────────────────────────────────────────
# GROUND ITEMS
# ─────────────────────────────────────────
@dataclass
class GroundItem:
    item: Item; x:int; y:int; z:int

def place_items(all_rooms, seed):
    rng=random.Random(seed+1234); ground=[]
    pool=["magazine","magazine","knife","bat","wolhon","bandage",
          "grenade","shotgun","rifle","shells","magazine"]
    for z,rooms in enumerate(all_rooms):
        for room in rooms[1:]:
            if rng.random()<0.6:
                k=rng.choice(pool); cx,cy=room.center()
                ground.append(GroundItem(ITEM_DB[k],
                    cx+rng.randint(-1,1), cy+rng.randint(-1,1), z))
    return ground

# ─────────────────────────────────────────
# ENEMIES
# ─────────────────────────────────────────
def place_enemies(all_rooms, seed):
    rng=random.Random(seed+5678); enemies=[]; eid=1
    fnames={"werebeast":["Werebeast","Wolf Soldier","Beast"],
            "vampire":  ["Vampire","Night Lord","Blood Fiend"],
            "hunter":   ["Rogue Hunter","Mercenary","Traitor"]}
    for z,rooms in enumerate(all_rooms):
        for room in rooms[2:]:
            if rng.random()<0.55:
                fac=rng.choice(list(fnames.keys()))
                nm=rng.choice(fnames[fac])
                cx,cy=room.center(); hp=rng.randint(40,100)
                e=Entity(name=f"{nm}_{eid}",x=cx,y=cy,z=z,
                         hp=hp,max_hp=hp,faction=fac)
                if fac=="hunter":   e.gun=ITEM_DB["glock17"]; e.ammo=rng.randint(6,15)
                elif fac=="werebeast": e.melee=ITEM_DB["knife"]
                else:               e.melee=ITEM_DB["bat"]
                enemies.append(e); eid+=1
    return enemies

# ─────────────────────────────────────────
# FOV
# ─────────────────────────────────────────
import math
def get_visible(tiles_z, px, py, radius=8):
    visible={(px,py)}
    for step in range(360):
        angle=step*math.pi/180
        rdx,rdy=math.cos(angle),math.sin(angle)
        rx,ry=float(px),float(py)
        for _ in range(radius):
            rx+=rdx; ry+=rdy
            ix,iy=int(rx),int(ry)
            if not(0<=ix<MAP_W and 0<=iy<MAP_H): break
            visible.add((ix,iy))
            if tiles_z[iy][ix].type==T_WALL: break
    return visible

# ─────────────────────────────────────────
# ASCII MAP
# ─────────────────────────────────────────
def render_map(floors, player, entities, ground_items, explored):
    tz=floors[player.z]
    vis=get_visible(tz,player.x,player.y)
    for vx,vy in vis: explored.add((vx,vy,player.z))
    lines=[f"  Floor {player.z+1}/{MAP_Z}  [seed:{SEED}]"]
    for y in range(MAP_H):
        row=""
        for x in range(MAP_W):
            p3=(x,y,player.z)
            iv=(x,y) in vis; ie=p3 in explored
            if x==player.x and y==player.y: row+="@"; continue
            drawn=False
            for e in entities:
                if e.alive and e.x==x and e.y==y and e.z==player.z:
                    row+=({"werebeast":"W","vampire":"V","hunter":"H"}.get(e.faction,"E") if iv else (tz[y][x].type if ie else " "))
                    drawn=True; break
            if drawn: continue
            gi_drawn=False
            for gi in ground_items:
                if gi.x==x and gi.y==y and gi.z==player.z and iv:
                    row+="i"; gi_drawn=True; break
            if gi_drawn: continue
            # show move path
            if (x,y) in [(px,py) for px,py in player.move_path] and iv:
                row+="*"; continue
            row+=(tz[y][x].type if iv else (tz[y][x].type if ie else " "))
        lines.append(row)
    lines+=["","@ you  W beast  V vampire  H hunter  i item  * your path",
            "< stair-up  > stair-dn  + door  , corridor"]
    return "\n".join(lines)

# ─────────────────────────────────────────
# DIRECTION HELPER
# ─────────────────────────────────────────
def dir_text(fx,fy,tx,ty):
    dx,dy=tx-fx,ty-fy
    if abs(dx)>=abs(dy): return "east" if dx>0 else "west"
    return "south" if dy>0 else "north"

DIR_MAP={
    "n":(0,-1,0),"north":(0,-1,0),"s":(0,1,0),"south":(0,1,0),
    "e":(1,0,0), "east": (1,0,0), "w":(-1,0,0),"west":(-1,0,0),
    "u":(0,0,1), "up":   (0,0,1), "d":(0,0,-1),"down":(0,0,-1),
}

# ─────────────────────────────────────────
# PHASE PROCESSING  (called every turn)
# ─────────────────────────────────────────
def tick_phase(player: Entity, entities, floors, ground_items, explored) -> str:
    """
    Advance any in-progress phase for the player.
    Returns narration string (may be empty).
    """
    msgs = []

    # ── multi-turn MOVEMENT in progress ──
    if player.move_path:
        if player.ap < 1:
            player.move_path = []
            return "Exhausted mid-move. You stop in your tracks."

        # check if next tile has an enemy
        nx, ny = player.move_path[0]
        occupied = any(e.alive and e.x==nx and e.y==ny and e.z==player.z
                       for e in entities)
        if occupied:
            player.move_path = []
            blocker = next(e for e in entities
                           if e.alive and e.x==nx and e.y==ny and e.z==player.z)
            return f"Path blocked by {blocker.name}! Movement cancelled."

        # advance one tile
        player.x, player.y = nx, ny
        player.ap -= 1
        explored.add((nx, ny, player.z))
        player.move_path.pop(0)

        if player.move_path:
            remaining = len(player.move_path)
            return f"Moving... ({remaining} tiles remaining)  AP:{player.ap}"
        else:
            # arrived — start arrival delay
            player.move_delay = PHASE["move"][1]
            if player.move_delay > 0:
                return f"Arrived. Catching breath... ({player.move_delay} turn delay)"
            return "Arrived."

    # ── arrival delay ──
    if player.move_delay > 0:
        player.move_delay -= 1
        if player.move_delay > 0:
            return f"[delay] Catching breath... ({player.move_delay} turns)"
        return "[delay] Ready."

    # ── PREP phase ──
    if player.phase == "prep":
        player.phase_timer -= 1
        msgs.append(f"[prep] {player.queued_action}... ({player.phase_timer} turns)")
        if player.phase_timer <= 0:
            # execute the queued action
            result = execute_queued(player, entities, floors, ground_items, explored)
            msgs.append(result)
        return "\n".join(msgs)

    # ── DELAY phase ──
    if player.phase == "delay":
        player.phase_timer -= 1
        msgs.append(f"[delay] Recovering... ({player.phase_timer} turns)")
        if player.phase_timer <= 0:
            player.phase = ""
            player.queued_action = ""
            msgs.append("[delay] Done.")
        return "\n".join(msgs)

    return ""

def execute_queued(player: Entity, entities, floors, ground_items, explored) -> str:
    """Execute the stored queued_action after prep phase ends."""
    action = player.queued_action
    args   = player.queued_args
    tiles_z = floors[player.z]

    result = ""

    if action == "shoot":
        target = resolve_target(args, player, entities)
        if not target or not target.alive:
            result = "Target gone."
        elif not player.gun or player.ammo <= 0:
            result = "No ammo."
        else:
            dist = abs(target.x-player.x)+abs(target.y-player.y)
            if dist > player.gun.range_:
                result = "Target moved out of range."
            else:
                dmg = player.gun.damage + random.randint(-4,6)
                target.hp -= dmg
                result = f"SHOT! {target.name} -{dmg}HP (HP:{max(0,target.hp)})"
                player.ammo -= 1
                if target.hp <= 0:
                    target.alive = False
                    result += f"\n{target.name} is down."

    elif action == "use_wolhon":
        for item in player.inventory:
            if item.effect == "ap_boost":
                player.inventory.remove(item)
                player.ap = min(player.max_ap, player.ap+12)
                player.wolhon_penalty += 3
                player.status = "WOLHON"
                result = "Wolhon hits. AP+12. Vision turns red."
                break

    elif action == "use_heal":
        for item in player.inventory:
            if item.effect.startswith("heal"):
                n=int(item.effect.split("+")[1])
                player.hp=min(player.max_hp,player.hp+n)
                player.inventory.remove(item)
                result = f"Bandaged. HP+{n} (HP:{player.hp})"
                break

    elif action == "use_grenade":
        for item in player.inventory:
            if item.effect=="explosion":
                player.inventory.remove(item)
                msgs2=["GRENADE!"]
                for e in entities:
                    if not e.alive or e.z!=player.z: continue
                    d=abs(e.x-player.x)+abs(e.y-player.y)
                    if d<=item.range_:
                        dmg=item.damage-d*8
                        e.hp-=dmg
                        msgs2.append(f"  {e.name} -{dmg}HP")
                        if e.hp<=0: e.alive=False; msgs2.append(f"  {e.name} eliminated.")
                result="\n".join(msgs2); break

    elif action == "break_wall":
        dx,dy=int(args[0]),int(args[1])
        nx,ny=player.x+dx,player.y+dy
        tile=tiles_z[ny][nx]
        tile.durability-=40
        if tile.durability<=0:
            tile.type=T_CORRIDOR
            explored.add((nx,ny,player.z))
            result="Wall broken! New passage."
        else:
            result=f"Wall cracked. ({tile.durability}% remains)"

    # set delay phase
    delay = PHASE.get(action, (0,0,0))[1]
    if delay > 0:
        player.phase       = "delay"
        player.phase_timer = delay
    else:
        player.phase       = ""
        player.queued_action = ""

    return result

def queue_action(player: Entity, action: str, args: list,
                 entities, floors, ground_items, explored) -> str:
    """
    Start a 3-phase action. If prep=0, execute immediately.
    Returns narration.
    """
    prep, delay, ap = PHASE.get(action, (0,0,0))

    # AP check
    if player.ap < ap:
        return f"Not enough AP. (need {ap}, have {player.ap})"

    player.ap -= ap

    if prep == 0:
        # execute immediately, then set delay
        player.queued_action = action
        player.queued_args   = args
        result = execute_queued(player, entities, floors, ground_items, explored)
        return result
    else:
        # enter prep phase
        player.phase         = "prep"
        player.phase_timer   = prep
        player.queued_action = action
        player.queued_args   = args
        action_names = {
            "shoot_sg": "Aiming shotgun",
            "shoot_ri": "Aiming rifle",
            "use_wolhon": "Swallowing Wolhon",
            "use_heal":   "Applying bandage",
            "use_grenade":"Pulling pin",
            "break_wall": "Winding up",
        }
        label = action_names.get(action, f"Preparing {action}")
        return f"{label}... ({prep} turn prep)"

# ─────────────────────────────────────────
# TARGET / ENTITY HELPERS
# ─────────────────────────────────────────
def resolve_target(args, player, entities):
    if not args: return closest_entity(player, entities)
    name=" ".join(args).lower()
    for e in entities:
        if e.alive and e.z==player.z and name in e.name.lower(): return e
    return closest_entity(player, entities)

def closest_entity(player, entities):
    alive=[e for e in entities if e.alive and e.z==player.z]
    if not alive: return None
    return min(alive,key=lambda e:abs(e.x-player.x)+abs(e.y-player.y))

# ─────────────────────────────────────────
# CONTEXT BAR
# ─────────────────────────────────────────
def context_bar(player, entities, ground_items, floors, rooms_z):
    tz=floors[player.z]; px,py,pz=player.x,player.y,player.z
    lines=[]

    # phase status
    if player.move_path:
        lines.append(f"Moving : {len(player.move_path)} tiles remaining  AP:{player.ap}")
    elif player.move_delay>0:
        lines.append(f"Delay  : arrival recovery {player.move_delay} turns")
    elif player.phase=="prep":
        lines.append(f"Prep   : {player.queued_action} in {player.phase_timer} turns  (type 'cancel' to abort)")
    elif player.phase=="delay":
        lines.append(f"Delay  : recovering {player.phase_timer} turns  (vulnerable!)")

    # exits
    DIR_LABELS=[("north",(0,-1)),("south",(0,1)),("east",(1,0)),("west",(-1,0))]
    exits=[]
    for dname,(dx,dy) in DIR_LABELS:
        goal=find_exit(tz, rooms_z, px, py, dx, dy)
        if goal:
            gx,gy=goal
            dist=len(astar(tz,px,py,gx,gy) or [])
            t=tz[py+dy][px+dx].type if 0<=px+dx<MAP_W and 0<=py+dy<MAP_H else T_WALL
            tag=""
            if t==T_DOOR: tag="(door)"
            elif t==T_STAIR_U: tag="(stair-up)"
            elif t==T_STAIR_D: tag="(stair-dn)"
            exits.append(f"{dname}{tag}[{dist}t,AP-{dist}]")
    cur=tz[py][px].type
    if cur==T_STAIR_U: exits.append("up(<)")
    if cur==T_STAIR_D: exits.append("down(>)")
    lines.append("Exits  : "+("  ".join(exits) if exits else "blocked"))

    # targets
    alive=[e for e in entities if e.alive and e.z==pz]
    if alive:
        tgts=[]
        for e in sorted(alive,key=lambda e:abs(e.x-px)+abs(e.y-py)):
            dist=abs(e.x-px)+abs(e.y-py)
            hp_pct=e.hp/e.max_hp
            cond=" !CRIT" if hp_pct<0.25 else " -wnd" if hp_pct<0.6 else ""
            in_m=dist<=1
            in_g=bool(player.gun and dist<=player.gun.range_)
            tag="[melee]" if in_m else "[shoot]" if in_g else f"[{dist}t-far]"
            tgts.append(f"{e.name}{cond}{tag}")
        lines.append("Targets: "+"  ".join(tgts))
    else:
        lines.append("Targets: none")

    # items
    nearby=[]
    for gi in ground_items:
        if gi.z!=pz: continue
        dist=abs(gi.x-px)+abs(gi.y-py)
        if dist<=2:
            loc="here" if dist==0 else dir_text(px,py,gi.x,gi.y)
            nearby.append(f"{gi.item.name}({loc})")
    if nearby: lines.append("Items  : "+"  ".join(nearby))

    # action hint
    busy = bool(player.move_path or player.move_delay or player.phase)
    if not busy:
        close=[e for e in alive if abs(e.x-px)+abs(e.y-py)<=1]
        mid  =[e for e in alive if 1<abs(e.x-px)+abs(e.y-py)<=(player.gun.range_ if player.gun else 0)]
        if close:
            lines.append("Action : attack  kk  kk shoot  dodge  cover  use wolhon")
        elif mid:
            lines.append("Action : shoot [name]  cover  reload  look  run [dir]")
        elif nearby:
            lines.append("Action : take [item]  look  listen  n/s/e/w")
        else:
            lines.append("Action : n/s/e/w  look  listen  map  inv  weapons  ?=help")

    return "\n".join(lines)

# ─────────────────────────────────────────
# DESCRIBE SURROUNDINGS
# ─────────────────────────────────────────
def describe_surroundings(floors, rooms_z, player, entities, ground_items):
    tz=floors[player.z]; px,py,pz=player.x,player.y,player.z
    desc=[]
    DIR_LABELS=[("north",(0,-1)),("south",(0,1)),("east",(1,0)),("west",(-1,0))]
    for dname,(dx,dy) in DIR_LABELS:
        goal=find_exit(tz,rooms_z,px,py,dx,dy)
        if goal:
            gx,gy=goal
            path=astar(tz,px,py,gx,gy)
            dist=len(path)
            t=tz[py+dy][px+dx].type if 0<=px+dx<MAP_W and 0<=py+dy<MAP_H else T_WALL
            suffix=""
            if t==T_DOOR: suffix=" (door)"
            elif t==T_STAIR_U: suffix=" (stairs up)"
            elif t==T_STAIR_D: suffix=" (stairs down)"
            desc.append(f"  {dname.capitalize()}: exit {dist} tiles away{suffix}  AP cost: {dist}")
        else:
            desc.append(f"  {dname.capitalize()}: wall")
    for e in entities:
        if not e.alive or e.z!=pz: continue
        dist=abs(e.x-px)+abs(e.y-py)
        if dist<=8:
            hp_pct=e.hp/e.max_hp
            cond="critically wounded" if hp_pct<0.25 else "wounded" if hp_pct<0.6 else ""
            cond_s=f" ({cond})" if cond else ""
            desc.append(f"  {e.name}{cond_s} — {dist} tiles {dir_text(px,py,e.x,e.y)}")
    for gi in ground_items:
        if gi.z!=pz: continue
        dist=abs(gi.x-px)+abs(gi.y-py)
        if dist<=3:
            desc.append(f"  {gi.item.name} on the ground — {dir_text(px,py,gi.x,gi.y)}")
    cur=tz[py][px].type
    if cur==T_STAIR_U: desc.append("  You stand on stairs leading UP. (u to climb)")
    if cur==T_STAIR_D: desc.append("  You stand on stairs leading DOWN. (d to descend)")
    return "\n".join(desc) if desc else "  The area is quiet."

# ─────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────
def status_bar(player, turn):
    hp_f=player.hp/player.max_hp
    hp_c=("CRITICAL" if hp_f<0.2 else "Wounded" if hp_f<0.5 else
          "Bruised" if hp_f<0.8 else "Fine")
    ap_c=("Exhausted" if player.ap<=3 else "Tired" if player.ap<=8 else "Normal")
    gun=player.gun.name if player.gun else "none"
    mel=player.melee.name if player.melee else "none"
    extras=[]
    if player.status: extras.append(player.status)
    if player.cover:  extras.append("COVER")
    ex="  ["+", ".join(extras)+"]" if extras else ""
    return (f"HP {player.hp}/{player.max_hp}({hp_c})  "
            f"AP {player.ap}/{player.max_ap}({ap_c})  "
            f"Ammo {player.ammo}  {gun}/{mel}{ex}  "
            f"Turn {turn}  Floor {player.z+1}")

# ─────────────────────────────────────────
# COMMAND PARSER
# ─────────────────────────────────────────
def parse(cmd):
    parts=cmd.strip().lower().split()
    if not parts: return "",[]
    return parts[0], parts[1:]

# ─────────────────────────────────────────
# MAIN ACTION HANDLER
# ─────────────────────────────────────────
def do_action(action, args, player, floors, all_rooms, entities, ground_items, explored):
    tz=floors[player.z]; rooms_z=all_rooms[player.z]

    # busy check — only allow cancel / look / listen / map / status
    busy = bool(player.move_path or player.move_delay>0 or player.phase)
    if busy and action not in ("cancel","look","l","listen","li","map","m","status","st","?","help"):
        phase_desc = (
            f"Moving ({len(player.move_path)} tiles left)" if player.move_path else
            f"Arrival delay ({player.move_delay} turns)"  if player.move_delay else
            f"{player.phase.upper()} phase ({player.phase_timer} turns)"
        )
        return f"Busy: {phase_desc}  (type 'cancel' to abort)", False

    # CANCEL
    if action == "cancel":
        if player.move_path or player.move_delay or player.phase:
            player.move_path=[]; player.move_delay=0
            player.phase=""; player.phase_timer=0
            player.queued_action=""
            return "Action cancelled.", False
        return "Nothing to cancel.", False

    # MOVEMENT (room-exit based)
    if action in DIR_MAP:
        dx,dy,dz = DIR_MAP[action]
        if dz != 0:
            cur=tz[player.y][player.x].type
            if dz==1 and cur!=T_STAIR_U: return "No stairs up here.",False
            if dz==-1 and cur!=T_STAIR_D: return "No stairs down here.",False
            if not (0<=player.z+dz<MAP_Z): return "No more floors.",False
            player.z+=dz; player.ap-=2
            return f"You climb to floor {player.z+1}.",False

        goal=find_exit(tz, rooms_z, player.x, player.y, dx, dy)
        if not goal: return "No exit in that direction.",False
        gx,gy=goal
        blocked={(e.x,e.y) for e in entities if e.alive and e.z==player.z}
        path=astar(tz, player.x, player.y, gx, gy, blocked)
        if not path: return "Path blocked.",False
        ap_needed=len(path)
        if player.ap < 1: return "Too exhausted to move.",False
        if player.ap < ap_needed:
            # partial move — go as far as AP allows
            path=path[:player.ap]
            msg=f"Not enough AP for full move. Partial: {len(path)} tiles."
        else:
            msg=f"Moving {len(path)} tiles {action}... (AP cost: {len(path)})"
        player.move_path=path
        player.cover=False
        return msg, False

    # RUN
    elif action=="run" and args:
        dir_=args[0]
        if dir_ not in DIR_MAP: return "Unknown direction.",False
        dx,dy,_=DIR_MAP[dir_]
        if player.ap<3: return "Not enough AP to run.",False
        player.ap-=3
        moved=0
        for _ in range(2):
            nx,ny=player.x+dx,player.y+dy
            if not(0<=nx<MAP_W and 0<=ny<MAP_H): break
            if tz[ny][nx].type not in PASSABLE: break
            if any(e.alive and e.x==nx and e.y==ny and e.z==player.z for e in entities):
                break
            player.x,player.y=nx,ny
            explored.add((nx,ny,player.z)); moved+=1
        player.move_delay=PHASE["run"][1]
        player.cover=False
        return f"Sprint! {moved} tiles {dir_}. Arrival delay {player.move_delay}t.",False

    # LOOK
    elif action in ("look","l"):
        if player.ap<1: return "Too exhausted.",False
        player.ap-=1
        return describe_surroundings(floors,rooms_z,player,entities,ground_items),False

    # LISTEN
    elif action in ("listen","li"):
        if player.ap<1: return "Not enough AP.",False
        player.ap-=1
        sounds=[]
        for e in entities:
            if not e.alive or e.z!=player.z: continue
            dist=abs(e.x-player.x)+abs(e.y-player.y)
            if dist<=10:
                sounds.append(f"  Footsteps {dir_text(player.x,player.y,e.x,e.y)}, ~{dist} tiles.")
        return "You listen...\n"+("\n".join(sounds) if sounds else "  Silence."),False

    # ATTACK
    elif action in ("attack","a"):
        target=resolve_target(args,player,entities)
        if not target: return "No target.",False
        dist=abs(target.x-player.x)+abs(target.y-player.y)
        if dist>1: return f"Too far. ({dist} tiles)",False
        if player.ap<2: return "Not enough AP.",False
        player.ap-=2
        wpn=player.melee; base=wpn.damage if wpn else 8
        dmg=base+random.randint(-3,4)
        target.hp-=dmg
        msgs=[f"You strike {target.name} for {dmg}! (HP:{max(0,target.hp)})"]
        if target.hp<=0: target.alive=False; msgs.append(f"{target.name} is dead.")
        # delay
        player.phase="delay"; player.phase_timer=PHASE["attack"][1]
        return "\n".join(msgs),False

    # KNOCKKICK
    elif action in ("kk","knockkick"):
        target=closest_entity(player,entities)
        if not target: return "No target.",False
        dist=abs(target.x-player.x)+abs(target.y-player.y)
        if dist>1: return f"Too far. ({dist} tiles)",False
        if player.ap<2: return "Not enough AP.",False
        player.ap-=2
        dmg=random.randint(8,16); target.hp-=dmg
        ddx=target.x-player.x; ddy=target.y-player.y
        px2,py2=target.x+ddx*2,target.y+ddy*2
        if(0<px2<MAP_W and 0<py2<MAP_H and tz[py2][px2].type in PASSABLE):
            target.x,target.y=px2,py2; push="Knocked back 2 tiles!"
        else: push="Slammed into the wall!"
        msgs=[f"KNOCKKICK! {dmg} dmg. {push}"]
        combo=args and args[0] in ("shoot","sh")
        if combo and player.gun and player.ammo>0:
            player.ammo-=1; dmg2=player.gun.damage+random.randint(-3,5); target.hp-=dmg2
            msgs.append(f"Follow-up shot! {dmg2} dmg! (HP:{max(0,target.hp)})")
            if target.hp<=0: target.alive=False; msgs.append(f"{target.name} down.")
        player.phase="delay"; player.phase_timer=PHASE["kk"][1]
        return "\n".join(msgs),False

    # SHOOT
    elif action in ("shoot","sh","fire"):
        if not player.gun: return "No firearm.",False
        if player.ammo<=0: return "Empty. Reload first.",False
        # suppression
        if args and args[0] in DIR_MAP and len(args)==1:
            if player.ap<1: return "Not enough AP.",False
            player.ap-=1; player.ammo-=1
            return f"Suppression fire {args[0]}. Ammo:{player.ammo}",False
        # shotgun?
        action_key="shoot_sg" if "shotgun" in player.gun.name.lower() else \
                   "shoot_ri" if "rifle"   in player.gun.name.lower() else "shoot"
        return queue_action(player, action_key, args, entities,
                            floors, ground_items, explored), False

    # RELOAD
    elif action in ("reload","r"):
        for item in player.inventory:
            if item.itype=="ammo":
                player.inventory.remove(item)
                n=int(item.effect.split("+")[1]) if "+" in item.effect else 0
                player.ammo=min(30,player.ammo+n)
                player.phase="delay"; player.phase_timer=PHASE["reload"][1]
                return f"Reloading... {PHASE['reload'][1]}-turn delay. Ammo:{player.ammo}",False
        return "No ammo in inventory.",False

    # COVER
    elif action in ("cover","c"):
        if player.ap<1: return "Not enough AP.",False
        player.ap-=1; player.cover=True
        return "You take cover.",False

    # DODGE
    elif action in ("dodge","dg"):
        if player.ap<2: return "Not enough AP.",False
        player.ap-=2; player.cover=False
        return "You roll aside.",False

    # TAKE
    elif action in ("take","t","pick"):
        if not args: return "Take what?",False
        name=" ".join(args).lower()
        for gi in ground_items[:]:
            if gi.z==player.z and name in gi.item.name.lower():
                dist=abs(gi.x-player.x)+abs(gi.y-player.y)
                if dist>1: return f"Too far. ({dist} tiles)",False
                player.inventory.append(gi.item); ground_items.remove(gi)
                if gi.item.itype=="weapon_gun" and not player.gun:
                    player.gun=gi.item; player.ammo=gi.item.ammo
                    player.phase="delay"; player.phase_timer=PHASE["take"][1]
                    return f"Picked up and equipped {gi.item.name}.",False
                if gi.item.itype=="weapon_melee" and not player.melee:
                    player.melee=gi.item
                player.phase="delay"; player.phase_timer=PHASE["take"][1]
                return f"Picked up {gi.item.name}. ({PHASE['take'][1]}-turn delay)",False
        return "No such item nearby.",False

    # USE
    elif action in ("use","u"):
        if not args: return "Use what?",False
        name=" ".join(args).lower()
        for item in player.inventory:
            if name in item.name.lower():
                if item.effect=="ap_boost":
                    return queue_action(player,"use_wolhon",[],entities,floors,ground_items,explored),False
                elif item.effect.startswith("heal"):
                    return queue_action(player,"use_heal",[],entities,floors,ground_items,explored),False
                elif item.effect=="explosion":
                    return queue_action(player,"use_grenade",[],entities,floors,ground_items,explored),False
                elif "ammo+" in item.effect:
                    n=int(item.effect.split("+")[1])
                    player.ammo=min(30,player.ammo+n)
                    player.inventory.remove(item)
                    return f"Loaded {n} rounds. Ammo:{player.ammo}",False
        return f"'{name}' not in inventory.",False

    # EQUIP
    elif action in ("equip","eq"):
        if not args: return "Equip what?",False
        name=" ".join(args).lower()
        for item in player.inventory:
            if name in item.name.lower():
                if item.itype=="weapon_gun":
                    player.gun=item; player.ammo=item.ammo; player.ap-=2
                    return f"Equipped {item.name}. AP-2",False
                if item.itype=="weapon_melee":
                    player.melee=item; player.ap-=1
                    return f"Equipped {item.name}.",False
        return "Not found.",False

    # INVENTORY
    elif action in ("inv","i","inventory"):
        lines=["━"*32,
               f"Primary : {player.gun.name if player.gun else 'none'} (ammo:{player.ammo})",
               f"Off-hand: {player.melee.name if player.melee else 'none'}",
               "Carried :"]
        lines+=[f"  {it.name}" for it in player.inventory] or ["  (empty)"]
        lines.append("━"*32)
        return "\n".join(lines),False

    # WEAPONS
    elif action in ("weapons","wp"):
        lines=["━"*32]
        if player.gun:
            g=player.gun
            lines+=[f"Primary : {g.name}",f"  Ammo  : {player.ammo}/30",
                    f"  Damage: {g.damage}  Range: {g.range_}"]
        else: lines.append("Primary : none")
        if player.melee:
            m=player.melee; lines.append(f"Off-hand: {m.name}  Dmg:{m.damage}")
        else: lines.append("Off-hand: none")
        lines.append("━"*32)
        return "\n".join(lines),False

    # STATUS
    elif action in ("status","st","me"):
        lines=["━"*32,
               f"HP   : {player.hp}/{player.max_hp}",
               f"AP   : {player.ap}/{player.max_ap}",
               f"Pos  : ({player.x},{player.y}) Floor {player.z+1}",
               f"Cover: {'yes' if player.cover else 'no'}"]
        if player.status: lines.append(f"Status: {player.status}")
        lines.append("━"*32)
        return "\n".join(lines),False

    # REST
    elif action in ("rest","wait","z"):
        player.phase="delay"; player.phase_timer=PHASE["rest"][1]
        rec=random.randint(5,9)
        player.ap=min(player.max_ap,player.ap+rec)
        return f"Resting... AP+{rec}. {PHASE['rest'][1]}-turn delay (vulnerable).",False

    # BREAK
    elif action in ("break","br"):
        dir_=args[0] if args else None
        if not dir_ or dir_ not in DIR_MAP: return "Break which direction?",False
        dx,dy,_=DIR_MAP[dir_]
        nx,ny=player.x+dx,player.y+dy
        if not(0<=nx<MAP_W and 0<=ny<MAP_H): return "Nothing there.",False
        if tz[ny][nx].type!=T_WALL: return "Nothing solid.",False
        return queue_action(player,"break_wall",[str(dx),str(dy)],
                            entities,floors,ground_items,explored),False

    # MAP
    elif action in ("map","m"):
        return None,True

    # HELP
    elif action in ("help","?","h"):
        return (
            "━"*32+"\n"
            "MOVE    n s e w  (room-exit, A* path)\n"
            "        u d      (stairs)\n"
            "        run [dir] (2-tile burst)\n"
            "        cancel   (abort move/action)\n"
            "COMBAT  attack  kk  kk shoot\n"
            "        shoot [target]  shoot [dir](suppress)\n"
            "        reload  cover  dodge\n"
            "ITEMS   take [item]  use [item]  equip [item]\n"
            "INFO    look  listen  inv  weapons  status  map\n"
            "MISC    rest  break [dir]  quit\n"
            "━"*32
        ),False

    else:
        return f"Unknown: '{action}'  (? for help)",False

# ─────────────────────────────────────────
# ENEMY AI
# ─────────────────────────────────────────
def enemy_ai(entity, player, floors):
    if not entity.alive or entity.z!=player.z: return ""
    tz=floors[entity.z]; msgs=[]
    dist=abs(entity.x-player.x)+abs(entity.y-player.y)

    # phase tick
    if entity.phase=="prep":
        entity.phase_timer-=1
        if entity.phase_timer<=0:
            # execute
            if entity.queued_action=="shoot" and entity.gun and entity.ammo>0:
                if dist<=entity.gun.range_:
                    dmg=entity.gun.damage+random.randint(-3,5)
                    player.hp-=max(1,dmg//2) if player.cover else dmg
                    entity.ammo-=1
                    msgs.append(f"  {entity.name} FIRES! HP-{dmg}")
                    if player.hp<=0: player.alive=False
            entity.phase="delay"; entity.phase_timer=1
        else:
            msgs.append(f"  {entity.name} aiming... ({entity.phase_timer}t)")
        return "\n".join(msgs)

    if entity.phase=="delay":
        entity.phase_timer-=1
        if entity.phase_timer<=0: entity.phase=""
        return ""

    # transform
    if entity.status=="TRANSFORMING":
        entity.transform_timer-=1
        msgs.append(f"  [{entity.name} transforming... {entity.transform_timer}t]")
        if entity.transform_timer<=0:
            entity.status="TRANSFORMED"
            entity.max_hp=int(entity.max_hp*1.5)
            entity.hp=min(entity.hp+20,entity.max_hp)
            msgs.append(f"  [{entity.name} TRANSFORMED!]")
        return "\n".join(msgs)

    if(entity.faction=="werebeast" and entity.hp<entity.max_hp*0.35
       and entity.status==""):
        entity.status="TRANSFORMING"; entity.transform_timer=3
        msgs.append(f"  {entity.name} begins transforming!")
        return "\n".join(msgs)

    # move toward player
    if dist>1:
        blocked={(e.x,e.y) for e in [] }  # simplified
        path=astar(tz,entity.x,entity.y,player.x,player.y)
        if path:
            nx,ny=path[0]
            if not any(e.alive and e.x==nx and e.y==ny and e.z==entity.z
                       for e in []):
                entity.x,entity.y=nx,ny
        msgs.append(f"  {entity.name} moves. ({dist-1}t away)")
        return "\n".join(msgs)

    # attack
    base=20 if entity.status=="TRANSFORMED" else 12
    if entity.faction=="hunter" and entity.gun and entity.ammo>0:
        base=entity.gun.damage; entity.ammo-=1
    dmg=base+random.randint(-3,6)
    if player.cover: dmg=max(1,dmg//2)
    player.hp-=dmg
    msgs.append(f"  {entity.name} attacks! HP-{dmg} (cover:{'yes' if player.cover else 'no'})")
    if player.hp<=0: player.alive=False
    entity.phase="delay"; entity.phase_timer=1
    return "\n".join(msgs)

# ─────────────────────────────────────────
# AP RECOVERY
# ─────────────────────────────────────────
def recover_ap(player):
    rate=3 if player.status=="WOLHON" else 2
    player.ap=min(player.max_ap,player.ap+rate)

# ─────────────────────────────────────────
# INTRO
# ─────────────────────────────────────────
def intro():
    print("="*58)
    print("       TEXT MUD v2  —  TODAY'S DUNGEON")
    print(f"       Seed:{SEED}   Floors:{MAP_Z}")
    print("="*58)
    print("\nFaction:\n  1. Hunter\n  2. Vampire\n  3. Werebeast\n")
    while True:
        c=input("> ").strip()
        if c=="1": return "hunter"
        if c=="2": return "vampire"
        if c=="3": return "werebeast"
        print("1/2/3 only.")

# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
def main():
    os.system("cls" if os.name=="nt" else "clear")
    faction=intro()
    print("\nGenerating dungeon...")
    floors,all_rooms=generate_map(SEED)
    ground_items=place_items(all_rooms,SEED)
    enemies=place_enemies(all_rooms,SEED)
    explored=set()

    sr=all_rooms[0][0]; sx,sy=sr.center()
    player=Entity(name="Player",x=sx,y=sy,z=0,faction=faction)
    player.gun=ITEM_DB["glock17"]; player.ammo=15
    player.inventory+=[ITEM_DB["wolhon"],ITEM_DB["magazine"]]
    explored.add((sx,sy,0))

    os.system("cls" if os.name=="nt" else "clear")
    print(f"\n  2026-03-12  Floor 1")
    intros={"hunter":"  Glock in hand. One Wolhon in your pocket.",
            "vampire":"  You taste blood on the air.",
            "werebeast":"  The beast stirs. You smell hunters."}
    print(intros.get(faction,"")); print()

    turn=0; show_map=False

    while player.alive:
        print()
        print("─"*58)
        print(status_bar(player,turn))
        print("─"*58)

        if show_map:
            print(); print(render_map(floors,player,enemies,ground_items,explored))
            show_map=False

        print(context_bar(player,enemies,ground_items,floors,all_rooms[player.z]))
        print()

        # ── tick phases before input ──
        phase_msg=tick_phase(player,enemies,floors,ground_items,explored)
        if phase_msg: print(phase_msg)

        # if still busy (moving), skip input this turn
        if player.move_path or player.move_delay>0 or player.phase:
            # enemy turns
            for e in enemies:
                em=enemy_ai(e,player,floors)
                if em: print(em)
            if not player.alive: break
            recover_ap(player); turn+=1
            continue

        # INPUT
        try: raw=input("  > ").strip()
        except (KeyboardInterrupt,EOFError): print("\nGame terminated."); break
        if not raw: continue
        if raw.lower() in ("quit","exit","q"):
            print("You retreat."); break

        action,args=parse(raw)
        msg,want_map=do_action(action,args,player,floors,all_rooms,
                               enemies,ground_items,explored)
        if want_map: show_map=True
        elif msg: print(); print(msg)

        if not player.alive:
            print("\n"+"═"*40)
            print(f"  YOU HAVE FALLEN.  Survived {turn} turns.")
            print("═"*40); break

        # enemy turns
        for e in enemies:
            em=enemy_ai(e,player,floors)
            if em: print(em)
        if not player.alive:
            print("\n"+"═"*40)
            print(f"  YOU HAVE FALLEN.  Survived {turn} turns.")
            print("═"*40); break

        floor_alive=[e for e in enemies if e.alive and e.z==player.z]
        if not floor_alive and turn>0:
            print(f"\n  Floor {player.z+1} cleared.")

        recover_ap(player); turn+=1

if __name__=="__main__":
    main()
