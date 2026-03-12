"""
TEXT MUD GAME - Terminal Version
BSP Room Generation + Corridors + Multi-room dungeon
"""

import random
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MAP_W  = 60
MAP_H  = 30
MAP_Z  = 3
SEED   = 20260312

MIN_ROOM_W = 6
MIN_ROOM_H = 4
MAX_ROOM_W = 14
MAX_ROOM_H = 8

# ─────────────────────────────────────────
# TILES
# ─────────────────────────────────────────
T_WALL     = '#'
T_FLOOR    = '.'
T_CORRIDOR = ','
T_DOOR     = '+'
T_STAIR_U  = '<'
T_STAIR_D  = '>'

@dataclass
class Tile:
    type: str      = T_WALL
    material: str  = "stone"
    durability: int = 100
    explored: bool = False

# ─────────────────────────────────────────
# ITEMS
# ─────────────────────────────────────────
@dataclass
class Item:
    name: str
    itype: str        # weapon_gun / weapon_melee / consumable / ammo
    damage: int  = 0
    range_: int  = 1
    ammo: int    = 0
    effect: str  = ""
    weight: int  = 1

ITEM_DB: Dict[str, Item] = {
    "glock17":    Item("Glock 17",    "weapon_gun",   damage=18, range_=6,  ammo=15),
    "shotgun":    Item("Shotgun",     "weapon_gun",   damage=45, range_=3,  ammo=0),
    "rifle":      Item("Rifle",       "weapon_gun",   damage=35, range_=10, ammo=8),
    "knife":      Item("Knife",       "weapon_melee", damage=22, range_=1),
    "bat":        Item("Iron Bat",    "weapon_melee", damage=18, range_=1),
    "magazine":   Item("Magazine",    "ammo",         effect="ammo+15"),
    "shells":     Item("Shell Box",   "ammo",         effect="ammo+6"),
    "wolhon":     Item("Wolhon",      "consumable",   effect="ap_boost"),
    "bandage":    Item("Bandage",     "consumable",   effect="heal+30"),
    "grenade":    Item("Grenade",     "consumable",   damage=60, range_=3, effect="explosion"),
}

# ─────────────────────────────────────────
# ENTITY
# ─────────────────────────────────────────
@dataclass
class Entity:
    name: str
    x: int = 0
    y: int = 0
    z: int = 0
    hp: int = 100
    max_hp: int = 100
    ap: int = 20
    max_ap: int = 20
    faction: str = "hunter"
    alive: bool = True
    inventory: List[Item] = field(default_factory=list)
    gun: Optional[Item]   = None
    melee: Optional[Item] = None
    ammo: int   = 0
    status: str = ""
    transform_timer: int = 0
    wolhon_penalty: int  = 0
    cover: bool = False

# ─────────────────────────────────────────
# BSP ROOM GENERATION
# ─────────────────────────────────────────
@dataclass
class Room:
    x: int
    y: int
    w: int
    h: int

    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def inner_tiles(self):
        for ry in range(self.y + 1, self.y + self.h - 1):
            for rx in range(self.x + 1, self.x + self.w - 1):
                yield rx, ry

    def overlaps(self, other: 'Room', margin: int = 1) -> bool:
        return not (
            self.x + self.w + margin <= other.x or
            other.x + other.w + margin <= self.x or
            self.y + self.h + margin <= other.y or
            other.y + other.h + margin <= self.y
        )

def bsp_split(x, y, w, h, depth, rooms, rng):
    # guard: too small to place any room
    if w < MIN_ROOM_W + 2 or h < MIN_ROOM_H + 2:
        return None

    if depth == 0 or (w < MIN_ROOM_W * 2 + 3 and h < MIN_ROOM_H * 2 + 3):
        # leaf: place a room
        max_rw = min(MAX_ROOM_W, w - 2)
        max_rh = min(MAX_ROOM_H, h - 2)
        if max_rw < MIN_ROOM_W or max_rh < MIN_ROOM_H:
            return None
        rw = rng.randint(MIN_ROOM_W, max_rw)
        rh = rng.randint(MIN_ROOM_H, max_rh)
        max_ox = w - rw - 1
        max_oy = h - rh - 1
        if max_ox < 1 or max_oy < 1:
            return None
        rx = x + rng.randint(1, max_ox)
        ry = y + rng.randint(1, max_oy)
        room = Room(rx, ry, rw, rh)
        for existing in rooms:
            if room.overlaps(existing, margin=2):
                return None
        rooms.append(room)
        return room

    # try to split
    can_split_h = w >= MIN_ROOM_W * 2 + 3
    can_split_v = h >= MIN_ROOM_H * 2 + 3

    if can_split_h and (not can_split_v or w >= h):
        split = rng.randint(MIN_ROOM_W + 1, w - MIN_ROOM_W - 1)
        left  = bsp_split(x,         y, split,     h, depth - 1, rooms, rng)
        right = bsp_split(x + split, y, w - split, h, depth - 1, rooms, rng)
        return left or right
    elif can_split_v:
        split = rng.randint(MIN_ROOM_H + 1, h - MIN_ROOM_H - 1)
        top    = bsp_split(x, y,         w, split,     depth - 1, rooms, rng)
        bottom = bsp_split(x, y + split, w, h - split, depth - 1, rooms, rng)
        return top or bottom
    else:
        # fallback leaf
        max_rw = min(MAX_ROOM_W, w - 2)
        max_rh = min(MAX_ROOM_H, h - 2)
        if max_rw < MIN_ROOM_W or max_rh < MIN_ROOM_H:
            return None
        rw = rng.randint(MIN_ROOM_W, max_rw)
        rh = rng.randint(MIN_ROOM_H, max_rh)
        rx = x + rng.randint(1, max(1, w - rw - 1))
        ry = y + rng.randint(1, max(1, h - rh - 1))
        room = Room(rx, ry, rw, rh)
        rooms.append(room)
        return room

def carve_corridor(tiles, x1, y1, x2, y2, rng):
    """L-shaped corridor between two points"""
    if rng.random() < 0.5:
        # horizontal first, then vertical
        for cx in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < cx < MAP_W - 1 and 0 < y1 < MAP_H - 1:
                if tiles[y1][cx].type == T_WALL:
                    tiles[y1][cx] = Tile(T_CORRIDOR)
        for cy in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < x2 < MAP_W - 1 and 0 < cy < MAP_H - 1:
                if tiles[cy][x2].type == T_WALL:
                    tiles[cy][x2] = Tile(T_CORRIDOR)
    else:
        # vertical first, then horizontal
        for cy in range(min(y1, y2), max(y1, y2) + 1):
            if 0 < x1 < MAP_W - 1 and 0 < cy < MAP_H - 1:
                if tiles[cy][x1].type == T_WALL:
                    tiles[cy][x1] = Tile(T_CORRIDOR)
        for cx in range(min(x1, x2), max(x1, x2) + 1):
            if 0 < cx < MAP_W - 1 and 0 < y2 < MAP_H - 1:
                if tiles[y2][cx].type == T_WALL:
                    tiles[y2][cx] = Tile(T_CORRIDOR)

def generate_floor(floor_z: int, seed: int):
    rng = random.Random(seed + floor_z * 9999)
    tiles = [[Tile(T_WALL) for _ in range(MAP_W)] for _ in range(MAP_H)]
    rooms: List[Room] = []

    # BSP room gen
    bsp_split(1, 1, MAP_W - 2, MAP_H - 2, depth=4, rooms=rooms, rng=rng)

    # carve rooms
    for room in rooms:
        for rx, ry in room.inner_tiles():
            tiles[ry][rx] = Tile(T_FLOOR)
        # walls around room (already walls by default)

    # connect rooms with corridors (minimum spanning tree style)
    if len(rooms) >= 2:
        connected = [rooms[0]]
        unconnected = rooms[1:]
        while unconnected:
            best_dist = 99999
            best_a = best_b = None
            for a in connected:
                ax, ay = a.center()
                for b in unconnected:
                    bx, by = b.center()
                    d = abs(ax - bx) + abs(ay - by)
                    if d < best_dist:
                        best_dist = d
                        best_a, best_b = a, b
            ax, ay = best_a.center()
            bx, by = best_b.center()
            carve_corridor(tiles, ax, ay, bx, by, rng)
            connected.append(best_b)
            unconnected.remove(best_b)

        # add a few extra corridors for loops
        for _ in range(max(1, len(rooms) // 3)):
            a = rng.choice(rooms)
            b = rng.choice(rooms)
            if a is not b:
                ax, ay = a.center()
                bx, by = b.center()
                carve_corridor(tiles, ax, ay, bx, by, rng)

    # place doors at corridor-to-room transitions
    for y in range(1, MAP_H - 1):
        for x in range(1, MAP_W - 1):
            if tiles[y][x].type == T_CORRIDOR:
                neighbors_floor = sum(
                    1 for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]
                    if tiles[y+dy][x+dx].type == T_FLOOR
                )
                if neighbors_floor >= 2 and rng.random() < 0.3:
                    tiles[y][x] = Tile(T_DOOR)

    # place stairs
    if len(rooms) >= 2:
        stair_up_room   = rooms[0]
        stair_down_room = rooms[-1]
        ux, uy = stair_up_room.center()
        dx_, dy_ = stair_down_room.center()
        tiles[uy][ux]   = Tile(T_STAIR_U)
        tiles[dy_][dx_] = Tile(T_STAIR_D)

    return tiles, rooms

def generate_map(seed: int):
    floors = []
    all_rooms = []
    for z in range(MAP_Z):
        tiles, rooms = generate_floor(z, seed)
        floors.append(tiles)
        all_rooms.append(rooms)
    return floors, all_rooms

# ─────────────────────────────────────────
# ITEM PLACEMENT
# ─────────────────────────────────────────
@dataclass
class GroundItem:
    item: Item
    x: int
    y: int
    z: int

def place_items(all_rooms, seed: int) -> List[GroundItem]:
    rng = random.Random(seed + 1234)
    ground: List[GroundItem] = []

    item_pool = [
        "magazine", "magazine", "magazine",
        "shells",
        "knife", "bat",
        "wolhon", "wolhon",
        "bandage", "bandage",
        "grenade",
        "shotgun", "rifle",
    ]

    for z, rooms in enumerate(all_rooms):
        for room in rooms[1:]:  # skip start room
            if rng.random() < 0.6:
                item_key = rng.choice(item_pool)
                item = ITEM_DB[item_key]
                cx, cy = room.center()
                offset_x = rng.randint(-1, 1)
                offset_y = rng.randint(-1, 1)
                ground.append(GroundItem(item, cx + offset_x, cy + offset_y, z))

    return ground

# ─────────────────────────────────────────
# ENEMY PLACEMENT
# ─────────────────────────────────────────
def place_enemies(all_rooms, seed: int) -> List[Entity]:
    rng = random.Random(seed + 5678)
    enemies: List[Entity] = []
    eid = 1

    factions = ["werebeast", "vampire", "hunter"]
    faction_names = {
        "werebeast": ["Werebeast", "Wolf Soldier", "Beast"],
        "vampire":   ["Vampire", "Night Lord", "Blood Fiend"],
        "hunter":    ["Rogue Hunter", "Mercenary", "Traitor"],
    }

    for z, rooms in enumerate(all_rooms):
        for room in rooms[2:]:   # skip first 2 rooms
            if rng.random() < 0.55:
                faction = rng.choice(factions)
                ename   = rng.choice(faction_names[faction])
                cx, cy  = room.center()
                hp      = rng.randint(40, 100)
                e = Entity(
                    name    = f"{ename}_{eid}",
                    x=cx, y=cy, z=z,
                    hp=hp, max_hp=hp,
                    faction = faction,
                )
                # arm enemy
                if faction == "hunter":
                    e.gun   = ITEM_DB["glock17"]
                    e.ammo  = rng.randint(6, 15)
                elif faction == "werebeast":
                    e.melee = ITEM_DB["knife"]
                else:
                    e.melee = ITEM_DB["bat"]
                enemies.append(e)
                eid += 1

    return enemies

# ─────────────────────────────────────────
# FOV / EXPLORED
# ─────────────────────────────────────────
def get_visible(tiles_z, px, py, radius=8):
    """Simple raycasting FOV"""
    visible = set()
    visible.add((px, py))
    for angle_step in range(360):
        angle = angle_step * 3.14159 / 180
        import math
        dx = math.cos(angle)
        dy = math.sin(angle)
        rx, ry = float(px), float(py)
        for _ in range(radius):
            rx += dx
            ry += dy
            ix, iy = int(rx), int(ry)
            if not (0 <= ix < MAP_W and 0 <= iy < MAP_H):
                break
            visible.add((ix, iy))
            if tiles_z[iy][ix].type == T_WALL:
                break
    return visible

# ─────────────────────────────────────────
# ASCII MAP RENDER
# ─────────────────────────────────────────
def render_map(floors, player: Entity, entities: List[Entity],
               ground_items: List[GroundItem], explored: set, show_fov=True):
    tiles_z = floors[player.z]
    visible = get_visible(tiles_z, player.x, player.y, radius=8)

    # mark explored
    for vx, vy in visible:
        explored.add((vx, vy, player.z))

    lines = []
    lines.append(f"  Floor {player.z + 1}/{MAP_Z}  [seed:{SEED}]")

    for y in range(MAP_H):
        row = ""
        for x in range(MAP_W):
            pos3 = (x, y, player.z)
            in_vis = (x, y) in visible
            in_exp = pos3 in explored

            # player
            if x == player.x and y == player.y:
                row += "@"
                continue

            # entities
            drawn = False
            for e in entities:
                if e.alive and e.x == x and e.y == y and e.z == player.z:
                    if in_vis:
                        sym = {"werebeast": "W", "vampire": "V", "hunter": "H"}.get(e.faction, "E")
                        row += sym
                    else:
                        row += " " if not in_exp else tiles_z[y][x].type
                    drawn = True
                    break
            if drawn:
                continue

            # ground items
            item_drawn = False
            for gi in ground_items:
                if gi.x == x and gi.y == y and gi.z == player.z and in_vis:
                    row += "i"
                    item_drawn = True
                    break
            if item_drawn:
                continue

            if in_vis:
                row += tiles_z[y][x].type
            elif in_exp:
                t = tiles_z[y][x].type
                row += t if t in (T_WALL, T_FLOOR, T_CORRIDOR, T_DOOR, T_STAIR_U, T_STAIR_D) else " "
            else:
                row += " "
        lines.append(row)

    lines.append("")
    lines.append("@ you  W werebeast  V vampire  H hunter  i item")
    lines.append(f"< up   > down   + door   , corridor")
    return "\n".join(lines)

# ─────────────────────────────────────────
# SURROUNDINGS TEXT
# ─────────────────────────────────────────
def describe_surroundings(floors, player: Entity, entities: List[Entity],
                           ground_items: List[GroundItem]) -> str:
    tiles_z = floors[player.z]
    px, py, pz = player.x, player.y, player.z
    desc = []

    DIRS = {"north": (0,-1), "south": (0,1), "east": (1,0), "west": (-1,0)}
    blocked = []
    for dname, (dx, dy) in DIRS.items():
        nx, ny = px + dx, py + dy
        if 0 <= nx < MAP_W and 0 <= ny < MAP_H:
            t = tiles_z[ny][nx].type
            if t == T_WALL:
                blocked.append(dname)
            elif t == T_DOOR:
                desc.append(f"There is a door to the {dname}.")
            elif t in (T_STAIR_U, T_STAIR_D):
                direction = "up" if t == T_STAIR_U else "down"
                desc.append(f"Stairs leading {direction} are to the {dname}.")

    if blocked:
        desc.append(f"Walls block: {', '.join(blocked)}.")

    # entities nearby
    for e in entities:
        if not e.alive or e.z != pz:
            continue
        dist = abs(e.x - px) + abs(e.y - py)
        if dist <= 8:
            dir_txt = dir_text(px, py, e.x, e.y)
            hp_pct  = e.hp / e.max_hp
            cond    = "critically wounded" if hp_pct < 0.25 else "wounded" if hp_pct < 0.6 else ""
            cond_s  = f" ({cond})" if cond else ""
            desc.append(f"{e.name}{cond_s} is {dist} tiles to the {dir_txt}.")

    # ground items nearby
    for gi in ground_items:
        if gi.z != pz:
            continue
        dist = abs(gi.x - px) + abs(gi.y - py)
        if dist <= 3:
            dir_txt = dir_text(px, py, gi.x, gi.y)
            desc.append(f"You see a {gi.item.name} to the {dir_txt}.")

    # stairs underfoot
    cur = tiles_z[py][px]
    if cur.type == T_STAIR_U:
        desc.append("You are standing on stairs leading UP (<).")
    elif cur.type == T_STAIR_D:
        desc.append("You are standing on stairs leading DOWN (>).")

    return "\n".join(desc) if desc else "The area is quiet."

def dir_text(fx, fy, tx, ty) -> str:
    dx, dy = tx - fx, ty - fy
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"

# ─────────────────────────────────────────
# COMMAND PARSER
# ─────────────────────────────────────────
DIR_MAP = {
    "n": (0,-1,0), "north": (0,-1,0),
    "s": (0, 1,0), "south": (0, 1,0),
    "e": (1, 0,0), "east":  (1, 0,0),
    "w": (-1,0,0), "west":  (-1,0,0),
    "u": (0, 0,1), "up":    (0, 0,1),
    "d": (0, 0,-1),"down":  (0, 0,-1),
}

def parse(cmd: str):
    parts = cmd.strip().lower().split()
    if not parts:
        return "", []
    return parts[0], parts[1:]

# ─────────────────────────────────────────
# ACTION HANDLER
# ─────────────────────────────────────────
def do_action(action, args, player: Entity,
              floors, entities: List[Entity],
              ground_items: List[GroundItem],
              explored: set) -> Tuple[str, bool]:
    """Returns (message, show_map_after)"""
    tiles_z = floors[player.z]

    # ── MOVEMENT ──
    if action in DIR_MAP:
        dx, dy, dz = DIR_MAP[action]
        nx, ny, nz = player.x + dx, player.y + dy, player.z + dz

        if dz != 0:
            # stair movement
            cur_tile = tiles_z[player.y][player.x].type
            if dz == 1 and cur_tile != T_STAIR_U:
                return "No stairs going up here.", False
            if dz == -1 and cur_tile != T_STAIR_D:
                return "No stairs going down here.", False
            if not (0 <= nz < MAP_Z):
                return "No more floors in that direction.", False
            player.z = nz
            player.ap -= 2
            return f"You climb the stairs to floor {nz + 1}.", False

        if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
            return "You can't go that way.", False

        target_tile = tiles_z[ny][nx].type
        if target_tile == T_WALL:
            return "A wall blocks your way.", False

        if player.ap < 1:
            return "You're too exhausted to move.", False

        player.x, player.y = nx, ny
        player.cover = False
        player.ap -= 1
        explored.add((nx, ny, player.z))

        # door flavour
        if target_tile == T_DOOR:
            return "You push through the door.", False
        if target_tile in (T_STAIR_U, T_STAIR_D):
            direction = "up" if target_tile == T_STAIR_U else "down"
            return f"You step onto stairs leading {direction}. (use 'u'/'d' to climb)", False

        moves = [
            "You move forward, footsteps echoing.",
            "You advance cautiously.",
            "You step into the darkness.",
        ]
        return random.choice(moves), False

    # ── RUN ──
    elif action == "run" and args:
        dir_ = args[0]
        if dir_ not in DIR_MAP:
            return "Unknown direction.", False
        if player.ap < 3:
            return "Not enough AP to run.", False
        dx, dy, _ = DIR_MAP[dir_]
        moved = 0
        for _ in range(3):
            nx, ny = player.x + dx, player.y + dy
            if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
                break
            if tiles_z[ny][nx].type == T_WALL:
                break
            player.x, player.y = nx, ny
            explored.add((nx, ny, player.z))
            moved += 1
        player.ap -= 3
        player.cover = False
        return f"You sprint {moved} tiles to the {dir_}.", False

    # ── LOOK ──
    elif action in ("look", "l"):
        if player.ap < 1:
            return "Too exhausted to look around.", False
        player.ap -= 1
        return describe_surroundings(floors, player, entities, ground_items), False

    # ── LISTEN ──
    elif action in ("listen", "li"):
        if player.ap < 1:
            return "Not enough AP.", False
        player.ap -= 1
        sounds = []
        for e in entities:
            if not e.alive or e.z != player.z:
                continue
            dist = abs(e.x - player.x) + abs(e.y - player.y)
            if dist <= 10:
                dir_txt = dir_text(player.x, player.y, e.x, e.y)
                sounds.append(f"  Footsteps to the {dir_txt}, ~{dist} tiles away.")
        if not sounds:
            sounds = ["  Water dripping. Wind. Nothing else."]
        return "You hold your breath and listen...\n" + "\n".join(sounds), False

    # ── SHOOT ──
    elif action in ("shoot", "sh", "fire"):
        if not player.gun:
            return "No firearm equipped.", False
        if player.ammo <= 0:
            return "Empty. Reload first. (reload)", False
        if player.ap < 3:
            return "Not enough AP to shoot.", False

        # direction suppression fire
        if args and args[0] in DIR_MAP and len(args) == 1:
            player.ap -= 1
            player.ammo -= 1
            return f"Suppression fire {args[0]}. Ammo: {player.ammo}", False

        # targeted shot
        target = resolve_target(args, player, entities)
        if not target:
            return "No target found.", False

        dist = abs(target.x - player.x) + abs(target.y - player.y)
        if dist > player.gun.range_:
            return f"Out of range. ({dist} tiles / max {player.gun.range_})", False

        # calc hit
        hit_base = 0.85 - dist * 0.07
        if player.cover:
            hit_base += 0.1
        shots = int(args[-1]) if args and args[-1].isdigit() else 2
        shots = min(shots, 3)

        msgs = []
        for i in range(shots):
            if player.ammo <= 0:
                msgs.append("Clip empty mid-burst!")
                break
            player.ammo -= 1
            player.ap -= max(1, 3 // shots)
            if random.random() < hit_base:
                dmg = player.gun.damage + random.randint(-4, 6)
                target.hp -= dmg
                msgs.append(f"Hit! {target.name} -{dmg}HP  (remaining: {max(0,target.hp)})")
                if target.hp <= 0:
                    target.alive = False
                    msgs.append(f"{target.name} goes down.")
                    break
            else:
                msgs.append("Miss.")

        return "\n".join(msgs), False

    # ── ATTACK (melee) ──
    elif action in ("attack", "a", "stab", "slash"):
        target = resolve_target(args, player, entities)
        if not target:
            target = closest_entity(player, entities)
        if not target:
            return "Nothing to attack.", False
        dist = abs(target.x - player.x) + abs(target.y - player.y)
        if dist > 1:
            return f"Too far away. ({dist} tiles)", False
        if player.ap < 2:
            return "Not enough AP.", False

        wpn  = player.melee
        base = wpn.damage if wpn else 8
        dmg  = base + random.randint(-3, 4)
        target.hp -= dmg
        player.ap -= 2
        msgs = [f"You strike {target.name} for {dmg} damage! (HP: {max(0,target.hp)})"]
        if target.hp <= 0:
            target.alive = False
            msgs.append(f"{target.name} is dead.")
        return "\n".join(msgs), False

    # ── KNOCKKICK ──
    elif action in ("kk", "knockkick"):
        target = closest_entity(player, entities)
        if not target:
            return "No target.", False
        dist = abs(target.x - player.x) + abs(target.y - player.y)
        if dist > 1:
            return f"Too far to kick. ({dist} tiles)", False
        if player.ap < 2:
            return "Not enough AP.", False

        player.ap -= 2
        dmg = random.randint(8, 16)
        target.hp -= dmg
        # push back 2 tiles
        ddx = target.x - player.x
        ddy = target.y - player.y
        px2, py2 = target.x + ddx * 2, target.y + ddy * 2
        if (0 < px2 < MAP_W and 0 < py2 < MAP_H and
                tiles_z[py2][px2].type not in (T_WALL,)):
            target.x, target.y = px2, py2
            push = f"Knocked back 2 tiles!"
        else:
            push = f"Slammed into the wall!"

        msgs = [f"KNOCKKICK! {dmg} damage. {push}"]

        # combo: kk shoot
        if args and args[0] in ("shoot", "sh") and player.gun and player.ammo > 0:
            player.ap -= 2
            player.ammo -= 1
            dmg2 = player.gun.damage + random.randint(-3, 5)
            target.hp -= dmg2
            msgs.append(f"Follow-up shot! {dmg2} damage! (HP: {max(0,target.hp)})")
            if target.hp <= 0:
                target.alive = False
                msgs.append(f"{target.name} is down.")

        return "\n".join(msgs), False

    # ── RELOAD ──
    elif action in ("reload", "r"):
        if player.ap < 3:
            return "Not enough AP.", False
        for item in player.inventory:
            if item.itype == "ammo":
                player.inventory.remove(item)
                if "ammo+" in item.effect:
                    n = int(item.effect.split("+")[1])
                    player.ammo = min(30, player.ammo + n)
                player.ap -= 3
                return f"Reloaded. Ammo: {player.ammo}  (you were vulnerable)", False
        return "No ammo in inventory.", False

    # ── COVER ──
    elif action in ("cover", "c"):
        if player.ap < 1:
            return "Not enough AP.", False
        player.ap -= 1
        player.cover = True
        return "You press yourself behind cover. (+hit chance bonus)", False

    # ── DODGE ──
    elif action in ("dodge", "dg"):
        if player.ap < 2:
            return "Not enough AP.", False
        player.ap -= 2
        player.cover = False
        return "You roll to the side.", False

    # ── TAKE ──
    elif action in ("take", "t", "pick", "grab"):
        if not args:
            return "Take what?", False
        name = " ".join(args).lower()
        for gi in ground_items[:]:
            if gi.z == player.z and name in gi.item.name.lower():
                dist = abs(gi.x - player.x) + abs(gi.y - player.y)
                if dist > 1:
                    return f"Too far to pick up. ({dist} tiles)", False
                player.inventory.append(gi.item)
                ground_items.remove(gi)
                player.ap -= 1
                # auto-equip
                if gi.item.itype == "weapon_gun" and not player.gun:
                    player.gun  = gi.item
                    player.ammo = gi.item.ammo
                    return f"Picked up and equipped {gi.item.name}.", False
                if gi.item.itype == "weapon_melee" and not player.melee:
                    player.melee = gi.item
                    return f"Picked up {gi.item.name} in off-hand.", False
                return f"Picked up {gi.item.name}.", False
        return "No such item nearby.", False

    # ── USE ──
    elif action in ("use", "u"):
        if not args:
            return "Use what?", False
        name = " ".join(args).lower()
        for item in player.inventory:
            if name in item.name.lower():
                if item.effect == "ap_boost":
                    player.inventory.remove(item)
                    player.ap = min(player.max_ap, player.ap + 12)
                    player.wolhon_penalty += 3
                    player.status = "WOLHON"
                    return ("Bitter taste hits your tongue.\n"
                            "Heart rate doubles. Vision turns crimson.\n"
                            f"AP +12 / Status: WOLHON / Next-run penalty: -{player.wolhon_penalty}AP"), False
                elif item.effect.startswith("heal"):
                    n = int(item.effect.split("+")[1])
                    player.hp = min(player.max_hp, player.hp + n)
                    player.inventory.remove(item)
                    return f"Bandaged wounds. HP +{n}  (HP: {player.hp})", False
                elif item.effect == "explosion":
                    player.inventory.remove(item)
                    player.ap -= 2
                    msgs = ["You pull the pin and throw!"]
                    for e in entities:
                        if not e.alive or e.z != player.z:
                            continue
                        dist = abs(e.x - player.x) + abs(e.y - player.y)
                        if dist <= item.range_:
                            dmg = item.damage - dist * 8
                            e.hp -= dmg
                            msgs.append(f"  {e.name} hit for {dmg}! (HP:{max(0,e.hp)})")
                            if e.hp <= 0:
                                e.alive = False
                                msgs.append(f"  {e.name} eliminated.")
                    return "\n".join(msgs), False
                elif "ammo+" in item.effect:
                    n = int(item.effect.split("+")[1])
                    player.ammo = min(30, player.ammo + n)
                    player.inventory.remove(item)
                    return f"Loaded {n} rounds. Ammo: {player.ammo}", False
        return f"'{name}' not in inventory.", False

    # ── EQUIP ──
    elif action in ("equip", "eq"):
        if not args:
            return "Equip what?", False
        name = " ".join(args).lower()
        for item in player.inventory:
            if name in item.name.lower():
                if item.itype == "weapon_gun":
                    player.gun  = item
                    player.ammo = item.ammo
                    player.ap  -= 2
                    return f"Equipped {item.name}. AP -2", False
                if item.itype == "weapon_melee":
                    player.melee = item
                    player.ap   -= 1
                    return f"Equipped {item.name} in off-hand.", False
        return f"'{name}' not found.", False

    # ── INVENTORY ──
    elif action in ("inv", "i", "inventory", "bag"):
        lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        lines.append(f"Primary : {player.gun.name if player.gun else 'none'}  (ammo: {player.ammo})")
        lines.append(f"Off-hand: {player.melee.name if player.melee else 'none'}")
        lines.append("Carried :")
        if not player.inventory:
            lines.append("  (empty)")
        for it in player.inventory:
            lines.append(f"  {it.name}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines), False

    # ── WEAPONS ──
    elif action in ("weapons", "wp", "arms"):
        lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        if player.gun:
            g = player.gun
            lines.append(f"Primary : {g.name}")
            lines.append(f"  Ammo  : {player.ammo}/30")
            lines.append(f"  Damage: {g.damage}  Range: {g.range_}")
        else:
            lines.append("Primary : none")
        if player.melee:
            m = player.melee
            lines.append(f"Off-hand: {m.name}  Damage: {m.damage}")
        else:
            lines.append("Off-hand: none")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines), False

    # ── STATUS ──
    elif action in ("status", "st", "me"):
        lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
        lines.append(f"HP   : {player.hp}/{player.max_hp}")
        lines.append(f"AP   : {player.ap}/{player.max_ap}")
        lines.append(f"Ammo : {player.ammo}")
        lines.append(f"Pos  : ({player.x},{player.y}) Floor {player.z+1}")
        lines.append(f"Cover: {'yes' if player.cover else 'no'}")
        if player.status:
            lines.append(f"Status: {player.status}")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines), False

    # ── REST ──
    elif action in ("rest", "wait", "z"):
        rec = random.randint(4, 7)
        player.ap = min(player.max_ap, player.ap + rec)
        return f"You catch your breath. AP +{rec}  (vulnerable)", False

    # ── BREAK (wall/floor) ──
    elif action in ("break", "smash", "br"):
        dir_ = args[0] if args else None
        if not dir_ or dir_ not in DIR_MAP:
            return "Break which direction?", False
        dx, dy, _ = DIR_MAP[dir_]
        nx, ny = player.x + dx, player.y + dy
        if not (0 <= nx < MAP_W and 0 <= ny < MAP_H):
            return "Nothing there.", False
        tile = tiles_z[ny][nx]
        if tile.type != T_WALL:
            return "Nothing solid to break.", False
        if player.ap < 5:
            return "Not enough AP.", False
        tile.durability -= 40
        player.ap -= 5
        if tile.durability <= 0:
            tile.type = T_CORRIDOR
            explored.add((nx, ny, player.z))
            return "You break through! New passage opened.", False
        return f"The wall cracks. ({tile.durability}% durability left)", False

    # ── MAP ──
    elif action in ("map", "m"):
        return None, True   # signal to show map

    # ── HELP ──
    elif action in ("help", "?", "h"):
        return (
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "MOVEMENT  n s e w u d\n"
            "          run [dir]  (3 tiles, AP-3)\n"
            "COMBAT    shoot [target] [shots]\n"
            "          shoot [dir]   (suppression)\n"
            "          attack [target]\n"
            "          kk            (knockkick)\n"
            "          kk shoot      (combo)\n"
            "          reload  cover  dodge\n"
            "ITEMS     take [item]   use [item]\n"
            "          equip [item]\n"
            "INFO      look  listen  inv  weapons\n"
            "          status  map\n"
            "MISC      rest  break [dir]  quit\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ), False

    else:
        return f"Unknown command: '{action}'  (? for help)", False

# ─────────────────────────────────────────
# TARGET RESOLUTION
# ─────────────────────────────────────────
def resolve_target(args, player: Entity, entities: List[Entity]) -> Optional[Entity]:
    if not args:
        return closest_entity(player, entities)
    name = args[0].lower()
    for e in entities:
        if e.alive and e.z == player.z and name in e.name.lower():
            return e
    return closest_entity(player, entities)

def closest_entity(player: Entity, entities: List[Entity]) -> Optional[Entity]:
    alive = [e for e in entities if e.alive and e.z == player.z]
    if not alive:
        return None
    return min(alive, key=lambda e: abs(e.x - player.x) + abs(e.y - player.y))

# ─────────────────────────────────────────
# ENEMY AI TURN
# ─────────────────────────────────────────
def enemy_ai(entity: Entity, player: Entity, floors) -> str:
    if not entity.alive or entity.z != player.z:
        return ""

    tiles_z = floors[entity.z]
    msgs = []
    dist = abs(entity.x - player.x) + abs(entity.y - player.y)

    # transform logic
    if entity.status == "TRANSFORMING":
        entity.transform_timer -= 1
        msgs.append(f"  [{entity.name} transforming... {entity.transform_timer}]")
        if entity.transform_timer <= 0:
            entity.status = "TRANSFORMED"
            entity.max_hp = int(entity.max_hp * 1.5)
            entity.hp = min(entity.hp + 20, entity.max_hp)
            msgs.append(f"  [{entity.name} TRANSFORMATION COMPLETE. Power surges.]")
        return "\n".join(msgs)

    # werebeast: transform when low HP
    if (entity.faction == "werebeast" and
            entity.hp < entity.max_hp * 0.35 and
            entity.status == ""):
        entity.status = "TRANSFORMING"
        entity.transform_timer = 3
        msgs.append(f"  {entity.name} begins to transform!")
        return "\n".join(msgs)

    # move toward player
    if dist > 1:
        dx = 1 if player.x > entity.x else (-1 if player.x < entity.x else 0)
        dy = 1 if player.y > entity.y else (-1 if player.y < entity.y else 0)
        # try primary direction
        moved = False
        for adx, ady in [(dx, dy), (dx, 0), (0, dy)]:
            if adx == 0 and ady == 0:
                continue
            nx, ny = entity.x + adx, entity.y + ady
            if (0 < nx < MAP_W and 0 < ny < MAP_H and
                    tiles_z[ny][nx].type not in (T_WALL,)):
                entity.x, entity.y = nx, ny
                moved = True
                break
        if moved:
            msgs.append(f"  {entity.name} closes in. ({dist-1} tiles)")
        return "\n".join(msgs)

    # attack at range 1
    base = 20 if entity.status == "TRANSFORMED" else 12
    if entity.faction == "hunter" and entity.gun and entity.ammo > 0:
        base = entity.gun.damage
        entity.ammo -= 1
    dmg = base + random.randint(-3, 6)
    player.hp -= dmg

    # player cover reduces damage
    if player.cover:
        dmg = max(1, dmg // 2)
        msgs.append(f"  {entity.name} attacks! Cover reduces damage. HP -{dmg}")
    else:
        msgs.append(f"  {entity.name} attacks! HP -{dmg}  (you: {max(0,player.hp)})")

    if player.hp <= 0:
        player.alive = False

    return "\n".join(msgs)

# ─────────────────────────────────────────
# AP RECOVERY
# ─────────────────────────────────────────
def recover_ap(player: Entity):
    rate = 3 if player.status == "WOLHON" else 2
    player.ap = min(player.max_ap, player.ap + rate)

# ─────────────────────────────────────────
# STATUS BAR
# ─────────────────────────────────────────
def status_bar(player: Entity, turn: int) -> str:
    hp_f  = player.hp / player.max_hp
    ap_f  = player.ap / player.max_ap
    hp_b  = int(hp_f * 12)
    ap_b  = int(ap_f * 12)
    hp_col = "!" if hp_f < 0.3 else ""
    gun   = player.gun.name if player.gun else "none"
    melee = player.melee.name if player.melee else "none"
    sta   = f" [{player.status}]" if player.status else ""
    cov   = " [COVER]" if player.cover else ""
    return (
        f"HP {'█'*hp_b}{'░'*(12-hp_b)} {player.hp}/{player.max_hp}{hp_col}  "
        f"AP {'█'*ap_b}{'░'*(12-ap_b)} {player.ap}/{player.max_ap}  "
        f"Ammo:{player.ammo}  "
        f"{gun} / {melee}{sta}{cov}  "
        f"T:{turn}  F:{player.z+1}"
    )

def hint_bar(player: Entity, entities: List[Entity]) -> str:
    alive = [e for e in entities if e.alive and e.z == player.z]
    if not alive:
        return ">> n/s/e/w  look  listen  inv  weapons  map"
    dist = min(abs(e.x - player.x) + abs(e.y - player.y) for e in alive)
    if dist <= 1:
        return ">> attack  kk  kk shoot  dodge  cover  run [dir]  use wolhon"
    elif dist <= 5:
        return ">> shoot [target]  shoot [dir](suppress)  cover  reload  look"
    else:
        return ">> n/s/e/w  look  listen  shoot [target]  map  inv"

# ─────────────────────────────────────────
# INTRO / FACTION SELECT
# ─────────────────────────────────────────
def intro() -> str:
    print("=" * 58)
    print("         TEXT MUD  —  TODAY'S DUNGEON")
    print(f"         Seed: {SEED}   Floor 1 of {MAP_Z}")
    print("=" * 58)
    print()
    print("Choose your faction:")
    print("  1. Hunter    (human, firearms, Wolhon)")
    print("  2. Vampire   (blood powers, night vision)")
    print("  3. Werebeast (transform, melee beast)")
    print()
    while True:
        c = input("> ").strip()
        if c == "1": return "hunter"
        if c == "2": return "vampire"
        if c == "3": return "werebeast"
        print("Enter 1, 2, or 3.")

# ─────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────
def main():
    os.system("cls" if os.name == "nt" else "clear")
    faction = intro()

    print("\nGenerating dungeon...")
    floors, all_rooms = generate_map(SEED)
    ground_items      = place_items(all_rooms, SEED)
    enemies           = place_enemies(all_rooms, SEED)
    explored          = set()

    # player start: center of first room, floor 0
    start_room = all_rooms[0][0]
    sx, sy = start_room.center()

    player = Entity(
        name="Player",
        x=sx, y=sy, z=0,
        faction=faction,
    )
    player.gun   = ITEM_DB["glock17"]
    player.ammo  = 15
    player.inventory.append(ITEM_DB["wolhon"])
    player.inventory.append(ITEM_DB["magazine"])
    explored.add((sx, sy, 0))

    # INTRO TEXT
    os.system("cls" if os.name == "nt" else "clear")
    print()
    print("  2026-03-12  00:00")
    print("  Somewhere beneath Seoul.")
    print()
    if faction == "hunter":
        print("  Damp air fills your lungs.")
        print("  The Glock 17 in your grip feels familiar.")
        print("  A single Wolhon tablet rolls in your pocket.")
    elif faction == "vampire":
        print("  You taste the air. Blood, rust, fear.")
        print("  Your eyes adjust instantly to the dark.")
    else:
        print("  Your skin prickles. The beast stirs inside.")
        print("  You smell hunters. Guns. Gunpowder.")
    print()

    turn       = 0
    show_map   = False

    while player.alive:
        # STATUS
        print()
        print("─" * 58)
        print(status_bar(player, turn))
        print("─" * 58)

        # MAP
        if show_map:
            print()
            print(render_map(floors, player, enemies, ground_items, explored))
            show_map = False

        # HINT
        print(hint_bar(player, enemies))
        print()

        # INPUT
        try:
            raw = input("  > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGame terminated.")
            break

        if not raw:
            continue
        if raw.lower() in ("quit", "exit", "q"):
            print("You retreat from the dungeon.")
            break

        action, args = parse(raw)
        msg, want_map = do_action(
            action, args, player,
            floors, enemies, ground_items, explored
        )

        if want_map:
            show_map = True
        elif msg:
            print()
            print(msg)

        # death check
        if not player.alive:
            print()
            print("═" * 40)
            print("  YOU HAVE FALLEN.")
            print(f"  Survived {turn} turns.")
            print(f"  Enemies eliminated: {sum(1 for e in enemies if not e.alive)}")
            print("═" * 40)
            break

        # all enemies on current floor dead?
        floor_enemies = [e for e in enemies if e.z == player.z and e.alive]
        if not floor_enemies and turn > 0:
            print()
            print(f"  Floor {player.z+1} cleared.")

        # ENEMY TURNS
        enemy_msgs = []
        for e in enemies:
            em = enemy_ai(e, player, floors)
            if em:
                enemy_msgs.append(em)
        if enemy_msgs:
            print()
            for em in enemy_msgs:
                print(em)

        # death after enemy turn
        if not player.alive:
            print()
            print("═" * 40)
            print("  YOU HAVE FALLEN.")
            print(f"  Survived {turn} turns.")
            print("═" * 40)
            break

        recover_ap(player)
        turn += 1

if __name__ == "__main__":
    main()
