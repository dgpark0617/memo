"""
텍스트 머드 게임 - 터미널 버전
오늘 플레이 씬 기반 MVP
"""

import random
import time
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
MAP_W = 10
MAP_H = 10
MAP_Z = 3
SEED  = 20260312

# ─────────────────────────────────────────
# 타일
# ─────────────────────────────────────────
TILE_FLOOR = "."
TILE_WALL  = "#"
TILE_VOID  = " "

@dataclass
class Tile:
    type: str = TILE_FLOOR
    durability: int = 100
    material: str = "stone"

# ─────────────────────────────────────────
# 아이템
# ─────────────────────────────────────────
@dataclass
class Item:
    name: str
    item_type: str        # weapon_gun / weapon_melee / consumable
    damage: int = 0
    range_: int = 1
    ammo: int = 0
    effect: str = ""

ITEMS = {
    "글록17":   Item("글록17",   "weapon_gun",   damage=18, range_=5, ammo=15),
    "나이프":   Item("나이프",   "weapon_melee", damage=22, range_=1),
    "산탄총":   Item("산탄총",   "weapon_gun",   damage=40, range_=3, ammo=0),
    "탄창":     Item("탄창",     "consumable",   effect="ammo+15"),
    "월혼":     Item("월혼",     "consumable",   effect="ap_boost"),
}

# ─────────────────────────────────────────
# 엔티티 (플레이어 / 몬스터 공통)
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
    inventory: list = field(default_factory=list)
    equipped_gun: Optional[Item] = None
    equipped_melee: Optional[Item] = None
    ammo: int = 15
    status: str = ""          # "변신중", "각성" 등
    transform_timer: int = 0  # 변신 카운트다운
    ap_penalty: int = 0       # 월혼 누적 패널티

# ─────────────────────────────────────────
# 맵 생성
# ─────────────────────────────────────────
def generate_map(seed: int):
    random.seed(seed)
    tiles = [[[Tile(TILE_WALL) for _ in range(MAP_W)]
               for _ in range(MAP_H)]
               for _ in range(MAP_Z)]

    # 각 층 기본 방 생성
    for z in range(MAP_Z):
        for y in range(1, MAP_H - 1):
            for x in range(1, MAP_W - 1):
                if random.random() > 0.25:
                    tiles[z][y][x] = Tile(TILE_FLOOR)

    # 시작 지점 주변 강제 오픈
    for z in range(MAP_Z):
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = 5 + dy, 3 + dx
                if 0 < ny < MAP_H-1 and 0 < nx < MAP_W-1:
                    tiles[z][ny][nx] = Tile(TILE_FLOOR)

    return tiles

# ─────────────────────────────────────────
# 아스키 맵 출력 (탐색된 영역만)
# ─────────────────────────────────────────
def render_map(tiles, player: Entity, entities: list, explored: set):
    lines = []
    z = player.z
    lines.append(f"[{z+1}층 / 씨드: {SEED}]")
    lines.append("┌" + "─" * (MAP_W * 2) + "┐")

    for y in range(MAP_H):
        row = "│"
        for x in range(MAP_W):
            pos = (x, y, z)
            # 플레이어
            if x == player.x and y == player.y:
                row += "@ "
                continue
            # 다른 엔티티
            found = False
            for e in entities:
                if e.x == x and e.y == y and e.z == z and e.alive:
                    if e.faction == "werebeast":
                        row += "W "
                    elif e.faction == "vampire":
                        row += "V "
                    else:
                        row += "E "
                    found = True
                    break
            if found:
                continue
            # 시체
            if pos in explored and tiles[z][y][x].type == TILE_FLOOR:
                row += ". "
            elif pos in explored and tiles[z][y][x].type == TILE_WALL:
                row += "# "
            else:
                row += "? "
        row += "│"
        lines.append(row)

    lines.append("└" + "─" * (MAP_W * 2) + "┘")

    # 범례
    lines.append(f"@ 나 (HP:{player.hp}/{player.max_hp})")
    for e in entities:
        if e.alive:
            lines.append(f"W {e.name} (HP:{e.hp}, {e.status})")

    return "\n".join(lines)

# ─────────────────────────────────────────
# 주변 텍스트 묘사
# ─────────────────────────────────────────
def describe_surroundings(tiles, player: Entity, entities: list, explored: set) -> str:
    z, y, x = player.z, player.y, player.x
    desc = []

    # 방향별 감지
    directions = {
        "북": (x,   y-1, z),
        "남": (x,   y+1, z),
        "동": (x+1, y,   z),
        "서": (x-1, y,   z),
        "위": (x,   y,   z+1),
        "아래": (x, y,   z-1),
    }

    for dir_name, (dx, dy, dz) in directions.items():
        if not (0 <= dz < MAP_Z and 0 <= dy < MAP_H and 0 <= dx < MAP_W):
            continue
        tile = tiles[dz][dy][dx]
        if tile.type == TILE_WALL:
            desc.append(f"{dir_name}쪽은 벽이다.")
        else:
            explored.add((dx, dy, dz))

    # 근처 엔티티 감지
    for e in entities:
        if not e.alive:
            continue
        dist = abs(e.x - x) + abs(e.y - y) + abs(e.z - z)
        if dist <= 3:
            dir_text = get_direction_text(x, y, e.x, e.y)
            desc.append(f"{dir_text} {dist}칸 거리에서 {e.name}의 기척이 느껴진다.")

    return "\n".join(desc) if desc else "주위는 조용하다."

def get_direction_text(fx, fy, tx, ty) -> str:
    dx = tx - fx
    dy = ty - fy
    if abs(dx) > abs(dy):
        return "동쪽" if dx > 0 else "서쪽"
    else:
        return "남쪽" if dy > 0 else "북쪽"

# ─────────────────────────────────────────
# 명령어 파서
# ─────────────────────────────────────────
DIR_MAP = {
    "북": (0, -1, 0), "남": (0, 1, 0),
    "동": (1, 0, 0),  "서": (-1, 0, 0),
    "위": (0, 0, 1),  "아래": (0, 0, -1),
    "n": (0,-1,0), "s": (0,1,0),
    "e": (1,0,0),  "w": (-1,0,0),
    "u": (0,0,1),  "d": (0,0,-1),
}

def parse_command(cmd: str):
    parts = cmd.strip().split()
    if not parts:
        return None, []
    action = parts[0]
    args = parts[1:]
    return action, args

# ─────────────────────────────────────────
# 액션 처리
# ─────────────────────────────────────────
def process_action(action, args, player: Entity, tiles, entities: list, explored: set) -> str:
    result = []

    # ── 이동 ──
    if action in DIR_MAP:
        dx, dy, dz = DIR_MAP[action]
        nx, ny, nz = player.x + dx, player.y + dy, player.z + dz
        if not (0 <= nz < MAP_Z and 0 <= ny < MAP_H and 0 <= nx < MAP_W):
            return "더 이상 갈 수 없다."
        if tiles[nz][ny][nx].type == TILE_WALL:
            return "벽이 막혀있다."
        if player.ap < 1:
            return "AP가 부족해 움직일 수 없다. 지쳐버렸다."
        player.x, player.y, player.z = nx, ny, nz
        player.ap -= 1
        explored.add((nx, ny, nz))
        move_texts = [
            "발걸음을 옮긴다.",
            "어둠 속으로 한 발 내딛는다.",
            "조심스럽게 이동한다.",
        ]
        return random.choice(move_texts)

    # ── 달려 ──
    elif action == "달려" and args:
        dir_ = args[0]
        if dir_ not in DIR_MAP:
            return "방향을 알 수 없다."
        if player.ap < 3:
            return "AP가 부족해 달릴 수 없다."
        dx, dy, dz = DIR_MAP[dir_]
        moved = 0
        for _ in range(2):
            nx, ny, nz = player.x + dx, player.y + dy, player.z + dz
            if not (0 <= nz < MAP_Z and 0 <= ny < MAP_H and 0 <= nx < MAP_W):
                break
            if tiles[nz][ny][nx].type == TILE_WALL:
                break
            player.x, player.y, player.z = nx, ny, nz
            explored.add((nx, ny, nz))
            moved += 1
        player.ap -= 3
        return f"전력으로 {dir_}쪽으로 {moved}칸 달렸다."

    # ── 보기 ──
    elif action in ("보기", "l", "look"):
        if player.ap < 1:
            return "너무 지쳐서 주위를 살필 여유가 없다."
        player.ap -= 1
        desc = describe_surroundings(tiles, player, entities, explored)
        return f"주위를 둘러본다.\n{desc}"

    # ── 듣기 ──
    elif action in ("듣기", "listen"):
        if player.ap < 1:
            return "AP가 부족하다."
        player.ap -= 1
        sounds = []
        for e in entities:
            if not e.alive:
                continue
            dist = abs(e.x - player.x) + abs(e.y - player.y) + abs(e.z - player.z)
            if dist <= 5:
                dir_text = get_direction_text(player.x, player.y, e.x, e.y)
                sounds.append(f"{dir_text} {dist}칸에서 발소리가 들린다.")
        if not sounds:
            sounds.append("물 떨어지는 소리. 바람 소리. 그게 전부다.")
        return "숨을 죽이고 귀를 기울인다.\n" + "\n".join(sounds)

    # ── 사격 ──
    elif action in ("사격", "shoot", "sh"):
        if not player.equipped_gun:
            return "총기를 장착하지 않았다."
        if player.ammo <= 0:
            return "탄약이 없다. 장전이 필요하다."
        if player.ap < 3:
            return "AP가 부족해 사격할 수 없다."

        target = find_target(args, player, entities)
        if not target:
            # 방향 견제 사격
            player.ap -= 1
            player.ammo -= 1
            return f"견제 사격. 탄약 소모. AP -{1}"

        dist = abs(target.x - player.x) + abs(target.y - player.y)
        if dist > player.equipped_gun.range_:
            return f"사거리 밖이다. ({dist}칸 / 최대 {player.equipped_gun.range_}칸)"

        player.ap -= 3
        player.ammo -= 2
        hit_chance = max(0.4, 1.0 - dist * 0.1)
        if random.random() < hit_chance:
            dmg = player.equipped_gun.damage + random.randint(-5, 5)
            target.hp -= dmg
            result.append(f"명중! {target.name} HP -{dmg} (잔여: {max(0, target.hp)})")
            if target.hp <= 0:
                target.alive = False
                result.append(f"{target.name}이 쓰러졌다.")
        else:
            result.append("빗나갔다.")

        return "\n".join(result)

    # ── 공격 ──
    elif action in ("공격", "attack", "a"):
        target = find_closest_entity(player, entities)
        if not target:
            return "공격할 대상이 없다."
        dist = abs(target.x - player.x) + abs(target.y - player.y)
        if dist > 1:
            return f"너무 멀다. ({dist}칸)"
        if player.ap < 2:
            return "AP가 부족하다."

        weapon = player.equipped_melee
        base_dmg = weapon.damage if weapon else 10
        dmg = base_dmg + random.randint(-3, 3)
        target.hp -= dmg
        player.ap -= 2
        result.append(f"{target.name}에게 {dmg} 데미지!")
        if target.hp <= 0:
            target.alive = False
            result.append(f"{target.name}이 쓰러졌다.")
        return "\n".join(result)

    # ── 넉킥 ──
    elif action == "넉킥":
        target = find_closest_entity(player, entities)
        if not target:
            return "대상이 없다."
        dist = abs(target.x - player.x) + abs(target.y - player.y)
        if dist > 1:
            return "너무 멀다."
        if player.ap < 2:
            return "AP가 부족하다."

        player.ap -= 2
        dmg = random.randint(5, 15)
        target.hp -= dmg
        # 적 밀어내기
        dx = target.x - player.x
        dy = target.y - player.y
        nx, ny = target.x + dx * 2, target.y + dy * 2
        if 0 < nx < MAP_W and 0 < ny < MAP_H and tiles[target.z][ny][nx].type == TILE_FLOOR:
            target.x, target.y = nx, ny
            push_text = "2칸 후퇴시켰다."
        else:
            push_text = "벽에 막혀 밀리지 않았다."

        result.append(f"발로 걷어찼다! {dmg} 데미지. {push_text}")

        # 넉킥 사격 콤보
        if args and args[0] == "사격" and player.equipped_gun and player.ammo > 0:
            player.ap -= 2
            player.ammo -= 1
            dmg2 = player.equipped_gun.damage + random.randint(-5, 5)
            target.hp -= dmg2
            result.append(f"연속 사격! {dmg2} 데미지! (잔여 HP: {max(0, target.hp)})")
            if target.hp <= 0:
                target.alive = False
                result.append(f"{target.name}이 쓰러졌다.")

        return "\n".join(result)

    # ── 장전 ──
    elif action in ("장전", "reload", "r"):
        if player.ap < 3:
            return "AP가 부족하다."
        # 인벤에 탄창 있는지 확인
        for item in player.inventory:
            if item.name == "탄창":
                player.inventory.remove(item)
                player.ammo = min(30, player.ammo + 15)
                player.ap -= 3
                return f"장전 완료. 탄약: {player.ammo} (무방비 상태였다)"
        return "탄창이 없다."

    # ── 엄폐 ──
    elif action in ("엄폐", "cover"):
        if player.ap < 1:
            return "AP가 부족하다."
        player.ap -= 1
        player.status = "엄폐중"
        return "주변 엄폐물 뒤로 몸을 낮췄다."

    # ── 회피 ──
    elif action in ("회피", "dodge"):
        if player.ap < 2:
            return "AP가 부족하다."
        player.ap -= 2
        return "몸을 옆으로 틀었다."

    # ── 줍기 ──
    elif action in ("줍기", "take", "t"):
        if not args:
            return "무엇을 줍겠는가?"
        item_name = args[0]
        if item_name in ITEMS:
            item = ITEMS[item_name]
            player.inventory.append(item)
            player.ap -= 1
            # 총기 자동 장착
            if item.item_type == "weapon_gun" and not player.equipped_gun:
                player.equipped_gun = item
                player.ammo = item.ammo
                return f"{item_name}을 줍고 장착했다."
            elif item.item_type == "weapon_melee" and not player.equipped_melee:
                player.equipped_melee = item
                return f"{item_name}을 줍고 왼손에 쥐었다."
            return f"{item_name}을 주웠다."
        return "그런 아이템이 없다."

    # ── 사용 ──
    elif action in ("사용", "use"):
        if not args:
            return "무엇을 사용하겠는가?"
        item_name = args[0]
        for item in player.inventory:
            if item.name == item_name:
                if item.effect == "ap_boost":
                    player.inventory.remove(item)
                    player.ap = min(player.max_ap, player.ap + 12)
                    player.ap_penalty += 3
                    player.status = "각성"
                    return ("월혼을 삼켰다.\n"
                            "심장이 두 배로 뛴다. 시야가 붉게 물든다.\n"
                            f"AP +12 / 각성 상태 / 다음 런 AP -{player.ap_penalty} 패널티")
                elif item.effect == "ammo+15":
                    player.inventory.remove(item)
                    player.ammo = min(30, player.ammo + 15)
                    return f"탄창 교체. 탄약: {player.ammo}"
        return f"{item_name}이 없다."

    # ── 장착 ──
    elif action in ("장착", "equip"):
        if not args:
            return "무엇을 장착하겠는가?"
        item_name = args[0]
        for item in player.inventory:
            if item.name == item_name:
                if item.item_type == "weapon_gun":
                    player.equipped_gun = item
                    player.ammo = item.ammo
                    player.ap -= 2
                    return f"{item_name} 장착. AP -2"
                elif item.item_type == "weapon_melee":
                    player.equipped_melee = item
                    player.ap -= 1
                    return f"{item_name} 장착."
        return f"{item_name}이 없다."

    # ── 가방 ──
    elif action in ("가방", "inv", "i"):
        if not player.inventory:
            return "가방이 비어있다."
        lines = ["━━━━━━━━━━━━━━━━"]
        lines.append(f"주무기: {player.equipped_gun.name if player.equipped_gun else '없음'} (탄약: {player.ammo})")
        lines.append(f"보조  : {player.equipped_melee.name if player.equipped_melee else '없음'}")
        lines.append("소지품:")
        for item in player.inventory:
            lines.append(f"  - {item.name}")
        lines.append("━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ── 무기 점검 ──
    elif action in ("무기점검", "무기"):
        lines = ["━━━━━━━━━━━━━━━━"]
        if player.equipped_gun:
            g = player.equipped_gun
            lines.append(f"주무기: {g.name}")
            lines.append(f"  탄약: {player.ammo}/30")
            lines.append(f"  사거리: {g.range_}칸")
        else:
            lines.append("주무기: 없음")
        if player.equipped_melee:
            m = player.equipped_melee
            lines.append(f"보조: {m.name} (데미지:{m.damage})")
        else:
            lines.append("보조: 없음")
        lines.append("━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    # ── 쉬기 ──
    elif action in ("쉬기", "rest"):
        recover = random.randint(4, 7)
        player.ap = min(player.max_ap, player.ap + recover)
        return f"숨을 고른다. AP +{recover} (무방비 상태)"

    # ── 확인 ──
    elif action in ("확인", "check"):
        if not args:
            return "무엇을 확인하겠는가?"
        target = args[0]
        if target == "시체":
            found = ["나이프", "월혼", "산탄총"]
            return ("시체를 빠르게 훑는다.\n"
                    "━━━━━━━━━━━━━━━━\n"
                    "발견:\n" +
                    "\n".join(f"  {i}" for i in found) +
                    "\n━━━━━━━━━━━━━━━━")
        return f"{target}에서 특별한 것을 찾지 못했다."

    # ── 맵 ──
    elif action in ("맵", "map"):
        return None  # 메인 루프에서 처리

    # ── 도움말 ──
    elif action in ("도움말", "help", "?"):
        return (
            "━━━━ 명령어 ━━━━\n"
            "이동: 북 남 동 서 위 아래\n"
            "달려 [방향]: 2칸 이동 (AP-3)\n"
            "보기 / 듣기\n"
            "사격 [대상] / 공격 / 넉킥\n"
            "넉킥 사격: 밀어내고 연속 사격\n"
            "장전 / 엄폐 / 회피\n"
            "줍기 [아이템] / 사용 [아이템]\n"
            "장착 [무기] / 가방 / 무기점검\n"
            "쉬기 / 확인 [대상] / 맵\n"
            "━━━━━━━━━━━━━━━━"
        )

    else:
        return f"알 수 없는 명령어: '{action}' (도움말: ?)"

# ─────────────────────────────────────────
# 타겟 찾기
# ─────────────────────────────────────────
def find_target(args, player: Entity, entities: list) -> Optional[Entity]:
    if not args:
        return find_closest_entity(player, entities)
    name = args[0]
    for e in entities:
        if e.alive and name in e.name:
            return e
    return find_closest_entity(player, entities)

def find_closest_entity(player: Entity, entities: list) -> Optional[Entity]:
    alive = [e for e in entities if e.alive]
    if not alive:
        return None
    return min(alive, key=lambda e: abs(e.x-player.x) + abs(e.y-player.y))

# ─────────────────────────────────────────
# 적 AI 행동
# ─────────────────────────────────────────
def enemy_turn(entity: Entity, player: Entity, tiles) -> str:
    if not entity.alive:
        return ""

    result = []
    dist = abs(entity.x - player.x) + abs(entity.y - player.y)

    # 변신 처리
    if entity.status == "변신중":
        entity.transform_timer -= 1
        result.append(f"[{entity.name} 변신중... {entity.transform_timer}턴 남음]")
        if entity.transform_timer <= 0:
            entity.status = "변신완료"
            entity.max_hp = int(entity.max_hp * 1.5)
            entity.hp = min(entity.hp + 30, entity.max_hp)
            result.append(f"[{entity.name} 변신 완료! 전투력 증가]")
        return "\n".join(result)

    # 체력 낮으면 변신 시도
    if entity.faction == "werebeast" and entity.hp < 30 and entity.status == "":
        entity.status = "변신중"
        entity.transform_timer = 3
        result.append(f"{entity.name}이 변신을 시작한다...")
        return "\n".join(result)

    # 이동
    if dist > 1:
        dx = 1 if player.x > entity.x else -1 if player.x < entity.x else 0
        dy = 1 if player.y > entity.y else -1 if player.y < entity.y else 0
        nx, ny = entity.x + dx, entity.y + dy
        if 0 < nx < MAP_W and 0 < ny < MAP_H and tiles[entity.z][ny][nx].type == TILE_FLOOR:
            entity.x, entity.y = nx, ny
            result.append(f"{entity.name}이 접근한다. ({dist-1}칸)")
    else:
        # 근접 공격
        base_dmg = 20 if entity.status == "변신완료" else 12
        dmg = base_dmg + random.randint(-3, 5)
        player.hp -= dmg
        result.append(f"{entity.name}의 공격! HP -{dmg} (잔여: {max(0, player.hp)})")
        if player.hp <= 0:
            player.alive = False

    return "\n".join(result)

# ─────────────────────────────────────────
# AP 회복 (턴마다)
# ─────────────────────────────────────────
def recover_ap(player: Entity):
    if player.ap < player.max_ap:
        player.ap = min(player.max_ap, player.ap + 2)

# ─────────────────────────────────────────
# 상태 표시
# ─────────────────────────────────────────
def status_bar(player: Entity) -> str:
    hp_bar = int(player.hp / player.max_hp * 10)
    ap_bar = int(player.ap / player.max_ap * 10)
    gun = player.equipped_gun.name if player.equipped_gun else "없음"
    melee = player.equipped_melee.name if player.equipped_melee else "없음"
    status = f" [{player.status}]" if player.status else ""
    return (
        f"HP: {'█'*hp_bar}{'░'*(10-hp_bar)} {player.hp}/{player.max_hp}  "
        f"AP: {'█'*ap_bar}{'░'*(10-ap_bar)} {player.ap}/{player.max_ap}  "
        f"탄약: {player.ammo}  "
        f"주무기: {gun}  보조: {melee}{status}"
    )

def hint_bar(player: Entity, entities: list) -> str:
    alive_enemies = [e for e in entities if e.alive]
    dist = abs(alive_enemies[0].x - player.x) + abs(alive_enemies[0].y - player.y) if alive_enemies else 99

    if not alive_enemies:
        return "💡 북 / 남 / 동 / 서 / 보기 / 듣기 / 가방"
    elif dist <= 1:
        return "💡 공격 / 사격 [대상] / 넉킥 / 넉킥 사격 / 회피 / 달려 남 / 사용 월혼"
    elif dist <= 5:
        return "💡 사격 [대상] / 엄폐 / 조준 / 달려 남 / 보기 / 듣기"
    else:
        return "💡 북 / 남 / 동 / 서 / 보기 / 듣기 / 가방 / 무기점검"

# ─────────────────────────────────────────
# 인트로
# ─────────────────────────────────────────
def intro():
    print("=" * 50)
    print("      텍스트 머드 - 오늘의 던전")
    print(f"      씨드: {SEED}")
    print("=" * 50)
    print()
    print("세력을 선택하세요")
    print("1. 헌터  (인간, 총기, 월혼)")
    print("2. 야족  (흡혈귀, 혈인능력)")
    print("3. 수인  (웨어비스트, 변신)")
    print()

    while True:
        choice = input("> ").strip()
        if choice == "1":
            return "hunter"
        elif choice == "2":
            return "vampire"
        elif choice == "3":
            return "werebeast"
        else:
            print("1, 2, 3 중 선택하세요.")

# ─────────────────────────────────────────
# 메인 게임 루프
# ─────────────────────────────────────────
def main():
    os.system("cls" if os.name == "nt" else "clear")

    faction = intro()

    # 맵 생성
    tiles = generate_map(SEED)
    explored = set()

    # 플레이어 생성
    player = Entity(name="나", x=3, y=5, z=0, faction=faction)
    player.equipped_gun = ITEMS["글록17"]
    player.ammo = 15
    explored.add((3, 5, 0))

    # 적 생성 (오늘의 플레이 재현)
    enemy = Entity(
        name="수인 플레이어",
        x=3, y=3, z=0,
        hp=100, max_hp=100,
        ap=20,
        faction="werebeast"
    )

    # 시체 아이템 (동쪽 위치)
    corpse_items = ["나이프", "월혼", "산탄총", "탄창"]

    entities = [enemy]

    # 시작 연출
    print()
    print("2026년 3월 12일 00:00")
    print("서울 지하 어딘가")
    print()
    print("눅눅한 공기가 폐를 채운다.")
    print("손에 쥔 글록의 무게가 익숙하다.")
    print("주머니 안에서 월혼 한 알이 구른다.")
    print()

    turn = 0
    show_map = False

    while player.alive:
        # 상태바
        print()
        print("─" * 50)
        print(status_bar(player))
        print("─" * 50)

        # 맵 표시 (요청시)
        if show_map:
            print()
            print(render_map(tiles, player, entities, explored))
            show_map = False

        # 힌트
        print(hint_bar(player, entities))
        print()

        # 입력
        try:
            cmd = input("액션 > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n게임을 종료합니다.")
            break

        if not cmd:
            continue

        if cmd in ("종료", "quit", "exit"):
            print("게임을 종료합니다.")
            break

        action, args = parse_command(cmd)

        # 맵 명령어
        if action in ("맵", "map"):
            show_map = True
            continue

        # 액션 처리
        result = process_action(action, args, player, tiles, entities, explored)
        if result:
            print()
            print(result)

        # 사망 체크
        if not player.alive:
            print()
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("당신은 쓰러졌다.")
            print(f"생존 턴: {turn}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            break

        # 적 턴
        for e in entities:
            if e.alive:
                enemy_result = enemy_turn(e, player, tiles)
                if enemy_result:
                    print()
                    print(enemy_result)

        # 사망 체크 (적 공격 후)
        if not player.alive:
            print()
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("당신은 쓰러졌다.")
            print(f"생존 턴: {turn}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            break

        # 모든 적 처치
        if all(not e.alive for e in entities):
            print()
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print("모든 적을 처치했다.")
            print(f"생존 HP: {player.hp} / 남은 AP: {player.ap}")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            break

        # AP 자연 회복
        recover_ap(player)
        turn += 1

if __name__ == "__main__":
    main()
