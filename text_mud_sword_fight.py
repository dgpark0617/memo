import random

# === Position Data ===
positions = {
    "Overhead": {"atk": 1.4, "def": 1.0},
    "FrontFoot": {"atk": 1.0, "def": 1.2},
    "LeftBody": {"atk": 1.1, "def": 1.1},
    "RightBody": {"atk": 1.1, "def": 1.1},
    "LeftFace": {"atk": 1.2, "def": 0.9},
    "RightFace": {"atk": 1.2, "def": 0.9},
}

# === Transition Data ===
# Format: (attack_bonus, defense_bonus, description)
transitions = {
    ("Overhead", "FrontFoot"): (0.1, -0.1, "A powerful downward cut from above to the lower guard."),
    ("Overhead", "LeftBody"): (0.0, 0.1, "A diagonal cut from above to the left side."),
    ("Overhead", "RightBody"): (0.0, 0.1, "A diagonal cut from above to the right side."),
    ("Overhead", "LeftFace"): (0.2, -0.2, "A thrust from above targeting the opponent's left side."),
    ("Overhead", "RightFace"): (0.2, -0.2, "A thrust from above targeting the opponent's right side."),

    ("FrontFoot", "Overhead"): (0.1, -0.1, "A rising cut from low guard to high guard."),
    ("FrontFoot", "LeftBody"): (0.0, 0.1, "A low sweep to the left side."),
    ("FrontFoot", "RightBody"): (0.0, 0.1, "A low sweep to the right side."),
    ("FrontFoot", "LeftFace"): (0.1, -0.2, "A quick thrust from low to the opponent's left face."),
    ("FrontFoot", "RightFace"): (0.1, -0.2, "A quick thrust from low to the opponent's right face."),

    ("LeftBody", "Overhead"): (0.1, -0.1, "A rising cut from the left side to high guard."),
    ("LeftBody", "FrontFoot"): (0.0, 0.1, "A downward sweep from the left side to low guard."),
    ("LeftBody", "RightBody"): (0.0, 0.0, "A horizontal cut from left to right side."),
    ("LeftBody", "LeftFace"): (0.1, -0.1, "A thrust from the left side to the opponent's face."),
    ("LeftBody", "RightFace"): (0.1, -0.1, "A thrust from the left side to the opponent's right face."),

    ("RightBody", "Overhead"): (0.1, -0.1, "A rising cut from the right side to high guard."),
    ("RightBody", "FrontFoot"): (0.0, 0.1, "A downward sweep from the right side to low guard."),
    ("RightBody", "LeftBody"): (0.0, 0.0, "A horizontal cut from right to left side."),
    ("RightBody", "LeftFace"): (0.1, -0.1, "A thrust from the right side to the opponent's left face."),
    ("RightBody", "RightFace"): (0.1, -0.1, "A thrust from the right side to the opponent's right face."),

    ("LeftFace", "Overhead"): (0.1, -0.1, "A rising cut from left face to high guard."),
    ("LeftFace", "FrontFoot"): (0.0, 0.1, "A downward sweep from left face to low guard."),
    ("LeftFace", "LeftBody"): (0.0, 0.0, "A short drop from left face to left body."),
    ("LeftFace", "RightBody"): (0.0, 0.0, "A diagonal cut from left face to right body."),
    ("LeftFace", "RightFace"): (0.0, 0.0, "A horizontal shift from left face to right face."),

    ("RightFace", "Overhead"): (0.1, -0.1, "A rising cut from right face to high guard."),
    ("RightFace", "FrontFoot"): (0.0, 0.1, "A downward sweep from right face to low guard."),
    ("RightFace", "LeftBody"): (0.0, 0.0, "A diagonal cut from right face to left body."),
    ("RightFace", "RightBody"): (0.0, 0.0, "A short drop from right face to right body."),
    ("RightFace", "LeftFace"): (0.0, 0.0, "A horizontal shift from right face to left face."),
}

# === Damage Calculation ===
def calculate_damage(attacker_pos, attacker_next, defender_pos, defender_next, base_atk=10):
    atk_coeff = positions[attacker_pos]["atk"] + transitions[(attacker_pos, attacker_next)][0]
    def_coeff = positions[defender_pos]["def"] + transitions[(defender_pos, defender_next)][1]
    damage = base_atk * atk_coeff * max(0, (1 - def_coeff))
    return round(damage, 1)

# === Game Loop ===
def game():
    player_hp = 50
    enemy_hp = 50
    player_pos = random.choice(list(positions.keys()))
    enemy_pos = random.choice(list(positions.keys()))

    print("=== Liechtenauer Swordfight (Text MUD) ===")
    print(f"Starting Position: You - {player_pos}, Enemy - {enemy_pos}")

    while player_hp > 0 and enemy_hp > 0:
        print(f"\nYour HP: {player_hp} | Enemy HP: {enemy_hp}")
        print(f"Current Position: {player_pos}")
        
        # Show choices
        choices = [p for p in positions.keys() if p != player_pos]
        for i, choice in enumerate(choices, 1):
            print(f"{i}. {choice}")
        
        # Player choice
        try:
            sel = int(input("Choose your next position: "))
            if sel < 1 or sel > len(choices):
                print("Invalid choice. Try again.")
                continue
            player_next = choices[sel - 1]
        except ValueError:
            print("Invalid input. Enter a number.")
            continue

        # Enemy choice
        enemy_choices = [p for p in positions.keys()]
