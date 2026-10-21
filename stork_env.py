import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw

# ---------- World settings (shared with the playable web version) ----------
W, H = 400, 600          # screen size in pixels
GROUND = 60              # height of the ground strip
STORK_X = 100            # the stork stays at this x; the world scrolls past
GRAVITY = 0.5            # added to falling speed every step
FLAP = -8.0              # upward speed right after a flap
MAX_FALL = 10.0
TOWER_W = 64             # minaret width
SPACING = 210            # horizontal distance between minarets
STORK_R = 13             # collision radius of the stork
MAX_STEPS = 3000         # 100 seconds at 30 steps per second


class StorkEnv(gym.Env):
    """A stork flies through gaps between minarets (below) and storm clouds (above).

    Actions: 0 = glide (do nothing), 1 = flap.
    Observation (5 numbers, roughly between -1 and 1):
      stork height, stork vertical speed, distance to the next gap,
      height of the gap centre relative to the stork, size of the gap.
    Reward: +0.1 for every step alive, +1 for each minaret passed, -1 for crashing.
    difficulty: 0 = easy (wide gaps, slow), 1 = hard (narrow gaps, fast).
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, difficulty=0.0, render_mode=None):
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(5,), dtype=np.float32)

    # gap size and speed both depend on difficulty
    def _gap(self):
        return 180 - 50 * self.difficulty

    def _speed(self):
        return 3.0 + 1.0 * self.difficulty

    def _new_tower(self, x, prev_gap_y=None):
        gap = self._gap()
        lo = 90 + gap / 2
        hi = H - GROUND - 70 - gap / 2
        if prev_gap_y is not None:               # keep each new gap within reach of the last one
            lo, hi = max(lo, prev_gap_y - 160), min(hi, prev_gap_y + 160)
        return {"x": float(x), "gap_y": float(self.np_random.uniform(lo, hi)), "passed": False}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.y = H / 2
        self.vy = 0.0
        self.steps = 0
        self.score = 0
        self.flap_timer = 0
        self.towers = []
        for i in range(3):
            prev = self.towers[-1]["gap_y"] if self.towers else None
            self.towers.append(self._new_tower(W + 40 + i * SPACING, prev))
        return self._obs(), {}

    def _next_tower(self):
        # the first minaret whose right edge is still ahead of the stork's tail
        return next(t for t in self.towers if t["x"] + TOWER_W > STORK_X - STORK_R)

    def _obs(self):
        t = self._next_tower()
        return np.array([
            (self.y - H / 2) / (H / 2),
            self.vy / MAX_FALL,
            (t["x"] - STORK_X) / W,
            (t["gap_y"] - self.y) / (H / 2),
            self._gap() / H,
        ], dtype=np.float32)

    def _crashed(self):
        if self.y + STORK_R >= H - GROUND or self.y - STORK_R <= 0:
            return True
        gap = self._gap()
        for t in self.towers:
            if t["x"] - STORK_R < STORK_X < t["x"] + TOWER_W + STORK_R:
                if self.y - STORK_R < t["gap_y"] - gap / 2 or self.y + STORK_R > t["gap_y"] + gap / 2:
                    return True
        return False

    def step(self, action):
        if action == 1:
            self.vy = FLAP
            self.flap_timer = 6
        self.vy = min(self.vy + GRAVITY, MAX_FALL)
        self.y += self.vy
        self.flap_timer = max(0, self.flap_timer - 1)

        reward = 0.1
        for t in self.towers:
            t["x"] -= self._speed()
            if not t["passed"] and t["x"] + TOWER_W < STORK_X - STORK_R:
                t["passed"] = True
                self.score += 1
                reward += 1.0
        if self.towers[0]["x"] + TOWER_W < 0:
            self.towers.pop(0)
            last = self.towers[-1]
            self.towers.append(self._new_tower(last["x"] + SPACING, last["gap_y"]))

        self.steps += 1
        terminated = self._crashed()
        if terminated:
            reward = -1.0
        truncated = self.steps >= MAX_STEPS
        return self._obs(), reward, terminated, truncated, {"score": self.score}

    # ---------- Drawing (simple shapes, all original) ----------
    def render(self):
        img = Image.new("RGB", (W, H))
        d = ImageDraw.Draw(img)
        for i in range(H - GROUND):                      # evening sky gradient
            k = i / (H - GROUND)
            d.line([(0, i), (W, i)], fill=(int(240 - 60 * k), int(170 + 40 * k), int(110 + 120 * k)))
        gap = self._gap()
        for t in self.towers:
            x0, x1 = t["x"], t["x"] + TOWER_W
            edge = t["gap_y"] + gap / 2                  # the collision edge = the lantern's tip
            body = edge + 36
            # minaret: lantern, small upper tower, crenellations, tiled band, sandstone body
            d.ellipse([x0 + 26, edge, x1 - 26, edge + 14], fill=(201, 160, 60))
            d.rectangle([x0 + 20, edge + 10, x1 - 20, body], fill=(196, 150, 95))
            for cx in range(int(x0), int(x1), 16):
                d.rectangle([cx, body - 8, cx + 9, body], fill=(214, 170, 112))
            d.rectangle([x0, body, x1, H - GROUND], fill=(214, 170, 112))
            d.rectangle([x0, body + 16, x1, body + 28], fill=(31, 122, 92))
            d.rectangle([x0 + 26, body + 48, x1 - 26, body + 78], fill=(120, 80, 50))
            # storm cloud from above
            bottom = t["gap_y"] - gap / 2
            d.rectangle([x0 - 6, 0, x1 + 6, bottom - 14], fill=(92, 98, 120))
            for j in range(4):
                cx = x0 - 6 + j * (TOWER_W + 12) / 3
                d.ellipse([cx - 16, bottom - 32, cx + 16, bottom], fill=(92, 98, 120))
        d.rectangle([0, H - GROUND, W, H], fill=(178, 120, 70))
        d.rectangle([0, H - GROUND, W, H - GROUND + 8], fill=(150, 98, 55))
        # stork: white body, black wing (up after a flap), orange beak and legs
        x, y = STORK_X, self.y
        d.line([(x - 4, y + 9), (x - 10, y + 22)], fill=(230, 110, 40), width=3)
        d.line([(x + 2, y + 9), (x - 4, y + 22)], fill=(230, 110, 40), width=3)
        d.ellipse([x - 16, y - 10, x + 14, y + 10], fill=(250, 250, 245))
        d.ellipse([x + 8, y - 16, x + 20, y - 4], fill=(250, 250, 245))
        d.polygon([(x + 19, y - 11), (x + 34, y - 8), (x + 19, y - 6)], fill=(230, 110, 40))
        d.ellipse([x + 13, y - 13, x + 16, y - 10], fill=(20, 20, 20))
        if self.flap_timer > 0:
            d.polygon([(x - 8, y - 4), (x + 4, y - 4), (x - 10, y - 26)], fill=(25, 25, 25))
        else:
            d.polygon([(x - 12, y - 2), (x + 6, y - 2), (x - 18, y + 8)], fill=(25, 25, 25))
        d.text((12, 10), f"Score {self.score}", fill=(255, 255, 255))
        return np.array(img)
