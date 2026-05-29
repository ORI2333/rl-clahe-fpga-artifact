# 文件名: clahe_env_dual_V1f.py
import os, random, numpy as np, cv2, torch
import gymnasium as gym
from gymnasium import spaces
import pyiqa  # <-- 导入到顶部


# ---------- 基础度量 (不变) ----------
def calculate_renyi_entropy(image: np.ndarray) -> float:
    hist = np.histogram(image, bins=256, range=(0, 256))[0].astype(np.float64)
    total = image.size
    if total == 0: return 0.0
    p = hist / total
    s = np.sum(p * p)
    return 0.0 if s <= 1e-12 else float(-np.log2(s))


def apply_clahe(image: np.ndarray, clip_limit: float) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    return clahe.apply(image)


def lap_var(image: np.ndarray) -> float:
    return float(cv2.Laplacian(image, cv2.CV_64F).var())


def rms_contrast(image: np.ndarray) -> float:
    return float(image.std())


class CLAHEEnvPro(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, image_folder='Train', max_steps=5,
                 obs_mode='lite', metrics_device='cuda', seed=None):
        super().__init__()
        assert obs_mode in ('rich', 'lite')
        assert metrics_device in ('cpu', 'cuda')
        self.obs_mode = obs_mode
        self.metrics_device = metrics_device

        # ---------- 关键修改：删除懒加载 ----------
        # self._brisque = None # (删除)
        # self._niqe    = None # (删除)
        # self._device  = None # (删除)

        if not os.path.isdir(image_folder):
            raise FileNotFoundError(f"image_folder 不存在: {image_folder}")
        self.image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif'))]
        if not self.image_paths:
            raise ValueError(f"图片文件夹 '{image_folder}' 为空")

        self.max_steps = int(max_steps)
        self.cl_min, self.cl_max = 0.1, 20.0
        self.action_space = spaces.Box(low=-2.0, high=2.0, shape=(1,), dtype=np.float32)

        if self.obs_mode == 'rich':
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)

        # (奖励系数不变)
        if self.obs_mode == 'rich':
            self.w_brisque, self.w_niqe, self.w_sharp, self.w_contrast = 1.0, 0.5, 0.01, 0.1
            self.pen_amp, self.pen_jerk, self.soft_mid_scale = 0.02, 0.01, 1.5
            self.tail_bonus_th, self.tail_bonus = 30.0, 2.0
            self.tail_bonus_strict_th, self.tail_bonus_strict = 20.0, 1.0
        else:
            self.w_brisque, self.w_niqe, self.w_sharp, self.w_contrast = 0.4, 0.2, 0.02, 0.2
            self.pen_amp, self.pen_jerk, self.soft_mid_scale = 0.03, 0.02, 1.8
            self.tail_bonus_th, self.tail_bonus = 32.0, 1.5
            self.tail_bonus_strict_th, self.tail_bonus_strict = 22.0, 0.8

        self.total_delta_cap = 6.0
        self.rng = np.random.default_rng(seed)
        self._reset_ep_state()

        # ---------- 关键修改：在 __init__ 中预加载 pyiqa ----------
        # 这是防止 SubprocVecEnv (多进程) 发生 1fps 死锁的关键
        use_cuda = (self.metrics_device == 'cuda') and torch.cuda.is_available()
        self._device = 'cuda' if use_cuda else 'cpu'

        # 'lite' 模式在奖励中也需要 IQA (w_brisque=0.4)，所以必须加载
        self._brisque = pyiqa.create_metric('brisque', device=self._device)
        self._niqe = pyiqa.create_metric('niqe', device=self._device)
        # ----------------------------------------------------

        # (线程限制保持不变, 这对多进程很重要)
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
        torch.set_num_threads(1)

    # ---------- 关键修改：删除 _ensure_metrics ----------
    # def _ensure_metrics(self):
    #     ( ... 此函数已删除 ... )

    def _iqa(self, img_gray):
        # self._ensure_metrics() # <-- 删除此行
        rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        ten = torch.tensor(rgb / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self._device)
        with torch.no_grad():
            b = float(self._brisque(ten).item())
            n = float(self._niqe(ten).item())
        return b, n

    def _calc_feats(self, img):
        # (函数不变)
        mu = float(np.mean(img))
        var = float(np.var(img))
        H2 = calculate_renyi_entropy(img)
        return (mu, var, H2)

    def _state_vec(self):
        # (函数不变)
        mu, var, H2 = self.mu_var_H2
        norm_mean = mu / 255.0
        norm_var = var / 10000.0
        norm_entropy = H2 / 8.0
        norm_cl = (self.cl - self.cl_min) / (self.cl_max - self.cl_min)
        norm_step = self.t / self.max_steps
        if self.obs_mode == 'rich':
            norm_b, norm_n = self.last_b / 100.0, self.last_n / 10.0
            s = np.array([norm_mean, norm_var, norm_entropy, norm_cl, norm_step, norm_b, norm_n], dtype=np.float32)
        else:
            s = np.array([norm_mean, norm_var, norm_entropy, norm_cl, norm_step], dtype=np.float32)
        return np.clip(s, 0.0, 1.0)

    def _reset_ep_state(self):
        # (函数不变)
        self.img_path = None;
        self.img = None;
        self.mu_var_H2 = (0.0, 0.0, 0.0)
        self.init_b = self.init_n = 0.0;
        self.init_s = self.init_c = 0.0
        self.cl = 0.0;
        self.t = 0;
        self.last_act = 0.0
        self.sum_abs_delta = 0.0;
        self.last_b = self.last_n = 0.0

    # ---------- Gym API (不变) ----------
    def reset(self, *, seed=None, options=None):
        # (函数不变)
        super().reset(seed=seed);
        self._reset_ep_state()
        self.img_path = random.choice(self.image_paths)
        self.img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        if self.img is None: raise ValueError(f"读取失败: {self.img_path}")
        self.mu_var_H2 = self._calc_feats(self.img)
        self.init_b, self.init_n = self._iqa(self.img)
        self.init_s = lap_var(self.img);
        self.init_c = rms_contrast(self.img)
        self.cl = float(self.rng.uniform(1.0, 10.0))
        self.t = 0;
        self.last_act = 0.0;
        self.sum_abs_delta = 0.0
        enh0 = apply_clahe(self.img, self.cl)
        self.last_b, self.last_n = self._iqa(enh0)
        return self._state_vec(), {}

    def step(self, action):
        # (函数不变)
        delta = float(np.clip(action[0], -2.0, 2.0));
        prev = self.last_act
        self.cl = float(np.clip(self.cl + delta, self.cl_min, self.cl_max))
        self.sum_abs_delta += abs(delta)
        enh = apply_clahe(self.img, self.cl)
        new_b, new_n = self._iqa(enh);
        new_s = lap_var(enh);
        new_c = rms_contrast(enh)
        r_b = self.w_brisque * (self.last_b - new_b)
        r_n = self.w_niqe * (self.last_n - new_n)
        r_s = self.w_sharp * (new_s - self.init_s) / 1000.0
        r_c = self.w_contrast * (new_c - self.init_c) / 10.0
        pen_amp = self.pen_amp * abs(delta)
        pen_jerk = self.pen_jerk * (delta - prev) * (delta - prev)
        mid_scale = self.soft_mid_scale if (2.0 <= self.cl <= 8.0) else 1.0
        rew = r_b + r_n + r_s + r_c - mid_scale * pen_amp - pen_jerk
        self.t += 1
        done = (self.t >= self.max_steps)
        if done:
            if new_b < self.tail_bonus_th and self.sum_abs_delta <= self.total_delta_cap:
                rew += self.tail_bonus
            if new_b < self.tail_bonus_strict_th:
                rew += self.tail_bonus_strict
            if new_b > 40.0:
                rew -= 50.0
        self.last_b, self.last_n = new_b, new_n;
        self.last_act = delta
        return self._state_vec(), float(rew), done, False, {
            'img': os.path.basename(self.img_path), 'clip_limit': self.cl, 'delta_cl': delta,
            'brisque': new_b, 'niqe': new_n, 'sharp': new_s, 'contrast': new_c
        }