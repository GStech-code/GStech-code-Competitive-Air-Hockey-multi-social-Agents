from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from air_hockey_ros import TeamPPOWrapper

DEFAULT_PPO = dict(
    gamma=0.995, gae_lambda=0.95, clip_eps=0.2,
    vf_coef=0.5, ent_coef=0.01, lr=3e-4, epochs=4,
    minibatch_size=1024
)

def train(ros_mock, orchestrator, config: Dict[str, Any]):
    """
    PPO trainer using WRAPPERS, with one-time initialization per team.
    - If team name == 'neural' (A or B), that side is trainable.
    - Wrappers mirror X for Team B for a consistent policy frame.
    - Load/randomize/teach run ONCE the first time each team becomes trainable.
    - Wrappers are reused across episodes so weights accumulate.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_episodes = int(_cfg(orchestrator.cfg, "episodes", 10))
    max_steps      = int(getattr(orchestrator, "steps_per_episode", 2000))
    goal_limit     = int(getattr(orchestrator, "goal_limit", 10))
    output_dir     = config.get("output_dir", os.path.join(_cfg(orchestrator.cfg,"output_dir","."), "checkpoints"))
    ckpt_every     = int(config.get("checkpoint_every", 5))
    ppo            = {**DEFAULT_PPO, **config.get("ppo", {})}

    # Persistent wrappers & init flags (so we don't re-initialize every episode)
    teamA_wrap: Optional[TeamPPOWrapper] = None
    teamB_wrap: Optional[TeamPPOWrapper] = None
    did_init_A = False
    did_init_B = False
    last_shape_A = None  # (na,)
    last_shape_B = None  # (nb,)

    for ep in range(total_episodes):
        # ---- Episode setup via orchestrator ----
        game = orchestrator.next_game()
        na = int(_get(game, "num_agents_team_a", 2))
        nb = int(_get(game, "num_agents_team_b", 2))
        team_a_name = _get(game, "team_a", "neural")
        team_b_name = _get(game, "team_b", "simple")
        rules = _get(game, "rules", {})

        # Reset RosMock with ORIGINAL policies for this episode
        ta_cls = orchestrator.get_team_policy_class(team_a_name)
        tb_cls = orchestrator.get_team_policy_class(team_b_name)
        ros_mock.reset(num_agents_team_a=na,
                       num_agents_team_b=nb,
                       team_a_policy_class=ta_cls,
                       team_b_policy_class=tb_cls,
                       **rules
        )

        # World constants for feature norms
        width = float(rules.get("width", 800))
        height = float(rules.get("height", 800))
        pmax = float(rules.get("puck_max_speed", 6.0))
        umove = float(rules.get("unit_speed_px", 4.0))
        max_speed = max(pmax, umove) * 1.05

        # Agent id layout as ros_mock expects (contiguous A then B)
        ids_a = list(range(na))
        ids_b = list(range(na, na + nb))
        tmates_a = [[i for i in ids_a if i != me] for me in ids_a]
        tmates_b = [[i for i in ids_b if i != me] for me in ids_b]
        opps_a = ids_b[:]  # A sees all B as opponents
        opps_b = ids_a[:]  # B sees all A as opponents

        train_A = (team_a_name == 'neural')
        train_B = (team_b_name == 'neural')

        # ---- Build or reuse wrappers for trainable sides ----
        # Team A
        if train_A:
            need_new_A = (teamA_wrap is None) or (last_shape_A != (na,))
            if need_new_A:
                teamA_wrap = TeamPPOWrapper('A', ids_a, width, height, max_speed,
                                            tmates_a, opps_a, device=str(device))
                last_shape_A = (na,)
                # one-time initialization
                if not did_init_A:
                    _apply_initialization(teamA_wrap, config.get("init_teamA", {}), ros_mock)
                    did_init_A = True
            ros_mock.team_a_policies = teamA_wrap.policies  # replace with wrappers

        # Team B
        if train_B:
            need_new_B = (teamB_wrap is None) or (last_shape_B != (nb,))
            if need_new_B:
                teamB_wrap = TeamPPOWrapper('B', ids_b, width, height, max_speed,
                                            tmates_b, opps_b, device=str(device))
                last_shape_B = (nb,)
                if not did_init_B:
                    _apply_initialization(teamB_wrap, config.get("init_teamB", {}), ros_mock)
                    did_init_B = True
            ros_mock.team_b_policies = teamB_wrap.policies

        # Toggle train mode
        if teamA_wrap and train_A: teamA_wrap.set_train_mode(True)
        if teamB_wrap and train_B: teamB_wrap.set_train_mode(True)

        # Reward shaping
        rw = orchestrator.reward_weights_for(game)
        w_score = float(rw.get("score_delta", 1.0))
        w_dist  = float(rw.get("dist_to_puck_penalty", 0.0))

        # Trajectories
        traj_A = [_empty_traj(i) for i in range(na)] if train_A else None
        traj_B = [_empty_traj(j) for j in range(nb)] if train_B else None

        # Initial scores
        ws = ros_mock.sim.get_world_state()
        last_a, last_b = int(ws.get("team_a_score", 0)), int(ws.get("team_b_score", 0))

        steps = 0
        for _ in range(max_steps):
            ws = ros_mock.step()
            a, b = int(ws.get("team_a_score", 0)), int(ws.get("team_b_score", 0))
            da, db = a - last_a, b - last_b
            last_a, last_b = a, b

            rA = float(w_score * (da - db))
            rB = float(w_score * (db - da))
            if w_dist:
                rA -= w_dist * _avg_dist(ws, 0, na)
                rB -= w_dist * _avg_dist(ws, na, nb)

            if train_A and teamA_wrap:
                for i, pol in enumerate(ros_mock.team_a_policies):
                    _push_step(traj_A[i], pol.last_feat, pol.last_ax, pol.last_ay,
                               pol.last_logp, pol.last_value, rA)
            if train_B and teamB_wrap:
                for j, pol in enumerate(ros_mock.team_b_policies):
                    _push_step(traj_B[j], pol.last_feat, pol.last_ax, pol.last_ay,
                               pol.last_logp, pol.last_value, rB)

            steps += 1
            if a >= goal_limit or b >= goal_limit:
                break

        if train_A and teamA_wrap:
            traj_A = _finalize_traj(traj_A)
            _ppo_update(teamA_wrap, traj_A, device, ppo)
        if train_B and teamB_wrap:
            traj_B = _finalize_traj(traj_B)
            _ppo_update(teamB_wrap, traj_B, device, ppo)

        print(f"EP {ep+1}/{total_episodes}  steps={steps}  score A:{last_a} B:{last_b}")

        if ckpt_every and ((ep + 1) % ckpt_every == 0):
            _save_ckpt(teamA_wrap, teamB_wrap, output_dir, ep + 1)

        try:
            ros_mock.close()
        except Exception:
            pass


# ───────────────────────── initialization (one-time per team) ─────────────────────────

def _apply_initialization(team_wrap: TeamPPOWrapper, init_cfg: Dict[str, Any], ros_mock):
    """
    Execute only the steps present in init_cfg; called ONCE per team.
      load_ckpt: "path.pt"        # optional; skipped if not set or not found
      rand_std: 0.01              # optional; 0 or omit = no randomization
      include_core: true|false    # whether to noise the shared encoder as well
      teach:
        enabled: true|false
        iters: 64
        lr: 0.001
    """
    if not team_wrap or not init_cfg:
        return

    # 1) load
    ckpt = init_cfg.get("load_ckpt")
    if ckpt:
        if os.path.exists(ckpt):
            _load_team_checkpoint(team_wrap, ckpt)
            print(f"[init] Loaded {ckpt}")
        else:
            print(f"[init] Skip load (not found): {ckpt}")

    # 2) randomize (after load, if any)
    std = float(init_cfg.get("rand_std", 0.0) or 0.0)
    if std > 0.0:
        _randomize_team_params(team_wrap, std, include_core=bool(init_cfg.get("include_core", True)))
        print(f"[init] Randomized params: std={std} include_core={init_cfg.get('include_core', True)}")

    # 3) teach warmup (optional)
    teach = init_cfg.get("teach", {})
    if teach.get("enabled", False):
        _supervised_teach_hit_puck(team_wrap, ros_mock,
                                   iters=int(teach.get("iters", 64)),
                                   lr=float(teach.get("lr", 1e-3)))
        print(f"[teach] Hit-puck warmup: iters={teach.get('iters', 64)} lr={teach.get('lr', 1e-3)}")


def _load_team_checkpoint(team_wrap: TeamPPOWrapper, ckpt_path: str, strict: bool = True):
    sd = torch.load(ckpt_path, map_location="cpu")
    if "core" in sd:
        team_wrap.core.load_state_dict(sd["core"], strict=strict)
    by_id = {p.agent_id: p for p in team_wrap.policies}
    for h in sd.get("heads", []):
        aid = int(h["agent_id"])
        pol = by_id.get(aid)
        if pol:
            pol.head_x.load_state_dict(h["head_x"], strict=strict)
            pol.head_y.load_state_dict(h["head_y"], strict=strict)
            pol.v_head.load_state_dict(h["v_head"], strict=strict)

def _randomize_team_params(team_wrap: TeamPPOWrapper, std: float, include_core: bool = True):
    with torch.no_grad():
        if include_core:
            for p in team_wrap.core.parameters():
                p.add_(std * torch.randn_like(p))
        for pol in team_wrap.policies:
            for p in pol.head_x.parameters(): p.add_(std * torch.randn_like(p))
            for p in pol.head_y.parameters(): p.add_(std * torch.randn_like(p))
            for p in pol.v_head.parameters():  p.add_(std * torch.randn_like(p))

def _supervised_teach_hit_puck(team_wrap: TeamPPOWrapper, ros_mock, iters: int = 64, lr: float = 1e-3):
    """
    Minimal behavior cloning: label each agent with "move toward puck" on X/Y and
    run CE updates on heads. Uses current sim state; no env changes.
    """
    if iters <= 0:
        return
    params = list(team_wrap.parameters())
    if not params:
        return
    opt = torch.optim.Adam(params, lr=lr)
    device = next(iter(params)).device

    for _ in range(iters):
        ws = ros_mock.sim.get_world_state()
        loss = 0.0
        for pol in team_wrap.policies:
            feat = pol._build_feature(ws)  # [1,F]; wrapper already handles mirroring internally
            px, py = float(ws["puck_x"]), float(ws["puck_y"])
            ax, ay = float(ws["agent_x"][pol.agent_id]), float(ws["agent_y"][pol.agent_id])
            tx = -1 if (px < ax) else (1 if px > ax else 0)
            ty = -1 if (py < ay) else (1 if py > ay else 0)
            target_ax = torch.tensor(0 if tx == -1 else 1 if tx == 0 else 2, device=device).unsqueeze(0)
            target_ay = torch.tensor(0 if ty == -1 else 1 if ty == 0 else 2, device=device).unsqueeze(0)
            logits_x = pol.head_x(feat)
            logits_y = pol.head_y(feat)
            loss = loss + F.cross_entropy(logits_x, target_ax) + F.cross_entropy(logits_y, target_ay)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()


# ───────────────────────── rollout/ppo helpers ─────────────────────────

def _cfg(cfg: Dict[str, Any], key: str, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else getattr(cfg, key, default)

def _get(obj_or_dict: Any, name: str, default=None):
    if isinstance(obj_or_dict, dict):
        return obj_or_dict.get(name, default)
    return getattr(obj_or_dict, name, default)

def _avg_dist(ws: Dict[str, Any], start: int, count: int) -> float:
    px, py = float(ws.get("puck_x", 0.0)), float(ws.get("puck_y", 0.0))
    ax, ay = ws.get("agent_x", []), ws.get("agent_y", [])
    if not count:
        return 0.0
    s = 0.0
    for i in range(start, start + count):
        if i < len(ax) and i < len(ay):
            dx, dy = float(ax[i]) - px, float(ay[i]) - py
            s += math.hypot(dx, dy)
    return s / max(1, count)

def _empty_traj(agent_idx: int):
    return dict(
        feat=[], ax=[], ay=[], logp=[], value=[], rew=[], done=[],
        agent_idx=agent_idx
    )

def _push_step(slot, feat, ax, ay, logp, value, rew):
    if feat is None or ax is None or ay is None or logp is None or value is None:
        return
    slot["feat"].append(feat.squeeze(0).detach().cpu().numpy())
    slot["ax"].append(ax); slot["ay"].append(ay)
    slot["logp"].append(float(logp.item()))
    slot["value"].append(float(value.item()))
    slot["rew"].append(float(rew)); slot["done"].append(0.0)

def _finalize_traj(per_agent: List[Dict]) -> Dict[str, torch.Tensor]:
    for x in per_agent:
        if x["done"]:
            x["done"][-1] = 1.0
    feats, axs, ays, lps, vals, rews, dns, agent_ids = [], [], [], [], [], [], [], []
    for k, x in enumerate(per_agent):
        if not x["feat"]:
            continue
        L = len(x["feat"])
        feats.append(np.stack(x["feat"], axis=0))
        axs.append(np.array(x["ax"], dtype=np.int64))
        ays.append(np.array(x["ay"], dtype=np.int64))
        lps.append(np.array(x["logp"], dtype=np.float32))
        vals.append(np.array(x["value"], dtype=np.float32))
        rews.append(np.array(x["rew"], dtype=np.float32))
        dns.append(np.array(x["done"], dtype=np.float32))
        agent_ids.append(np.full((L,), k, dtype=np.int64))
    if not feats:
        return dict(
            feat=torch.zeros(0, 1),
            ax=torch.zeros(0, dtype=torch.int64),
            ay=torch.zeros(0, dtype=torch.int64),
            logp=torch.zeros(0, dtype=torch.float32),
            value=torch.zeros(0, dtype=torch.float32),
            rew=torch.zeros(0, dtype=torch.float32),
            done=torch.zeros(0, dtype=torch.float32),
            agent_id=torch.zeros(0, dtype=torch.int64),
        )
    feat = np.concatenate(feats, axis=0)
    ax = np.concatenate(axs, axis=0)
    ay = np.concatenate(ays, axis=0)
    logp = np.concatenate(lps, axis=0)
    val = np.concatenate(vals, axis=0)
    rew = np.concatenate(rews, axis=0)
    done = np.concatenate(dns, axis=0)
    agent_id = np.concatenate(agent_ids, axis=0)
    return dict(
        feat=torch.as_tensor(feat, dtype=torch.float32),
        ax=torch.as_tensor(ax, dtype=torch.int64),
        ay=torch.as_tensor(ay, dtype=torch.int64),
        logp=torch.as_tensor(logp, dtype=torch.float32),
        value=torch.as_tensor(val, dtype=torch.float32),
        rew=torch.as_tensor(rew, dtype=torch.float32),
        done=torch.as_tensor(done, dtype=torch.float32),
        agent_id=torch.as_tensor(agent_id, dtype=torch.int64),
    )

def _gae_adv_ret(rew: torch.Tensor, val: torch.Tensor, done: torch.Tensor,
                 gamma: float, lam: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    T = rew.shape[0]
    adv = torch.zeros(T, dtype=torch.float32, device=device)
    last = 0.0
    val_ext = torch.cat([val, torch.zeros(1, device=device)], dim=0)
    for t in reversed(range(T)):
        nt = 1.0 - done[t]
        delta = rew[t] + gamma * val_ext[t + 1] * nt - val_ext[t]
        last = delta + gamma * lam * nt * last
        adv[t] = last
    ret = adv + val
    return adv, ret

def _ppo_update(team_wrap: TeamPPOWrapper, traj: Dict[str, torch.Tensor],
                device, ppo: Dict[str, Any]):
    if traj["feat"].shape[0] == 0:
        return
    params = list(team_wrap.parameters())
    optim = torch.optim.Adam(params, lr=float(ppo["lr"]))
    feat = traj["feat"].to(device)
    ax   = traj["ax"].to(device)
    ay   = traj["ay"].to(device)
    logp_old = traj["logp"].to(device)
    val  = traj["value"].to(device)
    rew  = traj["rew"].to(device)
    done = traj["done"].to(device)
    agent_id = traj["agent_id"].to(device)
    adv, ret = _gae_adv_ret(rew, val, done, float(ppo["gamma"]), float(ppo["gae_lambda"]), device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    B = feat.shape[0]
    mb = int(ppo["minibatch_size"])
    for _ in range(int(ppo["epochs"])):
        idx = torch.randperm(B, device=device)
        for s in range(0, B, mb):
            j = idx[s:s+mb]
            mb_feat, mb_ax, mb_ay = feat[j], ax[j], ay[j]
            mb_logp_old, mb_adv, mb_ret, mb_agent = logp_old[j], adv[j], ret[j], agent_id[j]
            logits_x_list, logits_y_list, values_list = [], [], []
            for k in range(mb_agent.shape[0]):
                aid = int(mb_agent[k].item())
                pol = team_wrap.policies[aid]
                f = mb_feat[k:k+1]
                logits_x_list.append(pol.head_x(f))
                logits_y_list.append(pol.head_y(f))
                values_list.append(pol.v_head(f))
            logits_x = torch.cat(logits_x_list, dim=0)
            logits_y = torch.cat(logits_y_list, dim=0)
            values   = torch.cat(values_list, dim=0).squeeze(-1)
            dist_x = torch.distributions.Categorical(logits=logits_x)
            dist_y = torch.distributions.Categorical(logits=logits_y)
            logp = dist_x.log_prob(mb_ax) + dist_y.log_prob(mb_ay)
            entropy = (dist_x.entropy() + dist_y.entropy()).mean()
            ratio = torch.exp(logp - mb_logp_old)
            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - float(ppo["clip_eps"]), 1.0 + float(ppo["clip_eps"])) * mb_adv
            pg_loss = -torch.min(unclipped, clipped).mean()
            v_loss  = 0.5 * (values - mb_ret).pow(2).mean()
            loss = pg_loss + float(ppo["vf_coef"]) * v_loss - float(ppo["ent_coef"]) * entropy
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()

def _save_ckpt(teamA: Optional[TeamPPOWrapper], teamB: Optional[TeamPPOWrapper],
               directory: str, ep: int):
    os.makedirs(directory, exist_ok=True)
    def save_team(team: TeamPPOWrapper, tag: str):
        if not team:
            return
        torch.save({
            "core": team.core.state_dict(),
            "heads": [dict(
                head_x=p.head_x.state_dict(),
                head_y=p.head_y.state_dict(),
                v_head=p.v_head.state_dict(),
                agent_id=p.agent_id,
            ) for p in team.policies]
        }, os.path.join(directory, f"{tag}_ep{ep}.pt"))
    save_team(teamA, "teamA")
    save_team(teamB, "teamB")
