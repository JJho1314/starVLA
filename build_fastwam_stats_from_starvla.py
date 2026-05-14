"""Translate starVLA training dataset_statistics.json into FastWAM eval stats schema.

starVLA schema:   {franka: {action: {min,max,...}, state: {min,max,...}, num_*}}
FastWAM schema:   {state: {default: {global_min,global_max,stepwise_*,q01,q99,...}},
                   action: {default: {global_min,global_max,...}}, num_episodes, num_transition}

The fastwam_eval pipeline reads the FastWAM-schema file and uses its
`state.default.global_min/global_max` and `action.default.global_min/global_max`
to normalize/denormalize at eval time. If we feed it FastWAM official stats
while the model was trained with starVLA stats, the conditioning is off and
SR drops to 0.

This script ports starVLA's min/max into the FastWAM schema so the eval
pipeline normalizes with the values the model actually saw at training.
"""
import argparse, json, copy
from pathlib import Path


def starvla_to_fastwam_stats(sv: dict, robot_key: str = "franka") -> dict:
    sv_robot = sv[robot_key]
    sv_state = sv_robot["state"]
    sv_action = sv_robot["action"]

    def block(s):
        # FastWAM schema needs "stepwise_*" and "global_*" min/max plus q01/q99.
        # We don't have stepwise from starVLA — use global value broadcast (single timestep).
        gmin, gmax = list(s["min"]), list(s["max"])
        return {
            "stepwise_min": [gmin],
            "stepwise_max": [gmax],
            "global_min": gmin,
            "global_max": gmax,
            "stepwise_q01": [list(s.get("q01", gmin))],
            "stepwise_q99": [list(s.get("q99", gmax))],
            "global_q01": list(s.get("q01", gmin)),
            "global_q99": list(s.get("q99", gmax)),
            "mean": list(s.get("mean", gmin)),
            "std":  list(s.get("std",  [1.0] * len(gmin))),
        }

    return {
        "state":  {"default": block(sv_state)},
        "action": {"default": block(sv_action)},
        "num_episodes":   sv_robot.get("num_trajectories", 0),
        "num_transition": sv_robot.get("num_transitions", 0),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="starVLA dataset_statistics.json")
    p.add_argument("--dst", required=True, help="output FastWAM-schema stats json")
    p.add_argument("--robot-key", default="franka")
    args = p.parse_args()
    sv = json.load(open(args.src))
    out = starvla_to_fastwam_stats(sv, args.robot_key)
    Path(args.dst).parent.mkdir(parents=True, exist_ok=True)
    with open(args.dst, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[+] wrote {args.dst}")
    print(f"    state.global_min[0:4] = {out['state']['default']['global_min'][:4]}")
    print(f"    action.global_min[0:4] = {out['action']['default']['global_min'][:4]}")


if __name__ == "__main__":
    main()
