import copy
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import logging

from evaluator import Evaluator
from holistic_models.DST.DST import DST
from holistic_models.GTL.GTL import GraphVlnAgentModel
from holistic_models.LANA.LANA import LANA
from holistic_models.StatefulConfidence import StatefulConfidenceWtaModule
from holistic_utils.data_utils import (
    construct_instrs_universal,
    split_agent_instruction_data,
)
from holistic_utils.dialog_control import (
    select_dialog_indices,
    select_goal_confirmation_indices,
)
from holistic_utils.distributed import init_distributed
from holistic_utils.misc import set_random_seed
from holistic_utils.temporal_localization import TemporalLocalizationReranker
from ModularGuide import ModularGuide
from ModularNavigator import ModularNavigator
from parser import parse_args


logging.set_verbosity_error()

GOAL_CONFIRMATION = (
    "You are already at the target location. Stop here now; "
    "do not move anywhere else."
)


def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("bert-base-uncased")


def load_instruction_data(args, target_envs, tokenizer):
    env_instructions = {}
    for split in target_envs:
        if split == "val_seen":
            annotation_paths = args.val_seen_anno_paths
        elif split == "val_unseen":
            annotation_paths = args.val_unseen_anno_paths
        elif split == "test":
            annotation_paths = args.test_anno_paths
        else:
            raise ValueError(f"invalid split: {split}")
        env_instructions[split] = construct_instrs_universal(
            annotation_paths.split(","),
            tokenizer,
            args.max_instr_len,
            prefix="target : ",
        )
    return env_instructions


def dialnav(navigator, guide, max_action_len=50):
    """Run one causal Navigator-Guide trajectory for the current batch."""
    navigator.set_next_batch()
    obs = navigator.get_obs()
    batch_size = len(obs)
    goals = guide.get_goals([ob["instr_id"] for ob in obs])
    trajectories = [
        {
            "scan": ob["scan"],
            "start_pano": ob["viewpoint"],
            "end_panos": goals[index],
            "target": ob["instruction"],
            "instr_id": ob["instr_id"],
            "path": [[ob["viewpoint"]]],
            "navigation_detail": [],
        }
        for index, ob in enumerate(obs)
    ]
    navigator.initialize_nav(obs)
    temporal = TemporalLocalizationReranker(
        batch_size,
        guide.shortest_distances,
        weight=guide.args.temporal_weight,
        top_k=guide.args.temporal_top_k,
    )

    for step in range(max_action_len):
        ended_before_step = navigator.get_ended()
        (
            next_viewpoints,
            ended,
            nav_probs,
            navigation_instructions,
            nav_outputs,
        ) = navigator.get_next_action(step, obs)
        next_viewpoints_before_dialog = copy.deepcopy(next_viewpoints)
        nav_probs_before_dialog = nav_probs.detach().clone()

        ask = np.asarray(navigator.wta(step, nav_probs, nav_outputs), dtype=bool)
        proposed_stop = nav_probs.argmax(dim=1).eq(0).detach().cpu().numpy()
        ask = np.logical_or(ask, proposed_stop)
        dialog_indices = select_dialog_indices(
            ask, ended_before_step, ended, proposed_stop
        )

        questions = None
        answers = None
        localized_viewpoints = None
        raw_localized_viewpoints = None
        temporal_expected = [None] * batch_size
        temporal_probability = [None] * batch_size
        question_seen_paths = None
        answer_seen_paths = None

        if dialog_indices:
            scan_ids = [item["scan"] for item in obs]
            viewpoints = [item["viewpoint"] for item in obs]

            # Only the Navigator invokes question generation.
            questions, question_seen_paths = navigator.ask(scan_ids, viewpoints)

            # The Guide consumes question text, localizes it, and prepares answers.
            localized_viewpoints = guide.localize(scan_ids, questions)
            raw_localized_viewpoints = list(localized_viewpoints)
            if len(guide.localization_metadata) != batch_size:
                raise ValueError("GTL did not return per-item top-k metadata")
            for index in dialog_indices:
                (
                    localized_viewpoints[index],
                    temporal_expected[index],
                    temporal_probability[index],
                ) = temporal.rerank(
                    index,
                    scan_ids[index],
                    raw_localized_viewpoints[index],
                    guide.localization_metadata[index],
                    step,
                )

            guide_paths = [
                guide._choose_path(scan, viewpoint, episode_goals)
                for scan, viewpoint, episode_goals in zip(
                    scan_ids, localized_viewpoints, goals
                )
            ]
            answers, answer_seen_paths = guide.answer(
                scan_ids, localized_viewpoints, guide_paths
            )
            temporal.update(step, dialog_indices, answer_seen_paths)

            confirmation_indices = select_goal_confirmation_indices(
                dialog_indices, localized_viewpoints, goals
            )
            for index in confirmation_indices:
                answers[index] = GOAL_CONFIRMATION

            # Only natural-language questions and answers cross into Navigator state.
            navigator.update_instruction(
                dialog_indices, questions, answers, append_behind=True
            )
            (
                next_viewpoints,
                ended,
                nav_probs,
                navigation_instructions,
                nav_outputs,
            ) = navigator.get_next_action(step, obs)

            if confirmation_indices:
                navigator.force_current_stop(confirmation_indices, obs)
                for index in confirmation_indices:
                    next_viewpoints[index] = None
                    ended[index] = True

        current_viewpoints = [item["viewpoint"] for item in obs]
        obs, _ = navigator.navigate(
            next_viewpoints, obs, ended, trajectories
        )
        just_ended = np.logical_and(ended, np.logical_not(ended_before_step))
        entropy = torch.distributions.Categorical(nav_probs).entropy()
        entropy_before = torch.distributions.Categorical(
            nav_probs_before_dialog
        ).entropy()

        for index in range(batch_size):
            if ended[index] and not just_ended[index]:
                continue
            detail = {
                "nav_idx": step,
                "ask": index in dialog_indices,
                "instruction": navigation_instructions[index],
                "gt_viewpoint": current_viewpoints[index],
                "next_vp_ids": next_viewpoints[index],
                "ended": bool(ended[index]),
                "entropy": entropy[index].item(),
            }
            if index in dialog_indices:
                detail.update(
                    {
                        "question": questions[index],
                        "localized_viewpoint": localized_viewpoints[index],
                        "answer": answers[index],
                        "vp_before_dialog": next_viewpoints_before_dialog[index],
                        "entropy_before_dialog": entropy_before[index].item(),
                        "localization_raw_viewpoint": raw_localized_viewpoints[index],
                        "localization_temporal_expected_viewpoint": temporal_expected[index],
                        "localization_temporal_reranked": (
                            raw_localized_viewpoints[index]
                            != localized_viewpoints[index]
                        ),
                        "localization_temporal_selected_probability": temporal_probability[index],
                    }
                )
                if question_seen_paths:
                    detail["question_seen_path"] = question_seen_paths[index]
                if answer_seen_paths:
                    detail["answer_seen_path"] = answer_seen_paths[index]
            trajectories[index]["navigation_detail"].append(detail)

        if all(navigator.get_ended()):
            break

    return trajectories


def run(navigator, guide, max_action_len, env_name, output_file):
    navigator.set_target_env(env_name)
    navigator.reset_epoch()
    results = {}
    finished = False
    while not finished:
        batch = dialnav(navigator, guide, max_action_len=max_action_len)
        for trajectory in batch:
            if trajectory["instr_id"] in results:
                finished = True
            if not finished:
                results[trajectory["instr_id"]] = trajectory
        output = [{"instr_id": key, **value} for key, value in results.items()]
        with open(output_file, "w") as handle:
            json.dump(output, handle, default=_json_default)
    return output


def flatten_path(path):
    flattened = []
    for step in path:
        if isinstance(step, list):
            flattened.extend(flatten_path(step))
        else:
            flattened.append(step)
    return flattened


def make_submit_output(output):
    submission = []
    for item in output:
        record = {key: value for key, value in item.items() if key != "navigation_detail"}
        record["path"] = flatten_path(item.get("path", []))
        for field in ("start_pano", "end_panos", "nav_error", "gt_path"):
            record.pop(field, None)
        record["dialog"] = [
            {
                "nav_idx": detail["nav_idx"],
                "question": detail["question"],
                "answer": detail["answer"],
                "localized_viewpoint": detail["localized_viewpoint"],
                "viewpoint": detail["gt_viewpoint"],
            }
            for detail in item.get("navigation_detail", [])
            if detail.get("ask")
        ]
        submission.append(record)
    return submission


def set_agents(args, target_envs, env_instructions, evaluator, scans):
    navigator_instructions, guide_goals = split_agent_instruction_data(
        env_instructions
    )
    dst_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../modules/nav/DST/map_nav_src",
    )
    sys.path.insert(0, dst_path)

    navigation_model = DST(
        args.basepath,
        {
            "batch_size": args.batch_size,
            "basepath": args.basepath,
            "resume_file": args.nav_resume_file,
            "act_visited_nodes": args.nav_act_visited_nodes,
            "question_weight": 0.5,
            "max_instr_len": args.max_instr_len,
            "max_action_len": args.max_action_len,
            "retain_dialog_history": False,
        },
    )
    navigation_model.eval()
    navigation_model.set_envs(target_envs, navigator_instructions)

    threshold, cooldown = args.wta_mode.split("_")[1:]
    wta_model = StatefulConfidenceWtaModule(
        threshold=float(threshold), cooldown_steps=int(cooldown)
    )
    question_model = LANA(
        args.basepath,
        {
            "scan_list": scans,
            "resume_file": args.qg_resume_file,
            "connectivity_dir": args.connectivity_dir,
            "bpe_path": args.qa_clip_tokenizer_path,
        },
        type="qg",
    )
    answer_model = LANA(
        args.basepath,
        {
            "scan_list": scans,
            "resume_file": args.ag_resume_file,
            "connectivity_dir": args.connectivity_dir,
            "bpe_path": args.qa_clip_tokenizer_path,
            "max_action_len": args.ag_max_answer_seen_path,
        },
        type="ag",
    )
    localization_model = GraphVlnAgentModel(
        args.basepath,
        {"resume_file": args.loc_resume_file, "scan_list": scans},
    )
    guide = ModularGuide(
        args,
        answer_model,
        localization_model,
        {
            "shortest_distances": evaluator.shortest_distances,
            "shortest_paths": evaluator.shortest_paths,
            "goals_by_instr_id": guide_goals,
        },
    )
    navigator = ModularNavigator(
        args, navigation_model, wta_model, question_model
    )
    return navigator, guide


def _json_default(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main():
    args = parse_args()
    if not args.wta_mode.startswith("ctc_"):
        raise ValueError("this release supports only ctc_THRESHOLD_COOLDOWN WTA")
    target_envs = args.env_names.split(",")
    os.makedirs(args.output_path, exist_ok=True)

    rank = 0
    if args.world_size > 1:
        rank = init_distributed(args)
        torch.cuda.set_device(args.local_rank)
    set_random_seed(args.seed + rank)

    tokenizer = get_tokenizer()
    env_instructions = load_instruction_data(args, target_envs, tokenizer)
    scans = sorted(
        {
            item["scan"]
            for env_name in target_envs
            for item in env_instructions[env_name]
        }
    )
    evaluator = Evaluator(
        args.connectivity_dir,
        scans,
        success_margin=args.success_margin,
        error_margin=args.error_margin,
    )
    navigator, guide = set_agents(
        args, target_envs, env_instructions, evaluator, scans
    )

    submission = {}
    for env_name in target_envs:
        output = run(
            navigator,
            guide,
            args.max_action_len,
            env_name,
            os.path.join(args.output_path, f"{env_name}.json"),
        )
        submission[env_name] = make_submit_output(output)
    with open(os.path.join(args.output_path, "submit.json"), "w") as handle:
        json.dump(submission, handle, default=_json_default)


if __name__ == "__main__":
    main()
