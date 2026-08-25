import json
import copy


FUTURE_ANNOTATION_FIELDS = (
    'nav_trajectory',
    'dialog',
    'stop_history',
)


def strip_future_annotations(item):
    """Remove human future behavior before an episode reaches either agent."""
    for field in FUTURE_ANNOTATION_FIELDS:
        item.pop(field, None)
    return item


def split_agent_instruction_data(env_instructions):
    """Keep disclosed goal nodes in Guide state, outside Navigator inputs."""
    navigator_instructions = copy.deepcopy(env_instructions)
    guide_goals = {}
    for records in navigator_instructions.values():
        for item in records:
            instr_id = str(item['instr_id'])
            if instr_id in guide_goals:
                raise ValueError(f"duplicate instruction ID: {instr_id}")
            goals = item.pop('end_panos', None)
            if not goals:
                raise ValueError(f"missing Guide goals for instruction {instr_id}")
            guide_goals[instr_id] = tuple(goals)
    return navigator_instructions, guide_goals

def load_instr_datasets(anno_paths):
    data = []
    for anno_path in anno_paths:
        with open(anno_path) as f:
            new_data = json.load(f)
        data += new_data
    return data

def construct_instrs(anno_paths, tokenizer, max_instr_len=512):
    data = []
    for item in load_instr_datasets(anno_paths):
        strip_future_annotations(item)
        item['path_id'] = f"{item['instr_id']}"
        instruction = "target : "+item['target']
        instr_encoding = tokenizer.encode(instruction)
        item['instr_encoding'] = instr_encoding[-max_instr_len:]
        item['instruction'] = instruction
        # item['path'] = item['nav_steps']
        item['heading'] = 3.14
        data.append(item)
    return data


def construct_instrs_universal(anno_paths, tokenizer, max_instr_len=512, prefix="target : "):
    data = []
    for item in load_instr_datasets(anno_paths):
        strip_future_annotations(item)
        item['path_id'] = f"{item['instr_id']}"
        instruction = prefix + item['target']
        instr_encoding = tokenizer.encode(instruction)
        item['instr_encoding'] = instr_encoding[-max_instr_len:]
        item['instruction'] = instruction
        item['heading'] = 3.14
        data.append(item)
    return data
