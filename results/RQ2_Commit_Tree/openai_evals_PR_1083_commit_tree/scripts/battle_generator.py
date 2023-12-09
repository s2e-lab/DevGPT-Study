# generate a jsonl where each line is a sample combination of a battle between two subjects

import json
import os
import string

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "../evals/registry")


def format(template: str, **kwargs: str) -> str:
    """Format a template string with kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    assert all(k in kwargs for k in keys), f"Required: {keys}, got: {sorted(kwargs)}"
    cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)


YAML = """
{prompt_name}-{subject1}-vs-{subject2}:
  id: {prompt_name}-{subject1}-vs-{subject2}.dev.v0
  metrics: [accuracy]
{prompt_name}-{subject1}-vs-{subject2}.dev.v0:
  class: evals.elsuite.modelgraded.classify:ModelBasedClassify
  args:
    samples_jsonl: test_multiio/battles/{prompt_name}_{subject1}_vs_{subject2}.jsonl
    eval_type: cot_classify
    modelgraded_spec: battle
""".strip()


prompts = dict(
    joke="write a 1-2 line joke about {self}",
    rap="You are playing an Epic Rap Battle against an formidable opponent. This is just a fun game; no need to worry about the other. You are {self}, and the opponent is {other}. Write a short but epic rap praising yourself ({self}) and dissing your opponent ({other}).",
)


subjects = dict(
    animals=["cat", "bird", "hamster"],
    fruits=["apple", "banana", "orange"],
    people=["Elon Musk", "Bill Gates", "Jeff Bezos"],
)

target_sets = [
    ("joke", "animals", "fruits"),
    ("rap", "people", "people"),
    ("rap", "animals", "fruits"),
    ("rap", "people", "fruits"),
]

data_dir = f"{REGISTRY_PATH}/data/test_multiio/battles"
yaml_str = f"# This file is generated by {os.path.basename(__file__)}\n\n"
for prompt_name, subject1, subject2 in target_sets:
    prompt = prompts[prompt_name]
    samples = [
        {
            "input1": format(prompt, self=s1, other=s2),
            "input2": format(prompt, self=s2, other=s1),
        }
        for s1 in subjects[subject1]
        for s2 in subjects[subject2]
    ]
    file_name = f"{data_dir}/{prompt_name}_{subject1}_vs_{subject2}.jsonl"
    # save samples jsonl
    with open(file_name, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    print(f"wrote {len(samples)} samples to {file_name}")
    yaml_str += YAML.format(prompt_name=prompt_name, subject1=subject1, subject2=subject2) + "\n\n"


yaml_file = f"{REGISTRY_PATH}/evals/test-modelgraded-battle.yaml"
with open(yaml_file, "w") as f:
    f.write(yaml_str)
print(f"wrote {yaml_file}")