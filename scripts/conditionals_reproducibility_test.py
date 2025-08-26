import argparse
import os

import litellm
import numpy as np
import torch
from scipy.special import softmax
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig

from naturalv2.models.lm import get_prompt_logprobs


torch.use_deterministic_algorithms(True)

# V1 only: Turn off multiprocessing to make the scheduling deterministic.
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

os.environ["HF_HOME"] = "/projects/natural/.cache"
SEED = 1337
TEMPERATURE = 1.0

# prompts
PROMPT = [
    """You are a medical assistant aiding a physician.
I am going to ask you a few multiple choice questions about a report on a patient's experience with a treatment.
Please, answer accordingly.

I will give you a report about an individual's experience with a treatment and its effect.
Then, I'll show you a few questions with their correct answers.
Finally, I'll give you some multiple choice questions and options to choose from. Pick the right answer.

Report:
```
I had major brain fog, tingling in my extremities, and hair loss. The hair loss took a few months to come and didn’t stop until after I was off topamax for a while. Topamax was pretty bad but it did help with migraines.

Questions and their correct answers:
Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Age, Continuous
A: Less than or equal to 38.0.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Sex: Female, Male
A: Female.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Race/Ethnicity, Customized
A: White.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Baseline Monthly Migraine Days (MMDs) categories
A: 8-14.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Duration
A: Less than or equal to 0 days 00:06:00.

Multiple Choice Questions

Q: Which of the following treatments did the individual described in the report take?
['Erenumab', 'Topiramate']
Options: a) Erenumab b) Topiramate
A: {treatment}

Q: Does the individual described in the report count positively towards: Proportion of Patients With Treatment Discontinuation Due to an Adverse Event (AE) During the Double-blind Treatment Epoch/Period (DBTE) after taking the treatment for 24 Weeks?
Options: a) No b) Yes
A: {discontinuation}
```""",
    """You are a medical assistant aiding a physician.
I am going to ask you a few multiple choice questions about a report on a patient's experience with a treatment.
Please, answer accordingly.

I will give you a report about an individual's experience with a treatment and its effect.
Then, I'll show you a few questions with their correct answers.
Finally, I'll give you some multiple choice questions and options to choose from. Pick the right answer.

Report:
```
Love triptans, miracle drugs really. It can knock a potentially week-long migraine to its knees in hours. I needed to take sumatriptan with an antinauseant but others have been fine without.

I didn't enjoy Effexor and the discontinuation syndrome was hell. It wouldn't be my first choice. Topiramate really reduced my migraines and I was able to go back to work full time, although I have now discontinued it because I couldn't hack the side effects any longer - and I found other solutions (behaviour, diet changes, Botox, nerve block) that work for me.

Some meds don't work for people, others allow you to breathe for the first time. It can be trial and error and tapering up slowly to reduce side effects, but I do hope you find a preventative that helps!

Questions and their correct answers:
Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Age, Continuous
A: Less than or equal to 38.0.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Sex: Female, Male
A: Female.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Race/Ethnicity, Customized
A: White.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Baseline Monthly Migraine Days (MMDs) categories
A: >=15.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Duration
A: Greater than 0 days 00:06:00.

Multiple Choice Questions

Q: Which of the following treatments did the individual described in the report take?
['Erenumab', 'Topiramate']
Options: a) Erenumab b) Topiramate
A: {treatment}

Q: Does the individual described in the report count positively towards: Proportion of Patients With Treatment Discontinuation Due to an Adverse Event (AE) During the Double-blind Treatment Epoch/Period (DBTE) after taking the treatment for 24 Weeks?
Options: a) No b) Yes
A: {discontinuation}
```""",
    """You are a medical assistant aiding a physician.
I am going to ask you a few multiple choice questions about a report on a patient's experience with a treatment.
Please, answer accordingly.

I will give you a report about an individual's experience with a treatment and its effect.
Then, I'll show you a few questions with their correct answers.
Finally, I'll give you some multiple choice questions and options to choose from. Pick the right answer.

Report:
```
Hey guys!

Wanted to start by saying I am not asking for medical advice, just experience of others.

In high school I was prescribed Topamax that worked fairly well at preventing migraine. In college I tried Topamax XR and had horrible suicidal ideation out of thin air that stopped after I stopped the medication.

Also tried an emergency med that I can’t remember the name of. It was a nasal spray..maybe aimovig? That caused horrible heart palpitations and nausea and didn’t help my migraines.

Have been managing migraine fairly well with magnesium for the past few years. It lessened the frequently of my migraines and also the intensity. With age and my body changing and different birth control methods, my migraines have changed a bit and I’m going back to a neurologist soon just to check in and am finding myself nervous to explore any prescription options after my experiences.

Has anyone had something similar happen but ended up finding a prescription med that was successful?

Questions and their correct answers:
Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Age, Continuous
A: Less than or equal to 38.0.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Sex: Female, Male
A: Female.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Race/Ethnicity, Customized
A: White.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Baseline Monthly Migraine Days (MMDs) categories
A: 8-14.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Duration
A: Greater than 0 days 00:06:00.

Multiple Choice Questions

Q: Which of the following treatments did the individual described in the report take?
['Erenumab', 'Topiramate']
Options: a) Erenumab b) Topiramate
A: {treatment}

Q: Does the individual described in the report count positively towards: Proportion of Patients With Treatment Discontinuation Due to an Adverse Event (AE) During the Double-blind Treatment Epoch/Period (DBTE) after taking the treatment for 24 Weeks?
Options: a) No b) Yes
A: {discontinuation}
```""",
    """You are a medical assistant aiding a physician.
I am going to ask you a few multiple choice questions about a report on a patient's experience with a treatment.
Please, answer accordingly.

I will give you a report about an individual's experience with a treatment and its effect.
Then, I'll show you a few questions with their correct answers.
Finally, I'll give you some multiple choice questions and options to choose from. Pick the right answer.

Report:
```
I was on Topimax for a few months, it was the first migraine medication I was ever prescribed (Now I take nurtec and ajovy). The worse months of my life. I had no migraines (only upside) but the depression, crippling anxiety, dread, and suicidal ideation, plus muscle spasms, memory loss, brain fog, and loss of appetite/weight loss was so bad. I quit cold turkey. My neurologist wanted to wain me off of it little by little, due to the dosage I was taking but I told them that he would have to put me on suicide watch if I had to continue to take it. All I have ever heard about the drug is negative things and how adverse it has been for people. Definitely talk to your neurologists about this! Do not wait.

Questions and their correct answers:
Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Age, Continuous
A: Less than or equal to 38.0.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Sex: Female, Male
A: Female.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Race/Ethnicity, Customized
A: White.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Baseline Monthly Migraine Days (MMDs) categories
A: 15-20.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Duration
A: Less than or equal to 0 days 00:06:00.

Multiple Choice Questions

Q: Which of the following treatments did the individual described in the report take?
['Erenumab', 'Topiramate']
Options: a) Erenumab b) Topiramate
A: {treatment}

Q: Does the individual described in the report count positively towards: Proportion of Patients With Treatment Discontinuation Due to an Adverse Event (AE) During the Double-blind Treatment Epoch/Period (DBTE) after taking the treatment for 24 Weeks?
Options: a) No b) Yes
A: {discontinuation}
```""",
    """You are a medical assistant aiding a physician.
I am going to ask you a few multiple choice questions about a report on a patient's experience with a treatment.
Please, answer accordingly.

I will give you a report about an individual's experience with a treatment and its effect.
Then, I'll show you a few questions with their correct answers.
Finally, I'll give you some multiple choice questions and options to choose from. Pick the right answer.

Report:
```
It tooks me several months to cope with side effects but now it is ok for me. My neurologist prescribed it with 25mg, then 50 and now 100mg a day. It takes few weeks before it has actual effects but when it had, it was a miracle for me (passing from daily migraine to two or three a month).
But I agree about the side effects, they can be very strong (I let know some people around me), and I spent the first six months almost crying at work, throwing up in shower or barely feeling my hands all day long ... Everything was tasting like metal or blood, I was disgusted by most of things I used to enjoyed...
But even if I continue taking it now (after 4-5y) all the side effects tend to reduce a lot (no more metal taste, almost no brainfog, my weight is stable, ...).
Do not hesitate to discuss your conditions with your therapist, they may adjust better treatment for you.

Questions and their correct answers:
Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Age, Continuous
A: Less than or equal to 38.0.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Sex: Female, Male
A: Female.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Race/Ethnicity, Customized
A: White.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Baseline Monthly Migraine Days (MMDs) categories
A: Other.

Q: Choose the correct answer from the provided options about the following feature for the individual described in the report:
Feature: Duration
A: Less than or equal to 0 days 00:06:00.

Multiple Choice Questions

Q: Which of the following treatments did the individual described in the report take?
['Erenumab', 'Topiramate']
Options: a) Erenumab b) Topiramate
A: {treatment}

Q: Does the individual described in the report count positively towards: Proportion of Patients With Treatment Discontinuation Due to an Adverse Event (AE) During the Double-blind Treatment Epoch/Period (DBTE) after taking the treatment for 24 Weeks?
Options: a) No b) Yes
A: {discontinuation}
```""",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline", action="store_true", help="Run the test using offline vLLM"
    )
    args = parser.parse_args()

    treatment_options = ["Erenumab", "Topiramate"]
    discontinuation_options = ["No", "Yes"]
    prompts = {}
    for idx, p in enumerate(PROMPT):
        for treatment in treatment_options:
            for discontinuation in discontinuation_options:
                prompts.setdefault(f"prompt_{idx}", []).append(
                    p.format(treatment=treatment, discontinuation=discontinuation)
                )

    if args.offline:
        # Initialization
        llm = LLM(
            model="/model-weights/Llama-3.3-70B-Instruct",
            seed=SEED,
            max_model_len=16384,
            tensor_parallel_size=4,
            trust_remote_code=True,
            max_num_batched_tokens=2048,
            compilation_config=CompilationConfig(
                level=3, cache_dir="/projects/natural/.cache"
            ),
        )
        tokenizer = llm.get_tokenizer()

        sampling_params = SamplingParams(
            temperature=TEMPERATURE,
            max_tokens=1,
            prompt_logprobs=0,
            seed=SEED,
        )

        outputs = {}
        for idx, ps in prompts.items():
            outputs[idx] = llm.generate(ps, sampling_params)

        logprobs = {}
        for idx, output_list in outputs.items():
            logprobs[idx] = []
            for output in output_list:
                input_tokens = tokenizer.encode(output.prompt, add_special_tokens=True)
                logprobs[idx].append(
                    [
                        i[j].logprob
                        for i, j in zip(output.prompt_logprobs[1:], input_tokens[1:])
                    ]
                )
    else:
        logprobs = {}
        for idx, ps in prompts.items():
            for p in ps:
                response = litellm.text_completion(
                    model="hosted_vllm/gpt-oss-120b",
                    prompt=p,
                    base_url="http://gpu190:8080/v1",
                    api_key="EMPTY",
                    prompt_logprobs=0,
                    max_tokens=1,
                    temperature=TEMPERATURE,
                    seed=SEED,
                )

                prompt_logprobs_obj = get_prompt_logprobs(response)
                if prompt_logprobs_obj is None:
                    print(f"No logprobs for prompt: {ps}")
                    continue
                logprobs.setdefault(idx, []).append(prompt_logprobs_obj.logprobs)

    # Save raw logprobs for later analysis
    os.makedirs("outputs", exist_ok=True)
    np.savez_compressed(
        "outputs/gpt-oss-conditionals_logprobs_offline.npz"
        if args.offline
        else "outputs/gpt-oss-conditionals_logprobs_online.npz",
        # flatten logprobs
        **{
            f"{k}_treatment_{ops[0]}_discontinuation_{ops[1]}": np.array(v[i])
            for k, v in logprobs.items()
            for i, ops in enumerate(
                [
                    (treatment, discontinuation)
                    for treatment in ["a", "b"]
                    for discontinuation in ["a", "b"]
                ]
            )
        },
    )

    logprob_sums = {idx: [sum(p) for p in lp] for idx, lp in logprobs.items()}
    print(
        "[Offline]" if args.offline else "[Online]", f"Logprob sums: \n{logprob_sums}"
    )
    probs = {idx: softmax(np.array(lp_sums)) for idx, lp_sums in logprob_sums.items()}
    print("[Offline]" if args.offline else "[Online]", f"Probabilities: \n{probs}")


if __name__ == "__main__":
    main()
