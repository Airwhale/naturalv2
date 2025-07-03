import json
import praw
import pandas as pd
from create_study import Study
from naturalv2.evals.experiment import Experiment
from naturalv2.sources.reddit_utils import get_sub_about_info

study = Study.from_yaml("/mfs1/u/nikita/naturalv2/studies/nervous_system_diseases_study.yaml")
train_ncts = [list(trial.keys())[0] for trial in study.train_trials]
val_ncts = [list(trial.keys())[0] for trial in study.val_trials]
test_ncts = [list(trial.keys())[0] for trial in study.test_trials]

splits = (
    ["train"] * len(train_ncts) + ["val"] * len(val_ncts) + ["test"] * len(test_ncts)
)
all_ncts = train_ncts + val_ncts + test_ncts

exp_subreddits: dict[str, list[str]] = {}
condition_subreddits: dict[str, list[str]] = {}

subs_about = get_sub_about_info("/mfs1/u/nikita/naturalv2/reddit_data")
pushshift_subreddits = subs_about["sub"].to_list()

# Structure for multi-row DataFrame
eval_rows = []

reddit = praw.Reddit(
    client_id="bp6e3QF-KwAIqcnCD_Z0GA", 
    client_secret="P-26vz3UFEou2en3xvXBGpTMKJTEyg", 
    password="naturalv2", username="natural_scaling", 
    user_agent="natural"
)
# all_ncts = ["NCT02191579"]
all_ncts = ["NCT03828539"]
splits = ["train"]
for nct_id, split in zip(all_ncts, splits):
    status = "active" if split == "test" else "completed"
    exp = Experiment("/mfs1/u/nikita/naturalv2/", nct_id, status=status)
    conditions = exp.conditions if exp.conditions else []
    trial_keywords = exp.trial_keywords if exp.trial_keywords else []
    treatment_names = exp.treatment_names if exp.treatment_names else []
    
    for condition in conditions:
        if condition not in condition_subreddits:
            # Search for subreddits related to the condition
            subreddits = [
                subreddit.display_name
                for subreddit in reddit.subreddits.search(condition)
            ]
            # set intersection
            relevant_subs = set(subreddits).intersection(pushshift_subreddits)
            condition_subreddits[condition] = list(relevant_subs)
        else:
            relevant_subs = condition_subreddits[condition]
        
        exp_subreddits.setdefault(exp.title, []).extend(relevant_subs)
    exp_subreddits[exp.title] = list(set(exp_subreddits[exp.title]))

    # For eval_df: one row per (trial_title, condition_words, subreddit)
    for condition in conditions:
        for subreddit in exp_subreddits[exp.title]:
            posts = [
                "**Title**: " + submission.title + "\n\n" + "**Post content**: " + submission.selftext[:1000]
                for submission in reddit.subreddit(subreddit).search(condition)
            ][:5]
            # posts_sample_str = "\n\n".join(posts)
            print(subreddit, condition)
            eval_rows.append({
                # "trial_title": exp.title,
                # "condition_words": condition,
                "subreddit": subreddit,
                "example_posts": posts
            })

    # Save eval_df 
    eval_df = pd.DataFrame(eval_rows, columns=["trial_title", "condition_words", "subreddit", "posts_sample"])
    eval_df.to_csv("scratch/NCT03828539_subreddit_eval.csv", index=False)

# Save experiment subreddits to a JSON file
with open("scratch/NCT03828539_experiment_subreddits.json", "w", encoding="utf-8") as f:
    json.dump(exp_subreddits, f, indent=4)

with open("scratch/NCT03828539_subreddit_examples.json", "w", encoding="utf-8") as f:
    json.dump(eval_rows, f, indent=4)