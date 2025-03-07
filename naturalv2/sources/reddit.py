import datetime
import os

import pandas as pd

from naturalv2.sources.reddit_utils import (
    download_sub_data,
    filter_by_date,
    get_context_post_df,
    get_reddit_synonyms,
    rule_based_filter,
    subreddit_relevance_llm,
)


class RedditSet:
    def __init__(self, data_path, trial, match_method, llm, download=False):
        self.data_path = data_path
        self.subs_about = pd.read_csv(self.data_path + "subs_about.csv", index_col=0)
        self.trial = trial
        self.llm = llm

        self.log_path = os.path.join(
            self.data_path, f"{trial.nctid}_{match_method}_{llm.model_name}_reddit.log"
        )
        if os.path.exists(self.log_path):
            with open(self.log_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if line.startswith("treatment_names:"):
                        self.treatment_names = eval(line.split(": ")[1])
                    elif line.startswith("outcome_words:"):
                        self.outcome_words = eval(line.split(": ")[1])
                    elif line.startswith("keywords:"):
                        self.trial_keywords = eval(line.split(": ")[1])
        else:
            self.treatment_names = get_reddit_synonyms(
                [i.title for i in trial.interventions], llm
            )
            self.outcome_words = get_reddit_synonyms(
                [o.title for o in trial.primary_endpoints], llm
            )
            self.trial_keywords = get_reddit_synonyms(
                trial.keywords + trial.conditions, llm
            )
            with open(self.log_path, "w") as f:
                f.write(f"treatment_names: {self.treatment_names}\n")
                f.write(f"outcome_words: {self.outcome_words}\n")
                f.write(f"keywords: {self.trial_keywords}\n")

        if download:
            self.subreddits = self.get_subreddits(match_method, trial)
            self.download_data()

    def get_subreddits(self, method, trial):
        relevant_subs = []
        trial_keywords = [self.trial_keywords, self.treatment_names, self.outcome_words]
        for row in self.subs_about.iterrows():
            sub_name, desc, public_desc = row[1].to_list()
            desc = f"Subreddit: r/{sub_name}.\nDescription: {desc}\nPublic description: {public_desc}"
            if method == "string_match":
                if any(
                    keyword.lower() in desc.lower() for keyword in self.treatment_names
                ) or any(
                    keyword.lower() in desc.lower() for keyword in self.trial_keywords
                ):
                    relevant_subs.append(sub_name)
            elif method == "llm":
                answer = subreddit_relevance_llm(desc, trial_keywords, self.llm)
                if answer.lower().startswith("yes"):
                    with open(self.log_path, "a") as f:
                        f.write(f"subreddit {sub_name} relevance: {answer}\n")
                    relevant_subs.append(sub_name)
        print(len(relevant_subs), "relevant subreddits found!")
        return relevant_subs

    def download_data(self):
        self.data_files = {}
        for sub in self.subreddits:
            submissions_path = self.data_path + f"{sub}_submissions.csv"
            comments_path = self.data_path + f"{sub}_comments.csv"
            if not os.path.exists(submissions_path):
                download_sub_data(sub, "submissions", self.data_path)
            if not os.path.exists(comments_path):
                download_sub_data(sub, "comments", self.data_path)
            self.data_files[f"{sub}_submissions"] = submissions_path
            self.data_files[f"{sub}_comments"] = comments_path

    def curate_data(self, date_filter=False):
        rule_filtered_df = pd.DataFrame()
        save_path = self.data_path + f"{self.trial.nctid}_reddit_rule_based.csv"
        # treatment_names = get_reddit_synonyms([i.title for i in self.trial.interventions], self.llm)
        # outcome_words = get_reddit_synonyms([o.title for o in self.trial.primary_endpoints], self.llm)
        if not os.path.exists(save_path):
            for sub in self.subreddits:
                submissions = pd.read_csv(self.data_files[f"{sub}_submissions"])
                comments = pd.read_csv(self.data_files[f"{sub}_comments"])
                if date_filter:
                    trial_date = datetime.datetime.strptime(
                        self.trial.results_first_posted, "%Y-%m-%d"
                    )
                    trial_date_utc = int(
                        trial_date.replace(tzinfo=datetime.timezone.utc).timestamp()
                    )
                    submissions = filter_by_date(submissions, trial_date_utc)
                    comments = filter_by_date(comments, trial_date_utc)
                submissions = rule_based_filter(submissions, "selftext")
                comments = rule_based_filter(comments, "body")
                merged_df = get_context_post_df(
                    submissions, comments, self.treatment_names, self.outcome_words
                )
                rule_filtered_df = pd.concat(
                    [rule_filtered_df, merged_df], ignore_index=True
                )
                rule_filtered_df.to_csv(save_path)
            rule_filtered_df = rule_filtered_df.drop_duplicates("post")
            rule_filtered_df.to_csv(save_path)
        else:
            rule_filtered_df = pd.read_csv(save_path, index_col=0)
        self.curated_data = rule_filtered_df
        return self.curated_data
