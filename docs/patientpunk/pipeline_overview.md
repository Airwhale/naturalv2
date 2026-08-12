# Predicting Clinical Trial Outcomes from Patient Communities

*Working draft — Shaun & Claude. Written for Polina; useful to anyone picking this up cold.*
*Last updated 2026-08-11.*

*This document is specifically aimed at Polina. Hi Polina!*

---

## 1. Orientation

We are trying to estimate what a clinical trial will find, using what patients wrote about the same
treatment **before the trial reported**.  This matches what Nikita et. al did with [https://www.cs.toronto.edu/~nikita/natural/](https://www.cs.toronto.edu/~nikita/natural/)  
, but with a different population.  This is a different approach then the between group differences that we showed in our pilot paper,  and should be much more exact.  For our data we are using three Long-COVID subreddits (r/covidlonghaulers, r/LongCovid, r/LongHaulersRecovery). The trials come from **ClinicalTrials.gov**, pulled through naturalv2's own downloader against the CT.gov API v2, with two additions of ours: a handful of **ISRCTN** trials adapted into the same schema, and a **papers-as-labels** path that recovers completed trials which never posted structured results by extracting the primary endpoint from the published paper.

After a bunch of fiddling, I have **the pipeline runing end-to-end, but I do not trust validity.** Natural's paper had a lot of things going for it that ours does not:

1. **Their endpoints are proportions; ours are continuous symptom scales.** Their headline result is
  "within 3 percentage points of ground truth" — a *percentage point* metric, i.e. binary/response-rate  
   outcomes. Binary endpoints are bounded in [0,1], are directly enumerable by the default estimator,  
   and are far easier to read off a post ("did they improve — yes or no?"). Our data is often in terms of subjective severaity,  fatigue reports, etc. **We cannot compare our error to their 3 points.**
2. **They report against Phase 3/4 trials.** Those are large, well-powered, and mostly test
  established drugs. Our Long-COVID set is small early-phase work — lithium's arm was n=24.
3. **Model scale.** The method assumes a large model; we are running a 7B smoke model (see §11.5). Once we get everything else working better we should ask Nikita to run it on UofT's machines or source some more heavy duty VRAM having machines to get an 80B paramater model or larger running it.
4. **Label cleanliness.** Several of our endpoints report a *change from baseline* while describing an
  absolute scale, which silently mismatches the prediction target (§10.1). A response rate has no
   equivalent failure mode.
5. **Treatment naming.** There is a LOT of variaty in how different treatments are named in the Long covid community.

Note that Nikita's current repo is here: [github.com/nikitadhawan/naturalv2](https://github.com/nikitadhawan/naturalv2).
I'm currently working out of a fork branch,
[Airwhale/naturalv2 @ shaun/patientpunk-integration](https://github.com/Airwhale/naturalv2/tree/shaun/patientpunk-integration)

This working branch is currently an AI vibe-coded mess,  and I will be cleaning it up.  I just wanted to get it running to figure out where things were failing in the main repo's expermental/scraping logic and be ready to talk about this if needed for thursday.  
(its `main` is byte-identical to hers, so the branch diff *is* the set of our changes),   

I am recording things to improve upon in her repo here:  
[docs/patientpunk/findings.md](https://github.com/Airwhale/naturalv2/blob/shaun/patientpunk-integration/docs/patientpunk/findings.md).

---

## 2. What NATURAL is:

**Paper:** Dhawan, Cotta, Ullrich, Krishnan & Maddison, *End-To-End Causal Effect Estimation from
Unstructured Natural Language Data*, arXiv:2407.07018 (NeurIPS 2024). We run pinned commit `7a2e006`.(naturalv2)

NATURAL uses an LLM as a *measurement instrument*. For a given trial it reads each patient report  
and emits the four variables a causal estimator needs: which of **this trial's** treatment arms the  
author took, what value they would have reported on **this trial's** primary outcome scale, the  
covariates describing them, and the probability they would have met the trial's inclusion criteria. It pulls from the log probs of the next predicted token,  and thus requires the use of open source models to get at this.  

Each simulated report becomes one row. That table is a synthetic observational dataset, and a textbook causal  
estimator i.e. inverse propensity weighting or outcome regression runs over it unchanged to produce  
an estimate per expermental arm. The paper's claim is that these land within **3 percentage points** of the  
randomised ground truth across six datasets, including real Phase 3/4 trials for the studies it looked at. 

Note that **There is no memory between runs.** Each trial is estimated independently, from its own community  
text, with nothing carried over: no weights, no cache, no learning from the previous trial. The  
trials are **benchmarks that supply ground truth**, not training data for a generalized predictive model.

**How this differs from what PatientPunk has done before.** Our pilot compared groups: aggregate  
sentiment differences between or within populations. That gives a correlational contrast and no  
adjustment for who ends up in which group. We can only make broad directional claims with this,  and I marvel  
that we were able to get as good of data as we did.    

NATURAL instead reconstructs *per-patient* rows and runs formal causal machinery over them; covariate adjustment, propensity or outcome modelling, and an  
explicit reweighting toward the trial-eligible population. It is a more exact claim that attempts to reproduce or predict specific findings of a study.

**How what we are doing differs from what Nikita did: The math is the same.** We run her code, on her  
pinned commit, with her estimators. Everything that differs is *input and context*, a different  
patient population (Long-COVID Reddit), continuous symptom scales rather than proportion outcomes, a  
7B model rather than a large one, and small early-phase trials rather than Phase 3/4. The §1[please link] list is  
the full inventory of those differences. **I generally am finding lots of errors that I think biological expermentalists would find, and poor scraping practices,  which makes sense looking at the work of a math heavy ML lab I suppose.**

---

## 3. The three stages of NATURAL

Each stage is a separate CLI and writes its output to disk, so they are run independently and
re-run cheaply. The whole system is a funnel: 2.45M documents in, a few dozen usable patient
reports per trial out.


| stage                  | CLI             | in → out                                                              | what it actually does                                                                                                                                                                                                                                                                                                                                  |
| ---------------------- | --------------- | --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **1. Build the study** | `create_study`  | trial registries → trials, criteria, splits, a per-trial `Experiment` | Pulls trial records, applies the eligibility filters (§4.1), sorts trials into train/val/test by date, and builds one `Experiment` per trial holding its treatments, outcomes, covariates and — if results were posted — the label we score against.                                                                                                   |
| **2. Curate evidence** | `filter_curate` | corpus → the patient reports relevant to one trial                    | Assembles each Reddit comment into a thread-aware "report" (comment + the post it replied to + that author's other replies), then keeps only reports mentioning one of *this trial's* treatments. Purely lexical string matching — no LLM, no cost. Output is a per-trial candidate pool: 3,136 reports for lithium, 42,617 for LIFT.                  |
| **3. Estimate**        | `estimate_ate`  | reports + trial → average potential outcome per arm, with CI          | Where all the LLM work happens. Six sub-stages progressively filter and read the reports (is this relevant? does it describe both the treatment and the outcome? what are this person's covariates? what outcome value did they report?), producing one synthetic patient per surviving report. A standard causal estimator then runs over that table. |


**Sources for stage 1.** Primarily **ClinicalTrials.gov** (API v2, via naturalv2's own downloader).
We add two paths of our own: a few **ISRCTN** trials[can you provide hyperlink?] adapted into the same schema, and  
**papers-as-labels**, which rescues completed trials that never posted structured results by
extracting the primary endpoint from the published paper. Those are flagged `registry_adapted` and
`paper_extracted` respectively, so they can be excluded from any analysis that wants CT.gov-only.

---

## 4. Stage 1 — Trial selection

### 4.1 The criteria used to select trials

**Tier 1 — Structural**

Two names to define first, since they recur throughout:

- `**trial_filters`** — the block in naturalv2's config holding the four booleans below. It is the
only place trial eligibility is decided upstream.
- `**noparallel_notbinary`** — the name we gave our filter settings, and the string that appears in
every output path and filename. It is purely descriptive: *no* `parallel` requirement, *not*
`binary` endpoints. Upstream's default demands both; we demand neither.

Every config key mentioned in this document is also collected in
[Appendix B — Glossary](#appendix-b--glossary-of-config-keys-and-stage-names).

These are naturalv2's own eligibility switches. Each is a boolean: **required** means the switch is
`True` and a trial is *dropped* unless it has that property; **relaxed** means we set it `False`, so
the property is no longer demanded and trials lacking it are kept. Upstream's default turns all four
on; we relax the last two, and the preset name `noparallel_notbinary` records exactly that.


| filter            | what the trial must have                                    | our setting  | why                                                                                                                                                                                                                                                   |
| ----------------- | ----------------------------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `randomized`      | participants randomly assigned to arms                      | **required** | without randomisation there is no causal contrast to predict against                                                                                                                                                                                  |
| `nonhealthy`      | enrolls patients with the condition, not healthy volunteers | **required** | healthy-volunteer trials (safety, PK) have no symptom outcome to compare                                                                                                                                                                              |
| `parallel`        | arms run concurrently, each participant in one arm          | **relaxed**  | crossover and factorial designs are common here — LIFT is a 2×2 factorial and would be dropped                                                                                                                                                        |
| `binary_endpoint` | primary outcome is yes/no or a response rate                | **relaxed**  | the consequential one: demanding binary endpoints collapses the set by **92%**, because Long-COVID primaries are continuous symptom scales. Relaxing it is what makes the whole set viable — and what forces us onto the NATURAL-MC estimator (§10.4) |


**Tier 2 — Condition** (ours; replaces upstream's substring matcher — [bug A1](#appendix-a--bug-index))

- A keyword classifier, not `c in tc or tc in c`. The substring test admitted 12 acute-COVID
hospitalisation trials (because `"covid"` ⊂ `"long covid"`) and dropped 7 genuine post-COVID ones
tagged `"post-acute covid-19 syndrome"`.  This way of doing things was causing a lot of problems, obviously,  and is something I want to document and bring to Nikita. 
- Broadened the CT.gov search scope. Upstream searched `query.cond="COVID"`; we search
`COVID OR SARS-CoV-2 OR PASC OR Post-Acute Sequelae of SARS-CoV-2 OR Post-COVID-19 Condition OR Chronic COVID OR Long-haul COVID`. A trial tagged only `"Post-Acute Sequelae of SARS-CoV-2"`
contains no "COVID" substring, and the registry does not expand the term for you
([bug B1](#appendix-a--bug-index)) — so the narrow query silently missed them. Broadening it
recovered +4 trials with results and +33 without.

**Tier 3 — Label availability**

A trial that passes Tiers 1 and 2 is in the study. This tier only decides **which of the three
splits it lands in**, and that depends entirely on whether the trial has published its results yet:


| trial state                                         | split           | what we do with it                                                                                        |
| --------------------------------------------------- | --------------- | --------------------------------------------------------------------------------------------------------- |
| posted structured results on CT.gov                 | **train / val** | we have the real answer, so we score our estimate against it. Split between train and val by readout date |
| active or recruiting, no results yet                | **test**        | nothing to score against — these are the **prediction targets**, and the actual product. LIFT is one      |
| completed, no posted results, but a published paper | **train / val** | rescued by extracting the endpoint from the paper (`paper_extracted`)                                     |


Note that "train" is a slight misnomer inherited from the config: nothing is fitted on those trials
(§2). They are the subset we look at while developing, held apart from val so we do not tune against
everything at once.

**Tier 4 — Evidence coverage** (our addition)

This filter actually looks at the data to see if there is enough to draw meaningful conclusions from an intervention of interest in a study. This is better than Nikita's approach which just asks an LLM is an intervention would reasonably be in the data. A trial can be perfectly designed and still be useless here if nobody on Reddit has taken the drug. We  
therefore measure the evidence available per treatment and drop trials below a floor.

- **≥ 50 on-target patient usage reports.**
- `effective_authors = distinct authors mentioning the drug × LLM-validated on-target fraction`
- "On-target" means *the author personally took it for their own Long COVID and described an
outcome* — not news, mechanism speculation, someone else's use, or acute-COVID use.
- This replaced an earlier `is_corpus_learnable` gate that also demanded the drug be
self-obtainable, single-agent, and blinded. That was wrong: NATURAL only needs a treatment to be
**discussed** by people who took it. Using this gate instead of the one Nikita was using took us from **5 to 19** usable trials to train the model on.

**Two things we record about each trial but deliberately do not filter on.**

- `obtainability` — can a patient get this themselves (a supplement), only by prescription, or only
in a clinic (an infusion, a nerve block)?
- `reliability` — a judgement that follows from it. When patients can obtain a treatment freely,
the people who try it are mostly self-selecting on how badly they want to try it. When a treatment
is only available in a clinic, who receives it is also determined by severity, money, insurance and
geography — so the Reddit population diverges further from the trial population.

It is tempting to drop or down-weight the clinic-only trials, since their evidence is more
confounded. We deliberately do not. Those trials are the hardest cases, and how badly the method
does on them is one of the things we most want to find out — excluding them would flatter the
results and hide the failure mode. So both fields travel with the trial as context for interpreting
its result, and neither affects whether the trial is included.

### 4.2 Core-5 — These pass according to Nikita's orginal criteria


| NCT                                                         | drug                           | split | readout | on-target | primary outcome                    | why it is in the clean set                                                |
| ----------------------------------------------------------- | ------------------------------ | ----- | ------- | --------- | ---------------------------------- | ------------------------------------------------------------------------- |
| [NCT05472090](https://clinicaltrials.gov/study/NCT05472090) | cyclobenzaprine                | train | 2024-11 | 92        | Daily Diary Pain NRS               | oral Rx, single agent, blinded, self-report endpoint                      |
| [NCT05047952](https://clinicaltrials.gov/study/NCT05047952) | vortioxetine                   | train | 2025-01 | 118       | DSST z-score change                | same, cognitive endpoint                                                  |
| [NCT05618587](https://clinicaltrials.gov/study/NCT05618587) | lithium                        | train | 2025-03 | 145       | Fatigue / Brain Fog Severity Scale | **our worked example**; labels −11.3 / −9.0 verified against the registry |
| [NCT04809974](https://clinicaltrials.gov/study/NCT04809974) | nicotinamide riboside (Niagen) | val   | 2025-06 | 125       | NAD+ and cognitive function        | OTC supplement — cleanest self-selection                                  |
| [NCT05874037](https://clinicaltrials.gov/study/NCT05874037) | fluvoxamine                    | val   | 2026-06 | 450       | Change in Total Symptom Scores     | highest coverage of the five                                              |


Split is temporal (3 train / 2 val), ordered by readout date.

### 4.3 Additional 14:  Despite being excluded in Nikita's old criteria,  we can see that people are talking about these interventions enough in the correct way just by looking in the data, and we can also use these studies to try and get at ground truth.


| NCT                                                         | drug         | on-target | access          | reliability | previously excluded for                 |
| ----------------------------------------------------------- | ------------ | --------- | --------------- | ----------- | --------------------------------------- |
| [NCT06419712](https://clinicaltrials.gov/study/NCT06419712) | vitamin D    | 2,729     | self-obtainable | high        | non-self-report endpoint (GPx activity) |
| [NCT04657484](https://clinicaltrials.gov/study/NCT04657484) | prednisolone | 1,562     | Rx oral         | high        | open-label / combination                |



|                                                                                                                                                                                         |                         |     |                 |          |                                            |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------- | --- | --------------- | -------- | ------------------------------------------ |
| [NCT05445427](https://clinicaltrials.gov/study/NCT05445427)                                                                                                                             | vagus nerve stim        | 765 | device          | caveated | not self-obtainable                        |
| [NCT04842448](https://clinicaltrials.gov/study/NCT04842448)                                                                                                                             | hyperbaric oxygen       | 757 | clinic          | caveated | not self-obtainable                        |
| [NCT05576662](https://clinicaltrials.gov/study/NCT05576662) · [NCT05595369](https://clinicaltrials.gov/study/NCT05595369) · [NCT05965726](https://clinicaltrials.gov/study/NCT05965726) | Paxlovid ×3             | 354 | Rx oral         | high     | **three trials, one shared evidence pool** |
| [NCT06253806](https://clinicaltrials.gov/study/NCT06253806)                                                                                                                             | stellate ganglion block | 311 | clinic          | caveated | procedure                                  |
| [NCT05445674](https://clinicaltrials.gov/study/NCT05445674) · [NCT05841498](https://clinicaltrials.gov/study/NCT05841498)                                                               | apheresis ×2            | 213 | clinic          | caveated | procedure; shared pool                     |
| [NCT07544186](https://clinicaltrials.gov/study/NCT07544186)                                                                                                                             | L-citrulline            | 144 | self-obtainable | high     | biomarker endpoint                         |
| [NCT05200858](https://clinicaltrials.gov/study/NCT05200858)                                                                                                                             | TENS                    | 127 | device          | caveated | device                                     |
| [NCT05840237](https://clinicaltrials.gov/study/NCT05840237)                                                                                                                             | oxaloacetate            | 125 | self-obtainable | high     | supplement                                 |
| [NCT05126563](https://clinicaltrials.gov/study/NCT05126563)                                                                                                                             | mesenchymal stem cells  | 66  | clinic          | caveated | procedure                                  |


Two things to keep in mind when reading any aggregate over these:

- **The rows are not independent.** 19 trials draw on **16 distinct drug signals**; Paxlovid has  
three trials and apheresis two, each sharing a single Reddit pool. NATURAL produces one estimate  
per drug, scored against several trial outcomes. There is probably an argument for removing some of these so that they are independent.
- **Two annotation errors, since corrected.** Our LLM drug classifier had put `oxaloacetate` in
`behavioral_or_device` (it is an oral supplement, bought over the counter) and `apheresis` there too
(it is a clinic procedure). Fixed: oxaloacetate is now `self_obtainable`, which also moves it to
`high_clean_self_selection`, so the reliability split is **12 clean / 7 access-confounded**, not
11/8. Neither field is read by the pipeline — they annotate results rather than filter trials — so
nothing about the runs changes.

### 4.4 Prediction targets — What we are trying to predict from this.


| NCT                                                                                                                                                                                     | trial                            | sponsor                            | target arms                                                   | corpus signal                                               | status                                                                         |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **[NCT06366724](https://clinicaltrials.gov/study/NCT06366724)**                                                                                                                         | **LIFT: Life Improvement Trial** | **Open Medicine Foundation (OMF)** | `Placebo/LDN`, `Pyridostigmine/Placebo`, `Pyridostigmine/LDN` | 5,183 raw / **3,417 effective** (LDN); 683 (pyridostigmine) | recruiting — **running now** **(Note that we are meeting with them Thursday)** |
| [NCT06095297](https://clinicaltrials.gov/study/NCT06095297) · [NCT06585254](https://clinicaltrials.gov/study/NCT06585254) · [NCT06968104](https://clinicaltrials.gov/study/NCT06968104) | tVNS ×3                          | —                                  | device arms [unclear]                                         | 3,091                                                       | recruiting / active                                                            |
| [NCT05852873](https://clinicaltrials.gov/study/NCT05852873) · [NCT06792214](https://clinicaltrials.gov/study/NCT06792214)                                                               | Paxlovid ×2                      | —                                  | nirmatrelvir-ritonavir                                        | 2,925                                                       | active / recruiting                                                            |
| [NCT06082518](https://clinicaltrials.gov/study/NCT06082518)                                                                                                                             | HBOT                             | —                                  | hyperbaric arm [unclear]                                      | 1,404                                                       | recruiting                                                                     |


**LIFT is probably the most important of the above studies to predict.**

1. Highest corpus signal of any Long-COVID intervention — LDN is the most-discussed treatment in
  the corpus by a wide margin, and Eli (and I) have a history of advocating for.
2. The 2×2 factorial isolates the **LDN main effect** (`Placebo/LDN`), which upstream's old
  arm-matching silently discarded ([bug A3](#appendix-a--bug-index)).
3. It exercises two of our findings directly: [A3](#appendix-a--bug-index) and [A4](#appendix-a--bug-index).
4. It is run by **OMF, who we meet on Thursday 13 August 2026**.

> **Note a signal warning.** §4.2 and §4.3 report **effective** (validated on-target) counts. §4.4 reports
> **raw** distinct authors, which is a larger number for the same evidence. LDN is 5,183 raw but  
> 3,417 effective (reported this way because the trial has multiple expermental arms. 

---

## 5. Stage 2 — Evidence curation

**The corpus.** 147,333 submissions and 2,304,485 comments from r/covidlonghaulers, r/LongCovid and  
r/LongHaulersRecovery, spanning 2020-07 to 2026-06. Stored in our S3.

I pulled data from Arctic shift because upstream's archive source (the-eye's Pushshift mirror) is dead, and has been for the past month. Our adapter `RedditPrebuiltParquet` feeds a pre-built corpus into the pipeline in place of the download stage.

### What a "report" is

Curation does not work on bare comments. `build_contextualized_dataset` first assembles each comment  
into a thread-aware document, because a comment alone is usually unusable; "it helped a lot" means  
nothing until you know what "it" was. A real example from the lithium set:

```
**Subreddit**      r/LongHaulersRecovery
**Initial Post**   the post this comment replied to (title + body),
                   plus the original poster's other replies in that thread
**Date created**   April 30, 2024
**Comment**        "I havent, i have heard things though ill try it!"
```

That assembled block is a single observational unit that eventually becomes one  
synthetic patient in the estimator (§6.1).

### The matching itself

Treatment matching uses an **Aho-Corasick automaton** [1]: all treatment names and aliases are
compiled into a single finite-state machine (a trie with failure links), and the corpus is streamed
through it once. The property that matters is that **search time is independent of how many patterns
you are looking for** — five aliases or five hundred cost the same single pass. The naive
alternative, scanning the corpus once per alias, would be O(documents × aliases), which across 2.45M
documents is the difference between minutes and hours.

So this stage is deterministic, reproducible, has no API cost and no sampling variance. It is a
**candidate generator**, not a filter: it decides what the expensive LLM stages are allowed to look
at, and nothing more.

### I(Shaun) think this way of doing things creates at least a lot of noise, and maybe undetectable bias.

Three consequences, and the first is visible in the example above.

**1. Matches can fire on context rather than the author.** In that example the matched term was
`lithium` — but "Lithium Orotate" appears nowhere in the comment. It is in the *original poster's*
replies, pulled in as thread context. The comment author's actual words are *"I havent … ill try
it!"* — they explicitly have **not** taken it. The report is attributed to them anyway, causing them to be a phantom patient in our data. This bias is systematic rather than random: it inflates whichever drugs are most discussed *in threads*, which is likely a lot of conversation.

**2. The LLM stages are carrying the entire semantic burden.** Curation's precision is poor by
design; the collapse from 3,136 reports to 61 for lithium (98%) is the LLM stages doing the real
filtering. That is the intended division of labour, but it means curation quality shows up as
*cost* (paying to score thousands of reports that will be discarded) rather than as an error you can
see.

**3. Recall is a hard ceiling, and failures are silent.** Anything the automaton misses is invisible
to every downstream stage — no LLM ever sees it, and nothing errors; there are simply fewer rows.
This is the failure mode that has bitten us twice: a partitioning mismatch that silently produced
*zero* records, and the three-character alias filter that silently cost 79% of LDN evidence
(§10.3, since fixed). **The free stage sets the recall ceiling; the expensive stages can only ever
remove** — so a bug here is invisible and unrecoverable, which is why it is worth disproportionate
attention for a stage that costs nothing to run.

**Where this leaves the curation rules.** Word-boundary checking has since been added (§10.3), so  
short drug abbreviations such as LDN are now matched. Matching remains purely lexical, with no  
morphology, and no spelling tolerance — a misspelled "naltroxone" is invisible. Improving it is  
tracked as §11.6, and measuring the phantom-patient rate is a concrete thing to check in the  
human-labelled sample proposed in §11.3.

- `Skipped treatment_synonyms` — needs OpenAI web-search, which I have not implemented. We hand-seed aliases instead, which is where  
problem §10.3 comes from.

**Scale, for intuition:** lithium curates to 3,136 reports; LIFT to 42,617.

---

## 6. Stage 3 — Estimation, and the maths

yay maths!

### 6.1 The shape of the method

```
reports ──LLM──▶ tabular pseudo-population (X, T, Y, w) ──causallib──▶ Ê[Y(t)] + CI
```

Every report becomes one synthetic patient. **The causal machinery downstream is textbook; all of
the risk lives in the arrow labelled LLM.** That framing is the single most useful thing to hold on
to about this system.

### 6.2 What quantity we are actually predicting

There are two different things you can predict about a trial, and it matters a great deal which one:

- the **average potential outcome (APO)** — what one arm's patients scored, on its own. "Patients on
lithium improved by 11.3 points."
- the **average treatment effect (ATE)** — the *difference* between arms. "Lithium improved patients
2.7 points **more than placebo** did."

ATE is what a clinician cares about, because it isolates the drug. APO includes everything that
would have happened anyway: placebo response, regression to the mean, natural recovery over the
trial period.

**We currently predict APO.** The config key is `ate: False`. For each arm we estimate the mean
outcome among patients who took that arm's treatment — written `E[Y(t)]`, read as "the expected
outcome `Y` if a patient were given treatment `t`".

This is why the run output reads *Predicted Response* against *True Response*: we are predicting the
lithium arm's own mean change of −11.3, not
lithium-minus-placebo. Easy to misread; worth reading twice.

### 6.3 What the LLM has to produce, per report


| symbol | meaning                                                                                                 | produced by                                                |
| ------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `X_i`  | 8 covariates: age (categorical and continuous), sex, ethnicity, race, region, country, illness duration | `knowns` (stated in the post) + `imputations` (not stated) |
| `T_i`  | which arm's treatment the author took                                                                   | `sample_ty`                                                |
| `Y_i`  | the trial's outcome value for that author                                                               | `sample_ty`                                                |
| `w_i`  | P(author would meet the trial's inclusion criteria)                                                     | `inclusion_prob`                                           |


All are then discretised: covariates to an index into their option list, treatment to an index into
`treatment_names`, and a continuous outcome to a numeric value (identity).

### 6.4 Two extraction modes

**(a) Conditional extraction — teacher-forced scoring.** Upstream's default, used by NATURAL-IPW.
Enumerate every candidate assignment `a = (x, t, y)` over the option grid and score each by summing
the token log-probabilities of the *prompt* (`prompt_logprobs`, `max_tokens=1` — nothing is
generated):

$$s_i(a)=\sum_k \log p(\text{tok}*k \mid \text{prefix}*{<k}), \qquad P_i(a)=\mathrm{softmax}_a s_i(a)$$ 

`length_norm: False`, so there is no division by token count. This yields a *distribution* over
assignments per report rather than a hard label.

It requires a **discrete option set for `Y`**, so it fails outright on continuous endpoints — see
§10.4. It is also the reason we need a GPU at all: `prompt_logprobs` is a vLLM extension, and no
chat API (OpenRouter, Anthropic, OpenAI) exposes it.

**(b) Monte Carlo — direct sampling.** NATURAL-MC, and **what we use**. The LLM emits
`{treatment, outcome}` as JSON, sampling `Y_i` as a real number. Handles continuous endpoints
natively.

### 6.5 Inclusion weighting

`inclusion_prob` scores `P(meets_inclusion_criteria = Yes | report)` by the same teacher-forced
method, and the weights reweight the community population toward the trial's eligible population.
This is the pipeline's one explicit correction for *"Reddit is not the trial cohort"*. Controlled by
`use_inclusion_weights: True`.

### 6.6 The estimators

Nothing here is bespoke. Verified against `naturalv2/models/causal_models.py`, the estimators are
**causallib** — an open-source causal-inference library from IBM Research that implements the
standard estimators as scikit-learn-style objects — with ordinary scikit-learn models doing the
actual fitting:


| name          | class                                  | the model it fits                |
| ------------- | -------------------------------------- | -------------------------------- |
| IPW           | `causallib.estimation.IPW`             | multinomial `LogisticRegression` |
| **OI (ours)** | `causallib.estimation.Standardization` | `LinearRegression`               |
| baseline      | `MarginalOutcomeEstimator`             | none — plain difference in means |


**IPW (inverse propensity weighting)** asks *"how surprising is it that this person took this
treatment, given who they are?"* It fits a model of the probability of receiving each treatment
given the covariates — the **propensity** — then weights each patient by the inverse of that
probability. People who took an unlikely-for-them treatment count for more, which rebalances the
sample toward what a randomised trial would have produced:

```
             sum over patients on arm t of  ( Y_i / propensity(t | X_i) )
E[Y(t)]  =   -----------------------------------------------------------
             sum over patients on arm t of  (   1 / propensity(t | X_i) )
```

**OI (outcome imputation, a.k.a. standardisation or the g-formula)** — the path we use — works the
other way round. It fits a model predicting the outcome from covariates *and* treatment, then asks
that model to predict what **every** patient would have scored had they all been given arm `t`, and
averages:

```
E[Y(t)]  =  mean over ALL patients i of  outcome_model( X_i , t )
```

Two things about this are worth flagging.

**The outcome model is a plain, unregularised linear regression** over the discretised covariates
plus a treatment indicator. With 8 covariates plus treatment, it is fitting roughly 9 or more
parameters. For lithium's Fatigue outcome we had **n = 9 patients**. That is as many parameters as
data points: the regression is saturated, it can fit the training rows exactly, and the coefficients
are essentially arbitrary — there is no residual variation left to estimate them from. Extrapolating
that model to every patient, as OI does, is then unstable in a way nothing in the pipeline warns
about, and it is a plausible contributor to the wild point estimate in §9. No regularisation, no
interaction terms, no check that n exceeds the number of parameters.

**Identification rests on the usual assumptions:** consistency, positivity, and **no unmeasured
confounding given those 8 covariates**. That last one is heroic on Reddit, and §11.4 is about
exactly how heroic.

I can probably find and pull more studys to use as ground truth/trial stuff.

### 6.7 Uncertainty

A percentile bootstrap with `bootstrap_size: 10` and `alpha: 0.05`, resampling the
**already-extracted** table.

This is the crux of §10.2. The bootstrap propagates sampling variability *of the pseudo-population*,
but not extraction error (how wrong `Y_i` is given the text), not LLM sampling variance (one draw
per report), and not selection bias. So the interval answers *"how much would this move if I drew a
different subset of the same extracted rows"* — which is not the question anyone is asking. Also,
with B = 10, the 2.5th and 97.5th percentiles are simply the minimum and maximum of ten numbers.

### 6.8 The knobs we set, and why


| knob                    | value                         | why                                                      |
| ----------------------- | ----------------------------- | -------------------------------------------------------- |
| `estimator`             | `natural_mc_oi`               | continuous endpoints; the IPW path cannot enumerate them |
| `ate`                   | `False`                       | average potential outcome per arm                        |
| `use_inclusion_weights` | `True`                        | reweight community → trial-eligible population           |
| `bootstrap_size`        | `10` (upstream default)       | **too small — the first thing to change**                |
| `probs_model`           | Qwen2.5-7B via vLLM           | the only role needing a GPU; **should be 70B**           |
| generative roles        | gemini-2.5-flash / flash-lite | cost; flash-lite for the two filter stages               |


---

## 7. How it actually runs right now

### 7.1 The picture

```
GitHub Actions  (Airwhale/PatientPunk, branch shaun/dispersed-vllm-image)
        │  builds dispersed/vllm-server/Dockerfile
        ▼
GHCR    ghcr.io/airwhale/pp-vllm:latest          ← public; dispersed pulls anonymously
        │
        ▼
Dispersed (Render Network)   PERSISTENT job, 1× RTX 5090, HMAC-signed REST API
        │  vLLM downloads Qwen2.5-7B from HuggingFace at boot (~15 GB, ~5 min)
        ▼
http://mlpipeN-<region>.proxy.dispersed.com:<port>/v1    ← OpenAI-compatible, IP-whitelisted
        ▲
        │  probs_model only
WSL Ubuntu (laptop) ── orchestrates everything else
        ├── OpenRouter → relevance, treatment/outcome, knowns, imputations, sample_ty
        ├── local CPU  → contextualise, curate, causallib estimator, bootstrap
        └── S3 / local → the corpus
```

### 7.2 Why a custom image

Dispersed's job spec accepts `image`, `tag`, `ports`, `env` and `volumes` — but **no command or
entrypoint override**, and there are no pre-built recipes. So the served model cannot be passed as a
flag to the stock `vllm/vllm-openai` image.

Our image adds a wrapper entrypoint that reads its configuration from **environment variables**,
which dispersed *can* set. One image therefore serves any model — 7B today, 72B later — with **no
rebuild**:


| env             | value                                          | why                                                    |
| --------------- | ---------------------------------------------- | ------------------------------------------------------ |
| `MODEL`         | `Qwen/Qwen2.5-7B-Instruct`                     | change this to change models                           |
| `MAX_MODEL_LEN` | `8192`                                         | bounds the logits tensor                               |
| `GPU_MEM_UTIL`  | `**0.55`**                                     | leaves headroom for `prompt_logprobs` — see §10.6      |
| `EXTRA_ARGS`    | `--max-num-seqs 1 --no-enable-chunked-prefill` | `prompt_logprobs` is incompatible with chunked prefill |


The base is pinned to `vllm/vllm-openai:v0.22.0` to match naturalv2's pinned `vllm==0.22.0`, so the
logprob wire format the server returns is exactly what the client expects. Model weights are **not**
baked into the image; vLLM pulls them from HuggingFace at container start, which keeps the image
small at the cost of ~5 minutes of startup.

### 7.3 Where each stage physically executes


| stage                      | runs on                  | why                     | cost driver                                        |
| -------------------------- | ------------------------ | ----------------------- | -------------------------------------------------- |
| contextualise corpus       | local CPU (polars)       | no model needed         | free                                               |
| curate                     | local CPU (Aho-Corasick) | rule-based              | free                                               |
| `relevance_filter`         | **OpenRouter**           | generative              | **dominates spend** — every report × every outcome |
| `treatment_outcome_filter` | OpenRouter               | generative              | moderate                                           |
| `knowns` / `imputations`   | OpenRouter               | generative              | small                                              |
| `sample_ty`                | OpenRouter               | generative              | small                                              |
| `**inclusion_prob`**       | **dispersed GPU**        | needs `prompt_logprobs` | GPU-hours                                          |
| estimator + bootstrap      | local CPU (causallib)    | scikit-learn            | free                                               |


**Exactly one stage needs a GPU.** That is the load-bearing architectural fact: it is why this runs
on a laptop plus a rented card rather than a cluster, and why lithium cost $1–3 end to end.

### 7.4 The run chain

`dispersed/run_estimate_chain.py` performs one supervised sequence:

1. **launch** — HMAC-signed `POST /v1/jobs` (PERSISTENT, `gpu_name`, `min_vram_gb`, port 8000,
  `allowed_ips` set to our egress IP /32).
2. **wait** — poll `/v1/job-runs` until `node_urls` appear. (The reachable host and port live on the
  job-*run*, not the job object.)
3. **probe** — one realistic ~5k-token `prompt_logprobs` request. Fails in seconds if the serving
  config is wrong, rather than after a full pipeline pass.
4. **run** — `estimate_ate` in WSL against that endpoint.
5. **auto-stop** — cancel the job in a `finally` block, so a crash cannot leave a GPU idling.

### 7.5 Operational realities

- Billing is hourly and starts at job creation: $0.35/hr (RTX 4090) to $4.14/hr (the 6×5090 node the
scheduler keeps assigning for a single-GPU request).
- We have $500 of credit at dispersed right now,  and I can ask for more.  LMK if you need login credentials again. 
- The web console and the job API disagree — an A6000 appears in the UI but not in
`/v1/gpu-registry`, which is what job specs match against.
- A crashed vLLM is indistinguishable from a reclaimed node (`connection refused` either way). That
ambiguity cost five debugging cycles.
- **Nothing persists on the GPU box.** It serves a model and nothing else; all state lives locally or
in S3. Losing a node costs minutes, not data.
- Stage results are cached as CSVs under `outputs/results/<NCT>_<experiment>/`, so a failed GPU stage
does not re-spend the OpenRouter budget on retry.

---

## 8. LIFT end to end — a worked trace

Following OMF's trial through all three stages, with real numbers.

**Stage 1 — enters as a `test` trial.** `status=active`, no posted results, so it produces a
prediction with no error metric.

- Four arms → **three kept**: `Pyridostigmine/LDN`, `Pyridostigmine/Placebo`, `Placebo/LDN`, all
typed `ACTIVE_COMPARATOR`. `Placebo/Placebo` is correctly dropped as `PLACEBO_COMPARATOR`.
- **[Bug A3](#appendix-a--bug-index) in action.** Upstream's old arm matching classified arms by *title*, so any arm whose
label contained "Placebo" was dropped — that is the LDN main effect *and* the pyridostigmine main
effect, silently. Now fixed upstream by classifying on the structured `ArmGroupType`.
- **[Bug A4](#appendix-a--bug-index) in action.** The test universe filter `status:act` means "active, not recruiting", so a
*recruiting* trial like LIFT would be excluded from prediction entirely.
- Four primary outcomes — FUNCAP55 functional capacity, OUES, VO₂, heart-rate recovery — and **all
four are change-from-baseline** (§10.1).

**Stage 2 — curation yields 42,617 reports** (against lithium's 3,136).

- Matched on `naltrexone`, `low dose naltrexone`, `mestinon`, `pyridostigmine`.
- **Problem §10.3 bit here.** The bare abbreviation "LDN" is three characters and was dropped by the
alias filter, putting **79% of LDN evidence out of reach**. This has since been fixed, but *this run
was launched before the fix* — so the LIFT numbers below rest on roughly a fifth of the available
LDN evidence. Re-running with the fix would raise the candidate pool from ~42,600 to ~138,600
reports, and triple the cost.
- No date cutoff is applied, which is correct for a target that has not reported.

**Stage 3 — Monte Carlo path**, four outcomes over 42,617 reports, producing an average potential
outcome per arm per outcome, with a confidence interval and **no ground truth**.

### 8.1 Status — first attempt stopped deliberately, re-run planned

We started this run and **stopped it after the first outcome**. Not a failure: the pipeline was
working, and outcome 1 completed the full funnel (42,456 relevance-filtered → 5,898 with treatment
and outcome → 3,127 sampled) before we cancelled during its `inclusion_prob` stage. Three things
made finishing the wrong call:

1. **It was running on ~21% of the available LDN evidence.** Launched before the boundary-matching
  fix (§10.3), so the answer would have been the weak version of itself and superseded immediately.
2. **We are not presenting a LIFT number on Thursday** (§12), so the result had no near-term use.
3. **Cost.** Outcome 1 alone took two hours; four would have been about eight, at roughly $65 more
  than we had already spent. Actual spend before cancelling: **$8.69 GPU plus roughly $12–14 of API**.

**A real inefficiency this exposed.** For those two hours the GPU sat idle: the pipeline spends most
of its wall-clock in the OpenRouter stages, and only touches the GPU at `inclusion_prob`. We were
paying $4.14/hr for hardware in use maybe a tenth of the time — and paying a six-GPU node rate for a
single-GPU job on top of that. The runner should acquire the GPU around the logprob stages only,
which would cut a LIFT run's GPU cost from roughly $33 to roughly $5.

**Planned re-run**, once these are in place:


| prerequisite                                | why                                                                                                    |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| boundary-matching fix applied (§10.3, done) | raises the candidate pool from ~42,600 to ~138,600 reports                                             |
| GPU acquired only for `inclusion_prob`      | removes ~85% of the GPU cost                                                                           |
| DrugBank aliases supplied (§11.6)           | free additional synonyms before we pay to scan anything                                                |
| a decision on outcome count                 | four outcomes quadruples the dominant `relevance_filter` cost; Functional Capacity alone may be enough |


Sequencing matters: this is a *prediction target*, so it is not on the critical path for
verification. The four held-out core-5 trials (§9.3) come first, since they are what tell us whether
any LIFT number would be worth reporting.

---

## 9. First run — lithium (development)

Lithium ([NCT05618587](https://clinicaltrials.gov/study/NCT05618587)) is the trial we built the
pipeline *against*. Every fix described in this document — change-from-baseline detection, the
outcome-description rewrite, the NATURAL-MC configuration, the vLLM serving settings — came from
running lithium, watching it fail, and adjusting. No model was trained; there is nothing to train
(§2). But the configuration was tuned by hand until this trial ran, and that is researcher
overfitting all the same.

So this section is a record of a **development run**. It shows the pipeline executes end to end and
returns a well-formed estimate. It is *not* evidence that the method works, and the numbers below
should not be quoted as though it were.

### 9.1 What the run produced


| outcome                  | predicted (95% CI)      | trial label     | absolute error | n   |
| ------------------------ | ----------------------- | --------------- | -------------- | --- |
| Fatigue Severity Scale   | −28.47 (−38.56, −18.37) | −11.3 (SD 12.6) | 17.17          | 9   |
| Brain Fog Severity Scale | −33.89 (−34.87, −32.92) | −9.0 (SD 13.8)  | 24.89          | 32  |


The one substantive thing this run established is that the units problem (§10.1) was real and is
fixed: correcting it flipped both predictions to **negative**, i.e. improvement, matching the
direction of the trial. It did not improve accuracy — mean absolute error moved from 21.9 to 21.0,
which is noise. Sign and magnitude are separate problems, and only the first was addressed.

### 9.2 What those numbers do not tell us

An error of 21 points is uninterpretable on its own, because nothing here says what a *good* number
would be. Computing that comparison — which we had not done until writing this section — is
uncomfortable. (To be clear about what is being compared: the last row is the NATURAL pipeline as
described in this document, not any earlier PatientPunk analysis. The rows above it are trivial
predictors that use no patient text at all.)

Start with what the trial actually found. Lithium reported both arms:


| outcome                  | lithium arm | placebo arm | true treatment effect |
| ------------------------ | ----------- | ----------- | --------------------- |
| Fatigue Severity Scale   | −11.3       | −8.6        | **−2.7**              |
| Brain Fog Severity Scale | −9.0        | −8.1        | **−0.9**              |


Lithium is close to a **null trial**. Almost all of that −11.3 is placebo response and regression to
the mean, not drug. Which means a predictor that knows nothing about lithium, and simply guesses that
patients in a trial improve by roughly nine points, does very well:


| predictor                               | Fatigue error | Brain Fog error | **MAE**  |
| --------------------------------------- | ------------- | --------------- | -------- |
| predict the **placebo arm's value**     | 2.7           | 0.9             | **1.8**  |
| predict **zero change**                 | 11.3          | 9.0             | **10.2** |
| **NATURAL as we are running it** (§9.1) | 17.2          | 24.9            | **21.0** |


That constant beats us roughly twelve-fold. On one trial, with a 7B model, this is not a verdict on
the method — but it does expose a structural problem with *what we are measuring*. Because we
estimate **APO, the mean outcome under one arm** (§6.2), and both arms move together, the target is
dominated by placebo response and time trend. **Mean absolute error on APO barely tests whether the
method detects a treatment effect at all.** The decision-relevant quantity is the contrast — −2.7
and −0.9 — and we have never tested it.

### 9.3 What verification will mean

**The verification set already exists.** Every trial that has posted results is one: we estimate,
then compare against the known answer. The prediction targets in §4.4 — LIFT included — can never
verify anything, because there is nothing to check against. So the 19 completed trials are the
verification set and the 3 targets are the product.

**We have used one of them.** Lithium, two outcomes, and it is spent: having been tuned against, it
can no longer serve as a clean test. The other four core-5 trials — cyclobenzaprine, vortioxetine,
Niagen, fluvoxamine — are built and untouched, and are now cheap to run, since corpus
contextualisation is cached and each needs only curation plus a short GPU pass.

**Four measurements, on those four trials, with the configuration frozen first:**

1. **Against baselines** — versus predict-zero and predict-placebo-response. Without a baseline an
  error figure means nothing, as §9.2 demonstrates.
2. **Sign agreement** — does it get the direction right? Currently 2 of 2, after the units fix.
3. **Interval coverage** — does the confidence interval contain the truth? Currently 0 of 2, which
  restates §10.2 as a metric rather than an observation.
4. **Contrast, not just APO** — can it separate lithium from placebo when the real gap is 0.9 points
  on a 49-point scale? This is the question that matters, and by far the hardest.

Freezing the configuration beforehand is the step that is easy to skip and expensive to lose. Any
further tuning against these four contaminates them exactly as lithium is contaminated, and there is
only one clean shot per trial.

---

## 10. Problems

Defects we have found and diagnosed. The full registry, with evidence, is in
`[bugs.md](findings.md)`; this is the readable summary.

### 10.1 Change-from-baseline labels versus absolute predictions

Lithium reports **−11.3** for an outcome whose description says *"Score range 1-49"*. A value of
−11.3 is impossible on that scale — the trial reported a *change* from baseline while describing an
*absolute* scale. Only the `timeFrame` field ("Change from baseline to day 21") says so.

The model is asked for a value "on the same scale as the outcome description", correctly returns an
absolute severity, and is then scored against a change. Guaranteed large error, telling us nothing
about estimate quality.

Our fix restates the description so the sampled quantity matches the label. Notably **our own
detector missed this** — it regexed the outcome *title*, which says nothing about change. All four
LIFT outcomes are change endpoints too.

### 10.2 Estimate quality and interval calibration — the lithium variance

Two separate problems live here.

**Systematic overshoot.** Brain Fog predicts −33.9 against a true −9.0, on a scale whose maximum
possible improvement is −48. The model is claiming near-total recovery. This is the signature of
selection bias (§11.2), not random error.

**Implausibly tight intervals.** Quantitatively:


| outcome   | n   | CI half-width |
| --------- | --- | ------------- |
| Fatigue   | 9   | ±10.1         |
| Brain Fog | 32  | **±0.98**     |


The ratio is 10.3×, but √(32/9) predicts only 1.9× — so the Brain Fog interval is roughly **five
times tighter than sample size alone explains**. Using the trial's own SD of 13.8, a naive standard
error on n = 32 would give about ±4.8, not ±1.

The mechanism is in §6.7: the bootstrap measures agreement among LLM samples rather than patient  
heterogeneity or extraction error, and B = 10 is far too few for a stable 95% interval.

### 10.3 Estimator and endpoint mismatch

Upstream's default estimator (`natural_ipw`) cannot run a `notbinary` study: `conditional_extraction`
enumerates multiple-choice options over the outcome, continuous endpoints have none, and it dies with
a bare `ZeroDivisionError`. We use NATURAL-MC instead.

### 10.4 Trial selection

[Bug A1](#appendix-a--bug-index) (the condition matcher) and [bug A4](#appendix-a--bug-index) (`status:act` excluding recruiting trials) both affect
which trials are eligible at all. [A4](#appendix-a--bug-index) would exclude LIFT.

### 10.5 Infrastructure

`prompt_logprobs` needs a transient `[prompt_tokens × vocab]` logits tensor — about 3.5 GB for a
5,700-token prompt at Qwen's 152k vocabulary — allocated *outside* the pool vLLM preallocates. The
usual `gpu_memory_utilization: 0.90` leaves no room and the engine dies. Because it is a *fraction*,
**moving to a bigger card does not help**: more VRAM simply buys a bigger KV cache. The fix is to
lower it to 0.55.

Separately: we are running a **7B** model where the method assumes ~70B.

---

## 11. Open problems

Distinct from §10. Those are defects with known causes and mostly known fixes. These are the
unsolved ones, where the right approach is genuinely undecided. Roughly ordered by how much each
blocks trusting an estimate.

### 11.1 Uncertainty quantification — what should a confidence interval here even mean?

The bootstrap resamples an already-extracted table, capturing one variance component and silently
ignoring the rest: extraction error, LLM sampling variance, selection variance, and
model-specification variance.

The dominant uncertainty is epistemic and LLM-shaped — *"how much do I trust this extraction"* has no
standard estimator. Taking multiple draws per report would capture sampling variance but not
systematic extraction bias.

**Progress looks like:** an interval whose empirical coverage across the 19 labelled trials is close
to nominal. That is measurable with what we already have, and nobody has measured it.
*Self-contained; needs no new data.*

### 11.2 Selection bias — people post about dramatic outcomes

The pseudo-population is self-selected in a way that correlates directly with the outcome. Dramatic
recoveries and dramatic failures are both over-represented; "mild and unremarkable" is not.

The overshoot in §10.2 is the signature. `inclusion_prob` corrects for *eligibility*, not for
*propensity to post*.

This is unobservable by construction — non-posters leave no data — so it needs either an external
anchor (a survey with known prevalence) or an explicit model of the posting mechanism.

**Progress looks like:** a correction that reduces overshoot on the labelled trials without simply
fitting to them. *Probably the deepest problem here.*

### 11.3 Extraction validity — is `Y_i` what the patient actually reported?

There is no gold standard, so extraction error is entangled with estimator error and selection bias.
Validating means humans reading posts and assigning outcome values, which is expensive — and
inter-rater agreement on *"what FSS score does this post imply?"* may itself be poor.

Underneath is a real question: **is there a ceiling?** If humans cannot agree on the outcome implied
by a Reddit post, no model can extract it reliably, and the method has an intrinsic noise floor worth
knowing about.

**Progress looks like:** a small human-labelled set of 100–200 posts, giving a per-report error
distribution — which feeds directly into §11.1.

### 11.4 Confounding — is the identification assumption ever plausible?

The estimator assumes no unmeasured confounding given eight covariates. But baseline severity,
comorbidity, prior treatment failures and treatment-seeking behaviour are all unmeasured, and all
drive both what people try and how they report it. Someone trying LDN after five failed treatments is
not exchangeable with someone trying it first.

**Open question:** can richer confounders be extracted from the text itself, and does adding them
actually move the estimate or just add noise?

**Progress looks like:** a sensitivity analysis — how large would an unmeasured confounder have to be
to explain the error we see?

### 11.5 Does model scale fix this?

We run Qwen2.5-**7B**; the method assumes ~70B. We do not know whether accuracy is model-limited or
evidence-limited, and that decides where all the remaining effort should go. If 70B closes the gap,
this is an infrastructure problem; if it does not, §11.1–11.4 are the whole story.

**Progress looks like:** the same trials at 7B and 70B. Cleanly testable — the serving image is
env-driven, so it is a config change and a bigger card (§7.2). *The cheapest decisive experiment
available, and arguably what to do first.*

### 11.6 How treatment mentions should be found at all — the matcher probably needs replacing

Fixing the boundary bug (§10.3) removed the worst symptom, but it patched a design that is
fundamentally limited. The current matcher is an Aho-Corasick automaton doing **exact substring
matching against a hand-supplied alias list**. Both halves of that are weak:

- **Exact matching.** No tolerance for misspellings ("naltroxone"), spacing ("low dose" vs
"lowdose"), or morphology. Patients write informally and often on a phone.
- **Hand-supplied aliases.** Upstream expands synonyms via web search; we hand-seed them. Neither
scales past a handful of drugs, and both fail *silently* — an unmatched report is simply absent,
with nothing to indicate it existed.

**A source already wired in but unused.** `naturalv2` imports `get_drugbank_aliases` and exposes
`Experiment.drugbank_names`, but expects a DrugBank database file (`full_database.xml.gz`) that we
have never supplied — the "DrugBank data file not found" warning in every run is this. Supplying it
would give canonical synonyms and brand names for free, and is the cheapest thing to try before
building anything.

**Open question: what should replace exact matching?** Plausible directions, roughly in order of
effort:

1. **Ontology-backed aliases** — DrugBank or RxNorm as the alias source rather than a hand list.
  Solves brand and generic names; not informal abbreviations.
2. **Fuzzy matching** — edit distance or phonetic keys over candidate tokens, which catches
  misspellings at the cost of new false positives.
3. **Embedding retrieval** — find reports semantically near a treatment description rather than
  string-matching. Handles paraphrase, but loses the deterministic, auditable, free properties that
   make the current stage cheap (§5).
4. **Entity linking** — a model that maps text spans to drug concepts directly. Strongest and most
  expensive, and it changes the stage from a free prefilter into an inference cost over 2.45M
   documents.

The tension is that the current stage's virtue *is* that it is free and deterministic, and that is
what lets us pay LLM cost only on candidates. Anything that replaces it has to preserve that or
justify the expense. Worth measuring first: how much evidence are we still missing after the
boundary fix? That is answerable by sampling reports the matcher rejected.

### 11.7 Prolific posters — the wrong unit of analysis

Every surviving report becomes one synthetic patient (§6.1), so a patient who posts fifty times
counts fifty times. For lithium's evidence:


|                              |                              |
| ---------------------------- | ---------------------------- |
| documents mentioning lithium | 1,012                        |
| distinct authors             | 365                          |
| median documents per author  | **1**                        |
| **maximum**                  | **99**                       |
| top 10 authors' share        | **30.3%**                    |
| single most prolific author  | **9.8% of all the evidence** |


One person supplies a tenth of it. Ten people supply nearly a third.

**We cannot currently measure this downstream, because `author` is discarded.** It exists in the raw
corpus but survives neither contextualisation nor curation, so nothing after stage 2 can
deduplicate, cluster or weight by patient — and we cannot tell how concentrated the 61 surviving
lithium reports are. The numbers above are pre-filter.

This is obviously very bad,  biasing in clear ways, and needs to be fixed.

### 11.8 What is this benchmark actually measuring?

19 trials reduce to **16 independent drug signals**, several with change-from-baseline label
mismatches. How many trials are needed before an aggregate error is meaningful? Should shared-pool
trials be pooled or down-weighted? And is per-arm APO even the right target, when the
decision-relevant quantity is the treatment *contrast*?

Without answers, "mean absolute error over 19 trials" is a number without an interpretation.

---

## 12. Stakes and timeline

**Thursday 13 August 2026 — meeting with OMF.** They run LIFT. Our LIFT output is a prospective
estimate for a trial owned by the people we are presenting to, which sets the bar for how it is
framed.

**Solid enough to present:**

- The corpus — 147k submissions and 2.3M comments over six years.
- The coverage measurement — 3,417 validated on-target LDN reports, a quantitative read on what
patients report about their own trial's interventions.
- The method, and that it runs end to end.
- The bug findings — A3 in particular, since it concerns *their* trial's factorial arms directly.

**Not solid enough to present as a forecast:** the point estimate. On lithium, where we can check,
the pipeline was off by roughly 2–3× with the sign initially inverted, its intervals are
demonstrably uncalibrated, we are on a 7B smoke model, problems §10.1–10.3 are open, and only ~21%
of LDN evidence is reachable.

**The position to take:** *"Here is what patients report, here is a pipeline that turns it into a
testable prediction, and here are three estimator bugs that affect factorial trials like yours.
Calibration is in progress."* That is both stronger and true.

---

## 13. Appendix

**Repositories**

- `PatientPunk` — `trial_superset/` holds our study construction, coverage measurement, sidecar and
docs; `dispersed/` holds the serving kit.
- Fork `Airwhale/naturalv2`, branch `shaun/patientpunk-integration` — our changes to the estimator.
`main` on that fork is byte-identical to Nikita's.

**Environment**

- Runs in WSL Ubuntu, not native Windows: the estimator's compiled dependencies are blocked by
Windows Smart App Control, and `contextualize` imports the Unix-only `resource` module.
- Python 3.12 venv managed with `uv` at `~/pp-natural/.venv`; the fork checkout at `~/pp-natural/fork`.

**Data**

- Corpus: `s3://patientpunk/trial_superset/natural_corpus_parquet/`
- Run outputs: `~/pp-natural/outputs/` — `studies/`, `experiments/`, `nct_reports[_test]/`,
`reddit_data/`, `results/`

**Further reading**

- [bugs.md](findings.md) — the full bug registry with evidence, structured as a fix-by-fix summary.
- [benchmark_19.md](benchmark_19.md) — how the 19-trial set was derived.
- [method_and_scope.md](method_and_scope.md) — the original scope decisions.

---

## Appendix A — Bug index

Short forms used throughout this document. "Ours" means the defect is in our code or our data;
everything else is in `naturalv2` and therefore affects Nikita's own results too. Full evidence for
each is in [bugs.md](findings.md); the same list, restructured as a fix-by-fix summary for her, is on the
fork at
[docs/patientpunk/findings.md](https://github.com/Airwhale/naturalv2/blob/shaun/patientpunk-integration/docs/patientpunk/findings.md).


| id     | what                                                                                                                                                                                             | where                                | status                                                            |
| ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------ | ----------------------------------------------------------------- |
| **A1** | Condition matcher over- and under-matches. A plain substring test in both directions admitted 12 acute-COVID trials into Long COVID and dropped 7 genuine post-COVID ones.                       | `find_condition_ncts`                | open upstream; we use a keyword classifier (§4.1 Tier 2)          |
| **A2** | `notbinary` labels computed as `value / N`, which is a response rate for binary endpoints but meaningless for a continuous mean.                                                                 | `Experiment`                         | **fixed upstream** in `7a2e006`                                   |
| **A3** | Factorial arms named `"X/Placebo"` classified as placebo by title and silently dropped — including LIFT's LDN main effect.                                                                       | `check_nonplacebo` → `check_arm`     | **fixed upstream**; arms now typed by `ArmGroupType`              |
| **A4** | Test universe uses `status:act`, which means "active, not recruiting" and excludes every *recruiting* trial — LIFT included.                                                                     | test aggFilters                      | open; we use a recruiting-inclusive universe                      |
| **A5** | The default estimator cannot run a `notbinary` study: `conditional_extraction` enumerates options over the outcome, continuous endpoints have none, and it dies with a bare `ZeroDivisionError`. | `conditional_extraction`             | open; we supply `conf/estimate_mc.yaml` (§10.4)                   |
| **A6** | Change-from-baseline labels compared against absolute predictions (§10.1).                                                                                                                       | `Experiment` + our tooling           | open — a design question for Nikita                               |
| **A7** | `detokenize=False` is hardcoded on every `prompt_logprobs` call; harmless in-process, but it crashes a *hosted* vLLM server.                                                                     | `conditional_extraction`             | **fix written** — an in-process guard, on our branch              |
| **A8** | Bootstrap intervals implausibly tight — ±1 on n=32 (§10.2).                                                                                                                                      | `bootstrap_size`                     | open — needs her view                                             |
| **A9** | Treatment matching dropped ≤3-character aliases, costing 79% of LDN evidence (§10.3).                                                                                                            | `build_treatment_automaton` + curate | **fixed** — boundary-aware matching added; strongest PR candidate |
| **B1** | `query.cond="COVID"` misses trials tagged `SARS-CoV-2` / `PASC`, because CT.gov does not expand the term.                                                                                        | our CT.gov scope layer               | fixed in `seed_terms`                                             |
|        |                                                                                                                                                                                                  |                                      |                                                                   |


---

## Appendix B — Glossary of config keys and stage names

Every name that appears in a code font in this document, in one place.

**Trial selection**


| name                                                      | meaning                                                                                                                                                                                     |
| --------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `trial_filters`                                           | the config block holding the four eligibility booleans (§4.1 Tier 1)                                                                                                                        |
| `noparallel_notbinary`                                    | our filter preset, and the string in every output path: no `parallel` requirement, not `binary` endpoints                                                                                   |
| `randomized`, `parallel`, `nonhealthy`, `binary_endpoint` | the four booleans themselves — see the table in §4.1                                                                                                                                        |
| `status:act`                                              | a ClinicalTrials.gov filter meaning "active, **not** recruiting". Excludes recruiting trials, which is bug A4                                                                               |
| `ArmGroupType`                                            | CT.gov's structured arm label — `EXPERIMENTAL`, `ACTIVE_COMPARATOR`, `PLACEBO_COMPARATOR`, `SHAM_COMPARATOR`, `NO_INTERVENTION`. Now used to classify arms instead of their titles (bug A3) |
| `registry_adapted` / `paper_extracted`                    | our flags for trials sourced from ISRCTN, or whose label came from a published paper rather than posted results                                                                             |
| `effective_authors`                                       | our coverage metric: distinct authors mentioning a drug × the LLM-validated fraction that are genuine on-target usage reports                                                               |


**The `Experiment` object** — one per trial, built in stage 1, holding its treatments, outcomes,
covariates, option lists and (for completed trials) the label. Everything downstream reads it.

**Pipeline stages** (all inside stage 3 unless noted)


| name                                              | what it does                                                                                                                     | runs on     |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| `create_study` / `filter_curate` / `estimate_ate` | the three top-level CLIs (§3)                                                                                                    | —           |
| `relevance_filter`                                | is this report relevant to the trial's outcome at all?                                                                           | LLM (cheap) |
| `treatment_outcome_filter`                        | does it describe *both* the treatment and the outcome? The hard cut — 3,136 → 61 for lithium                                     | LLM (cheap) |
| `knowns`                                          | covariates the report states explicitly                                                                                          | LLM         |
| `imputations`                                     | covariates it does not state, inferred                                                                                           | LLM         |
| `sample_ty`                                       | sample the treatment taken and the outcome value (`T`, `Y`). The NATURAL-MC path                                                 | LLM         |
| `conditional_extraction`                          | the alternative to `sample_ty`: enumerate candidate answers and score them. Needs discrete outcomes, so unusable for us (bug A5) | vLLM        |
| `inclusion_prob`                                  | probability the author would have met the trial's inclusion criteria; used as a weight                                           | vLLM        |


**Model roles** — naturalv2 lets a different model serve each job:


| role                | what it serves                                                          | ours                    |
| ------------------- | ----------------------------------------------------------------------- | ----------------------- |
| `cheap_model`       | the two filter stages, which see every report                           | `gemini-2.5-flash-lite` |
| `sample_model`      | `knowns`, `sample_ty`                                                   | `gemini-2.5-flash`      |
| `imputations_model` | `imputations`                                                           | `gemini-2.5-flash`      |
| `probs_model`       | the stages needing `prompt_logprobs`; **the only role requiring a GPU** | Qwen2.5-7B on vLLM      |


**Estimation**


| name                    | meaning                                                                                                                                                                                                         |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `prompt_logprobs`       | a vLLM feature returning the log-probability of every token *in the prompt*, so a candidate answer can be scored without generating anything. No chat API offers it, which is the entire reason a GPU is needed |
| **APO**                 | average potential outcome — the mean outcome under one arm, `E[Y(t)]`. What we estimate                                                                                                                         |
| **ATE**                 | average treatment effect — the *difference* between arms. What we do **not** currently estimate (§6.2)                                                                                                          |
| `ate: False`            | the config key selecting APO over ATE                                                                                                                                                                           |
| `use_inclusion_weights` | whether to apply the `inclusion_prob` weights                                                                                                                                                                   |
| `bootstrap_size`        | number of bootstrap resamples for the confidence interval. Upstream default 10 — too few (§10.2)                                                                                                                |
| `natural_mc_oi`         | the estimator we use: Monte-Carlo extraction (`sample_ty`) plus outcome imputation                                                                                                                              |
| `natural_ipw`           | upstream's default: enumerated extraction plus inverse propensity weighting. Cannot run continuous outcomes                                                                                                     |


**Serving** — `MODEL`, `MAX_MODEL_LEN`, `GPU_MEM_UTIL` and `EXTRA_ARGS` are environment variables
read by our container entrypoint; see the table in §7.2.

---

## References

[1] Aho, A. V. and Corasick, M. J. (1975). *Efficient string matching: an aid to bibliographic
search.* Communications of the ACM, 18(6), 333–340. [doi:10.1145/360825.360855](https://doi.org/10.1145/360825.360855)

[2] Dhawan, N., Cotta, L., Ullrich, K., Krishnan, R. G. and Maddison, C. J. (2024). *End-To-End
Causal Effect Estimation from Unstructured Natural Language Data.* NeurIPS 2024.
[arXiv:2407.07018](https://arxiv.org/abs/2407.07018) ·
[project page](https://www.cs.toronto.edu/~nikita/natural/)