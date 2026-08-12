# Predicting Clinical Trial Outcomes from Patient Communities

*Working draft — Shaun & Claude. Written for Polina; useful to anyone picking this up cold.*
*Last updated 2026-08-11.*

*This document is specifically aimed at Polina. Hi Polina!*

---

## 1. Orientation

We are trying to estimate what a clinical trial will find, using what patients wrote about the same
treatment **before the trial reported**.  This matches what Nikita et al. did with [https://www.cs.toronto.edu/~nikita/natural/](https://www.cs.toronto.edu/~nikita/natural/)  
, but with a different population.  This is a different approach than the between-group differences that we showed in our pilot paper,  and should be much more exact.  For our data we are using three Long-COVID subreddits (r/covidlonghaulers, r/LongCovid, r/LongHaulersRecovery). The trials come from **ClinicalTrials.gov**, pulled through naturalv2's own downloader against the CT.gov API v2, with two additions of ours: a handful of **ISRCTN** trials adapted into the same schema, and a **papers-as-labels** path that recovers completed trials which never posted structured results by extracting the primary endpoint from the published paper.

After a bunch of fiddling, I have **the pipeline running end-to-end, but I do not trust validity.** Natural's paper had a lot of things going for it that ours does not:

1. **Their endpoints are proportions; ours are continuous symptom scales.** Their headline result is
  "within 3 percentage points of ground truth" — a *percentage point* metric, i.e. binary/response-rate  
   outcomes. Binary endpoints are bounded in [0,1], are directly enumerable by the default estimator,  
   and are far easier to read off a post ("did they improve — yes or no?"). Our data is often in terms of subjective severity,  fatigue reports, etc. **We cannot compare our error to their 3 points.**
2. **They report against Phase 3/4 trials.** Those are large, well-powered, and mostly test
  established drugs. Our Long-COVID set is small early-phase work, for example a trial we are looking at involving lithium has an arm with n=24.
3. **Model scale.** The method assumes a large model; we are running a 7B smoke model (see §11.4). Once we get everything else working better we should ask Nikita to run it on UofT's machines or source some more heavy duty VRAM having machines to get an 80B parameter model or larger running it.
4. **Label cleanliness.** Several of our endpoints report a *change from baseline* while describing an
  absolute scale, which silently mismatches the prediction target (§10.1). A response rate has no
   equivalent failure mode.
5. **Treatment naming.** There is a LOT of variety in how different treatments are named in the Long covid community.

Note that Nikita's current repo is here: [github.com/nikitadhawan/naturalv2](https://github.com/nikitadhawan/naturalv2).
I'm currently working out of a fork branch,
[Airwhale/naturalv2 @ shaun/patientpunk-integration](https://github.com/Airwhale/naturalv2/tree/shaun/patientpunk-integration)

This working branch is currently an AI vibe-coded mess,  and I will be cleaning it up.  I just wanted to get it running to figure out where things were failing in the main repo's experimental/scraping logic and be ready to talk about this if needed for thursday.  
(its `main` is byte-identical to hers, so the branch diff *is* the set of our changes),   

I am recording things to improve upon in her repo here:  
[docs/patientpunk/findings.md](https://github.com/Airwhale/naturalv2/blob/shaun/patientpunk-integration/docs/patientpunk/findings.md).

---

## 2. What NATURAL is:

**Paper:** Dhawan, Cotta, Ullrich, Krishnan & Maddison, *End-To-End Causal Effect Estimation from
Unstructured Natural Language Data*, arXiv:2407.07018 (NeurIPS 2024). We run pinned commit `7a2e006` of `naturalv2`.

NATURAL uses an LLM as a *measurement instrument*. For a given trial it reads each patient report  
and emits the four variables a causal estimator needs: which of **this trial's** treatment arms the  
author took, what value they would have reported on **this trial's** primary outcome scale, the  
covariates describing them, and the probability they would have met the trial's inclusion criteria. It pulls from the log probs of the next predicted token,  and thus requires the use of open source models to get at this.  

Each simulated report becomes one row. That table is a synthetic observational dataset, and a textbook causal  
estimator i.e. inverse propensity weighting or outcome regression runs over it unchanged to produce  
an estimate per experimental arm. The paper's claim is that these land within **3 percentage points** of the  
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
7B model rather than a large one, and small early-phase trials rather than Phase 3/4. The [§1](#1-orientation) list is  
the full inventory of those differences. **The causal machinery is the strong part of that work, and**  
**we are using it unchanged.** We have run into some problems with scraping and expermental design as documented below:  I don't think we are done finding these problems. Check out[findings.md](findings.md) for more details on this.

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
We add two paths of our own: a few trials from the **[ISRCTN registry](https://www.isrctn.com/)** adapted into the same schema, and  
**papers-as-labels**, which rescues completed trials that never posted structured results by
extracting the primary endpoint from the published paper. Those are flagged `registry_adapted` and
`paper_extracted` respectively, so they can be excluded from any analysis that wants CT.gov-only.

---

## 4. Stage 1 — Trial selection

### 4.1 The criteria used to select trials

**Tier 1 — Structural**

Two names to define first, since they recur throughout:

- `trial_filters` — the block in naturalv2's config holding the four booleans below. It is the
only place trial eligibility is decided upstream.
- `noparallel_notbinary` — the name we gave our filter settings, and the string that appears in
every output path and filename. It is purely descriptive: *no* `parallel` requirement, *not*
`binary` endpoints. Upstream's default demands both; we demand neither.

Every config key mentioned in this document is also collected in
[Appendix B — Glossary](#appendix-b--glossary-of-config-keys-and-stage-names).

These are naturalv2's own eligibility switches. Each is a boolean: **required** means the switch is
`True` and a trial is *dropped* unless it has that property; **relaxed** means we set it `False`, so
the property is no longer demanded and trials lacking it are kept. Upstream's default turns all four
on; we relax the last two, and the preset name `noparallel_notbinary` records exactly that.


| filter            | what the trial must have                                    | our setting      | why                                                                                                                                                                                                                                                   |
| ----------------- | ----------------------------------------------------------- | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `randomized`      | participants randomly assigned to arms                      | **required**     | without randomisation there is no causal contrast to predict against                                                                                                                                                                                  |
| `nonhealthy`      | enrolls patients with the condition, not healthy volunteers | **required**     | healthy-volunteer trials (safety, PK) have no symptom outcome to compare                                                                                                                                                                              |
| `parallel`        | arms run concurrently, each participant in one arm          | **Not required** | crossover and factorial designs are common here — LIFT is a 2×2 factorial and would be dropped                                                                                                                                                        |
| `binary_endpoint` | primary outcome is yes/no or a response rate                | **Not required** | the consequential one: demanding binary endpoints collapses the set by **92%**, because Long-COVID primaries are continuous symptom scales. Relaxing it is what makes the whole set viable — and what forces us onto the NATURAL-MC estimator (§10.3) |


**Tier 2 — Condition** (ours; replaces upstream's substring matcher — [bug A1](#appendix-a--bug-index))

- A keyword classifier, not `c in tc or tc in c`. The substring test admitted 12 acute-COVID
hospitalisation trials (because `"covid"` ⊂ `"long covid"`) and dropped 7 genuine post-COVID ones
tagged `"post-acute covid-19 syndrome"`. Substring matching on free-text condition names is brittle
in both directions. Tracked as [bug A1](#appendix-a--bug-index), and one of the main things to raise
with Nikita — it is still open upstream, so it affects her own results as well as ours.
- We should probably have Eli or another expert go through these trials and make sure they are suitable for establishing ground truth.
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

This filter measures the corpus directly, asking whether there is enough real discussion of an
intervention to draw conclusions from. It complements the upstream check, which asks a model whether
an intervention would plausibly be discussed; here we count what is actually there. A trial can be perfectly designed and still be useless here if nobody on Reddit has taken the drug. We  
therefore measure the evidence available per treatment and drop trials below a floor.

- **≥ 50 on-target patient usage reports.**
- `effective_authors = distinct authors mentioning the drug × LLM-validated on-target fraction`
- "On-target" means *the author personally took it for their own Long COVID and described an
outcome* — not news, mechanism speculation, someone else's use, or acute-COVID use.
- This replaced an earlier `is_corpus_learnable` gate that also demanded the drug be
self-obtainable, single-agent, and blinded. That was wrong: NATURAL only needs a treatment to be
**discussed** by people who took it. Using this gate instead of the one Nikita was using took us from **5 to 19** usable trials. (Usable as *benchmark* trials — nothing is trained on them, see §2.)

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

### 4.2 Core-5 — the trials that pass the original upstream criteria


| NCT                                                         | drug                           | split | readout | on-target | primary outcome                    | why it is in the clean set                                                |
| ----------------------------------------------------------- | ------------------------------ | ----- | ------- | --------- | ---------------------------------- | ------------------------------------------------------------------------- |
| [NCT05472090](https://clinicaltrials.gov/study/NCT05472090) | cyclobenzaprine                | train | 2024-11 | 92        | Daily Diary Pain NRS               | oral Rx, single agent, blinded, self-report endpoint                      |
| [NCT05047952](https://clinicaltrials.gov/study/NCT05047952) | vortioxetine                   | train | 2025-01 | 118       | DSST z-score change                | same, cognitive endpoint                                                  |
| [NCT05618587](https://clinicaltrials.gov/study/NCT05618587) | lithium                        | train | 2025-03 | 145       | Fatigue / Brain Fog Severity Scale | **our worked example**; labels −11.3 / −9.0 verified against the registry |
| [NCT04809974](https://clinicaltrials.gov/study/NCT04809974) | nicotinamide riboside (Niagen) | val   | 2025-06 | 125       | NAD+ and cognitive function        | OTC supplement — cleanest self-selection                                  |
| [NCT05874037](https://clinicaltrials.gov/study/NCT05874037) | fluvoxamine                    | val   | 2026-06 | 450       | Change in Total Symptom Scores     | highest coverage of the five                                              |


Split is temporal (3 train / 2 val), ordered by readout date.

### 4.3 The other 14 — recovered by measuring the corpus

These fall outside the original criteria, but the corpus shows people discussing these interventions
in the right way and in sufficient numbers, so they can serve as ground truth too.


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


| NCT                                                                                                                                                                                     | trial                            | sponsor                            | target arms                                                        | corpus signal                                               | status                                                 |
| --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------ |
| **[NCT06366724](https://clinicaltrials.gov/study/NCT06366724)**                                                                                                                         | **LIFT: Life Improvement Trial** | **Open Medicine Foundation (OMF)** | `Placebo/LDN`, `Pyridostigmine/Placebo`, `Pyridostigmine/LDN`      | 5,183 raw / **3,417 effective** (LDN); 683 (pyridostigmine) | recruiting — planned, see §8. **Meeting OMF Thursday** |
| [NCT06095297](https://clinicaltrials.gov/study/NCT06095297) · [NCT06585254](https://clinicaltrials.gov/study/NCT06585254) · [NCT06968104](https://clinicaltrials.gov/study/NCT06968104) | tVNS ×3                          | —                                  | CICT / BFT (± VR); patient- vs device-controlled; Intervention A/B | 3,091                                                       | recruiting / active                                    |
| [NCT05852873](https://clinicaltrials.gov/study/NCT05852873) · [NCT06792214](https://clinicaltrials.gov/study/NCT06792214)                                                               | Paxlovid ×2                      | —                                  | nirmatrelvir-ritonavir                                             | 2,925                                                       | active / recruiting                                    |
| [NCT06082518](https://clinicaltrials.gov/study/NCT06082518)                                                                                                                             | HBOT                             | —                                  | immediate vs delayed start of hyperbaric treatment                 | 1,404                                                       | recruiting                                             |


**LIFT is probably the most important of the above studies to predict.**

1. Highest corpus signal of any Long-COVID intervention — LDN is the most-discussed treatment in
  the corpus by a wide margin, and Eli (and I) have a history of advocating for it.
2. The 2×2 factorial isolates the **LDN main effect** (`Placebo/LDN`), which upstream's old
  arm-matching silently discarded ([bug A3](#appendix-a--bug-index)).
3. It exercises two of our findings directly: [A3](#appendix-a--bug-index) and [A4](#appendix-a--bug-index).
4. It is run by **OMF, who we meet on Thursday 13 August 2026**.

> **Signal warning: two different counts.** §4.2 and §4.3 report **effective** counts — distinct
> authors, multiplied by the LLM-validated fraction that are genuine on-target usage reports. §4.4
> reports **raw** distinct authors, before that validation, which is always the larger number for the
> same evidence. LDN is 5,183 raw but 3,417 effective. Reading across the tables without noticing
> makes LIFT look ~35× better supplied than lithium when the real ratio is ~23×.

---

## 5. Stage 2 — Evidence curation

**The corpus.** 147,333 submissions and 2,304,485 comments from r/covidlonghaulers, r/LongCovid and  
r/LongHaulersRecovery, spanning 2020-07 to 2026-06. Stored in our S3.

I pulled data from Arctic Shift because upstream's archive source (the-eye's Pushshift mirror) is dead, and has been for the past month. Our adapter `RedditPrebuiltParquet` feeds a pre-built corpus into the pipeline in place of the download stage.

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

**2. The LLM stages are carrying the entire semantic burden.** The collapse from 3,136 reports to 61 for lithium (98%) is the LLM stages doing the real filtering. That is the intended division of labour, but it means curation quality shows up as  
*cost* (paying to score thousands of reports that will be discarded).  The primary step is just creating bias as to what the LLM sees. 

**3. There is no good way to mesure the amount of bias or noise this creates.**

---

## 6. Stage 3 — Estimation, and the maths

Yay maths! Read §6.1 and §6.2 if you read nothing else.

### 6.1 The shape of the method

```
reports ──LLM──▶ tabular pseudo-population (X, T, Y, w) ──causallib──▶ Ê[Y(t)] + CI
```

Every report becomes one synthetic patient. **The causal machinery downstream is textbook; all of
the risk lives in the arrow labelled LLM.** That framing is the single most useful thing to hold on
to about this system.

### 6.2 What quantity we are actually predicting

There are two different things you can predict about a trial.

- the **average potential outcome (APO)** — what one arm's patients scored, on its own. "Patients on
lithium improved by 11.3 points."
- the **average treatment effect (ATE)** — the *difference* between arms. "Lithium improved patients
2.7 points **more than placebo** did."

ATE is a stronger, harder target. It is what a clinician cares about, because it isolates the drug. 

**We currently predict APO.** The config key is `ate: False`. For each arm we estimate the mean
outcome among patients who took that arm's treatment — written `E[Y(t)]`, read as "the expected
outcome `Y` if a patient were given treatment `t`".

**We should be predicting the effect, not the level — that is the goal, and it is §11.8.** The
obstacle is not the config flag but that Reddit has no placebo arm to contrast against, so the
difference the corpus can support is drug-versus-untreated where the trial reports
drug-versus-placebo.

This is why the run output reads *Predicted Response* against *True Response*: we are predicting the
lithium arm's own mean change of −11.3, not lithium-minus-placebo. Easy to misread, and §9.2 shows
what it costs — on a near-null trial, APO is dominated by the placebo response, so error against it
barely tests whether the method detects a drug effect at all.

### 6.3 What the LLM has to produce, per report


| symbol | meaning                                                                                                 | produced by                                                |
| ------ | ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| `X_i`  | 8 covariates: age (categorical and continuous), sex, ethnicity, race, region, country, illness duration | `knowns` (stated in the post) + `imputations` (not stated) |
| `T_i`  | which arm's treatment the author took                                                                   | `sample_ty`                                                |
| `Y_i`  | the trial's outcome value for that author                                                               | `sample_ty`                                                |
| `w_i`  | P(author would meet the trial's inclusion criteria)                                                     | `inclusion_prob`                                           |


Covariates and treatment are then encoded as indices — into their option list and into
`treatment_names` respectively. A continuous outcome is passed through unchanged; the "discretise"
step is the identity function for it, which is worth knowing because the name suggests otherwise.

### 6.4 Two extraction modes

The pipeline is a list of stages, and which stages run is set by a config file. Nikita's default
config is `conf/estimate_ate.yaml`; ours is `conf/estimate_mc.yaml`, and it differs in exactly **one
swap**. (Her filename says `ate`, but that is only the default — whether a run estimates an ATE or an
APO is the `ate:` flag inside, which we set to `False`.)


| #   | stage                      | Nikita's default          | ours                |
| --- | -------------------------- | ------------------------- | ------------------- |
| 1   | `relevance_filter`         | yes                       | yes                 |
| 2   | `treatment_outcome_filter` | yes                       | yes                 |
| 3   | `knowns`                   | yes                       | yes                 |
| 4   | `imputations`              | yes                       | yes                 |
| 5   | `sample_ty`                | present but commented out | **added**           |
| 6   | `inclusion_prob`           | yes                       | yes — **unchanged** |
| 7   | `conditional_extraction`   | yes                       | **removed**         |


Stages 5 and 7 are the two modes below. Stage 6 is unchanged, and that matters more than it looks.

**Mode 1 — conditional extraction, i.e. scoring the grid.** `ConditionalExtractionStage`, stage 7, and
upstream's default. **This is the part that works over a grid.** It enumerates every candidate
assignment `a = (x, t, y)` the option lists allow, and scores each one by summing the token
log-probabilities of the *prompt* (`prompt_logprobs`, `max_tokens=1` — nothing is generated):

```
for each candidate answer a = (x, t, y):
    score(a) = sum over every token k in the prompt of
                   log P( token_k | all tokens before it )

P(a) = softmax over all candidates of score(a)
```

In words: paste each candidate answer into the prompt, ask the model how unsurprising that whole
prompt was, and turn those scores into a probability distribution over the candidates. The model is
only ever asked to *score* text it was given, never to write any — which is what `prompt_logprobs`  
returns, and what no chat API exposes. A collapsed single answer can't return this. 

**Mode 2 — Monte Carlo, i.e. sampling.** `SampleTYStage`, stage 5. NATURAL-MC. Rather  
than scoring an enumerated grid for predicting results as in the standard pipeline,  
the model is asked the question once and replies with `{treatment, outcome}` as JSON, `Y_i` a real number. This is nessasary because the endpoints we are looking at are continuous, not discreate. 

This Monte Carlo prediction cannot express uncertainty. Scoring the grid in conditional extraction returns a probability across candidates, 

**Conditional extraction is still needed for trial inclusion determination even if using Monti Carlo:  We can't get away from the open model as things are currently written.** 

### 6.5 Inclusion weighting

Reddit is not the trial's cohort, and this is the pipeline's one explicit correction for that.
`inclusion_prob` scores `P(meets_inclusion_criteria = Yes | report)` by the teacher-forced method of  
§6.4(a), and each report then enters the average weighted by probability — in  
`estimate_ate.py`, literally `np.average(responses, weights=probs)`. Someone who reads as clearly
trial-eligible counts nearly fully; someone who reads as ineligible counts nearly zero. Controlled by
`use_inclusion_weights: True`.

Two things about how it behaved on lithium.

**The probability is entirely inferred.** The extracted `meets_inclusion_criteria` field was
`Unknown` on all 32 Brain Fog reports and all 9 Fatigue ones — no Reddit post says whether its author
would pass a trial screen. The weight is the model's read of the report, not something a patient
stated.

**It does something, but nothing dramatic.** Weighting costs roughly a fifth of the sample:


| outcome   | rows | effective sample size | heaviest single row |
| --------- | ---- | --------------------- | ------------------- |
| Brain Fog | 32   | 25.5 (80%)            | 4.9%                |
| Fatigue   | 9    | 6.2 (69%)             | 24.4%               |


*(Effective sample size is `(Σw)² / Σw²` — how many equally-weighted reports would carry the same
information.)* 

### 6.6 The estimators

The estimators come from **causallib** — an open-source causal-inference library from IBM Research
that wraps the standard estimators as scikit-learn-style objects — with ordinary scikit-learn models
doing the actual fitting. Checked against `naturalv2/models/causal_models.py`:


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
confounding given those 8 covariates**. That is a strong assumption anywhere, and a particularly
demanding one on Reddit; §11.3 sets out why.

### 6.7 Uncertainty

A percentile bootstrap with `bootstrap_size: 10` and `alpha: 0.05`, resampling the
**already-extracted** table.

The key word is *already*. The bootstrap propagates sampling variability of the pseudo-population,
but not extraction error (how wrong `Y_i` is given the text), not LLM sampling variance (one draw per
report), and not selection bias. So the interval answers *"how much would this move if I drew a
different subset of the same extracted rows"* — which is not the question anyone is asking. And with
B = 10, the 2.5th and 97.5th percentiles are simply the minimum and maximum of ten numbers.

Both limitations are real, but neither is what collapsed the lithium intervals. Resampling a column
in which 21 of 32 values are identical returns the same number however large B is — the values
themselves were degenerate, for the reason in §10.2. Raising `bootstrap_size` is still worth doing;
it just will not fix an interval computed over the wrong quantity.

### 6.8 The knobs we set, and why


| knob                    | value                         | why                                                      |
| ----------------------- | ----------------------------- | -------------------------------------------------------- |
| `estimator`             | `natural_mc_oi`               | continuous endpoints; the IPW path cannot enumerate them |
| `ate`                   | `False`                       | average potential outcome per arm                        |
| `use_inclusion_weights` | `True`                        | reweight community → trial-eligible population           |
| `bootstrap_size`        | `10` (upstream default)       | too small, but A11 outranks it (§10.2)                   |
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
| `GPU_MEM_UTIL`  | `0.55`                                         | leaves headroom for `prompt_logprobs` — see §7.5       |
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
| `inclusion_prob`           | **dispersed GPU**        | needs `prompt_logprobs` | GPU-hours                                          |
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
- **A low `gpu_memory_utilization` is required** — 0.55, not the usual 0.90. Scoring a
prompt materialises a `[prompt_tokens × vocab]` logits tensor (~3.5 GB for a realistic report)
*outside* the pool vLLM preallocates, and because the setting is a **fraction rather than an amount,
a bigger card does not help**: the pool grows with the card and the leftover does not. The fix is to
make the number smaller, which is the opposite of what an out-of-memory error usually suggests. Full
account in D1 of [findings.md](findings.md).
- A crashed vLLM is indistinguishable from a reclaimed node (`connection refused` either way), and a
short smoke test passes either way — a 7-token prompt needs ~4 MB of logits and never exercises the
failure. Between them, that cost five debugging cycles, which is why the run chain now probes with a
realistic-length prompt first.
- **Nothing persists on the GPU box.** It serves a model and nothing else; all state lives locally or
in S3. Losing a node costs minutes, not data.
- Stage results are cached as CSVs under `outputs/results/<NCT>_<experiment>/`, so a failed GPU stage
does not re-spend the OpenRouter budget on retry.

---

## 8. LIFT — the planned prediction study

LIFT ([NCT06366724](https://clinicaltrials.gov/study/NCT06366724), run by OMF) is the target we most
want to predict. It is **planned, not done**: we started a run, stopped it after the first outcome,
and will run it properly once the prerequisites below are met.

**What it looks like in the pipeline.** It enters as a `test` trial — active, no posted results — so
it yields a prediction with no error metric. Three of its four arms are kept (`Pyridostigmine/LDN`,
`Pyridostigmine/Placebo`, `Placebo/LDN`, all typed `ACTIVE_COMPARATOR`); `Placebo/Placebo` is
correctly dropped. All four primary outcomes are change-from-baseline (§10.1). Curation yielded
42,617 candidate reports, against lithium's 3,136.

It also exercises two findings directly: [A3](#appendix-a--bug-index), since the older title-based
arm matching would have dropped both main-effect arms, and [A4](#appendix-a--bug-index), since
`status:act` would exclude a recruiting trial from prediction altogether.

**Why we stopped.** Outcome 1 completed its funnel (42,456 → 5,898 → 3,127) and the pipeline was
working, but the run had been launched before the boundary-matching fix ([A9](#appendix-a--bug-index)), so it was using
roughly a fifth of the available LDN evidence. Finishing would have cost about $65 more for an
answer we had already decided not to present (§12) and would supersede immediately. Spend before
stopping: **$8.69 GPU plus roughly $12–14 of API**.

**Prerequisites for the real run:**


| prerequisite                                               | why                                                           |
| ---------------------------------------------------------- | ------------------------------------------------------------- |
| boundary-matching fix ([A9](#appendix-a--bug-index), done) | raises the candidate pool from ~42,600 to ~138,600 reports    |
| acquire the GPU only for `inclusion_prob`                  | it sat idle ~90% of the run; removes ~85% of GPU cost         |
| DrugBank aliases (§11.5)                                   | free additional synonyms before paying to scan anything       |
| decide the outcome count                                   | four outcomes quadruples the dominant `relevance_filter` cost |


**Sequencing:** LIFT is a prediction target, so it is not on the critical path. The four held-out
core-5 trials (§9.3) come first — they are what tell us whether any LIFT number is worth reporting.

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

> **These two numbers are now known to be artefacts.** Reading the extracted rows back (§10.2)
> showed that 21 of the 32 Brain Fog values are identical, inherited from the recovery threads the
> comments replied to rather than extracted from the commenters themselves
> ([A11](#appendix-a--bug-index)). The table is kept because the *diagnosis* is the result; the
> estimates are not.  
> There is very clearly bias and noise to the point of making us question everything here.

---

## 10. Problems

Defects **still open** in `naturalv2` — including ones we route around locally, since a
workaround in our fork does not fix the pipeline for anyone else. Things we have genuinely closed are
not here: the serving gotchas are in §7.5, and the complete registry of open *and* closed items, with
evidence, is [findings.md](findings.md).

### 10.1 Change-from-baseline labels versus absolute predictions — [A6](#appendix-a--bug-index)

**In one line: we were estimating how ill someone is, when the trial reported how much they
improved.** Two different quantities, compared as though they were the same.

The Fatigue Severity Scale is a questionnaire scored **1–49**, higher meaning worse  
fatigue. Lithium's trial did not report "patients ended up at 25 on the scale". It reported  
**−11.3** — patients *improved by* 11.3 points. A change, not a level.

We are mesuring abolute values here,  when we should according to the study be mesuring changes I think.  This probably requires both prompt rejigering and actual changes to the algorithm. **The fix** rewrites the description so it asks for the movement rather than the level ("report the CHANGE, follow-up minus baseline"), which makes the model's answer and the trial's label the same quantity. 

It is the same mistake as comparing *"the patient weighs 80 kg"* with *"the patient lost 5 kg"*.  
Both are in kilograms; comparing them is meaningless. The error is guaranteed and says nothing about  
how good the estimate was.

### 10.2 The estimate is wrong, and confident about it — [A11](#appendix-a--bug-index), [A8](#appendix-a--bug-index)

Two things failed independently, and it is worth separating them because they have different causes
and different fixes.

**1 — The answer is wrong, in a consistent direction.** Brain Fog predicts −33.9 against a true −9.0,
on a scale whose maximum possible improvement is −48. The model is claiming near-total recovery.

**2 — The error bar around that wrong answer is tiny.** ±0.98 on a 49-point scale. The interval
`(−34.87, −32.92)` misses −9.0 by about 25 half-widths. Neither outcome's interval contains the
truth: **coverage is 0 of 2**.

These answers seem suspicious, and it's because of bad attribution as to who is taking the drug.

**Take for example the following comments, each scored −35 as a reply within one post's history:**

```
"Yes! Those are horrible when they happen.. Thanks for sharing."
"Hey! We're you able to eventually ween off the antihistamines?"
"It's cheap. Will give it a try. I take generic zyrtec and Benadryl at night"
```

None of them reports an outcome, and none is about lithium. Every comment's prompt embeds the **full
initial post**, and the question asks what *"the individual described in the report"* would score —
which, in a comment, is two people, with the original poster described far more fully. The model
answers about the wrong one.

The separation is clean:


| lithium Brain Fog rows        | n   | mean extracted value |
| ----------------------------- | --- | -------------------- |
| own comment discusses lithium | 4   | **−2.5**             |
| only the thread mentions it   | 28  | **−33.2**            |
| *trial's answer*              |     | *−9.0*               |


Nor is this a lithium quirk: across LIFT's 3,127 rows, **76% never name the drug in their own text**.

This referent binding problem has been solved in our native pipeline, and we can probably just port
the fix over here — the `REPLY CHAIN` block in `src/prompts/intervention_config.py`, which demotes
upstream text to context and lets the model return *neutral* when a reply expresses no experience of
its own. The one adaptation needed is that `sample_ty` returns a continuous value, so it needs a real
null (`"Not stated"`) rather than another category.

**The same thing explains the tiny error bar.** 21 of the 32 Brain Fog values are the identical
−35.0, inherited from the same handful of threads. Resampling a column that is mostly one number
returns that number, so the interval collapses however many resamples you take. `bootstrap_size: 10`
is genuinely too low (§6.7) and worth raising — but it is not what caused this, and raising it on its
own would not have widened the interval by much.

### 10.3 Estimator and endpoint mismatch — [A5](#appendix-a--bug-index)

Upstream's default estimator (`natural_ipw`) cannot run a `notbinary` study: `conditional_extraction`
enumerates multiple-choice options over the outcome, continuous endpoints have none, and it dies with
a bare `ZeroDivisionError`. We route around it with `conf/estimate_mc.yaml`, so it no longer blocks  
us — but I'm not sure that this is the correct path forward.

---

## 11. Open problems

Distinct from §10. Those are defects with known causes and mostly known fixes. These are the
unsolved ones, where the right approach is genuinely undecided. Roughly ordered by how much each
blocks trusting an estimate — except the last, §11.8, which is not a blocker but the **goal** the
other eight are in service of.

### 11.1 Uncertainty quantification — what should a confidence interval here even mean?

Our interval measures exactly one thing: **how much the estimate would move if we drew a different
subset of the same extracted rows.** The bootstrap resamples a table that has already been extracted,
so every source of error that acted before that table existed sits outside the interval:

- **extraction error** — the model misreading a post. [A11](#appendix-a--bug-index) is one confirmed
case, and it moved the estimate by more than 20 points.
- **sampling variance in the LLM** — we take one draw per report, so a report the model was unsure
about is indistinguishable from one it was certain about (§6.4).
- **selection variance** — which patients chose to write anything down (§11.2).
- **model-specification variance** — the outcome model being an underdetermined linear regression
(§6.6).

If possible, we need to measure error in each of these.

**But we do not have to separate them to catch the problem: a 95% interval promises the true answer falls inside it 95% of the time, and we can just check whether it does.**

For each of the 19 trials whose real results we already have:

1. run the pipeline and record the interval it produces;
2. check whether the trial's published value falls inside that interval;
3. count how often it did.

**Progress looks like:** that count, for all 19 trials. It needs no new data, no labelling and no
GPU beyond the runs themselves — the trials are built, the answers are known, and nobody has done it.
It is also the cheapest way to find out whether any interval this pipeline reports can be quoted to
anyone.

### 11.2 Selection bias — people post about dramatic outcomes

The pseudo-population is self-selected in a way that correlates directly with the outcome. Dramatic  
recoveries and dramatic failures are both over-represented; "mild and unremarkable" is not.  
`inclusion_prob` corrects for *eligibility*, not for *propensity to post*.

We need to mesure and account for **selection bias if possible**. It needs either an external anchor (a survey with  
known prevalence) or an explicit model of the posting mechanism.  

**Progress looks like:** mesuring and normalizing magnatude of changes/effects reported somehow. 

### 11.3 Confounding — are we comparing like with like?

The estimator assumes **no unmeasured confounding** given eight covariates, all of them demographic:
condition on those and the treated are exchangeable with the untreated.

A confounder is anything that drives *both* the choice of treatment and the outcome. Baseline
severity, comorbidity and prior treatment failures all qualify, and none is among the eight. Someone
reaching for LDN after five failures is not exchangeable with someone trying it first.

**Open question:** can those be extracted from the text itself, and does adding them move the
estimate or just add noise?

**Progress looks like:** running a sensitivity analysis — how large would an unmeasured difference have to be  
to explain our error? "Implausibly large" clears confounding; "mild" implicates it.

### 11.4 Does model scale fix this?

Things could magically get better with a bigger more modern model!

### 11.5 How treatment mentions should be found at all — the matcher probably needs replacing

Fixing the boundary bug ([A9](#appendix-a--bug-index)) removed the worst symptom, but it patched a design that is
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



### 11.6 Prolific posters — the wrong unit of analysis

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

**This is VERY BAD for obvious reasons.** 

### 11.7 What is this benchmark actually measuring?

19 trials reduce to **16 independent drug signals**, several with change-from-baseline label
mismatches. How many trials are needed before an aggregate error is meaningful? Should shared-pool
trials be pooled or down-weighted? And is per-arm APO even the right target, when the
decision-relevant quantity is the treatment *contrast*?

Without answers, "mean absolute error over 19 trials" is a number without an interpretation. The
"what quantity" half of that question is now stated as a goal in its own right — §11.8.

### 11.8 The goal — predict the treatment effect, not the arm

We might want to look at what it would take to predict ATE, not ATO.

**Progress looks like:** one completed trial with both arms reported, estimated as a contrast and
scored against the trial's own ATE, alongside the drug-versus-untreated number so the gap between
them is visible rather than assumed. Until then §9.2's baseline table is the honest framing of what
our error figures mean — and A11 has to be fixed first, since a contrast between two mismeasured
arms is worse than a level, not better.

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
- The bug findings — A3 in particular, since it concerns *their* trial's factorial arms directly,
and **A11**, which is the most substantive thing we have: a reproducible demonstration that the
extraction step reads the wrong person's outcome, measured on 3,127 of their own trial's evidence
rows.

**Not solid enough to present as a forecast:** the point estimate — and as of A11 that is no longer
a calibration caveat but a correctness one. The lithium estimates are artefacts of comments
inheriting their thread's outcome, 76% of LIFT's rows never mention the drug in their own text, and
14.8% of its extracted values fall outside the endpoint's own scale. Add the 7B smoke model, the
open §10.1, and ~21% LDN evidence reach.

**The position to take:** *"Here is what patients report, here is a pipeline that turns it into a
testable prediction, and here is what we found by auditing our own output — including a measurement
bug that would have made any estimate from it meaningless, and two estimator bugs that affect
factorial trials like yours."* Finding A11 before quoting a number is a better story than the number
would have been, and it is true.

---

## 13. Appendix

**Repositories**

- Fork `Airwhale/naturalv2`, branch `shaun/patientpunk-integration` — **everything**: the estimator
changes, the study-construction and coverage-measurement tooling under `patientpunk/analysis/`, the
serving kit under `patientpunk/serving/`, and these docs. `main` on that fork is byte-identical to
Nikita's, so the branch diff is exactly our contribution.
- `PatientPunk` — `trial_superset/` is **retired**; it was the original home of the analysis tooling
and is kept read-only with a pointer. Nothing there is current.
- **No data lives in either repository.** The corpus, study definition and trial records are on S3
under `s3://patientpunk/trial_superset/`; `patientpunk/scripts/00_fetch_inputs.sh` pulls them.

**Environment**

- Runs in WSL Ubuntu, not native Windows: the estimator's compiled dependencies are blocked by
Windows Smart App Control, and `contextualize` imports the Unix-only `resource` module.
- Python 3.12 venv (managed with `uv`, since Ubuntu 26.04 ships 3.14 with no pip and some scientific
wheels lag). Point `PP_PYTHON` at it; see [patientpunk/README.md](../../patientpunk/README.md).

**Data**

- Corpus: `s3://patientpunk/trial_superset/natural_corpus_parquet/`
- Run outputs: `$SAVE_PATH` (defaults to `<repo>/outputs`) — `studies/`, `experiments/`, `nct_reports[_test]/`,
`reddit_data/`, `results/`

**Further reading**

- [findings.md](findings.md) — the full bug registry with evidence, as a fix-by-fix summary.
- [benchmark_19.md](benchmark_19.md) — how the 19-trial set was derived.
- [method_and_scope.md](method_and_scope.md) — the original scope decisions.

---

## Appendix A — Bug index

Short forms used throughout this document. "Ours" means the defect is in our code or our data;
everything else is in `naturalv2` and therefore affects Nikita's own results too.

This is a glossary, not the registry. **[findings.md](findings.md)** holds the full account of every
bug — evidence, root cause, and what was done about it — structured fix-by-fix, and carries entries
in the C and D series that this document does not use.


| id      | what                                                                                                                                                                                                                              | where                                          | status                                                                 |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- | ---------------------------------------------------------------------- |
| **A1**  | Condition matcher over- and under-matches. A plain substring test in both directions admitted 12 acute-COVID trials into Long COVID and dropped 7 genuine post-COVID ones.                                                        | `find_condition_ncts`                          | open upstream; we use a keyword classifier (§4.1 Tier 2)               |
| **A2**  | `notbinary` labels computed as `value / N`, which is a response rate for binary endpoints but meaningless for a continuous mean.                                                                                                  | `Experiment`                                   | **fixed upstream** in `7a2e006`                                        |
| **A3**  | Factorial arms named `"X/Placebo"` classified as placebo by title and silently dropped — including LIFT's LDN main effect.                                                                                                        | `check_nonplacebo` → `check_arm`               | **fixed upstream**; arms now typed by `ArmGroupType`                   |
| **A4**  | Test universe uses `status:act`, which means "active, not recruiting" and excludes every *recruiting* trial — LIFT included.                                                                                                      | test aggFilters                                | open; we use a recruiting-inclusive universe                           |
| **A5**  | The default estimator cannot run a `notbinary` study: `conditional_extraction` enumerates options over the outcome, continuous endpoints have none, and it dies with a bare `ZeroDivisionError`.                                  | `conditional_extraction`                       | open; we supply `conf/estimate_mc.yaml` (§10.3)                        |
| **A6**  | Change-from-baseline labels compared against absolute predictions (§10.1).                                                                                                                                                        | `Experiment` + our tooling                     | open — a design question for Nikita                                    |
| **A7**  | `detokenize=False` is hardcoded on every `prompt_logprobs` call; harmless in-process, but it crashes a *hosted* vLLM server.                                                                                                      | `conditional_extraction`                       | **fix written** — an in-process guard, on our branch                   |
| **A8**  | Bootstrap intervals implausibly tight — ±1 on n=32 (§10.2).                                                                                                                                                                       | `bootstrap_size`                               | open — needs her view                                                  |
| **A9**  | Treatment matching dropped ≤3-character aliases, costing 79% of LDN evidence.                                                                                                                                                     | `build_treatment_automaton` + curate           | **fixed** — boundary-aware matching added; strongest PR candidate      |
| **A10** | `author` is discarded after stage 2, so one prolific poster becomes many synthetic patients — 99 posts from one person became 99 "patients" (§11.6).                                                                              | contextualise + curate                         | open — a salted author hash is small; the statistics are the hard part |
| **A11** | A comment inherits the outcome of the **thread it replies to**. 76–88% of our evidence rows never mention the treatment in their own text (§10.2).                                                                                | `curate.py` `fmt_comment` + `sample_ty` prompt | open — the largest single defect we have found                         |
| **A12** | Extracted outcome values are never checked against the endpoint's range. LIFT produced 4,444,000 on a 0–55 scale; 14.8% of values fell outside it.                                                                                | `sample_ty` parsing                            | open — mechanical, and the range is already known                      |
| **B1**  | `query.cond="COVID"` misses trials tagged `SARS-CoV-2` / `PASC`, because CT.gov does not expand the term.                                                                                                                         | our CT.gov scope layer                         | fixed in `seed_terms`                                                  |
| **A13** | `NaturalMC`'s own docstring says not to use it for APOs — and `ate: False` is exactly that. The `oi` branch looks correct for an APO; the `ipw` branch does not (§6.4).                                                           | `natural_mc.py`                                | open — a question for Nikita, upstream of every estimate we have       |
| **A14** | A bare `except:` in the same class turns any fit failure into a silent `NaN`. Most likely to fire on the small-n outcomes, where the outcome model is already saturated (§6.6).                                                   | `natural_mc.py`                                | open — small; our runs did not hit it                                  |
| **D1**  | `gpu_memory_utilization` starves `prompt_logprobs`. The transient logits tensor sits outside vLLM's preallocated pool, so the usual `0.90` crashes the engine — and because it is a fraction, a bigger card does not help (§7.5). | serving config (ours)                          | **fixed** — `0.55`, plus a realistic-length probe before every run     |


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
| `bootstrap_size`        | number of bootstrap resamples for the confidence interval. Upstream default 10 — too few, though not what collapsed our intervals (§10.2)                                                                       |
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