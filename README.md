# NATURAL-v2

</div>

This repository extends [NATURAL](https://arxiv.org/abs/2407.07018) to larger data and evaluation scales.

______________________________________________________________________

## Set-up

```bash
git clone https://github.com/nikitadhawan/naturalv2.git
cd naturalv2
python setup.py develop
```

Create a user file `conf/user/{your_name}.yaml`. See [nikita.yaml](https://github.com/nikitadhawan/naturalv2/tree/main/conf/user/nikita.yaml) for an example.
______________________________________________________________________

## Retrospective Study

To create a retrospective study for some `condition` (e.g. "diabetes"), with temporally split training and validation clinical trials, run:

```bash
python create_study.py condition={condition}
```

______________________________________________________________________

## Data Filtering and Curation


______________________________________________________________________

## Estimating NATURAL ATEs
