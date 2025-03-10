import subprocess
import re
import pandas as pd
import os
import argparse

# List of condition lists
condition_lists = [
    ['diabetes', 't2dm', 'obesity', 'weight'],
    ['copd', 'pulmonary', 'asthma', 'respiratory', 'lung', 'pneumonia', 'rhinitis', 'fibrosis'],
    ['cancer', 'leukemia', 'lymphoma', 'myeloma', 'carcinoma', 'sarcoma', 'malignant', 'tumor', 'melanoma'],
    ['heart', 'cardiac', 'coronary', 'cardio', 'myocardial', 'atrial'],
    ['stroke', 'cerebrovascular', 'brain', 'infarction', 'ischemic', 'cerebral'],
    ['psoriasis', 'psoriatic'],
    ['hepatitis', 'hep', 'hbv', 'hcv'],
    ['hiv', 'aids'],
    ['tuberculosis', 'tb', 'mycobacterium'],
    ['covid', 'sars-cov-2', 'coronavirus'],
    ['hypertension', 'blood pressure'],
    ['acne', 'derma', 'eczema', 'skin', 'rosacea'],
    ['alzheimer', 'dementia', 'cognitive', 'parkinson', 'epilepsy', 'seizure', 'neuro', 'huntington'],
    ['lupus', 'sclerosis', 'ms', 'autoimmune', 'immune disorder'],
    ['migraine', 'headache'],
    ['stomach', 'digest', 'gastro', 'intestin', 'bowel', 'gastric', 'peptic', 'duodenal', 'ulcer', 'crohn'],
    ['liver', 'hepatic', 'cirrhosis'],
    ['kidney', 'renal', 'ckd', 'esrd', 'dialysis'],
    ['urinary', 'bladder', 'incontinence', 'cystitis', 'uti'],
    ['depress', 'schizo', 'psycho', 'anxiety', 'panic', 'sleep', 'insomnia', 'bipolar'],
    ['substance', 'opioid', 'alcohol', 'smoking', 'tobacco', 'addiction', 'abuse', 'nicotine'],
    ['gout', 'arthritis', 'arthritic', 'spondyl', 'osteo', 'bone', 'spine', 'spinal', 'disc', 'orthotic', 'carpal'],
    ['anemia', 'iron', 'hemo', 'thrombo', 'coagulation', 'platelet', 'purpura', 'clot'],
    ['eye', 'ocular', 'retina', 'retinopathy', 'macular', 'glaucoma', 'ophthalmic', 'cataract', 'uveitis', 'conjunctivitis', 'myopia'],
    ['uterine', 'ovarian', 'cervical', 'menstrual', 'endometriosis', 'fibroids', 'menopause', 'postmenopausal'],
    ['allergy', 'allergic', 'hypersensitivity'],
    ['pain', 'nociceptive', 'neuropathic'],
    ['transplant', 'graft', 'allograft', 'rejection'],
    ['genetic', 'congenital', 'inherited'],
    ['pregnancy', 'pregnant', 'prenatal', 'maternal', 'pediatric', 'child', 'infant'],
    ['vaccine', 'vaccination', 'immunization'],
    ['surgical', 'postoperative', 'complication'],
]

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run create_study with different conditions and record stats')
parser.add_argument('--script_path', type=str, default='create_study.py',
                    help='Absolute or relative path to the create_study.py')
parser.add_argument('--output_dir', type=str, default='.',
                    help='Directory to save the output files')
args = parser.parse_args()

# Ensure output directory exists
os.makedirs(args.output_dir, exist_ok=True)

# Function to run the study script and extract the results
def run_study(conditions):
    # Convert the list to a string for the command
    conditions_str = str(conditions).replace(' ', '')
    
    # Run the command using the provided path
    cmd = f"python {args.script_path} conditions={conditions_str}"
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Get the output
    output = result.stdout
    
    # Parse the output to extract the numbers
    train_trials = re.search(r'Train: (\d+) trials', output)
    train_labels = re.search(r'Train: \d+ trials, (\d+) labels', output)
    val_trials = re.search(r'Val: (\d+) trials', output)
    val_labels = re.search(r'Val: \d+ trials, (\d+) labels', output)
    test_trials = re.search(r'Test: (\d+) trials', output)
    test_labels = re.search(r'Test: \d+ trials, (\d+) labels', output)
    
    # Extract the numbers or set to None if not found
    train_trials = int(train_trials.group(1)) if train_trials else None
    train_labels = int(train_labels.group(1)) if train_labels else None
    val_trials = int(val_trials.group(1)) if val_trials else None
    val_labels = int(val_labels.group(1)) if val_labels else None
    test_trials = int(test_trials.group(1)) if test_trials else None
    test_labels = int(test_labels.group(1)) if test_labels else None
    
    return {
        'conditions': conditions,
        'train_trials': train_trials,
        'train_labels': train_labels,
        'val_trials': val_trials,
        'val_labels': val_labels,
        'test_trials': test_trials,
        'test_labels': test_labels
    }

# Run all the studies and collect results
results = []
for conditions in condition_lists:
    results.append(run_study(conditions))

# Create a DataFrame for easier manipulation
df = pd.DataFrame(results)

# Save to CSV
csv_path = os.path.join(args.output_dir, 'create_study_results.csv')
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")