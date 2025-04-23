# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # experiment 1 preprocess

# ##### This script converts .mp4 or .webm files to .wav files and uses MFA to extract sentence onset time.

# ### load packages and set up group-level file directories

# + editable=true slideshow={"slide_type": ""}
import pandas as pd
import os
import subprocess
import textgrid
import matplotlib.pyplot as plt
import seaborn as sns
import whisper
from glob import glob
import warnings
warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import zipfile
import wave

stimuli_file = ['expt3_b1.csv', 'expt3_b2.csv', 'expt3_b3.csv', 'expt3_b4.csv']
data_dir = '/Users/yiwei403/Documents/SONA_data/Exp3/'
df_first = pd.read_csv(os.path.join(data_dir,stimuli_file[0]))
                       
df_all = [df_first]
for stimuli in stimuli_file[1:]:
    df = pd.read_csv(os.path.join(data_dir, stimuli), skiprows=1, header=None)
    df.columns = df_first.columns
    df_all.append(df)

stimuli_df = pd.concat(df_all, ignore_index=True)
# -

# ### set up subject-level file directories

# + editable=true slideshow={"slide_type": ""}
subject_id = 'SP045'
subject_folder = os.path.join(data_dir,subject_id)
wavs_folder = os.makedirs(os.path.join(data_dir,subject_id,'wavs'), exist_ok=True)
wavs_dir = os.path.join(data_dir,subject_id,'wavs')
# transcription_file = f"{subject}_transcription.csv"
pcibex_log = f'{subject_id}_Exp3_pcibex_clean.xlsx'
pcibex_log_df = pd.read_excel(os.path.join(subject_folder, pcibex_log))
row_index = pcibex_log_df[pcibex_log_df['Label']=='exp3_b1'].index[0]
unique_id = pcibex_log_df.at[row_index, 'uniqueID']
print(f'participant`s unique id is {unique_id}')
# -

# ### convert .mp4 or .webm files to .wav files

# +
for file in os.listdir(os.path.join(subject_folder,'raw_audio')):
    if file.endswith(('.mp4', '.webm')):
        file_name = (os.path.splitext(os.path.basename(file))[0])
        # print(file_name)
        subprocess.call(['ffmpeg', '-loglevel', 'error', '-i', f'{subject_folder}/raw_audio/{file}', f'{wavs_dir}/{file_name}.wav'])

print('ffmpeg process done')

# + [markdown] editable=true slideshow={"slide_type": ""}
# ### use stimuli file as transcription
# -

# Loop over the stimuli file rows
for index, row in stimuli_df.iterrows():
    
    content = row['Target']
    # Define the name of the text file
    txt_file_name = f"{row['Condition']}_{row['Item']}_{unique_id}.txt"
    
    # Define the path to the text file
    txt_file_path = os.path.join(wavs_dir, txt_file_name)
    # Add the key-value pair to the dictionary, where the value is the third column of the current row
    with open(txt_file_path, 'w') as files:
        files.write(content)

# ### apply MFA to each .txt/.wav file pair and get a .TextGrid file with word onset time for each target word

# + editable=true slideshow={"slide_type": ""}
# Define the command and arguments
command = 'mfa'
args1 = ['server', 'start']
args2 = ['align', '--clean', '--use_postgres', '--overwrite', wavs_dir, 'english_mfa', 'english_mfa', wavs_dir]

# Run the command
subprocess.run([command] + args1)
subprocess.run([command] + args2)

# remove all the .txt files in the folder. 
# subprocess.run(f'rm {wavs_dir}/*.txt', shell=True)

mfa_output_files = os.listdir(wavs_dir)
textgrid_files = [f for f in mfa_output_files if f.endswith('.TextGrid')]

print(f'MFA done! {str(len(textgrid_files))} generated out of 120.')

# -

# ### extract word onset time from each .TextGrid file and generate a .csv file (wordOnsetTime.csv)

# +
# Create an empty DataFrame to store the results
results = pd.DataFrame()
from decimal import Decimal
for file in os.listdir(wavs_dir):
    
    # Check if the file is a TextGrid file
    if file.endswith('.TextGrid'):
        
        # Extract the item code from the file name
        file_name = file[:-9]
        trial_type = file_name.split('_')[0]
        item_code = file_name.split('_')[1]  
        
        if item_code in pcibex_log_df['itemcode'].values:
            # get the df of the current trial
            trial_df = pcibex_log_df[pcibex_log_df['itemcode'] == item_code]
            # trial_df['event_time'] = trial_df['event_time'].apply(lambda x: int(Decimal(x)))

            recording_start_row = trial_df[(trial_df['Parameter'] == 'Recording') & (trial_df['Value'] == 'Start')]
            recording_start_time = recording_start_row['EventTime'].values[0]
            # print(recording_start_time)
         
            target_print_row = trial_df[(trial_df['PennElementName'] == 'Target') & (trial_df['Parameter'] == 'Print')]
            target_print_time = target_print_row['EventTime'].values[0]
            # print(target_print_time)
            
            stim_time = int(target_print_time) - int(recording_start_time)

            # Get the transcription of the target word corresponding to the item code
            target = pcibex_log_df.loc[pcibex_log_df['itemcode'] == item_code, 'target'].values[0] 
            
            new_row = pd.DataFrame({
                    'filename': [file_name],
                    'subject': [subject_id],
                    'itemcode': [item_code],
                    'trialtype':[trial_type],                
                    'target': [target],
                    'stim_time':[stim_time],
                })
            
            # Read the TextGrid file
            textGridFile = textgrid.TextGrid.fromFile(os.path.join(wavs_dir, file))
            
            # Extract the word tier from the TextGrid file, and find the onset time of the target word
            wordTier = textGridFile.tiers[0]
            for interval in wordTier:
                if interval.mark != "":
                    # Create a new row with the item code, target word, file name, and onset time
                    new_row['MFA_sentence_onset'] = interval.minTime - stim_time/1000
                    break
                else:
                    new_row['MFA_sentence_onset'] = None
                    
            results = pd.concat([results, new_row], ignore_index=True)
            # results = pd.concat([results, new_row])
        else:
            print(f'{item_code} not in stimuli_df')


# Save the results to a .csv file
results.to_csv(os.path.join(data_dir, subject_id, f'{subject_id}_Exp3_adjustedSentenceOnsetTime.csv'), index=False)
# -

# ### plot individual subject' word onset times by condition 

# +
subject_id = subject_id
# whether to plot by 'trialtype' or 'tense'
condition = 'trialtype'

results = pd.read_csv(os.path.join(data_dir,subject_id, f'{subject_id}_Exp3_adjustedSentenceOnsetTime.csv'))
mean_mfa_onset_time = results.groupby(condition).agg({'MFA_sentence_onset':'mean'}).reset_index()
# mean_mfa_onset_time['MFA_sentence_onset']=mean_mfa_onset_time['MFA_sentence_onset']

# plot using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=condition, y='MFA_sentence_onset', data=mean_mfa_onset_time, errorbar='sd')
plt.title(f"{subject_id}: mean mfa onset time by {condition}")
plt.xlabel(condition)
plt.ylabel("mean mfa sentence onset time")
# plt.ylim(0.75,1.5)
plt.xticks(rotation=45)
plt.show()

# +
subject_folders = [item for item in os.listdir(data_dir) if item.startswith('SP')]
# print(subject_folders)

group_df = []

for subject_folder in subject_folders:
    subject_folder_path = os.path.join(data_dir, subject_folder)
    # print(subject_folder_path)
    csv_file = glob(os.path.join(subject_folder_path, "*_adjustedSentenceOnsetTime.csv"))
    # print(csv_file)
    if csv_file:
        csv_file = csv_file[0]
        # print(csv_file)
        df = pd.read_csv(csv_file)
    group_df.append(df)

all_data = pd.concat(group_df, ignore_index=True)

group_average = all_data.groupby(condition)['MFA_sentence_onset'].mean()
subject_average = all_data.groupby(['subject', condition])['MFA_sentence_onset'].mean()

group_plot = group_average.plot(kind='bar', color = '#6A5ACD')

for IV in group_average.index:
    # print(IV)
    IV_data = subject_average[subject_average.index.get_level_values(condition) == IV]
    # print(IV_data)
    group_plot.scatter([IV] * len(IV_data), IV_data.values, color = '#FF8C00')
    # print(IV_data.values)

plt.xlabel('condition')
plt.ylabel('mean MFA onset time(s)')
plt.title('Exp1 group mean MFA onset time by condition')
plt.xticks(rotation=45)
plt.show()           
# -


