import pandas as pd
import os
import zipfile
import shutil
import re
import sys
import io

study = 'Exp5'
dates = '0314-0410-25'
data_dir = r'U:\yfshah\Aphasia Research\Projects\Agrammatism MEG R01\R01 Experiments\SONA_Data'
server_data_dir = os.path.join(data_dir, study, dates)
pcibex_data_dir = os.path.join(data_dir, study, 'all_pcibex')
archive_pcibex_dir = os.path.join(data_dir, study, 'archive_pcibex')
unzipped_file_folder = os.makedirs(os.path.join(server_data_dir, 'unzipped_files'), exist_ok=True)
unzipped_file_dir = os.path.join(server_data_dir, 'unzipped_files')
log_file = os.path.join(study, f'python_log_{dates}.txt')


# Custom class to duplicate output
# so that all the print() output is displayed both in the console
# and will be printed to a .txt file at the end of this script
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

# console output
original_stdout = sys.stdout
# output that will be captured and later printed to a .txt file
captured_output = io.StringIO()
# outputs will be displayed and captured at the same time
sys.stdout = Tee(original_stdout, captured_output)


def read_pcibex(filepath, auto_colnames=True,
                fun_col=lambda col, cols: [col + ".Ibex" if c == col else c for c in cols]):
    # Check if the file is empty
    if os.stat(filepath).st_size == 0:
        print(f"The file {filepath} is empty.")
        return None

    # Determine the maximum number of columns
    with open(filepath, 'r') as f:
        n_cols = max(len(line.split(',')) for line in f if line.strip() and not line.startswith('#'))
    if auto_colnames:
        cols = []
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    match = re.match(r'^# (\d+)\. (.+)\.$', line)
                    if match:
                        index = int(match.group(1))
                        value = match.group(2)
                        if callable(fun_col):
                            cols = fun_col(value, cols)
                        if len(cols) < index:
                            cols.extend([None] * (index - len(cols)))
                        cols[index - 1] = value
                        if index == n_cols:
                            break
        return pd.read_csv(filepath, comment='#', header=None, names=cols)
    else:
        return pd.read_csv(filepath, comment='#', header=None, names=range(1, n_cols + 1))


# function to extract the list of subfolder names if they exist
def get_subfolder_names_if_any(folder_path):
    subfolders = []
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if entry.is_dir():
                subfolders.append(entry.name)
    return subfolders


print(f'Start organizing data for {study}, server data: {dates}')

# unzipping all the files in the server data folder
for zip_file in os.listdir(server_data_dir):
    if zip_file.endswith('.zip'):
        zip_file_path = os.path.join(server_data_dir, zip_file)
        try:
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                for to_unzip_file in zip_ref.namelist():
                    to_unzip_file_path = os.path.join(unzipped_file_dir, to_unzip_file)
                    if not os.path.exists(to_unzip_file_path):
                        zip_ref.extract(to_unzip_file, unzipped_file_dir)
        except zipfile.BadZipFile:
            print(f'invalid zip file: {zip_file}')
print('All server files unzipped')

# save a set of unique_id extracted from all the unzipped file
unzipped_file_unique_id = set()
for file in os.listdir(unzipped_file_dir):
    if not file.startswith('audiotest'):
        filename = file.rsplit('.', 1)[0]
        unique_id = filename.split('_')[2]
        unzipped_file_unique_id.add(unique_id)


# extract unique_id, subject_id from each pcibex log file.
# if the unique id is also found in the server data,
# create a new folder for the corresponding subject id,
# transform this pcibex log file, rename and move it to its corresponding subject's folder
# then move this pcibex log file to its corresponding archive folder.
# (what's left in the all_pcibex folder are pcibex log files that don't have corresponding server data)

subject_id_to_unique_ids = {}
subfolders = get_subfolder_names_if_any(pcibex_data_dir)

if not subfolders:
    subfolders = ['']

# looping through all the pcibex subfolders (if any), extract subject_id and unique_id from each .csv file.
for subfolder in subfolders:
    subfolder_path = os.path.join(pcibex_data_dir, subfolder) if subfolder else pcibex_data_dir
    for filename in os.listdir(subfolder_path):
        if filename.endswith('.csv') and not filename.startswith('~$'):
            csv_file = os.path.join(subfolder_path, filename)
            results = read_pcibex(csv_file)
            # check if the pcibex log is empty
            if results is None:
                print(f'WARNING: {filename} is empty')
            else:
                unique_id = next(uid for uid in results['uniqueID'].unique() if pd.notna(uid))
                subject_id = next(sid for sid in results['ParticipantID'].unique() if pd.notna(sid))
                # creating a dictionary contains linking subject id to its corresponding unique id(s) and subfolder(s).
                if subject_id not in subject_id_to_unique_ids:
                    subject_id_to_unique_ids[subject_id] = []
                subject_id_to_unique_ids[subject_id].append((unique_id, subfolder))

                # if the unique_id is also found in the server data.
                if unique_id in unzipped_file_unique_id:

                    # create a subject folder with subject_id,
                    subject_folder_path = os.path.join(data_dir, study, subject_id, subfolder)
                    os.makedirs(subject_folder_path, exist_ok=True)

                    # depending on whether there are subfolders or not
                    # move the pcibex .csv file to archive folder (or its subfolder)
                    # print results to .xlsx file under the subject folder (or its subfolder)
                    if subfolders == ['']:
                        new_xlsx_path = os.path.join(subject_folder_path,
                                                     f'{subject_id}_{study}_pcibex_clean.xlsx')
                        shutil.move(csv_file, archive_pcibex_dir)
                    else:
                        new_xlsx_path = os.path.join(subject_folder_path,
                                                     f'{subject_id}_{study}_{subfolder}_pcibex_clean.xlsx')
                        archive_dir = os.path.join(archive_pcibex_dir, f'{subfolder}')
                        shutil.move(csv_file, archive_dir)
                    results.to_excel(new_xlsx_path, index=False)

                    print(f'creating folder for {subject_id} and moving cleaned pcibex log file')
                else:
                    print(f'{subject_id}:{(unique_id, subfolder)} from {filename} not found in server data')

# print the subject_id_to_unique_ids dictionary
for key, value in subject_id_to_unique_ids.items():
    print(f"{key}: {value}")
print('All new subjects` folders created, all pcibex log files renamed and moved')

# check if unique ids found in the server data don't have corresponding pcibex log files.
pcibex_unique_ids = {uid for subject_ids in subject_id_to_unique_ids.values() for uid, _ in subject_ids}
extra_uid_in_server_data = unzipped_file_unique_id - pcibex_unique_ids
if not extra_uid_in_server_data:
    print("unique ids agree between pcibex and server data")
else:
    print('WARNING: missing pcibex file for these unique id found in server data:')
    for uid in extra_uid_in_server_data:
        print(uid)


# going through all the unzipped server data,
# move each file to its corresponding subject folder (subfolder) using using the subject_id_to_unique_ids dictionary.
for file in os.listdir(unzipped_file_dir):
    if not file.startswith('audiotest'):
        filename = file.rsplit('.', 1)[0]
        unique_id = filename.split('_')[2]
        for subject_id, unique_id_session_info in subject_id_to_unique_ids.items():
            for uid, folder_name in unique_id_session_info:
                if unique_id == uid:
                    new_unzipped_file_dir = os.path.join(data_dir, study, subject_id, folder_name, 'raw_audio')
                    os.makedirs(new_unzipped_file_dir, exist_ok=True)
                    # somehow using my own move_file function generates permission issue here,
                    # but shutil.move works fine
                    shutil.move(os.path.join(unzipped_file_dir, file), new_unzipped_file_dir)

print(f'Data organization finished for {study}, date range: {dates}')

# reset the sys.stdout to its original value to restore normal console display
sys.stdout = original_stdout

# print all the console output to a .txt file.
with open(log_file, "w") as file:
    file.write(captured_output.getvalue())
