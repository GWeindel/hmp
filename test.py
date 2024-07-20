import os
import re
import mne
import numpy
import pandas as pd

# Define the base path where the numbered folders are located
base_path = '~/Downloads/'
base_path = os.path.expanduser(base_path)

# Define the channel types to be set
channel_types = {
    'HEOG_left': 'eog',
    'HEOG_right': 'eog',
    'VEOG_lower': 'eog',
    '(corr) HEOG': 'eog',
    '(corr) VEOG': 'eog',
    '(uncorr) HEOG': 'eog',
    '(uncorr) VEOG': 'eog'
}

mapping_scheme = {
    111: ['blue', 1, 'left', 1, 'top', 1],
    112: ['blue', 1, 'left', 1, 'bottom', 2],
    121: ['blue', 1, 'right', 2, 'top', 1],
    122: ['blue', 1, 'right', 2, 'bottom', 2],
    211: ['pink', 2, 'left', 1, 'top', 1],
    212: ['pink', 2, 'left', 1, 'bottom', 2],
    221: ['pink', 2, 'right', 2, 'top', 1],
    222: ['pink', 2, 'right', 2, 'bottom', 2]
}

# Function to translate ecode using the mapping scheme
def translate_ecode(ecode, mapping_scheme):
    return mapping_scheme.get(ecode, [None] * 6)



# Iterate over the numbered folders
for folder in os.listdir(base_path):
    if folder.isdigit():
        folder_path = os.path.join(base_path, folder)
        
        # Read the EEG epochs
        epochs_path = os.path.join(folder_path, f'{folder}_N2pc_shifted_ds_reref_ucbip_hpfilt_ica_corr_cbip_elist_bins_epoch_interp_ar.set')
        epochs = mne.read_epochs_eeglab(epochs_path)
        
        # Set the channel types
        epochs.set_channel_types(channel_types)
        
        # Pick only EEG channels
        epochs = epochs.pick_types(eeg=True)
        
        # Read the event list file (if needed)
        event_list_path = os.path.join(folder_path, f'{folder}_N2pc_Eventlist_RTs.txt')
        with open(event_list_path, 'r') as file:
            lines = file.readlines()

        # Find the line where the actual data starts
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith('# item'):
                data_start = i + 2  # data starts two lines after the header line
                break
        
        # Process the data lines into a list of dictionaries
        data = []
        for line in lines[data_start:]:
            if not line.strip():  # Skip empty lines
                continue
            # Split line by whitespace
            split_line = line.split()
            # Combine all parts after the 10th element (index 9) into one field
            combined_last_field = ' '.join(split_line[10:])
            # Extract digits from the brackets
            digits_in_brackets = re.findall(r'\d+', combined_last_field)
            # Append the first 10 elements and the list of digits
            data.append({
                'item': split_line[0],
                'bepoch': split_line[1],
                'ecode': split_line[2],
                'label': split_line[3],
                'onset': split_line[4],
                'diff': split_line[5],
                'dura': split_line[6],
                'b_flags': split_line[7],
                'a_flags': split_line[8],
                'enable': split_line[9],
                'bin': digits_in_brackets
            })
        # MNE read trigger description
        mapping_dict = epochs.event_id
        # Create the DataFrame
        df = pd.DataFrame(data)
        df['rt'] = df['diff'].shift(-1)
        
        metadata = df
        metadata.bepoch = metadata.bepoch.astype(int)
        metadata.ecode = metadata.ecode.astype(int)
        metadata['rt'] = metadata['rt'].astype(float)
        
    
        arr = epochs.events.copy()
        # Invert keys and values of mapping_dict, keeping only the three-digit code in parentheses
        inverted_mapping = {v: k.split('(')[1].split(')')[0] for k, v in mapping_dict.items() if '(' in k}
        
        # Replace last column using inverted mapping dictionary
        arr[:, -1] = [int(inverted_mapping.get(row[2])) for row in arr]
        
        # Find indices where metadata_ecode matches arr_last_column
        indices = []
        filtered_metadata = metadata.copy()
        i = 0
        for value in arr[:,-1]:
            while i < len(metadata.ecode) and metadata.ecode.values[i] != value:
                i += 1
            indices.append(i)
            i += 1
        
        assert (arr[:,-1] ==  metadata.ecode.values[indices]).all()

        # Create a DataFrame from metadata_ecode and translate using the mapping scheme
        translated_columns = ['color', 'num1', 'side', 'num2', 'position', 'num3']
        metadata[translated_columns] = metadata['ecode'].apply(lambda x: pd.Series(translate_ecode(x, mapping_scheme)))

        
        epochs.metadata = metadata.iloc[indices]
        # Remove epoch discarded during AR
        filtered_metadata = epochs.metadata[epochs.metadata['bin'].apply(lambda x: len(x) > 1)]
        epochs = epochs[filtered_metadata.index]

        epochs.save(f'{folder}-epo.fif', overwrite=True)
        print(f'Saved {folder}')
