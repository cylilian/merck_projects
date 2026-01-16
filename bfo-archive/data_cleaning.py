# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 09:48:05 2024

@author: ruanzhiy
"""

import pandas as pd

# Load the data
data_file_path = "data/DPS Protein Formulation Characterization - All data.csv"
df = pd.read_csv(data_file_path,encoding = 'utf-16',delimiter='\t')
SAMPLE_IDENTIFIER = ['Project', 'Protein', 'Modality', 'Stress Condition', 'Temp. (°C)', 'Time (Days)','Protein Conc (mg/mL)']
FOMULATION_SPACE = ['pH', 'Buffer','NaCl (mM)', 'PS80 (%)', 'Sucrose (%)','Trehalose (%)', 'Additional Excipient Name']
RESPONSE = ['UP-SEC - %Monomer', 'UP-SEC - %LMW', 'UP-SEC - %HMW'] # %Monomer + %LMW + %HMW = 100%,Goal: Keep %LMW and %HMW as low as possible

COLUMN_OF_INTEREST = SAMPLE_IDENTIFIER + FOMULATION_SPACE + RESPONSE
df_train = df[COLUMN_OF_INTEREST]
