#! /usr/bin/env python

import sys

import numpy as np
import pandas as pd


def load_dose_response():
    path = 'combined_drug_growth.ALMANAC'
    df = pd.read_table(path, engine='c', dtype=str)
    df = df.rename(columns={'SOURCE': 'Source', 'CELL': 'Sample',
                            'DRUG1': 'Drug1', 'DRUG2': 'Drug2',
                            'DOSE1': 'Dose1', 'DOSE2': 'Dose2',
                            'GROWTH': 'Growth', 'STUDY': 'Study'})
    df.Growth = df.Growth.astype(np.float32)
    return df


def load_cell_info():
    path = 'NCI60_cells'
    df = pd.read_table(path, engine='c')
    return df


def load_drug_info():
    path = 'ALMANAC_drugs'
    df = pd.read_table(path, engine='c')
    return df


def cell_name_to_id(name, df_cell):
    df = df_cell
    hits = df[(df['NCI60.ID'] == name.upper()) |
              (df['CELLNAME'] == name.upper()) |
              (df['Name'] == name.upper())]
    if not hits.shape[0]:
        hits = df[df['NCI60.ID'].str.contains(name.upper()) |
                  df['CELLNAME'].str.contains(name.upper()) |
                  df['Name'].str.contains(name.upper())]
    return hits.iloc[0]['NCI60.ID']


def drug_name_to_id(name, df_drug):
    if name.startswith('NSC.'):
        return name
    else:
        return df_drug[df_drug.Name.str.lower() == name.lower()].iloc[0]['NSC.ID']


def expected_growth(growth_1, growth_2):
    if growth_1 <= 0 or growth_2 <= 0:
        growth = min(growth_1, growth_2)
    else:
        growth = min(growth_1, 100) * min(growth_2, 100) / 100
    return growth


def custom_combo_score(combined_growth, growth_1, growth_2):
    expected = expected_growth(growth_1, growth_2)
    custom_score = expected - combined_growth
    return custom_score


def load_data(cell, drug1, drug2, df_response, df_cell, df_drug):
    cell = cell_name_to_id(cell, df_cell)
    drug1 = drug_name_to_id(drug1, df_drug)
    drug2 = drug_name_to_id(drug2, df_drug)
    df = df_response

    df_combo = df[(df.Sample == cell) & (df.Drug1 == drug1) & (df.Drug2 == drug2)]
    if not df_combo.shape[0]:
        df_combo = df[(df.Sample == cell) & (df.Drug1 == drug2) & (df.Drug2 == drug1)]
        drug1, drug2 = drug2, drug1

    study = df_combo.iloc[0]['Study']

    df1 = df[(df.Sample == cell) & (df.Drug1 == drug1) & df.Drug2.isnull()]
    df1 = df1[df1.Dose1.isin(df_combo.Dose1)]
    df1_agg = df1.groupby('Dose1')['Growth'].agg(['mean', 'std', 'count']).reset_index(drop=True)
    df1 = df1[df1.Study == study].reset_index(drop=True)
    df1 = pd.concat([df1, df1_agg], axis=1).sort_values(by='Dose1', ascending=False)

    df2 = df[(df.Sample == cell) & (df.Drug1 == drug2) & df.Drug2.isnull()]
    df2 = df2[df2.Dose1.isin(df_combo.Dose2)]
    df2_agg = df2.groupby('Dose1')['Growth'].agg(['mean', 'std', 'count']).reset_index(drop=True)
    df2 = df2[df2.Study == study].reset_index(drop=True)
    df2 = pd.concat([df2, df2_agg], axis=1).sort_values(by='Dose1', ascending=False)

    df_combo = df_combo.assign(ExpectedGrowth = df_combo.apply(lambda x: expected_growth(df1[df1.Dose1 == x.Dose1].Growth.iloc[0],
                                                                                         df2[df2.Dose1 == x.Dose2].Growth.iloc[0]), axis=1).astype(np.float32))

    df_combo = df_combo.reset_index(drop=True).sort_values(by=['Dose1', 'Dose2'], ascending=False)

    return df_combo, df1, df2


def main():
    if len(sys.argv) > 3:
        cell = sys.argv[1]
        drug1 = sys.argv[2]
        drug2 = sys.argv[3]
    else:
        cell = 'MDA-MB-468'
        drug1 = 'Nilotinib'
        drug2 = 'Paclitaxel'

    df_response = load_dose_response()
    df_cell = load_cell_info()
    df_drug = load_drug_info()

    # df_combo, df1, df2 = load_data('MDA-MB-468', 'Nilotinib', 'Paclitaxel', df_response, df_cell, df_drug)
    # df_combo, df1, df2 = load_data('SK-MEL-28', 'NSC.733504', 'NSC.226080', df_response, df_cell, df_drug)

    df_combo, df1, df2 = load_data(cell, drug1, drug2, df_response, df_cell, df_drug)

    n = df1.shape[0]
    growth = df_combo.Growth.values.reshape((n, n))
    expected = df_combo.ExpectedGrowth.values.reshape((n, n))
    score = expected - growth

    drug1_doses = df1.Dose1.tolist()
    drug2_doses = df2.Dose1.tolist()

    print(df_combo)
    print('Experimental growth:\n', growth)
    print('Expected growth:\n', expected)
    print('Drug1 doses:', drug1_doses)
    print('Drug2 doses:', drug2_doses)


if __name__ == '__main__':
    main()
