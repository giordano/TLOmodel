from scripts.rti.rti_create_graphs import create_rti_graphs
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from tlo.analysis.utils import parse_log_file

no_cast_dalys = []
for logfile in os.listdir("outputs/blocked_interventions/casts"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/casts/" + logfile)
        model_pop_size = parsed['tlo.methods.demography']['population']['total'].tolist()
        data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
        sim_start_year = parsed['tlo.simulation']['info']['date'].iloc[0].year
        sim_end_year = parsed['tlo.simulation']['info']['date'].iloc[- 1].year
        sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year))
        Data_Pop = data.groupby(by="Year")["Count"].sum()
        Data_Pop = Data_Pop.loc[sim_year_range]
        scaling_df = pd.DataFrame({'total': model_pop_size})
        scaling_df['pred_pop_size'] = Data_Pop.to_list()
        scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
        scaling_df.index = sim_year_range
        dalys_per_year = parsed['tlo.methods.healthburden']['dalys'].groupby('year').sum()
        dalys_per_year = dalys_per_year.loc[scaling_df.index]
        dalys_per_year['scaled_dalys'] = dalys_per_year['Transport Injuries'] * scaling_df['scale_for_each_year']
        no_cast_dalys.append(dalys_per_year['scaled_dalys'].sum())
no_minor_dalys = []
for logfile in os.listdir("outputs/blocked_interventions/minor"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/minor/" + logfile)
        model_pop_size = parsed['tlo.methods.demography']['population']['total'].tolist()
        data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
        sim_start_year = parsed['tlo.simulation']['info']['date'].iloc[0].year
        sim_end_year = parsed['tlo.simulation']['info']['date'].iloc[- 1].year
        sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year))
        Data_Pop = data.groupby(by="Year")["Count"].sum()
        Data_Pop = Data_Pop.loc[sim_year_range]
        scaling_df = pd.DataFrame({'total': model_pop_size})
        scaling_df['pred_pop_size'] = Data_Pop.to_list()
        scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
        scaling_df.index = sim_year_range
        dalys_per_year = parsed['tlo.methods.healthburden']['dalys'].groupby('year').sum()
        dalys_per_year = dalys_per_year.loc[scaling_df.index]
        dalys_per_year['scaled_dalys'] = dalys_per_year['Transport Injuries'] * scaling_df['scale_for_each_year']
        no_minor_dalys.append(dalys_per_year['scaled_dalys'].sum())

no_major_dalys = []
for logfile in os.listdir("outputs/blocked_interventions/major"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/major/" + logfile)
        model_pop_size = parsed['tlo.methods.demography']['population']['total'].tolist()
        data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
        sim_start_year = parsed['tlo.simulation']['info']['date'].iloc[0].year
        sim_end_year = parsed['tlo.simulation']['info']['date'].iloc[- 1].year
        sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year))
        Data_Pop = data.groupby(by="Year")["Count"].sum()
        Data_Pop = Data_Pop.loc[sim_year_range]
        scaling_df = pd.DataFrame({'total': model_pop_size})
        scaling_df['pred_pop_size'] = Data_Pop.to_list()
        scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
        scaling_df.index = sim_year_range
        dalys_per_year = parsed['tlo.methods.healthburden']['dalys'].groupby('year').sum()
        dalys_per_year = dalys_per_year.loc[scaling_df.index]
        dalys_per_year['scaled_dalys'] = dalys_per_year['Transport Injuries'] * scaling_df['scale_for_each_year']
        no_major_dalys.append(dalys_per_year['scaled_dalys'].sum())

