from scripts.rti.rti_create_graphs import create_rti_graphs
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from tlo.analysis.utils import parse_log_file

normal_run_dalys = []
normal_run_deaths = []
for logfile in os.listdir("outputs/blocked_interventions/all"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/all/" + logfile)
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
        normal_run_dalys.append(dalys_per_year['scaled_dalys'].sum())
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        normal_run_deaths.append(deaths['scaled'].sum())

no_cast_dalys = []
no_cast_deaths = []
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
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_cast_deaths.append(deaths['scaled'].sum())
no_minor_dalys = []
no_minor_deaths = []
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
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_minor_deaths.append(deaths['scaled'].sum())

no_major_dalys = []
no_major_deaths = []
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
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_major_deaths.append(deaths['scaled'].sum())


mean_dalys_per_scenario = [np.mean(normal_run_dalys), np.mean(no_cast_dalys), np.mean(no_minor_dalys),
                           np.mean(no_major_dalys)]
scenarios = ['Normal', 'No fracture\ncasts', 'No minor\nsurgery', 'No major\nsurgery']
plt.bar(np.arange(len(mean_dalys_per_scenario)), mean_dalys_per_scenario,
        color=['lightsteelblue', 'lightsalmon', 'thistle', 'peachpuff'])
plt.xticks(np.arange(len(mean_dalys_per_scenario)), scenarios)
plt.ylabel('DALYs')
plt.title('The number of DALYs produced in simulations\n where certain treatments were blocked')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/BlockedInterventions/'
            'DALYs_blocked_treatments.png', bbox_inches='tight')
plt.clf()
mean_deaths_per_scenario = [np.mean(normal_run_deaths), np.mean(no_cast_deaths), np.mean(no_minor_deaths),
                            np.mean(no_major_deaths)]
plt.bar(np.arange(len(mean_deaths_per_scenario)), mean_deaths_per_scenario,
        color=['lightsteelblue', 'lightsalmon', 'thistle', 'peachpuff'])
plt.xticks(np.arange(len(mean_deaths_per_scenario)), scenarios)
plt.ylabel('Deaths')
plt.title('The number of deaths produced in simulations\n where certain treatments were blocked')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/BlockedInterventions/'
            'Deaths_blocked_treatments.png', bbox_inches='tight')
