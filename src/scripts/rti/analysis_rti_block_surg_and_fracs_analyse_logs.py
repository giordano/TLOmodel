from pathlib import Path

from scripts.rti.rti_create_graphs import create_rti_graphs
import os
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from tlo.analysis.utils import (
    parse_log_file,
    extract_params,
    extract_results,
    get_scenario_info,
    get_scenario_outputs,
    load_pickled_dataframes,
    summarize,
)

def extract_yll_yld(results_folder):
    yll = pd.DataFrame()
    yld = pd.DataFrame()
    info = get_scenario_info(results_folder)
    for draw in range(info['number_of_draws']):
        yll_this_draw = []
        yld_this_draw = []
        for run in range(info['runs_per_draw']):
            try:
                yll_df: pd.DataFrame = \
                    load_pickled_dataframes(
                        results_folder, draw, run, "tlo.methods.healthburden"
                        )["tlo.methods.healthburden"]
                yll_df = yll_df['yll_by_causes_of_death_stacked']
                yll_df = yll_df.groupby('year').sum()
                rti_columns = [col for col in yll_df.columns if 'RTI' in col]
                yll_df['yll_rti'] = [0.0] * len(yll_df)
                for col in rti_columns:
                    yll_df['yll_rti'] += yll_df[col]
                sim_start_year = min(yll_df.index)
                sim_end_year = max(yll_df.index) - 1
                sim_year_range = pd.Index(np.arange(sim_start_year, sim_end_year + 1))
                pop_size_df: pd.DataFrame = \
                    load_pickled_dataframes(
                        results_folder, draw, run, "tlo.methods.demography"
                    )["tlo.methods.demography"]
                pop_size_df = pop_size_df['population']
                pop_size_df['year'] = pop_size_df['date'].dt.year
                pop_size_df = pop_size_df.loc[pop_size_df['year'].isin(sim_year_range)]
                scaling_df = pd.DataFrame({'total': pop_size_df['total']})
                data = pd.read_csv("resources/demography/ResourceFile_Pop_Annual_WPP.csv")
                Data_Pop = data.groupby(by="Year")["Count"].sum()
                Data_Pop = Data_Pop.loc[sim_year_range]
                scaling_df['pred_pop_size'] = Data_Pop.to_list()
                scaling_df['scale_for_each_year'] = scaling_df['pred_pop_size'] / scaling_df['total']
                scaling_df.index = sim_year_range
                yll_df = yll_df.loc[sim_year_range]
                yll_df['scaled_yll'] = yll_df['yll_rti'] * scaling_df['scale_for_each_year']
                total_yll = yll_df['scaled_yll'].sum()
                yll_this_draw.append(total_yll)
                yld_df: pd.DataFrame = \
                    load_pickled_dataframes(results_folder, draw, run, "tlo.methods.rti")["tlo.methods.rti"]
                yld_df = yld_df['rti_health_burden_per_day']
                yld_df['year'] = yld_df['date'].dt.year
                yld_df = yld_df.groupby('year').sum()
                yld_df['total_daily_healthburden'] = \
                    [sum(daly_weights) for daly_weights in yld_df['daly_weights'].to_list()]
                yld_df['scaled_healthburden'] = yld_df['total_daily_healthburden'] * \
                                                scaling_df['scale_for_each_year'] / 365
                total_yld = yld_df['scaled_healthburden'].sum()
                yld_this_draw.append(total_yld)
            except KeyError:
                yll_this_draw.append(np.mean(yll_this_draw))
                yld_this_draw.append(np.mean(yld_this_draw))
        yll[str(draw)] = yll_this_draw
        yld[str(draw)] = yld_this_draw
    return yll, yld


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

outputspath = Path('./outputs/rmjlra2@ucl.ac.uk/')
results_folder = get_scenario_outputs('rti_analysis_full_calibrated.py', outputspath)[- 6]
# look at one log (so can decide what to extract)
log = load_pickled_dataframes(results_folder)

# get basic information about the results
info = get_scenario_info(results_folder)

# 1) Extract the parameters that have varied over the set of simulations
params = extract_params(results_folder)
# get main paper results, incidence of RTI, incidence of death and DALYs
extracted_incidence_of_death = extract_results(results_folder,
                                               module="tlo.methods.rti",
                                               key="summary_1m",
                                               column="incidence of rti death per 100,000",
                                               index="date"
                                               )
extracted_incidence_of_RTI = extract_results(results_folder,
                                             module="tlo.methods.rti",
                                             key="summary_1m",
                                             column="incidence of rti per 100,000",
                                             index="date"
                                             )
yll, yld = extract_yll_yld(results_folder)
mean_incidence_of_death = summarize(extracted_incidence_of_death, only_mean=True).mean()
mean_incidence_of_RTI = summarize(extracted_incidence_of_RTI, only_mean=True).mean()
# mean yll, yld per draw
mean_yll = yll.mean()
mean_yld = yld.mean()
batch_all_dalys = mean_yll[4] + mean_yld[4]
scale_to_dalys = batch_all_dalys / np.mean(normal_run_dalys)
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


no_hs_results = get_scenario_outputs('rti_analysis_full_calibrated_no_hs.py', outputspath)[- 1]
no_hs_extracted_incidence_of_death = extract_results(no_hs_results,
                                                     module="tlo.methods.rti",
                                                     key="summary_1m",
                                                     column="incidence of rti death per 100,000",
                                                     index="date"
                                                     )
no_hs_extracted_incidence_of_RTI = extract_results(no_hs_results,
                                                   module="tlo.methods.rti",
                                                   key="summary_1m",
                                                   column="incidence of rti per 100,000",
                                                   index="date"
                                                   )
no_hs_yll, no_hs_yld = extract_yll_yld(no_hs_results)
no_hs_mean_inc_rti = summarize(no_hs_extracted_incidence_of_RTI, only_mean=True).mean()
no_hs_mean_inc_death = summarize(no_hs_extracted_incidence_of_death, only_mean=True).mean()
gbd_inc = 954.2
scale_for_no_hs = np.divide(gbd_inc, no_hs_mean_inc_rti)
no_hs_scaled_inc = no_hs_mean_inc_rti * scale_for_no_hs
no_hs_scaled_inc_death = no_hs_mean_inc_death * scale_for_no_hs
no_hs_dalys = no_hs_yll.mean() + no_hs_yld.mean()
no_hs_scaled_dalys = np.multiply(list(no_hs_dalys), list(scale_for_no_hs))
no_hs_dalys_1 = []
no_hs_deaths = []
for logfile in os.listdir("outputs/blocked_interventions/no_hs"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/no_hs/" + logfile)
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
        no_hs_dalys_1.append(dalys_per_year['scaled_dalys'].sum())
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_hs_deaths.append(deaths['scaled'].sum())

no_suture_dalys = []
no_suture_deaths = []
for logfile in os.listdir("outputs/blocked_interventions/suture"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/suture/" + logfile)
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
        no_suture_dalys.append(dalys_per_year['scaled_dalys'].sum())
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_suture_deaths.append(deaths['scaled'].sum())

no_burn_dalys = []
no_burn_deaths = []
for logfile in os.listdir("outputs/blocked_interventions/burn"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/burn/" + logfile)
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
        no_burn_dalys.append(dalys_per_year['scaled_dalys'].sum())
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_burn_deaths.append(deaths['scaled'].sum())

no_open_dalys = []
no_open_deaths = []
for logfile in os.listdir("outputs/blocked_interventions/open_fracture"):
    if logfile.title().startswith('Log'):
        parsed = parse_log_file("outputs/blocked_interventions/open_fracture/" + logfile)
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
        no_open_dalys.append(dalys_per_year['scaled_dalys'].sum())
        rti_deaths = parsed['tlo.methods.demography']['death'].loc[
            parsed['tlo.methods.demography']['death']['label'] != 'Other']
        rti_deaths['rti_death_count'] = [1] * len(rti_deaths)
        rti_deaths['year'] = rti_deaths['date'].dt.year
        deaths = rti_deaths.groupby('year').sum()
        deaths['scaled'] = deaths['rti_death_count'] * scaling_df['scale_for_each_year']
        no_open_deaths.append(deaths['scaled'].sum())


mean_dalys_per_scenario = [batch_all_dalys, np.mean(no_cast_dalys) * scale_to_dalys,
                           np.mean(no_minor_dalys) * scale_to_dalys, np.mean(no_major_dalys) * scale_to_dalys,
                           np.mean(no_suture_dalys) * scale_to_dalys, np.mean(no_burn_dalys) * scale_to_dalys,
                           np.mean(no_open_dalys) * scale_to_dalys, np.mean(no_hs_dalys)]
scenarios = ['Normal', 'No\nfracture\ncasts', 'No\nminor\nsurgery', 'No\nmajor\nsurgery', 'No\nsuture',
             'No\nburn\ntreatment', 'No\nopen\nfracture\ntreatment', 'No\nhealth\nsystem']
plt.bar(np.arange(len(mean_dalys_per_scenario)), mean_dalys_per_scenario,
        color=['lightsteelblue', 'lightsalmon', 'thistle', 'peachpuff', 'slategrey', 'orchid', 'palegreen',
               'mediumpurple'])
plt.xticks(np.arange(len(mean_dalys_per_scenario)), scenarios, fontsize=8)
plt.ylabel('DALYs')
plt.title('The number of DALYs produced in simulations\n where certain treatments were blocked')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/BlockedInterventions/'
            'DALYs_blocked_treatments.png', bbox_inches='tight')
plt.clf()
additional_dalys_due_to_blocked_treatment = [i - mean_dalys_per_scenario[0] for i in mean_dalys_per_scenario[0: -1]]
cum_sum_of_dalys = np.cumsum(additional_dalys_due_to_blocked_treatment)
colors = ['lightsteelblue', 'lightsalmon', 'thistle', 'peachpuff', 'slategrey', 'orchid', 'palegreen',
          'mediumpurple']
scenarios = ['Normal', 'No fracture casts', 'No minor surgery', 'No major surgery', 'No suture',
             'No burn treatment', 'No open fracture treatment', 'No health system']
for i in range(0, len(cum_sum_of_dalys)):
    if i > 0:
        plt.bar([1], cum_sum_of_dalys[i], color=colors[i],
                bottom=mean_dalys_per_scenario[0] + cum_sum_of_dalys[i - 1], label=scenarios[i])
    else:
        plt.bar([1], mean_dalys_per_scenario[0] + cum_sum_of_dalys[i], color=colors[i], label=scenarios[i])
plt.bar([2], mean_dalys_per_scenario[-1], color=colors[-1], label='No health system')
plt.ylabel('DALYs')
plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
plt.xticks([1, 2], ['DALYs attributed\nto each\nintervention', 'No health\nsystem'])
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/BlockedInterventions/'
            'Daly_stacked.png', bbox_inches='tight')

mean_deaths_per_scenario = [np.mean(normal_run_deaths), np.mean(no_cast_deaths), np.mean(no_minor_deaths),
                            np.mean(no_major_deaths)]
plt.bar(np.arange(len(mean_deaths_per_scenario)), mean_deaths_per_scenario,
        color=['lightsteelblue', 'lightsalmon', 'thistle', 'peachpuff'])
plt.xticks(np.arange(len(mean_deaths_per_scenario)), scenarios)
plt.ylabel('Deaths')
plt.title('The number of deaths produced in simulations\n where certain treatments were blocked')
plt.savefig('C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/BlockedInterventions/'
            'Deaths_blocked_treatments.png', bbox_inches='tight')
