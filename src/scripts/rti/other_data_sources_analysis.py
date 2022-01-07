import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# ====================== look at DHS data ==============================================================================
data = pd.read_stata("C:/Users/Robbie Manning Smith/Desktop/ihs data/DHS data/2017/MWPR7IDT/MWPR7IFL.DTA",
                     convert_categoricals=False)

# ===================== Look at the data from other countries in the GBD study =========================================
data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/gbddata/all_countries_inc_data.csv")
death_inc_data = data.loc[data['measure'] == 'Deaths']
death_inc_data = death_inc_data.groupby('location').mean()
death_inc_data = death_inc_data.sort_values('val', ascending=True)
colors = ['lightsteelblue'] * len(death_inc_data)
malawi_ranking = np.where(death_inc_data.index == 'Malawi')[0][0]
colors[malawi_ranking] = 'gold'
plt.bar(np.arange(len(death_inc_data)), death_inc_data['val'], color=colors)
plt.ylabel('Incidence of death per 100,000')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.xlabel('Countries in GBD study')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/gbd_all_countries_inc_death.png",
            bbox_inches='tight')
plt.clf()
inc_data = data.loc[data['measure'] == 'Incidence']
inc_data = inc_data.groupby('location').mean()
inc_data = inc_data.sort_values('val', ascending=True)
colors = ['lightsteelblue'] * len(inc_data)
malawi_ranking = np.where(inc_data.index == 'Malawi')[0][0]
colors[malawi_ranking] = 'gold'
plt.bar(np.arange(len(inc_data)), inc_data['val'], color=colors)
plt.ylabel('Incidence of RTI per 100,000')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.xlabel('Countries in GBD study')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/gbd_all_countries_inc_rti.png",
            bbox_inches='tight')
plt.clf()
inc_data = inc_data.reindex(death_inc_data.index)
inc_data['cfr'] = np.divide(death_inc_data.val, inc_data.val)
inc_data = inc_data.sort_values('cfr', ascending=True)
colors = ['lightsteelblue'] * len(inc_data)
malawi_ranking = np.where(inc_data.index == 'Malawi')[0][0]
colors[malawi_ranking] = 'gold'
plt.bar(np.arange(len(inc_data)), inc_data['cfr'], color=colors)
plt.ylabel('% Fatal')
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
plt.xlabel('Countries in GBD study')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/gbd_all_countries_rti_cfr.png",
            bbox_inches='tight')
plt.clf()
# ======================= Create an estimate of the incidence of RTI from the world health survey 2003 =================

whs_pop_size = 5297
n_injury = 5297 * 0.035
study_length_in_years = 1
average_years_until_injury = 0.5
incidence_of_RTI_per_person_year = \
    n_injury / (whs_pop_size * study_length_in_years - average_years_until_injury * n_injury)
whs_incidence_of_RTI_per_100000_person_years = incidence_of_RTI_per_person_year * 100000
# ========================== plot the estimates for the incidence of RTI in Malawi =====================================
GBD_est_inc = 954.2
inc_data = [whs_incidence_of_RTI_per_100000_person_years, GBD_est_inc]
inc_xlabels = ['World\nHealth\nSurvey\n2003', 'GBD\nStudy']
plt.bar(np.arange(len(inc_data)), inc_data, color=['lightsteelblue', 'lightsalmon'])
plt.xticks(np.arange(len(inc_data)), inc_xlabels)
plt.ylabel('Incidence per 100,000 p.y.')
plt.title('Incidence of RTI')
plt.savefig("C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/Incidence_of_RTI_other_est.png",
            bbox_inches='tight')
plt.clf()

# ======================= Create an estimate of the incidence of death from the household survey 2016 ==================
# get the IHS household info data
household_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/ihs data/IHS data/2016/household/hh_mod_b.csv")
# create a variable to count the number of people in the household
household_data['count'] = [1] * len(household_data)
# group by household to get number of people in house
household_counts = household_data.groupby('HHID').sum()
# calculate mean number of people in house
mean_n_in_house = household_counts['count'].mean()
# get IHS death data
death_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/ihs data/IHS data/2016/household/hh_mod_w.csv")

# create a variable to count the number of RTI deaths in the household
death_data['count'] = death_data['hh_w10'] == 1
# group by household
death_counts = death_data.groupby('HHID').sum()
# set the index for death counts to be the same as household_counts
death_counts = death_counts.reindex(household_counts.index)
# get the number of people per household into death_counts
death_counts['n_in_household'] = household_counts['count']
# calculate person years in study (assume that those who died did so after 1 year)
death_counts['person_years'] = np.multiply(death_counts['n_in_household'], 2) - death_counts['count']

total_rti_deaths = death_data['count'].sum()
total_person_years = death_counts['person_years'].sum()
incidence_per_person_year = total_rti_deaths / total_person_years
ihs_2016_incidence_per_100000_person_years = incidence_per_person_year * 100000
# ======================= Create an estimate of the incidence of death from the household survey 2019 ==================
# get the IHS household info data
household_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/ihs data/IHS data/2019/HH_MOD_B.csv")
# create a variable to count the number of people in the household
household_data['count'] = [1] * len(household_data)
# group by household to get number of people in house
household_counts = household_data.groupby('HHID').sum()
# calculate mean number of people in house
mean_n_in_house = household_counts['count'].mean()
# get IHS death data
death_data = pd.read_csv("C:/Users/Robbie Manning Smith/Desktop/ihs data/IHS data/2019/HH_MOD_W.csv")

# create a variable to count the number of RTI deaths in the household
death_data['count'] = death_data['hh_w10'] == 'TRAFFIC ACCIDENT'
# group by household
death_counts = death_data.groupby('HHID').sum()
# set the index for death counts to be the same as household_counts
death_counts = death_counts.reindex(household_counts.index)
# get the number of people per household into death_counts
death_counts['n_in_household'] = household_counts['count']
# calculate person years in study (assume that those who died did so after 1 year)
death_counts['person_years'] = np.multiply(death_counts['n_in_household'], 2) - death_counts['count']

total_rti_deaths = death_data['count'].sum()
total_person_years = death_counts['person_years'].sum()
incidence_per_person_year = total_rti_deaths / total_person_years
ihs_2019_incidence_per_100000_person_years = incidence_per_person_year * 100000
# ============================ Plot the estimates for the incidence of death due to RTI ================================
samuel_est_inc_death = 20.9
who_est_inc_death = 35
GBD_est_inc_death = 12.1
inc_death_data = [GBD_est_inc_death, ihs_2016_incidence_per_100000_person_years,
                  ihs_2019_incidence_per_100000_person_years, samuel_est_inc_death, who_est_inc_death]
inc_death_xlabels = ['GBD\nStudy', 'IHS\n2016', 'IHS\n2019', 'Samuel et al.\n 2012', 'WHO']
plt.bar(np.arange(len(inc_death_data)), inc_death_data,
        color=['lightsteelblue', 'lightsalmon', 'steelblue', 'lemonchiffon', 'peachpuff'])
plt.xticks(np.arange(len(inc_death_data)), inc_death_xlabels)
plt.ylabel('Incidence per 100,000 p.y.')
plt.title('Incidence of RTI death')
plt.savefig(
    "C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/FinalPaperOutput/Incidence_of_RTI_death_other_est.png",
    bbox_inches='tight'
)
plt.clf()
