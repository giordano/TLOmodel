"""
Deviance measure

This module runs at the end of a simulation and calculates a weighted deviance measure
for a given set of parameters using outputs from the demography (deaths), HIV and TB modules

"""
from pathlib import Path
import math
import numpy as np
import pandas as pd

from tlo import Date, Module, Parameter, Types


class HealthSeekingBehaviour(Module):
    """
    This modules reads in logged outputs from HIV, TB and demography and compares them with reported data
    a deviance measure is calculated and returned on simulation end
    """

    INIT_DEPENDENCIES = {'Demography', 'Hiv', 'Tb'}
    ADDITIONAL_DEPENDENCIES = {}

    # Declare Metadata
    METADATA = {}

    # No parameters to declare
    PARAMETERS = {}

    # No properties to declare
    PROPERTIES = {}

    def __init__(self, name=None, resourcefilepath=None, force_any_symptom_to_lead_to_healthcareseeking=False):
        super().__init__(name)
        self.resourcefilepath = resourcefilepath

        self.data_dict = dict()
        self.model_dict = dict()

    def read_parameters(self, data_folder):
        """Read in ResourceFile"""
        # Load parameters from resource file:
        # todo could read in params here instead of read_data_files
        # todo create data_dict / model_dict in def init??

    def initialise_population(self, population):
        """Nothing to initialise in the population
        """
        pass

    def initialise_simulation(self, sim):
        """ nothing to initialise
        """
        pass

    def on_birth(self, mother_id, child_id):
        """Nothing to handle on_birth
        """
        pass

    def read_data_files(self, resourcefilepath):
        # make a dict of all data to be used in calculating calibration score
        self.data_dict = {}

        # HIV read in resource files for data
        xls = pd.ExcelFile(resourcefilepath / "ResourceFile_HIV.xlsx")

        # MPHIA HIV data - age-structured
        data_hiv_mphia_inc = pd.read_excel(xls, sheet_name="MPHIA_incidence2015")
        data_hiv_mphia_prev = pd.read_excel(xls, sheet_name="MPHIA_prevalence_art2015")

        self.data_dict["mphia_inc_2015"] = data_hiv_mphia_inc.loc[
            (data_hiv_mphia_inc.age == "15-49"), "total_percent_annual_incidence"
        ].values[
            0
        ]  # inc
        self.data_dict["mphia_prev_2015"] = data_hiv_mphia_prev.loc[
            data_hiv_mphia_prev.age == "Total 15-49", "total percent hiv positive"
        ].values[
            0
        ]  # prev

        # DHS HIV data
        data_hiv_dhs_prev = pd.read_excel(xls, sheet_name="DHS_prevalence")
        self.data_dict["dhs_prev_2010"] = data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year == 2010), "HIV prevalence among general population 15-49"
        ].values[0]
        self.data_dict["dhs_prev_2015"] = data_hiv_dhs_prev.loc[
            (data_hiv_dhs_prev.Year == 2015), "HIV prevalence among general population 15-49"
        ].values[0]

        # UNAIDS AIDS deaths data: 2010-
        data_hiv_unaids_deaths = pd.read_excel(xls, sheet_name="unaids_mortality_dalys2021")
        self.data_dict["unaids_deaths_per_100k"] = data_hiv_unaids_deaths["AIDS_mortality_per_100k"]

        # TB
        # TB WHO data: 2010-
        xls_tb = pd.ExcelFile(resourcefilepath / "ResourceFile_TB.xlsx")

        # TB active incidence per 100k 2010-2017
        data_tb_who = pd.read_excel(xls_tb, sheet_name="WHO_activeTB2020")
        self.data_dict["who_tb_inc_per_100k"] = data_tb_who.loc[
            (data_tb_who.year >= 2010), "incidence_per_100k"
        ]

        # TB latent data (Houben & Dodd 2016)
        data_tb_latent = pd.read_excel(xls_tb, sheet_name="latent_TB2014_summary")
        data_tb_latent_all_ages = data_tb_latent.loc[data_tb_latent.Age_group == "0_80"]
        self.data_dict["who_tb_latent_prev"] = data_tb_latent_all_ages.proportion_latent_TB.values[0]

        # TB case notification rate NTP: 2012-2019
        data_tb_ntp = pd.read_excel(xls_tb, sheet_name="NTP2019")
        self.data_dict["ntp_case_notification_per_100k"] = data_tb_ntp.loc[
            (data_tb_ntp.year >= 2012), "case_notification_rate_per_100k"
        ]

        # TB mortality per 100k excluding HIV: 2010-2017
        self.data_dict["who_tb_deaths_per_100k"] = data_tb_who.loc[
            (data_tb_who.year >= 2010), "mortality_tb_excl_hiv_per_100k"
        ]

    def read_model_outputs(self):

        hiv = self.sim.modules['Hiv'].hiv_outputs
        tb = self.sim.modules['Tb'].tb_outputs
        demog = self.sim.modules['Demography'].demog_outputs

        # get logged outputs for calibration into dict
        self.model_dict = {}

        # these are the outputs
        # self.module.hiv_outputs["hiv_prev_adult_1549"].append(adult_prev_1549)
        # self.module.hiv_outputs["hiv_adult_inc_1549"].append(adult_inc_1549)
        # self.module.hiv_outputs["hiv_prev_child"].append(child_prev)

        # tb ["num_new_active_tb","tbPrevLatent"]

        # self.demog_outputs["death"]

        # HIV - prevalence among in adults aged 15-49
        model_hiv_prev = hiv["summary_inc_and_prev_for_adults_and_children_and_fsw"]
        self.model_dict["hiv_prev_adult_2010"] = (
                                                model_hiv_prev.loc[
                                                    (model_hiv_prev.date == "2011-01-01"), "hiv_prev_adult_1549"
                                                ].values[0]
                                            ) * 100
        self.model_dict["hiv_prev_adult_2015"] = (
                                                model_hiv_prev.loc[
                                                    (model_hiv_prev.date == "2015-01-01"), "hiv_prev_adult_1549"
                                                ].values[0]
                                            ) * 100

        # hiv incidence in adults aged 15-49
        self.model_dict["hiv_inc_adult_2015"] = (
                                               model_hiv_prev.loc[
                                                   (model_hiv_prev.date == "2015-01-01"), "hiv_adult_inc_1549"
                                               ].values[0]
                                           ) * 100

        # aids deaths
        # deaths
        deaths = demog["death"].copy()  # outputs individual deaths
        deaths = deaths.set_index("date")

        # AIDS DEATHS
        # person-years all ages (irrespective of HIV status)
        py_ = output["tlo.methods.demography"]["person_years"]
        years = pd.to_datetime(py_["date"]).dt.year
        py = pd.Series(dtype="int64", index=years)
        for year in years:
            tot_py = (
                (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["M"]).apply(pd.Series)
                + (py_.loc[pd.to_datetime(py_["date"]).dt.year == year]["F"]).apply(pd.Series)
            ).transpose()
            py[year] = tot_py.sum().values[0]

        py.index = pd.to_datetime(years, format="%Y")

        # limit to deaths among aged 15+, include HIV/TB deaths
        keep = (deaths.age >= 15) & (
            (deaths.cause == "AIDS_TB") | (deaths.cause == "AIDS_non_TB")
        )
        deaths_AIDS = deaths.loc[keep].copy()
        deaths_AIDS["year"] = deaths_AIDS.index.year
        tot_aids_deaths = deaths_AIDS.groupby(by=["year"]).size()
        tot_aids_deaths.index = pd.to_datetime(tot_aids_deaths.index, format="%Y")

        # aids mortality rates per 1000 person-years
        self.model_dict["AIDS_mortality_per_100k"] = (tot_aids_deaths / py) * 100000

        # tb active incidence per 100k - all ages
        TB_inc = output["tlo.methods.tb"]["tb_incidence"]
        TB_inc = TB_inc.set_index("date")
        TB_inc.index = pd.to_datetime(TB_inc.index)
        self.model_dict["TB_active_inc_per100k"] = (TB_inc["num_new_active_tb"] / py) * 100000

        # tb latent prevalence
        latentTB_prev = output["tlo.methods.tb"]["tb_prevalence"]
        self.model_dict["TB_latent_prev"] = latentTB_prev.loc[
            (latentTB_prev.date == "2014-01-01"), "tbPrevLatent"].values[0]

        # tb case notifications
        tb_notifications = output["tlo.methods.tb"]["tb_treatment"]
        tb_notifications = tb_notifications.set_index("date")
        tb_notifications.index = pd.to_datetime(tb_notifications.index)
        self.model_dict["TB_case_notifications_per100k"] = (
                                                          tb_notifications["tbNewDiagnosis"] / py
                                                      ) * 100000

        # tb deaths (non-hiv only)
        keep = deaths.cause == "TB"
        deaths_TB = deaths.loc[keep].copy()
        deaths_TB["year"] = deaths_TB.index.year  # count by year
        tot_tb_non_hiv_deaths = deaths_TB.groupby(by=["year"]).size()
        tot_tb_non_hiv_deaths.index = pd.to_datetime(tot_tb_non_hiv_deaths.index, format="%Y")
        # tb mortality rates per 100k person-years
        self.model_dict["TB_mortality_per_100k"] = (tot_tb_non_hiv_deaths / py) * 100000

    def weighted_mean(self, model_dict, data_dict):
        # assert model_output is not empty

        # return calibration score (weighted mean deviance)
        # sqrt( (observed data – model output)^2 | / observed data)
        # sum these for each data item (all male prevalence over time) and divide by # items
        # then weighted sum of all components -> calibration score

        # need weights for each data item
        model_weight = 0.5

        def deviance_function(data, model):
            deviance = math.sqrt((data - model) ** 2) / data

            return deviance

        # hiv prevalence in adults 15-49: dhs 2010, 2015
        hiv_prev_dhs = (
                           deviance_function(data_dict["dhs_prev_2010"], model_dict["hiv_prev_adult_2010"])
                           + deviance_function(
                           data_dict["dhs_prev_2015"], model_dict["hiv_prev_adult_2015"]
                       )
                       ) / 2

        # hiv prevalence mphia
        hiv_prev_mphia = deviance_function(
            data_dict["mphia_prev_2015"], model_dict["hiv_prev_adult_2015"]
        )

        # hiv incidence mphia
        hiv_inc_mphia = deviance_function(
            data_dict["mphia_inc_2015"], model_dict["hiv_inc_adult_2015"]
        )

        # aids deaths unaids 2010-2019
        hiv_deaths_unaids = (
                                deviance_function(
                                    data_dict["unaids_deaths_per_100k"][0],
                                    model_dict["AIDS_mortality_per_100k"][0],
                                )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][1],
                                model_dict["AIDS_mortality_per_100k"][1],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][2],
                                model_dict["AIDS_mortality_per_100k"][2],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][3],
                                model_dict["AIDS_mortality_per_100k"][3],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][4],
                                model_dict["AIDS_mortality_per_100k"][4],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][5],
                                model_dict["AIDS_mortality_per_100k"][5],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][6],
                                model_dict["AIDS_mortality_per_100k"][6],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][7],
                                model_dict["AIDS_mortality_per_100k"][7],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][8],
                                model_dict["AIDS_mortality_per_100k"][8],
                            )
                                + deviance_function(
                                data_dict["unaids_deaths_per_100k"][9],
                                model_dict["AIDS_mortality_per_100k"][9],
                            )
                            ) / 10

        # tb active incidence (WHO estimates) 2010 -2017
        tb_incidence_who = (
                               # deviance_function(
                               #     data_dict["who_tb_inc_per_100k"].values[0], model_dict["TB_active_inc_per100k"][0]
                               # )
                               # +
                               deviance_function(
                                   data_dict["who_tb_inc_per_100k"].values[1], model_dict["TB_active_inc_per100k"][1]
                               )
                               + deviance_function(
                               data_dict["who_tb_inc_per_100k"].values[2], model_dict["TB_active_inc_per100k"][2]
                           )
                               + deviance_function(
                               data_dict["who_tb_inc_per_100k"].values[3], model_dict["TB_active_inc_per100k"][3]
                           )
                               + deviance_function(
                               data_dict["who_tb_inc_per_100k"].values[4], model_dict["TB_active_inc_per100k"][4]
                           )
                               + deviance_function(
                               data_dict["who_tb_inc_per_100k"].values[5], model_dict["TB_active_inc_per100k"][5]
                           )
                               + deviance_function(
                               data_dict["who_tb_inc_per_100k"].values[6], model_dict["TB_active_inc_per100k"][6]
                           )
                               + deviance_function(
                               data_dict["who_tb_inc_per_100k"].values[7], model_dict["TB_active_inc_per100k"][7]
                           )
                           ) / 8

        # tb latent prevalence
        tb_latent_prev = deviance_function(
            data_dict["who_tb_latent_prev"], model_dict["TB_latent_prev"]
        )

        # tb case notification rate per 100k: 2012-2019
        tb_cnr_ntp = (
                         deviance_function(
                             data_dict["ntp_case_notification_per_100k"].values[0],
                             model_dict["TB_case_notifications_per100k"][2],
                         )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[1],
                         model_dict["TB_case_notifications_per100k"][3],
                     )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[2],
                         model_dict["TB_case_notifications_per100k"][4],
                     )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[3],
                         model_dict["TB_case_notifications_per100k"][5],
                     )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[4],
                         model_dict["TB_case_notifications_per100k"][6],
                     )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[5],
                         model_dict["TB_case_notifications_per100k"][7],
                     )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[6],
                         model_dict["TB_case_notifications_per100k"][8],
                     )
                         + deviance_function(
                         data_dict["ntp_case_notification_per_100k"].values[7],
                         model_dict["TB_case_notifications_per100k"][9],
                     )
                     ) / 8

        # tb death rate who 2010-2017
        tb_mortality_who = (
                               deviance_function(
                                   data_dict["who_tb_deaths_per_100k"].values[0],
                                   model_dict["TB_mortality_per_100k"][0],
                               )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[1],
                               model_dict["TB_mortality_per_100k"][1],
                           )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[2],
                               model_dict["TB_mortality_per_100k"][2],
                           )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[3],
                               model_dict["TB_mortality_per_100k"][3],
                           )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[4],
                               model_dict["TB_mortality_per_100k"][4],
                           )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[5],
                               model_dict["TB_mortality_per_100k"][5],
                           )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[6],
                               model_dict["TB_mortality_per_100k"][6],
                           )
                               + deviance_function(
                               data_dict["who_tb_deaths_per_100k"].values[7],
                               model_dict["TB_mortality_per_100k"][7],
                           )
                           ) / 8

        calibration_score = (
            hiv_prev_dhs
            + hiv_prev_mphia
            + hiv_inc_mphia
            + (hiv_deaths_unaids * model_weight)
            + tb_incidence_who
            + tb_latent_prev
            + tb_cnr_ntp
            + tb_mortality_who
        )

        transmission_rates = [None, None]  # store hiv and tb transmission rates

        return_values = {calibration_score, transmission_rates}

        return return_values

    def on_simulation_end(self):

        self.read_model_outputs()
        self.read_data_files()
        deviance_measure = self.weighted_mean(model_dict=self.model_dict, data_dict=self.data_dict)

        # todo logging deviance measure
