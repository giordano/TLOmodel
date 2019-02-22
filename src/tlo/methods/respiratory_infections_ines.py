"""
Respiratory Infection module
Documentation: 04 - Methods Repository/Method_Respiratory_Infection.xlsx
"""
import logging

import pandas as pd
from tlo import DateOffset, Module, Parameter, Property, Types
from tlo.events import PopulationScopeEventMixin, RegularEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class RespiratoryInfections(Module):

    PARAMETERS = {
        'base_incidence_rate_cold': Parameter(Types.REAL, 'probability per 1 week of incident cold at baseline'
                                                            ),
        'base_incidence_rate_pneumonia': Parameter(Types.REAL, 'probability per 1 week of incident pneumonia at baseline'
                                                   ),
        'base_incidence_rate_severe_pneumonia': Parameter(Types.REAL, 'probability per 1 week of incident pneumonia at baseline'
                                                   ),
        'init_prevalence_cold': Parameter(Types.REAL, 'prevalence of common cold at baseline for under 5s'
                                               ),
        'init_prevalence_pneumonia': Parameter(Types.REAL, 'prevalence of pneumonia at baseline for under 5s'
                                               ),
        'init_prevalence_severe_pneumonia': Parameter(Types.REAL, 'prevalence of severe pneumonia at baseline for under 5s'
                                               ),
        'ri_prob_progress_to_severe_pneumonia': Parameter(Types.REAL, 'probability of progressing from pneumonia to severe penumonia'
                                                          ),
        'rr_cold_age_under2months': Parameter(Types.REAL, 'relative rate of common cold for age  < 2 months'
                                              ),
        'rr_cold_age_2-11months': Parameter(Types.REAL, 'relative rate of common cold for age between 2 to 11 months'
                                              ),
        'rr_cold_age_12-23months': Parameter(Types.REAL, 'relative rate of common cold for age between 12 to 23 months'
                                            ),
        'rr_cold_age_24-59months': Parameter(Types.REAL, 'relative rate of common cold for age between 24 to 59 months'
                                            ),
        'rr_pneumonia_age_under2months': Parameter(Types.REAL, 'relative rate of pneumonia for age  < 2 months'
                                              ),
        'rr_pneumonia_age_2-11months': Parameter(Types.REAL, 'relative rate of pneumonia for age between 2 to 11 months'
                                            ),
        'rr_pneumonia_age_12-23months': Parameter(Types.REAL, 'relative rate of pneumonia for age between 12 to 23 months'
                                             ),
        'rr_pneumonia_age_24-59months': Parameter(Types.REAL, 'relative rate of pneumonia for age between 24 to 59 months'
                                             ),
        'rr_severe_pneumonia_age_under2months': Parameter(Types.REAL, 'relative rate of severe pneumonia for age <2 months'
                                              ),
        'rr_severe_pneumonia_age_2-11months': Parameter(Types.REAL, 'relative rate of severe pneumonia for age between 2 to 11 months'
                                            ),
        'rr_severe_pneumonia_age_12-23months': Parameter(Types.REAL, 'relative rate of severe pneumonia for age between 12 to 23 months'
                                             ),
        'rr_severe_pneumonia_age_24-59months': Parameter(Types.REAL,
                     'relative rate of severe pneumonia for age between 24 to 59 months'
                                             ),
        'base_death_rate_pneumonia': Parameter(Types.REAL, 'death rate from pneumonia/severe pneumonia at baseline'),
        'rr_cold_HIV': Parameter(Types.REAL, 'relative rate of common cold for HIV positive status'),
        'rr_pneumonia_HIV': Parameter(Types.REAL, 'relative rate of pneumonia for HIV positive status'),
        'rr_severe_pneumonia_HIV': Parameter(Types.REAL, 'relative rate of severe pneumonia for HIV positive status'),
        'rr_cold_wealth': Parameter(Types.REAL, 'relative rate of common cold for higher wealth level'),
        'rr_pneumonia_wealth': Parameter(Types.REAL, 'relative rate of pneumonia for higher wealth level'),
        'rr_severe_pneumonia_wealth': Parameter(Types.REAL, 'relative rate of severe pneumonia for higher wealth level'),
        'rr_cold_mother_education': Parameter(Types.REAL, 'relative rate of common cold if mother has higher education level'),
        'rr_pneumonia_mother_education': Parameter(Types.REAL, 'relative rate of pneumonia if mother has higher education level'),
        'rr_severe_pneumonia_mother_education': Parameter(Types.REAL, 'relative rate of severe pneumonia if mother has higher education level'),
        'rr_pneumonia_malnutrition': Parameter(Types.REAL,
                                                   'relative rate of pneumonia for acute malnutrition'),
        'rr_severe_pneumonia_malnutrition': Parameter(Types.REAL,
                                                   'relative rate of severe pneumonia for acute malnutrition'),
        'base_recovery_incidence_rate_cold': Parameter(Types.REAL, 'recovery rate from common cold at baseline'),
        'base_recovery_incidence_rate_pneumonia': Parameter(Types.REAL, 'recovery rate from pneumonia at baseline'),
        'base_recovery_incidence_rate_severe_pneumonia': Parameter(Types.REAL, 'recovery rate from severe pneumonia at baseline'),
        'rr_recovery_cold_age_2-11months': Parameter(Types.REAL, 'relative rate of recovery from common cold for age between 2 to 11 months'
                                            ),
        'rr_recovery_cold_age_12-23months': Parameter(Types.REAL, 'relative rate of recovery from common cold for age between 12 to 23 months'
                                             ),
        'rr_recovery_cold_age_24-59months': Parameter(Types.REAL, 'relative rate of recovery from common cold for age between 24 to 59 months'
                                             ),
        'rr_recovery_pneumonia_age_under2months': Parameter(Types.REAL, 'relative rate of recovery from pneumonia for age  < 2 months'
                                                   ),
        'rr_recovery_pneumonia_age_2-11months': Parameter(Types.REAL, 'relative rate of recovery from pneumonia for age between 2 to 11 months'
                                                 ),
        'rr_recovery_pneumonia_age_12-23months': Parameter(Types.REAL,
                                                  'relative rate of recovery from pneumonia for age between 12 to 23 months'
                                                  ),
        'rr_recovery_pneumonia_age_24-59months': Parameter(Types.REAL,
                                                  'relative rate of recovery from pneumonia for age between 24 to 59 months'
                                                  ),
        'rr_recovery_severe_pneumonia_age_under2months': Parameter(Types.REAL,
                                                          'relative rate of recovery from severe pneumonia for age <2 months'
                                                          ),
        'rr_recovery_severe_pneumonia_age_2-11months': Parameter(Types.REAL,
                                                        'relative rate of recovery from severe pneumonia for age between 2 to 11 months'
                                                        ),
        'rr_recovery_severe_pneumonia_age_12-23months': Parameter(Types.REAL,
                                                         'relative rate of recovery from severe pneumonia for age between 12 to 23 months'
                                                         ),
        'rr_recovery_severe_pneumonia_age_24-59months': Parameter(Types.REAL,
                                                         'relative rate of recovery from severe pneumonia for age between 24 to 59 months'
                                                         ),
        'rr_recovery_cold_HIV': Parameter(Types.REAL,
                                                      'relative rate of recovery from common cold for HIV positive status'
                                                      ),
        'rr_recovery_pneumonia_HIV': Parameter(Types.REAL,
                                          'relative rate of recovery from pneumonia for HIV positive status'
                                          ),
        'rr_recovery_severe_pneumonia_HIV': Parameter(Types.REAL,
                                          'relative rate of recovery from severe pneumonia for HIV positive status'
                                          ),
        'rr_recovery_cold_malnutrition': Parameter(Types.REAL,
                                          'relative rate of recovery from common cold for acute malnutrition'
                                          ),
        'rr_recovery_pneumonia_malnutrition': Parameter(Types.REAL,
                                                   'relative rate of recovery from pneumonia for acute malnutrition'
                                                   ),
        'rr_recovery_severe_pneumonia_malnutrition': Parameter(Types.REAL,
                                                   'relative rate of recovery from severe pneumonia for acute malnutrition'
                                                   ),
        'rr_recovery_pneumonia_treatment_adherence': Parameter(Types.REAL,
                                                   'relative rate of recovery from pneumonia if imcompleted treatment'
                                                   ),
        'rr_recovery_severe_pneumonia_treatment_adherence': Parameter(Types.REAL,
                                                               'relative rate of recovery from severe pneumonia if imcompleted treatment'
                                                               ),
        'rr_death_rate_pneumonia_treatment_adherence': Parameter(Types.REAL, 'death rate from pneumonia if completed treatment'),
        'rr_death_rate_pneumonia_HIV': Parameter(Types.REAL, 'death rate from pneumonia for HIV positive status'),
        'rr_death_rate_pneumonia_malnutrition': Parameter(Types.REAL, 'death rate from pneumonia for acute malnutrition'),
        'rr_death_pneumonia_age_under2months': Parameter(Types.REAL,
                                                            'relative rate of death from pneumonia for age  < 2 months'
                                                            ),
        'rr_death_pneumonia_age_2-11months': Parameter(Types.REAL,
                                                          'relative rate of death from pneumonia for age between 2 to 11 months'
                                                          ),
        'rr_death_pneumonia_age_12-23months': Parameter(Types.REAL,
                                                           'relative rate of death from pneumonia for age between 12 to 23 months'
                                                           ),
        'rr_death_pneumonia_age_24-59months': Parameter(Types.REAL,
                                                           'relative rate of death from pneumonia for age between 24 to 59 months'
                                                           )

    }

    # Next we declare the properties of individuals that this module provides.
    # Again each has a name, type and description. In addition, properties may be marked
    # as optional if they can be undefined for a given individual.
    PROPERTIES = {
        'ri_respiratory_infection_status': Property(Types.CATEGORICAL, 'respiratory infection status',
                                  categories=['none', 'common cold', 'pneumonia', 'severe pneumonia']),
        'ri_cough': Property(Types.BOOL, 'respiratory infection symptoms cough'),
        'ri_fever': Property(Types.BOOL, 'respiratory infection symptoms fever'),
        'ri_fast_breathing': Property(Types.BOOL, 'respiratory infection symptoms fast breathing'),
        'ri_chest_indraw': Property(Types.BOOL, 'respiratory infection symptoms chest indrawing'),
        'ri_stridor': Property(Types.BOOL, 'respiratory infection symptoms stridor'),
        'ri_not_able_drink_breastfeed': Property(Types.BOOL, 'resp infection symptoms not able to drink or breastfeed'),
        'ri_convulsions': Property(Types.BOOL, 'respiratory infection symptoms convulsions'),
        'ri_lethargic_unconscious': Property(Types.BOOL, 'respiratory infection symptoms lethargic or unconscious'),
        'ri_diagnosis': Property(Types.BOOL, 'respiratory infection diagnosis')
    }

    def read_parameters(self, data_folder):
        """Setup parameters used by the module
        """
        p = self.parameters
        p['base_incidence_rate_pneumonia'] = 0.001
        p['init_prevalence_pneumonia'] = 0.2
        p['rp_pneumonia_agelt1'] = 5
        p['rr_pneumonia_agelt1'] = 5

    def initialise_population(self, population):
        """Set our property values for the initial population.
        :param population: the population of individuals
        """
        df = population.props  # a shortcut to the data-frame storing data for individuals
        m = self
        rng = m.rng

        # -------------------- DEFAULTS ------------------------------------------------------------

        df['ri_respiratory_infection_status'] = 'none'

        # -------------------- ASSIGN VALUES OF RESPIRATORY INFECTION STATUS AT BASELINE -----------

        agelt5_idx = df.index[(df.age_years <= 5) & df.is_alive]
        agelt1_idx = df.index[(df.age_years <= 1) & df.is_alive]

        eff_prob_pneumonia = pd.Series(m.init_prevalence_pneumonia,
                                             index=df.index[(df.age_years <= 5) & df.is_alive])

        eff_prob_pneumonia.loc[agelt1_idx] *= m.rp_pneumonia_agelt1

        random_draw = pd.Series(rng.random_sample(size=len(agelt5_idx)),
                                index=df.index[(df.age_years <= 5) & df.is_alive])

        dfx = pd.concat([eff_prob_pneumonia, random_draw], axis=1)
        dfx.columns = ['eff_prob_pneumonia', 'random_draw']
        has_pneumonia_idx = dfx.index[dfx.eff_prob_pneumonia > dfx.random_draw]
        df.loc[has_pneumonia_idx, 'ri_respiratory_infection_status'] = 'pneumonia'

    def initialise_simulation(self, sim):
        """Add lifestyle events to the simulation
        """
        event = OesCancerEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=1))

        event = OesCancerLoggingEvent(self)
        sim.schedule_event(event, sim.date + DateOffset(weeks=0))

    def on_birth(self, mother_id, child_id):
        """Initialise properties for a newborn individual.
        :param mother_id: the mother for this child
        :param child_id: the new child
        """

        df = self.sim.population.props

        df.at[child_id, 'ca_oesophagus'] = 'none'
        df.at[child_id, 'ca_oesophagus_diagnosed'] = False
        df.at[child_id, 'ca_oesophagus_curative_treatment'] = 'never'
        df.at[child_id, 'ca_oesophageal_cancer_death'] = 'never'
        df.at[child_id, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False


class OesCancerEvent(RegularEvent, PopulationScopeEventMixin):
    """
    Regular event that updates all oesophagealcancer properties for population
    """
    def __init__(self, module):
        """schedule to run every 3 months
        note: if change this offset from 3 months need to consider code conditioning on age.years_exact
        :param module: the module that created this event
        """
        super().__init__(module, frequency=DateOffset(months=3))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        df = population.props
        m = self.module
        rng = m.rng

        # -------------------- UPDATING of CA-OESOPHAGUS OVER TIME -----------------------------------

        # updating for peopl aged over 20 with current status 'none'

        ca_oes_current_none_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20)]
        ca_oes_current_none_f_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           (df.sex == 'F')]
        ca_oes_current_none_tob_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           df.li_tob]
        ca_oes_current_none_ex_alc_idx = df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20) &
                                           df.li_ex_alc]

        eff_prob_low_grade_dysp = pd.Series(m.r_low_grade_dysplasia_none,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'none') & (df.age_years >= 20)])

        eff_prob_low_grade_dysp.loc[ca_oes_current_none_f_idx] *= m.rr_low_grade_dysplasia_none_female
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_tob_idx] *= m.rr_low_grade_dysplasia_none_tobacco
        eff_prob_low_grade_dysp.loc[ca_oes_current_none_ex_alc_idx] *= m.rr_low_grade_dysplasia_none_ex_alc

        p_oes_dys_can_age_muliplier = pd.Series(m.rr_low_grade_dysplasia_none_per_year_older ** (df.age_years - 20),
                                                index=ca_oes_current_none_idx)

        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_none_idx)),
                                   index=df.index[(df.age_years >= 20) & df.is_alive & (df.ca_oesophagus == 'none')])

        dfx = pd.concat([eff_prob_low_grade_dysp, p_oes_dys_can_age_muliplier, random_draw], axis=1)
        dfx.columns = ['eff_prob_low_grade_dysp', 'p_oes_dys_can_age_muliplier', 'random_draw']
        dfx.eff_prob_low_grade_dysp *= p_oes_dys_can_age_muliplier
        idx_incident_low_grade_dysp = dfx.index[dfx.eff_prob_low_grade_dysp > dfx.random_draw]
        df.loc[idx_incident_low_grade_dysp, 'ca_oesophagus'] = 'low_grade_dysplasia'

        # updating for people aged over 20 with current status 'low grade dysplasia'

        ca_oes_current_low_grade_dysp_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                     (df.age_years >= 20)]
        ca_oes_current_low_grade_dysp_treated_idx = df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                     (df.age_years >= 20)
                                                     & (df.ca_oesophagus_curative_treatment == 'low_grade_dysplasia')]
        eff_prob_high_grade_dysp = pd.Series(m.r_high_grade_dysplasia_low_grade_dysp,
                                             index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia')
                                                            & (df.age_years >= 20)])
        eff_prob_high_grade_dysp.loc[ca_oes_current_low_grade_dysp_treated_idx] \
            *= m.rr_high_grade_dysp_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_low_grade_dysp_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'low_grade_dysplasia')])
        dfx = pd.concat([eff_prob_high_grade_dysp, random_draw], axis=1)
        dfx.columns = ['eff_prob_high_grade_dysp', 'random_draw']
        idx_incident_high_grade_dysp = dfx.index[dfx.eff_prob_high_grade_dysp > dfx.random_draw]
        df.loc[idx_incident_high_grade_dysp, 'ca_oesophagus'] = 'high_grade_dysplasia'

        # updating for people aged over 20 with current status 'high grade dysplasia'

        ca_oes_current_high_grade_dysp_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                     (df.age_years >= 20)]
        ca_oes_current_high_grade_dysp_treated_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                     (df.age_years >= 20)
                                                     & (df.ca_oesophagus_curative_treatment == 'high_grade_dysplasia')]
        eff_prob_stage1 = pd.Series(m.r_stage1_high_grade_dysp,
                                             index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia')
                                                            & (df.age_years >= 20)])
        eff_prob_stage1.loc[ca_oes_current_high_grade_dysp_treated_idx] \
            *= m.rr_stage1_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_high_grade_dysp_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'high_grade_dysplasia')])
        dfx = pd.concat([eff_prob_stage1, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage1', 'random_draw']
        idx_incident_stage1 = dfx.index[dfx.eff_prob_stage1 > dfx.random_draw]
        df.loc[idx_incident_stage1, 'ca_oesophagus'] = 'stage1'

        # updating for people aged over 20 with current status stage 1 oes cancer

        ca_oes_current_stage1_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage1_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage1') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage1')]
        eff_prob_stage2 = pd.Series(m.r_stage2_stage1,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1')
                                                   & (df.age_years >= 20)])
        eff_prob_stage2.loc[ca_oes_current_stage1_treated_idx] \
            *= m.rr_stage2_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage1_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage1')])
        dfx = pd.concat([eff_prob_stage2, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage2', 'random_draw']
        idx_incident_stage2 = dfx.index[dfx.eff_prob_stage2 > dfx.random_draw]
        df.loc[idx_incident_stage2, 'ca_oesophagus'] = 'stage2'

        # updating for people aged over 20 with current status stage 2 oes cancer

        ca_oes_current_stage2_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage2_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage2') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage2')]
        eff_prob_stage3 = pd.Series(m.r_stage3_stage2,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2')
                                                   & (df.age_years >= 20)])
        eff_prob_stage3.loc[ca_oes_current_stage2_treated_idx] \
            *= m.rr_stage3_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage2_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage2')])
        dfx = pd.concat([eff_prob_stage3, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage3', 'random_draw']
        idx_incident_stage3 = dfx.index[dfx.eff_prob_stage3 > dfx.random_draw]
        df.loc[idx_incident_stage3, 'ca_oesophagus'] = 'stage3'

        # updating for people aged over 20 with current status stage 3 oes cancer

        ca_oes_current_stage3_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                             (df.age_years >= 20)]
        ca_oes_current_stage3_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage3') &
            (df.age_years >= 20)
            & (df.ca_oesophagus_curative_treatment == 'stage3')]
        eff_prob_stage4 = pd.Series(m.r_stage4_stage3,
                                    index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3')
                                                   & (df.age_years >= 20)])
        eff_prob_stage4.loc[ca_oes_current_stage3_treated_idx] \
            *= m.rr_stage4_undergone_curative_treatment
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_current_stage3_idx)),
                                index=df.index[(df.age_years >= 20) & df.is_alive &
                                               (df.ca_oesophagus == 'stage3')])
        dfx = pd.concat([eff_prob_stage4, random_draw], axis=1)
        dfx.columns = ['eff_prob_stage4', 'random_draw']
        idx_incident_stage4 = dfx.index[dfx.eff_prob_stage4 > dfx.random_draw]
        df.loc[idx_incident_stage4, 'ca_oesophagus'] = 'stage4'

        # -------------------- UPDATING OF CA_OESOPHAGUS DIAGNOSED OVER TIME --------------------------------

        df['ca_incident_oes_cancer_diagnosis_this_3_month_period'] = False

        # update diagnosis status for undiagnosed people with low grade dysplasia

        ca_oes_current_low_grade_dysp_not_diag_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_low_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_low_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with high grade dysplasia

        ca_oes_current_high_grade_dysp_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_high_grade_dysp,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_high_grade_dysp_not_diag_idx))
        df.loc[ca_oes_current_high_grade_dysp_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_high_grade_dysp_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 1 oes cancer

        ca_oes_current_stage1_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                      (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp * m.rr_diagnosis_stage1,
                                  index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                 (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage1_not_diag_idx))
        df.loc[ca_oes_current_stage1_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage1_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 2 oes cancer

        ca_oes_current_stage2_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_stage2,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage2_not_diag_idx))
        df.loc[ca_oes_current_stage2_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage2_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 3 oes cancer

        ca_oes_current_stage3_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_stage3,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage3_not_diag_idx))
        df.loc[ca_oes_current_stage3_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage3_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # update diagnosis status for undiagnosed people with stage 4 oes cancer

        ca_oes_current_stage4_not_diag_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4') &
                                                          (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed]
        eff_prob_diag = pd.Series(m.r_diagnosis_low_grade_dysp*m.rr_diagnosis_stage4,
                              index=df.index[df.is_alive & (df.ca_oesophagus == 'stage4') &
                                             (df.age_years >= 20) & ~df.ca_oesophagus_diagnosed])
        random_draw = rng.random_sample(size=len(ca_oes_current_stage4_not_diag_idx))
        df.loc[ca_oes_current_stage4_not_diag_idx, 'ca_oesophagus_diagnosed'] = (random_draw < eff_prob_diag)
        df.loc[ca_oes_current_stage4_not_diag_idx, 'ca_incident_oes_cancer_diagnosis_this_3_month_period'] \
            = (random_draw < eff_prob_diag)

        # -------------------- UPDATING VALUES OF CA_OESOPHAGUS_CURATIVE_TREATMENT -------------------

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with low grade dysplasia w

        ca_oes_diag_low_grade_dysp_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                              df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_low_grade_dysp_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'low_grade_dysplasia') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'low_grade_dysplasia'

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with high grade dysplasia w

        ca_oes_diag_high_grade_dysp_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_high_grade_dysp,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                              df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_high_grade_dysp_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'high_grade_dysplasia') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'high_grade_dysplasia'

        # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 1

        ca_oes_diag_stage1_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage1') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp * m.rr_curative_treatment_stage1,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never')])
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage1_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage1') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                   df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'stage1'

       # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 2

        ca_oes_diag_stage2_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage2') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_stage2,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never') ] )
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage2_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage2') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'stage2'

       # update ca_oesophagus_curative_treatment for diagnosed, untreated people with stage 3

        ca_oes_diag_stage3_not_treated_idx = df.index[
            df.is_alive & (df.ca_oesophagus == 'stage3') &
            (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (df.ca_oesophagus_curative_treatment == 'never')]
        eff_prob_treatment = pd.Series(m.r_curative_treatment_low_grade_dysp*m.rr_curative_treatment_stage3,
                                       index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                                      (df.age_years >= 20) & df.ca_oesophagus_diagnosed &
                                                      (df.ca_oesophagus_curative_treatment == 'never') ] )
        random_draw = pd.Series(rng.random_sample(size=len(ca_oes_diag_stage3_not_treated_idx)),
                                index=df.index[df.is_alive & (df.ca_oesophagus == 'stage3') &
                                               (df.age_years >= 20) & df.ca_oesophagus_diagnosed & (
                                                       df.ca_oesophagus_curative_treatment == 'never')])
        dfx = pd.concat([eff_prob_treatment, random_draw], axis=1)
        dfx.columns = ['eff_prob_treatment', 'random_draw']
        idx_incident_treatment = dfx.index[dfx.eff_prob_treatment > dfx.random_draw]
        df.loc[idx_incident_treatment, 'ca_oesophagus_curative_treatment'] = 'stage3'

        # -------------------- DEATH FROM OESOPHAGEAL CANCER ---------------------------------------

        stage4_idx = df.index[df.is_alive & (df.ca_oesophagus == 'stage4')]
        random_draw = m.rng.random_sample(size=len(stage4_idx))
        df.loc[stage4_idx, 'ca_oesophageal_cancer_death'] = (random_draw < m.r_death_oesoph_cancer)

        # todo - this code dealth with centrally
        dead_oes_can_idx = df.index[df.ca_oesophageal_cancer_death]
        df.loc[dead_oes_can_idx, 'is_alive'] = False


class OesCancerLoggingEvent(RegularEvent, PopulationScopeEventMixin):
    """Handles lifestyle logging"""
    def __init__(self, module):
        """schedule logging to repeat every 3 months
        """
        self.repeat = 3
        super().__init__(module, frequency=DateOffset(months=self.repeat))

    def apply(self, population):
        """Apply this event to the population.
        :param population: the current population
        """
        # get some summary statistics
        df = population.props

        # calculate incidence of oesophageal cancer diagnosis in people aged > 60+
        # (this includes people diagnosed with dysplasia, but diagnosis rate at this stage is very low)

        incident_oes_cancer_diagnosis_agege60_idx = df.index[df.ca_incident_oes_cancer_diagnosis_this_3_month_period
        & (df.age_years >= 60)]
        agege60_without_diagnosed_oes_cancer_idx = df.index[(df.age_years >= 60) & ~df.ca_oesophagus_diagnosed]

        incidence_per_year_oes_cancer_diagnosis = (4 * 100000 * len(incident_oes_cancer_diagnosis_agege60_idx))/\
                                                  len(agege60_without_diagnosed_oes_cancer_idx)

        incidence_per_year_oes_cancer_diagnosis = round(incidence_per_year_oes_cancer_diagnosis, 3)

 #      logger.debug('%s|person_one|%s',
 #                     self.sim.date,
 #                     df.loc[0].to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['ca_oesophagus']).size().to_dict())

        # note below remove is_alive
#       logger.info('%s|ca_oesophagus_death|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophageal_cancer_death']).size().to_dict())


        logger.info('%s|ca_incident_oes_cancer_diagnosis_this_3_month_period|%s',
                    self.sim.date,
                    incidence_per_year_oes_cancer_diagnosis)


#       logger.info('%s|ca_oesophagus_diagnosed|%s',
#                   self.sim.date,
#                   df[df.age_years >= 20].groupby(['ca_oesophagus', 'ca_oesophagus_diagnosed']).size().to_dict())

#       logger.info('%s|ca_oesophagus|%s',
#                   self.sim.date,
#                   df[df.is_alive].groupby(['age_range', 'ca_oesophagus']).size().to_dict())
