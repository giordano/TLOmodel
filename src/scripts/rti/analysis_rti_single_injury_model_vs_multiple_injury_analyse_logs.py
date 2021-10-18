from scripts.rti.rti_create_graphs import create_rti_graphs

create_rti_graphs('outputs/single_injury_model_vs_multiple_injury/single_injury',
                  'C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/Single_Injury',
                  'Single_Injury')
create_rti_graphs('outputs/single_injury_model_vs_multiple_injury/multiple_injury',
                  'C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/Multiple_Injury',
                  'Multiple_Injury')
create_rti_graphs('outputs/single_injury_model_vs_multiple_injury/multiple_injury_no_hs',
                  'C:/Users/Robbie Manning Smith/Pictures/TLO model outputs/SingleVsMultipleInjury/No_healthsystem',
                  'Multiple_Inj_No_hs')
