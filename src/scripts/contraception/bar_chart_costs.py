from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Where will output fig go - by default, wherever this script is run
outputpath = Path("./outputs")  # folder for convenience of storing outputs
# Width of the bars
width = 0.3


def plot_costs(in_id, in_suffix, in_x_labels, in_cons_costs_without, in_cons_costs_with,
               in_pop_interv_costs_with, in_ppfp_interv_costs_with, in_reduce_magnitude=1e3):

    # Prepare data for the plots
    def reduce_magnitude(in_list, in_in_reduce_magnitude):
        return [x / in_in_reduce_magnitude for x in in_list]

    cons_costs_without = reduce_magnitude(in_cons_costs_without, in_reduce_magnitude)
    #
    cons_costs_with = reduce_magnitude(in_cons_costs_with, in_reduce_magnitude)
    pop_interv_costs_with = reduce_magnitude(in_pop_interv_costs_with, in_reduce_magnitude)
    ppfp_interv_costs_with = reduce_magnitude(in_ppfp_interv_costs_with, in_reduce_magnitude)
    ppfp_bottom = [x + y for x, y in zip(cons_costs_with, pop_interv_costs_with)]

    # %%% PLot all time periods + total
    x_labels = in_x_labels.copy()
    x_labels[-1] = "TOTAL (" + x_labels[-1] + ")"
    x = np.arange(len(x_labels))  # the x_label locations
    fig, ax = plt.subplots()
    # bar_without
    ax.bar(x - width/2, cons_costs_without, width, label='consumables without intervention', color=(0.918, .255, 0.47))
    if int(in_x_labels[0].split("-")[0]) < 2023:
        with_label = 'consumables with intervention since 2023'
    else:
        with_label = 'consumables with intervention'
    # bar_with
    ax.bar(x + width/2, cons_costs_with, width, label=with_label, color=(0.698, 0.875, 0.541))
    # bar_with_pop_interv
    ax.bar(x + width/2, pop_interv_costs_with, width, bottom=cons_costs_with, label='Pop intervention',
           color=(0.122, .471, 0.706))
    # bar_with_ppfp_interv
    ax.bar(x + width/2, ppfp_interv_costs_with, width, bottom=ppfp_bottom, label='PPFP intervention',
           color=(0.651, .808, 0.89))

    # title, custom x-axis tick labels, set y-axis label and add legend
    ax.set_title('Consumables & Interventions Costs', fontweight="bold")
    #
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontweight="bold")
    #
    ax.set_ylabel('MWK (1e9) ~ USD (1e6)', fontweight="bold")
    #
    ax.legend()
    # TODO: more values on y-axis

    # the below needs at least 3.4 version of matplotlib package (we have 3.3.4)
    # ax.bar_label(bar_without, padding=3)
    # ax.bar_label(bar_with, padding=3)

    fig.tight_layout()

    plt.grid(axis='y')

    plt.savefig(outputpath / ('Consumables and Interventions Costs ' + in_id[0] + "_" + in_id[1] +
                              in_suffix + '.png'), format='png')

    # %%% PLot total only
    x2_labels = ["2023-2050"]  # TODO: use in_x_labels[-1]
    x2 = np.arange(len(x2_labels))  # the x_label locations
    fig, ax = plt.subplots()
    # bar_without
    ax.bar(x2 - width/6, cons_costs_without[-1], width/3, label='consumables without intervention',
           color=(0.918, .255, 0.47))
    # bar_with
    ax.bar(x2 + width/6, cons_costs_with[-1], width/3, label='consumables with intervention',
           color=(0.698, 0.875, 0.541))
    # bar_with_pop_interv
    ax.bar(x2 + width/6, pop_interv_costs_with[-1], width/3, bottom=cons_costs_with[-1], label='Pop intervention',
           color=(0.122, .471, 0.706))
    # bar_with_ppfp_interv
    ax.bar(x2 + width/6, ppfp_interv_costs_with[-1], width/3, bottom=ppfp_bottom[-1], label='PPFP intervention',
           color=(0.651, .808, 0.89))

    # title, custom x-axis tick labels, set y-axis label and add legend
    ax.set_title('TOTAL Consumables & Interventions Costs', fontweight="bold")
    #
    ax.set_xticks(x2)
    ax.set_xticklabels(x2_labels, fontweight="bold")
    #
    ax.set_ylabel('MWK (1e9) ~ USD (1e6)', fontweight="bold")
    #
    ax.legend(loc='upper left', bbox_to_anchor=(1, 0.9999))

    fig.tight_layout()

    plt.grid(axis='y')

    plt.savefig(outputpath / ('Total Consumables and Interventions Costs ' + in_id[0] + "_" + in_id[1] +
                              in_suffix + '.png'), format='png')

    print("Fig: Consumables and Interventions Costs Over time saved.")
