import matplotlib.pyplot as plt
import rl.graph_includes as graph_inc
import os
import collections as c
import numpy as np
def print_scatter(results_root, budget):
    r = graph_inc.get_files(results_root, budget)
    sizes={"axis":30, "label":32, "title":36}
    names = ['ba-degree-ineq', 'ba-degree-inverse-ineq', 'cl-degree-ineq', 'cl-degree-inverse-ineq', 'er-degree-ineq', 'er-degree-inverse-ineq', 'sbm-degree-ineq', 'sbm-degree-inverse-ineq']
    field_names = {"original": "orig", "baseline": "baseline", "model":"model"}
    names_plot = ["PA High Degree", "PA Low Degree", "CL High Degree", "CL Low Degree", "ER High Degree", "ER Low Degree", "SBM High Degree", "SBM Low Degree"]
    colors = ["b", "b", "g", "g", "c", "c", "k", "k"]
    ginis = {}
    utilities = c.defaultdict(int)
    counts = c.defaultdict(int)
    for ff_print, ff in field_names.items():
        fig, ax1 = plt.subplots(figsize=(20, 20))
        ax1.tick_params(axis='both', which='major', labelsize=sizes["axis"])
        for kk, kk_print in zip(names, names_plot):
            for k,v in r[kk]["results"].items():
                if k[0] == ff and k[1].startswith("flow"):
                    utilities[kk_print] += v[0]
                    counts[kk_print] += 1
                if k[0] == ff and k[1] == "gini":
                    ginis[kk_print] = v[0]

        for i,(x,y,l,kk, cc) in enumerate(zip(utilities.values(), ginis.values(), counts.values(), ginis.keys(), colors)):
            if i % 2:
                mm = "o"
            else:
                mm = "X"
            plt.scatter(x/l,y, label=kk, s=2000,marker=mm, c=cc )

        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.title("Utility vs. Gini Index by graph: {}, budget: {}".format(ff_print, budget) , fontsize=sizes["title"])
        plt.ylabel("Gini Index", fontsize=sizes["axis"])
        plt.xlabel("Utility", fontsize=sizes["axis"])
        plt.legend(fontsize=sizes["label"])
        plt.savefig(os.path.join(results_root, "scatter_" + ff_print + ".png"))


def print_budget_vs(results_root, budgets = [25, 50, 75, 100], vs="gini", model="model"):
    sizes = {"axis": 30, "label": 32, "title": 36}
    names = ['ba-degree-ineq', 'ba-degree-inverse-ineq', 'cl-degree-ineq', 'cl-degree-inverse-ineq',
             'er-degree-ineq', 'er-degree-inverse-ineq', 'sbm-degree-ineq', 'sbm-degree-inverse-ineq']
    field_names = {"original": "orig", "baseline": "baseline", "model": "model"}
    names_plot = ["PA High Degree", "PA Low Degree", "CL High Degree", "CL Low Degree", "ER High Degree",
                  "ER Low Degree", "SBM High Degree", "SBM Low Degree"]
    colors = ["b", "b", "g", "g", "c", "c", "k", "k"]
    fig, ax1 = plt.subplots(figsize=(20, 20))
    ax1.tick_params(axis='both', which='major', labelsize=sizes["axis"])

    ginis = c.defaultdict(list)
    utilities = c.defaultdict(list)
    plusses = c.defaultdict(list)
    for i, b in enumerate(budgets):
        r = graph_inc.get_files(results_root, b)



        for kk, kk_print in zip(names, names_plot):
            if model == "model":
                plusses[kk_print].append(r[kk]["history"]['edit_num_edits_exceeded'][-1])
            else:
                plusses[kk_print].append(0)
            for k,v in r[kk]["results"].items():
                if k[0] == model and k[1].startswith("flow"):
                    utilities[(kk_print,b)].append(v[0])

                if k[0] == model and k[1] == "gini":
                    ginis[kk_print].append(v[0])
    utilities_new = c.defaultdict(list)
    for (kk_print, b), v in utilities.items():
        utilities_new[kk_print].append(np.mean(v))

    if vs == "gini":
        measure = ginis
        vs_print = "Gini Index"
    else:
        measure = utilities_new
        vs_print = "Utility"

    for i,(y,kk, cc, pp) in enumerate(zip(measure.values(), measure.keys(), colors, plusses.values())):
        if i % 2:
            mm = "o"
        else:
            mm = "X"
        plt.plot(np.array(budgets)+ np.array(pp), y, color=cc, marker=mm, linewidth=4, markersize=18, label=kk )


    plt.ylim([0,1])
    plt.title("Budget vs. {} by graph: {}".format(vs_print, model) , fontsize=sizes["title"])
    plt.ylabel(vs_print, fontsize=sizes["axis"])
    plt.xlabel("Budget", fontsize=sizes["axis"])
    plt.legend(fontsize=sizes["label"])
    plt.savefig(os.path.join(results_root, "budget_vs_{}_{}.png".format(vs, model)))





def print_training_trajectory(results_root, key="edit_num_edits_exceeded", max_size=750):
    sizes = {"axis": 30, "label": 32, "title": 36}
    names = ['ba-degree-ineq', 'ba-degree-inverse-ineq', 'cl-degree-ineq', 'cl-degree-inverse-ineq',
             'er-degree-ineq', 'er-degree-inverse-ineq', 'sbm-degree-ineq', 'sbm-degree-inverse-ineq']
    names_plot = ["PA High Degree", "PA Low Degree", "CL High Degree", "CL Low Degree", "ER High Degree",
                  "ER Low Degree", "SBM High Degree", "SBM Low Degree"]
    colors = ["b", "b", "g", "g", "c", "c", "k", "k"]
    fig, ax1 = plt.subplots(figsize=(20, 20))
    ax1.tick_params(axis='both', which='major', labelsize=sizes["axis"])
    print_vals = {"value_diff_bw_groups": "Mean Utility Difference Between Groups",
            "edit_num_edits_exceeded": "Edits Over Budget",
            "value_mean_value": "Mean Utility"}
    r = graph_inc.get_files(results_root, budget=100)
    for i, (kk, kk_print, cc) in enumerate(zip(names, names_plot, colors)):
        if i % 2:
            mm = "o"
        else:
            mm = "X"
        plt.plot(r[kk]["history"][key][0:max_size], color=cc, marker=mm, linewidth=4, markersize=1, label=kk_print)

    if key != "edit_num_edits_exceeded":
        plt.ylim([0, 1])
    plt.title("Training trajectory: {} by graph".format(print_vals[key]), fontsize=sizes["title"])
    plt.ylabel(print_vals[key], fontsize=sizes["axis"])
    plt.xlabel("Epoch", fontsize=sizes["axis"])
    plt.legend(fontsize=sizes["label"])
    plt.savefig(os.path.join(results_root, "training_trajectory_{}.png".format(key)))





def print_budget_vs_placement(results_root, budgets = [3, 6, 9, 12, 15, 18, 21], vs="gini", model="model"):
    sizes = {"axis": 30, "label": 32, "title": 36}
    names = ['sbm-degree-ineq']
    field_names = {"original": "orig", "baseline": "baseline", "model": "model"}
    names_plot = ["PA High Degree"]
    colors = ["b", "b", "g", "g", "c", "c", "k", "k"]



    ginis = c.defaultdict(list)
    utilities = c.defaultdict(list)
    plusses = c.defaultdict(list)
    for i, b in enumerate(budgets):
        r = graph_inc.get_files(results_root, b)



        for kk, kk_print in zip(names, names_plot):
            if model == "model":
                plusses[kk_print].append(r[kk]["history"]['edit_num_edits_exceeded'][-1])
            else:
                plusses[kk_print].append(0)
            for k,v in r[kk]["results"].items():
                if k[0] == model and k[1].startswith("flow"):
                    utilities[(kk_print,b)].append(v[0])

                if k[0] == model and k[1] == "gini":
                    ginis[kk_print].append(v[0])
    utilities_new = c.defaultdict(list)
    for (kk_print, b), v in utilities.items():
        utilities_new[kk_print].append(np.mean(v))

    if vs == "gini":
        measure = ginis
        vs_print = "Gini Index"
    else:
        measure = utilities_new
        vs_print = "Utility"

    fig, ax1 = plt.subplots(figsize=(20, 20))
    plt.xlabel("Budget", fontsize=sizes["axis"])
    ax1.tick_params(axis='both', which='major', labelsize=sizes["axis"])
    colors = ['k', 'k']
    for i,(y,kk, cc, pp) in enumerate(zip(measure.values(), measure.keys(), colors, plusses.values())):
        mm = "o"
        plt.plot(np.array(budgets) + np.array(pp), y, color=cc, marker=mm, linewidth=4, markersize=18, label=kk)

    ax2 = ax1.twinx()
    for i,(y,kk, cc, pp) in enumerate(zip(utilities_new.values(), measure.keys(), colors, plusses.values())):
        mm = "o"
        ax2.plot(np.array(budgets) + np.array(pp), y, color='b', marker=mm, linewidth=4, markersize=18, label=kk)
    ax2.tick_params(axis='both', which='major', labelsize=sizes["axis"])
    ax2.tick_params(axis='y', labelcolor='b')

    ax1.set_ylabel("Gini Index", fontsize=sizes["label"])
    ax2.set_ylabel("Average Utility", color='b', fontsize=sizes["label"])

    plt.ylim([0,1])
    plt.title("Facility Placement:\nBudget vs. {} by graph: {}".format("Utility and Gini Index", kk_print) , fontsize=sizes["title"])
    #plt.ylabel(vs_print, fontsize=sizes["axis"])

    #plt.legend(fontsize=sizes["label"])
    plt.savefig(os.path.join(results_root, "facility_{}.png".format("base")))