import argparse
import csv
import datetime
import itertools
import matplotlib.pyplot as plt
import numpy as np

def read_data(csv):
    result = []
    for row in csv:
        result.append((datetime.datetime.strptime(row[0], '%d/%m/%Y').date(), float(row[1]), row[2]))
    return result

def get_day(tuple_):
    return tuple_[0].strftime('%Y.%m.%d')

def get_month(tuple_):
    return tuple_[0].strftime('%Y.%m')

def get_year(tuple_):
    return tuple_[0].strftime('%Y')

def get_category(tuple_):
    return tuple_[2]

def groupby(what, how):
    return itertools.groupby(sorted(what, key = how), key = how)

def sum_amounts(x, (date, amount, category)):
    return round(x + amount, 2)

def compute_totals(l):
    running_total = 0
    expenses = []
    for (date, amount, category) in l:
        running_total = round(running_total + amount, 2)
        if len(expenses) > 0 and expenses[-1][0] == date and expenses[-1][2] == category: # Collapse expenses sustained on the same day if they belong to the same category.
            last_expense = expenses.pop()
            amount = round(last_expense[1] + amount, 2)
        expenses.append((date, amount, category, running_total))
    return expenses, running_total

def generate_breakdowns(data, first_breakdown, second_breakdown):
    chart_data = []
    data_fb = groupby(data, first_breakdown)
    for key_fb, group_fb in data_fb:
        list_fb = sorted(list(group_fb), key = get_day)
        point_running_total, total_fb = compute_totals(list_fb)
        pie_chart_data = []
        for key_sb, group_sb in groupby(list_fb, second_breakdown):
            list_sb = list(group_sb)
            total_sb = reduce(sum_amounts, list_sb, 0)
            pie_chart_data.append((key_sb, total_sb, round((total_sb * 100) / total_fb, 2)))
        chart_data.append((key_fb, total_fb, point_running_total, pie_chart_data))
    return chart_data

def get_monthly_income(month, incomes):
    for income in incomes:
        if income[0] == month:
            return income[1]
    return 0

def collapse(dates, running_totals):
    collapsed_running_totals = []
    for (date, running_total) in zip(dates, running_totals):
        if len(collapsed_running_totals) > 0 and collapsed_running_totals[-1][0] == date:
            collapsed_running_totals.pop()
        collapsed_running_totals.append((date, running_total))
    _, collapsed_running_totals = zip(*collapsed_running_totals)
    return collapsed_running_totals

def fill_in_void_dates(partial_tuples, dates):
    amounts = []
    partials = dict(partial_tuples)
    for d in dates:
        try:
            amounts.append(partials[d])
        except KeyError:
            amounts.append(0)
    return amounts

def points_labels(graph, (xs, ys)):
    for p in zip(xs, ys):
        graph.text(p[0] - (graph.get_xticks()[-1] * 1 / 100), p[1] + (graph.get_yticks()[-1] * 1 / 100), '%.2f' % p[1], ha = 'center', va = 'bottom')

def bars_labels(graph, rects):
    for rect in rects:
        height = rect.get_height()
        if height != 0:
            graph.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%.2f' % height, ha = 'center', va = 'bottom')

def generate_visualization((title, total, line_bar_data, pie_data), (aggregate_label, aggregate_amount), breakdown_function, prefix = ''):
    # Image size: apparently pixels == inches * 1000.
    plt.figure(figsize = (12.8, 10.24))
    # Color scheme generated from: http://colorschemedesigner.com/ .
    colors = ['#F9FE4A', '#38BE9C', '#FF8E4B', '#973FC1', '#B3B65A', '#438877', '#B67C5A', '#75478A', '#959913', '#0E7259', '#994513', '#541074', '#FCFF9D', '#9AF9E1', '#FFC19D', '#DC9EFA', '#FEFFDF', '#DAF9F2', '#FFEBDF', '#F0DCFA']

    dates, amounts, categories, running_totals = zip(*line_bar_data)
    # Collapse same-day running totals.
    running_totals = collapse(dates, running_totals)
    # Take just one instance of each date.
    dates = sorted(list(set(dates)))

    # Title.
    plt.gcf().suptitle('{0} [{1} euro]'.format(title, total), fontsize = 14)

    line_bar_graph = plt.subplot2grid((2, 2), (0, 0), colspan = 2)
    width = 0.1
    color_counter = 0
    for key, group in groupby(line_bar_data, breakdown_function):
        g_amounts = fill_in_void_dates([(g[0], g[1]) for g in group], dates)
        bars = line_bar_graph.bar(range(len(dates)), g_amounts, width, color = colors[color_counter], log = True)
        bars_labels(line_bar_graph, bars)
        color_counter += 1
    line_bar_graph.set_yscale('log')

    # Rotate xticklabels.
    plt.gcf().autofmt_xdate()
    # Hide log scale.
    plt.gca().get_yaxis().set_visible(False)

    line_bar_graph_line = line_bar_graph.twinx()
    lines = line_bar_graph_line.plot(np.arange(len(dates)) + width / 2, running_totals, 'r-o')
    # Move linear yticklabels to the left.
    line_bar_graph_line.yaxis.tick_left()

    # Force x and y limits, otherwise some values go out of scale.
    line_bar_graph.set_xlim(-width, len(dates) - 1 + width * 2)
    line_bar_graph.set_ylim(1, running_totals[-1] + 1)
    # Force x ticks.
    line_bar_graph.set_xticks(np.arange(len(dates)) + width / 2)
    line_bar_graph.set_xticklabels(dates)

    # Show grid lines (only shows y lines, don't know why).
    plt.grid(True)

    points_labels(line_bar_graph_line, lines[0].get_data())

    pie_graph = plt.subplot2grid((2, 2), (1, 0))
    labels, amounts, percentages = zip(*pie_data)
    pie_graph.pie(percentages, labels = map(lambda (l, a): '{0} [{1} euro]'.format(l, a), zip(labels, amounts)), shadow = True, autopct = '%.2f%%', startangle = 90, colors = colors)

    if aggregate_amount != 0:
        pie_graph = plt.subplot2grid((2, 2), (1, 1))
        amounts = [sum(amounts), aggregate_amount - sum(amounts)]
        pie_graph.pie(map(lambda a: (a * 100) / aggregate_amount, amounts), labels = map(lambda (l, a): '{0} [{1} euro]'.format(l, a), zip([title, aggregate_label], amounts)), shadow = True, autopct = '%.2f%%', startangle = 90, colors = ('#8EEC6A', '#268E00'))

    plt.rcParams['font.size'] = 7.0
    plt.tight_layout()
    plt.savefig(prefix + title + ".png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expenses_file', type = argparse.FileType('r'), help = 'expenses file')
    parser.add_argument('bills_file', type = argparse.FileType('r'), help = 'bills file')
    parser.add_argument('income_file', type = argparse.FileType('r'), help = 'income file')
    parser.add_argument('high_speed_train_file', type = argparse.FileType('r'), help = 'high speed train fares file')
    args = parser.parse_args()

    expenses = read_data(csv.reader(args.expenses_file))
    bills = read_data(csv.reader(args.bills_file))
    income = read_data(csv.reader(args.income_file))
    high_speed_train = read_data(csv.reader(args.high_speed_train_file))

    combined_expenses = expenses + bills
    combined_expenses_amount = reduce(sum_amounts, combined_expenses, 0)
    income_monthly_breakdown = generate_breakdowns(income, get_month, get_category)

    for mb in generate_breakdowns(combined_expenses, get_month, get_category):
        generate_visualization(mb, ('savings', get_monthly_income(mb[0], income_monthly_breakdown)), get_category)
    for cb in generate_breakdowns(combined_expenses, get_category, get_month):
        generate_visualization(cb, ('other expenses', combined_expenses_amount), get_month)

    # Should we visualize income? How?
    # Yearly breakdowns?

    for hstmb in generate_breakdowns(high_speed_train, get_month, get_category):
        generate_visualization(hstmb, ('', 0), get_category, 'high_speed_train_')
    for hstcb in generate_breakdowns(high_speed_train, get_category, get_month):
        generate_visualization(hstcb, ('', 0), get_month, 'high_speed_train_')

    args.expenses_file.close()
    args.bills_file.close()
    args.income_file.close()
    args.high_speed_train_file.close()

if __name__ == '__main__':
    main()
