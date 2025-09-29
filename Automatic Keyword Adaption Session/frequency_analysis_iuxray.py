# frequency_analysis_iuxray.py
import argparse
import collections
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_csv', required=True)
parser.add_argument('--test_csv', required=True)
parser.add_argument('--val_csv', required=True)
parser.add_argument('--bins', default='10,100,1000', help='Comma-separated bin lowers')
parser.add_argument('--output_dir', default='Encode_Dataset/')
args = parser.parse_args()

bins = [int(b) for b in args.bins.split(',')]
bins.sort()
bins.append(float('inf'))  # Add upper bound

train_keyword_csv = pd.read_csv(args.train_csv)
test_keyword_csv = pd.read_csv(args.test_csv)
val_keyword_csv = pd.read_csv(args.val_csv)

level1_keyword_total_list = []

for df in [train_keyword_csv]:
    for i in range(len(df)):
        print(i)  # Progress
        temp_str = df['Level1_keywords'][i]
        if pd.isna(temp_str):
            continue
        else:
            temp_str = temp_str.replace("[", "").replace("]", "")
            temp_list = temp_str.split(", ")
            level1_keyword_total_list.extend(temp_list)

frequency = collections.Counter(level1_keyword_total_list)
frequency= collections.OrderedDict(frequency.most_common())
freq_df = pd.DataFrame({'keyword': list(frequency.keys()), 'frequency': list(frequency.values())})
freq_df.to_csv(f"{args.output_dir}/total_keyword_frequency_iuxray.csv", index=False)

bin_lists = [[] for _ in bins[:-1]]
bin_values = [[] for _ in bins[:-1]]

for kw, freq in frequency.items():
    for idx in range(len(bins) - 1):
        low, high = bins[idx], bins[idx + 1]
        if low < freq <= high:
            bin_lists[idx].append(kw)
            bin_values[idx].append(freq)
            break

for idx, low in enumerate(bins[:-1]):
    case_final = pd.DataFrame({'keyword': bin_lists[idx], 'frequency': bin_values[idx]})
    case_final.to_csv(f"{args.output_dir}/total_keyword_frequency_iuxray_total_{low}.csv", index=False)

# Encoding part (similar to original, but for each bin)
def generate_encoding(df, bin_keywords_list):
    encoding_lists = [[] for _ in bin_keywords_list]
    for i in range(len(df)):
        temp_str = df['Level1_keywords'][i]
        if pd.isna(temp_str):
            for enc in encoding_lists:
                enc.append('')
            continue
        else:
            temp_str = temp_str.replace("[", "").replace("]", "")
            temp_list = temp_str.split(", ")
            for b_idx, keywords in enumerate(bin_keywords_list):
                connect_str = ''
                for kw in keywords:
                    connect_str += '1' if kw in temp_list else '0'
                encoding_lists[b_idx].append(connect_str)
    return encoding_lists

bin_keywords_list = bin_lists  # List of keyword lists per bin

for split, df in [('train', train_keyword_csv), ('test', test_keyword_csv), ('val', val_keyword_csv)]:
    encoding_lists = generate_encoding(df, bin_keywords_list)
    columns = ['Case_num', 'Level1_keywords'] + [f"{bins[i]}_coding_str" for i in range(len(bins[:-1]))]
    data = [df['Case_num'].tolist(), df['Level1_keywords'].tolist()] + encoding_lists
    case_final = pd.DataFrame(dict(zip(columns, data)))
    case_final.to_csv(f"{args.output_dir}/total_keyword_frequency_iuxray_{split}_coding.csv", index=False)