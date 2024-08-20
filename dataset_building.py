# pip install datasets
# pip install ordered-set

from datasets import load_dataset
from ordered_set import OrderedSet
import nltk
import json


def get_data(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset


def get_sent_dict(sample_paragraph):
    sentences = nltk.sent_tokenize(sample_paragraph)
    new_dict = {}
    for i in range(len(sentences)):
        start_index = sample_paragraph.find(sentences[i])
        end_index = sample_paragraph.find(sentences[i]) + len(sentences[i])
        new_dict[str(i)] = {"start_index": start_index, "end_index": end_index}
    return new_dict


def combine_paragraph_by_title(squad_dataset, max_context_limit):
    # Group Paragraph by title
    grouped_by_title = {}
    block_list = []
    for datapoint in squad_dataset:
        title = datapoint['title']
        if title in block_list:
            continue

        if title not in grouped_by_title:
            grouped_by_title[title] = {'context': OrderedSet(), 'qas': []}

        grouped_by_title[title]['context'].add(datapoint['context'])
        context_counts = len(grouped_by_title[title]['context'])
        current_context_len = sum([len(i) for i in grouped_by_title[title]['context']])
        if context_counts >= max_context_limit:
            block_list.append(title)

        grouped_by_title[title]['qas'].append({'question': datapoint['question'],
                                               'answers': datapoint['answers'],
                                               "context_counts": context_counts,
                                               "current_context_len": current_context_len})

    # Combine all the data points
    combined_datapoints = []
    for title, data in grouped_by_title.items():
        contexts = '\n'.join(data['context'])
        sent_info = get_sent_dict(contexts)
        combined_datapoints.append({
            'title': title,
            'context': contexts,
            'qas': data['qas'],
            "sentences": sent_info
        })

    return combined_datapoints, grouped_by_title


def get_answer_index_full_paragraph(data_combined):
    # Get the Index for answer in full paragraph.
    for data in data_combined:
        context_len_list = sorted(set([i["current_context_len"] for i in data['qas']]))
        for i in range(len(data['qas'])):

            if data['qas'][i]['context_counts'] == 1:  # If the context is only one
                ind = data['qas'][i]["answers"]['answer_start']
            else:
                get_prev_context_len = context_len_list.index(data['qas'][i]['current_context_len']) - 1
                ind = context_len_list[get_prev_context_len] + data['qas'][i]['context_counts'] - 1 + \
                      data["qas"][i]["answers"]["answer_start"][0]
                ind = [ind]

            new_dict = {"answer_start_full_para": ind}
            data["qas"][i]["answers"].update(new_dict)

    return data_combined


if __name__ == '__main__':
    dataset_name = "squad"
    dataset = get_data(dataset_name)
    max_context_limit = 5
    combined_data, data_dict = combine_paragraph_by_title(dataset['train'], max_context_limit)
    result_data = get_answer_index_full_paragraph(combined_data)

    with open("processed_squad2.json", "w") as outfile:
        json_string = json.dumps(result_data, indent=4)
        outfile.write(json_string)
