#!/usr/bin/env python
# coding: utf-8

import os
import sys
import argparse
from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu

parser_txt = """Perform a qualitative assessment of a candidate summarization by comparing it to a reference response. Metrics calculated: 
BLEU - Measures the overlap of n-grams between a generated text and reference text, emphasizing precision and brevity in machine translation and text generation tasks.
ROUGE N - Evaluates the overlap of n-grams between generated and reference texts, focusing on recall to assess content similarity in summarization tasks.
BERTScore - Uses contextual embeddings from BERT to compare semantic similarity between generated and reference texts, capturing meaning beyond exact matches."""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=parser_txt)
    parser.add_argument('-c', '--candidate_response', type=str,
                        help='Path to candidate summarization response to test.',
                        default="candidate_summarization.txt")
    parser.add_argument('-r', '--reference_response', type=str,
                        help='Path to reference summarization.',
                        default="reference_summarization.txt")
    args = parser.parse_args(sys.argv[1:])    

    # Load reference response
    if os.path.exists(args.reference_response):
        with open(args.reference_response, 'r') as FH:
            reference_summary = FH.read()
    else:
        print(f"File does not exist: {args.reference_response}")
        exit()

    # Load candidate response
    if os.path.exists(args.candidate_response):
        with open(args.candidate_response, 'r') as FH:
            candidate_summary = FH.read()
    else:
        print(f"File does not exist: {args.candidate_response}")
        exit()

    # Calculate BLEU Scores. Gram level comparison. Useful for translation tasks.
    bleu_score = sentence_bleu([reference_summary], candidate_summary)

    # Calculate ROGUE scores. Gram level comparison. Useful for summerization tasks that are well defined.
    rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = rouge_scorer.score(reference_summary, candidate_summary)

    # Calculate BERTScore. Comparison in embedding space. Useful for summarization tasks, comparing samantic meaning. 
    bs_percision, bs_recall, bs_F1_score = score([candidate_summary],
                                                 [reference_summary],
                                                 lang='en', verbose=False)

    # Report out
    print(f'-----BLEU-----\n{bleu_score}')
    print('\n-----ROUGE-----')
    for key in rouge_scores:
        print(f"{key}: {rouge_scores[key].fmeasure}")
    print(f'\n-----BERTScore-----\n{float(bs_F1_score)}')


