from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from typing import Dict

# try:
#     from nltk import word_tokenize
# except ImportError:
#     import nltk
#     nltk.download('punkt')
#     nltk.download('wordnet')
#     from nltk import word_tokenize
# import nltk
# nltk.download('punkt')

def bleu_score(gen_caption: str, ref_captions: list) -> float:
    ref_tokens = [cap.split() for cap in ref_captions]
    gen_tokens = gen_caption.split()
    return sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))

def rouge_score_fn(gen_caption: str, ref_caption: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ref_caption.lower(), gen_caption.lower())['rougeL'].fmeasure

def caption_compute_score(gen_caption: str, ref_caption: str) -> Dict[str, float]:
    ref_captions = [ref_caption]
    try:
        bleu = bleu_score(gen_caption, ref_captions)
    except ZeroDivisionError:
        bleu = 0.0
    rough = rouge_score_fn(gen_caption, ref_caption)
    return {
        'overall': 0.7 * rough + 0.3 * bleu,
        "bleu": bleu,
        "rouge": rough,
    }
