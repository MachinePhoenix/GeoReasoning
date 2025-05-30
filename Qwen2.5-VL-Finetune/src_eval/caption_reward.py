from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from typing import Dict

# 确保nltk资源已下载（首次运行需要）
# try:
#     from nltk import word_tokenize
# except ImportError:
#     import nltk
#     nltk.download('punkt')
#     nltk.download('wordnet')
#     from nltk import word_tokenize
# import nltk
# nltk.download('punkt')

# ================== 奖励函数模块 ==================
def bleu_score(gen_caption: str, ref_captions: list) -> float:
    """计算BLEU-4分数"""
    ref_tokens = [cap.split() for cap in ref_captions]
    gen_tokens = gen_caption.split()
    return sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25))

def rouge_score_fn(gen_caption: str, ref_caption: str) -> float:
    """计算ROUGE-L F1分数（修正版）"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ref_caption.lower(), gen_caption.lower())['rougeL'].fmeasure

def caption_compute_score(gen_caption: str, ref_caption: str) -> Dict[str, float]:
    """综合评分计算（优化版）"""
    ref_captions = [ref_caption]
    try:
        bleu = bleu_score(gen_caption, ref_captions)
    except ZeroDivisionError:  # 处理空文本情况
        bleu = 0.0
    rough = rouge_score_fn(gen_caption, ref_caption)
    return {
        'overall': 0.7 * rough + 0.3 * bleu,  # 调整权重比例
        "bleu": bleu,
        "rouge": rough,
    }