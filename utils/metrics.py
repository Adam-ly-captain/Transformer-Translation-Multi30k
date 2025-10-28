from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch


class NLPMetrics:

    @staticmethod
    def clean_sequence(seq, sos_token_id=1, eos_token_id=2, pad_token_id=0):
        cleaned = []
        for token in seq:
            if token in [sos_token_id, pad_token_id]:
                continue
            if token == eos_token_id:
                break  # 到 eos 就停下
            cleaned.append(token)
        return cleaned

    @staticmethod
    def perplexity(loss):
        """Compute perplexity from loss."""
        return torch.exp(loss)

    @staticmethod
    def bleu_score(reference, hypothesis, sos_token_id=1, eos_token_id=2, pad_token_id=0):
        """Compute BLEU score between reference and hypothesis sequences."""
        reference_cleaned = NLPMetrics.clean_sequence(reference, sos_token_id, eos_token_id, pad_token_id)
        hypothesis_cleaned = NLPMetrics.clean_sequence(hypothesis, sos_token_id, eos_token_id, pad_token_id)

        # 使用 NLTK 的句子级 BLEU 计算
        smoothie = SmoothingFunction().method4
        score = sentence_bleu([reference_cleaned], hypothesis_cleaned, smoothing_function=smoothie)
        return score

    @staticmethod
    def bleu_score_batch(references, hypotheses, sos_token_id=1, eos_token_id=2, pad_token_id=0):
        """Compute average BLEU score for a batch of reference and hypothesis sequences."""
        total_score = 0.0
        batch_size = len(references)

        for ref, hyp in zip(references, hypotheses):
            total_score += NLPMetrics.bleu_score(ref, hyp, sos_token_id, eos_token_id, pad_token_id)

        return total_score / batch_size

# reference = [[3, 7]]
# candidate = [3, 7]
# print(NLPMetrics.bleu_score(reference[0], candidate))
