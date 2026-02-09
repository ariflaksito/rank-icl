import evaluate # type: ignore

def compute_metrics(preds, refs):
    bertmodel = 'bert-large-uncased'

    # Load metrics
    rouge_metric = evaluate.load('rouge')
    meteor_metric = evaluate.load('meteor')
    bertscore_metric = evaluate.load('bertscore')

    # Evaluate metrics
    rouge_result = rouge_metric.compute(predictions=preds, references=refs)
    meteor_result = meteor_metric.compute(predictions=preds, references=refs)
    bertscore_true = bertscore_metric.compute(predictions=preds, references=refs, 
                                              lang='en', model_type=bertmodel, 
                                              rescale_with_baseline=True)
    bertscore_false = bertscore_metric.compute(predictions=preds, references=refs, 
                                               lang='en', model_type=bertmodel, 
                                               rescale_with_baseline=False)

    # Display results
    print('--- Evaluation Results ---', flush=True)

    print("", flush=True)
    print(f"ROUGE-1: {rouge_result['rouge1']:.4f}") # type: ignore
    print(f"BERTScore F1: {sum(bertscore_false['f1']) / len(bertscore_false['f1']):.4f}") # type: ignore
    print(f"BERTScore F1 (baseline): {sum(bertscore_true['f1']) / len(bertscore_true['f1']):.4f}") # type: ignore
    print(f"METEOR: {meteor_result['meteor']:.4f}") # type: ignore