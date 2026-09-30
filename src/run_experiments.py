"""
Ejecutor principal de los Experimentos 1, 2 y 3 para la Práctica Calificada 1.
Cesar Augusto Sanchez Malaspina - Agente Auditor de Literatura Científica.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from src.evaluator import ScientificReviewEvaluator


def load_dataset(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_template(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def calculate_metrics(results_list, gold_classes=["agree", "disagree", "neutral"]):
    n = len(results_list)
    if n == 0:
        return {}
        
    raw_parse_count = sum(1 for r in results_list if r["raw_json_parse"])
    recovered_parse_count = sum(1 for r in results_list if r["recovered_json_parse"])
    schema_valid_count = sum(1 for r in results_list if r["schema_valid"])
    correct_count = sum(1 for r in results_list if r["is_correct"])
    
    y_true = [r["gold_relation"] for r in results_list]
    y_pred = [r["predicted_relation"] for r in results_list]
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=gold_classes, average="macro", zero_division=0)
    
    avg_latency = np.mean([r["latency_ms"] for r in results_list])
    
    return {
        "n_samples": n,
        "raw_json_parse_rate": raw_parse_count / n,
        "recovered_json_parse_rate": recovered_parse_count / n,
        "schema_valid_rate": schema_valid_count / n,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_latency_ms": avg_latency
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "benchmark_citation_disagreement.json")
    prompt_p0_path = os.path.join(base_dir, "prompts", "baseline.txt")
    prompt_p1_path = os.path.join(base_dir, "prompts", "improved.txt")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    dataset = load_dataset(data_path)
    p0_template = load_prompt_template(prompt_p0_path)
    p1_template = load_prompt_template(prompt_p1_path)
    
    # Comprobar si se puede ejecutar LLM real
    run_real = os.environ.get("CC0F4_RUN_REAL_LLM", "1") == "1"
    evaluator = ScientificReviewEvaluator(model_name="Qwen/Qwen2.5-0.5B-Instruct", run_real_llm=run_real)
    
    decoding_greedy = {"do_sample": False, "max_new_tokens": 256}
    decoding_sampling = {"do_sample": True, "temperature": 0.8, "top_p": 0.9, "top_k": 50, "seed": 42, "max_new_tokens": 256}

    print("=" * 70)
    print("EJECUTANDO EXPERIMENTO 1: PROMPT BASELINE (P0) VS PROMPT MEJORADO (P1)")
    print("=" * 70)
    
    exp1_p0_results = []
    exp1_p1_results = []
    
    for item in dataset:
        # P0 Baseline
        p0_text = p0_template.format(
            context=item["context_base"],
            paper_a_claim=item["paper_a_claim"],
            paper_b_claim=item["paper_b_claim"]
        )
        res_p0 = evaluator.evaluate_sample(p0_text, decoding_greedy, item["gold_label"])
        res_p0["case_id"] = item["id"]
        exp1_p0_results.append(res_p0)

        # P1 Improved
        p1_text = p1_template.format(
            context=item["context_base"],
            paper_a_claim=item["paper_a_claim"],
            paper_b_claim=item["paper_b_claim"]
        )
        res_p1 = evaluator.evaluate_sample(p1_text, decoding_greedy, item["gold_label"])
        res_p1["case_id"] = item["id"]
        exp1_p1_results.append(res_p1)

    m1_p0 = calculate_metrics(exp1_p0_results)
    m1_p1 = calculate_metrics(exp1_p1_results)

    print(f"Exp 1 - Baseline (P0): Schema Valid = {m1_p0['schema_valid_rate']:.2%}, Acc = {m1_p0['accuracy']:.2%}, Macro-F1 = {m1_p0['macro_f1']:.4f}")
    print(f"Exp 1 - Improved (P1): Schema Valid = {m1_p1['schema_valid_rate']:.2%}, Acc = {m1_p1['accuracy']:.2%}, Macro-F1 = {m1_p1['macro_f1']:.4f}")

    print("\n" + "=" * 70)
    print("EJECUTANDO EXPERIMENTO 2: CONTEXTO BASE (C0) VS CONTEXTO MODIFICADO (C1)")
    print("=" * 70)
    
    exp2_c0_results = exp1_p1_results # C0 coincide con P1 + Contexto Base
    exp2_c1_results = []
    
    for item in dataset:
        p1_c1_text = p1_template.format(
            context=item["context_modified"],
            paper_a_claim=item["paper_a_claim"],
            paper_b_claim=item["paper_b_claim"]
        )
        res_c1 = evaluator.evaluate_sample(p1_c1_text, decoding_greedy, item["gold_label"])
        res_c1["case_id"] = item["id"]
        exp2_c1_results.append(res_c1)

    m2_c0 = m1_p1
    m2_c1 = calculate_metrics(exp2_c1_results)

    print(f"Exp 2 - Contexto Base (C0): Schema Valid = {m2_c0['schema_valid_rate']:.2%}, Acc = {m2_c0['accuracy']:.2%}, Macro-F1 = {m2_c0['macro_f1']:.4f}")
    print(f"Exp 2 - Contexto Modif (C1): Schema Valid = {m2_c1['schema_valid_rate']:.2%}, Acc = {m2_c1['accuracy']:.2%}, Macro-F1 = {m2_c1['macro_f1']:.4f}")

    print("\n" + "=" * 70)
    print("EJECUTANDO EXPERIMENTO 3: DECODING GREEDY (D0) VS SAMPLING (D1)")
    print("=" * 70)
    
    exp3_d0_results = exp1_p1_results # D0 coincide con P1 + Greedy
    exp3_d1_results = []
    
    for item in dataset:
        p1_text = p1_template.format(
            context=item["context_base"],
            paper_a_claim=item["paper_a_claim"],
            paper_b_claim=item["paper_b_claim"]
        )
        res_d1 = evaluator.evaluate_sample(p1_text, decoding_sampling, item["gold_label"])
        res_d1["case_id"] = item["id"]
        exp3_d1_results.append(res_d1)

    m3_d0 = m1_p1
    m3_d1 = calculate_metrics(exp3_d1_results)

    print(f"Exp 3 - Greedy (D0): Schema Valid = {m3_d0['schema_valid_rate']:.2%}, Acc = {m3_d0['accuracy']:.2%}, Macro-F1 = {m3_d0['macro_f1']:.4f}")
    print(f"Exp 3 - Sampling (D1): Schema Valid = {m3_d1['schema_valid_rate']:.2%}, Acc = {m3_d1['accuracy']:.2%}, Macro-F1 = {m3_d1['macro_f1']:.4f}")

    # Guardar resumen en JSON
    summary_data = {
        "exp1_prompt": {"P0_baseline": m1_p0, "P1_improved": m1_p1},
        "exp2_context": {"C0_base": m2_c0, "C1_modified": m2_c1},
        "exp3_decoding": {"D0_greedy": m3_d0, "D1_sampling": m3_d1}
    }
    
    with open(os.path.join(results_dir, "experiments_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Crear Tabla Resumen pandas
    table_rows = [
        {"Experimento": "Exp 1 - Prompt", "Condición": "P0 (Baseline)", "Schema Valid Rate": f"{m1_p0['schema_valid_rate']:.2%}", "Accuracy": f"{m1_p0['accuracy']:.2%}", "Macro-F1": f"{m1_p0['macro_f1']:.4f}"},
        {"Experimento": "Exp 1 - Prompt", "Condición": "P1 (Mejorado)", "Schema Valid Rate": f"{m1_p1['schema_valid_rate']:.2%}", "Accuracy": f"{m1_p1['accuracy']:.2%}", "Macro-F1": f"{m1_p1['macro_f1']:.4f}"},
        {"Experimento": "Exp 2 - Contexto", "Condición": "C0 (Base)", "Schema Valid Rate": f"{m2_c0['schema_valid_rate']:.2%}", "Accuracy": f"{m2_c0['accuracy']:.2%}", "Macro-F1": f"{m2_c0['macro_f1']:.4f}"},
        {"Experimento": "Exp 2 - Contexto", "Condición": "C1 (Modificado)", "Schema Valid Rate": f"{m2_c1['schema_valid_rate']:.2%}", "Accuracy": f"{m2_c1['accuracy']:.2%}", "Macro-F1": f"{m2_c1['macro_f1']:.4f}"},
        {"Experimento": "Exp 3 - Decoding", "Condición": "D0 (Greedy)", "Schema Valid Rate": f"{m3_d0['schema_valid_rate']:.2%}", "Accuracy": f"{m3_d0['accuracy']:.2%}", "Macro-F1": f"{m3_d0['macro_f1']:.4f}"},
        {"Experimento": "Exp 3 - Decoding", "Condición": "D1 (Sampling t=0.8)", "Schema Valid Rate": f"{m3_d1['schema_valid_rate']:.2%}", "Accuracy": f"{m3_d1['accuracy']:.2%}", "Macro-F1": f"{m3_d1['macro_f1']:.4f}"},
    ]
    
    df_summary = pd.DataFrame(table_rows)
    df_summary.to_csv(os.path.join(results_dir, "experiments_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("TABLA RESUMEN DE RESULTADOS")
    print("=" * 70)
    print(df_summary.to_string(index=False))
    print(f"\nResultados detallados guardados en: {results_dir}")


if __name__ == "__main__":
    main()
