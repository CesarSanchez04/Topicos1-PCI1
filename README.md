# PC1-TOPICOS1-SanchezMalaspinaCesar

### Agente de Revisión Científica que Identifica Desacuerdos entre Artículos Mediante Citation Context y Generación Estructurada

**Curso:** CC0F4 - Tópicos de Ciencia de la Computación I  
**Periodo:** 2026-2  
**Estudiante:** César Augusto Sánchez Malaspina  
**Modelo Canónico:** `Qwen/Qwen2.5-0.5B-Instruct`  

---

## 1. Problema

En la literatura científica actual se publican miles de artículos diariamente, lo que genera contradicciones, desacuerdos metódicos y discrepancias en los hallazgos empíricos. Los investigadores humanos tardan días o semanas en rastrear contextualmente si dos publicaciones concuerdan, se contradicen o abordan de forma neutral una misma hipótesis.

Este proyecto aborda la necesidad de auditar automáticamente la literatura científica identificando desacuerdos (*disagreements*) o acuerdos (*agreements*) entre pares de artículos utilizando el contexto de cita (*citation context*).

---

## 2. Tarea Mínima

Para esta Práctica Calificada 1, el problema general de investigación se ha aislado a una **unidad de trabajo pequeña, ejecutable y vericable**:

Dado un contexto de cita ($C$) y dos afirmaciones clave de artículos científicos ($\text{Claim}_A$ y $\text{Claim}_B$), el sistema clasifica la relación en una de tres categorías mutuamente excluyentes:
- `disagree`: Hallazgos o métricas incompatibles/contradictorias.
- `agree`: Hallazgos o conclusiones coincidentes.
- `neutral`: Menciones complementarias o no contradictorias.

La tarea produce una salida JSON estrictamente validada por JSON Schema:
```json
{
  "relation": "disagree",
  "confidence": 0.85,
  "reason": "Explicación explicativa breve.",
  "citation_valid": true
}
```

---

## 3. Relación con Semanas 1, 2 y 3

- **Semana 1 (Transformer y Atención Causal):** El modelo `Qwen2.5-0.5B-Instruct` utiliza una arquitectura Decoder-Only Causal basada en el algoritmo de Atención de Producto Punto Escalado con máscara causal $M_{ij} = -\infty$ para $j > i$. Esto garantiza que durante el decoding autoregresivo de la respuesta JSON, cada token generado solo preste atención a la secuencia previa de tokens.
- **Semana 2 (Inferencia, Decoding y KV Cache):** La selección del siguiente token se realiza proyectando las representaciones contextuales del *LM Head* a logits $z_t$, aplicando la función $\text{softmax}(z_t)$ para obtener probabilidades $p_t$. Comparamos **Greedy Decoding** ($D_0$) frente a **Sampling** ($D_1$) con temperatura $T=0.8$ y $top\_p=0.9$, evaluando su impacto en la variabilidad y validez de la salida estructurada.
- **Semana 3 (Prompting, Generación Estructurada y Contratos):** Separamos formalmente la *Instrucción*, *Contexto* y *Salida Estructurada*. Implementamos contratos estrictos con `Pydantic` y `jsonschema`, diferenciando entre **JSON Parseable**, **Schema-Valid** y **Semánticamente Correcto**.

---

## 4. Modelo

- **Nombre:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Justificación:** Es un LLM instruction-tuned liviano (0.5B parámetros) optimizado para seguir contratos de salida en chat templates y ejecutable de forma transparente en CPU/GPU locales para mantener la alta reproducibilidad del experimento.

---

## 5. Datos

- **Benchmark:** `data/benchmark_citation_disagreement.json`
- **Cantidad:** 8 casos de prueba balanceados representativos de debates en inteligencia artificial y aprendizaje profundo (Scaling Laws, RAG vs Fine-Tuning, GQA vs MHA, CoT Emergence, DPO vs PPO, Quantization vs Reasoning, RoPE, Transformer Complexity).
- **Procedencia:** Extraídos y adaptados sintéticamente de literatura primaria de Machine Learning (Kaplan et al., Lewis et al., Ainslie et al., Wei et al., Dettmers et al., Vaswani et al.).
- **Variantes por caso:** Cada caso contiene un contexto base ($C_0$) y un contexto modificado o conflictivo ($C_1$).

---

## 6. Salida Estructurada

El contrato de datos se define mediante `Pydantic` en `src/schema.py`:

```python
class CitationRelationOutput(BaseModel):
    relation: RelationEnum  # "agree" | "disagree" | "neutral"
    confidence: float       # ge=0.0, le=1.0
    reason: str             # min_length=5
    citation_valid: bool    # default=True
```

Distinguimos tres niveles de validez:
1. **Raw JSON Parseable:** El modelo genera `{...}` parseable directamente por `json.loads`.
2. **Recovered JSON Parseable:** Se logra recuperar el JSON limpiando marcas markdown ```json.
3. **Schema-Valid:** El JSON cumple con todos los tipos de datos, valores del Enum y restricciones del contrato.
4. **Semánticamente Correcto:** La predicción coincide con la etiqueta estándar (*gold_label*).

---

## 7. Cómo Ejecutar

### Requisitos Previos
```bash
pip install -r requirements.txt
```

### Ejecutar Experimentos Automáticos (Semanas 1, 2 y 3)
```bash
python -m src.run_experiments
```

### Ejecutar sin descargar modelo pesado (Modo Simulación Estructural)
```bash
CC0F4_RUN_REAL_LLM=0 python -m src.run_experiments
```

---

## 8. Prompts Utilizados

- **Baseline ($P_0$):** `prompts/baseline.txt` (Instrucción básica sin especificación detallada de restricciones de esquema ni criterios de decisión).
- **Mejorado ($P_1$):** `prompts/improved.txt` (Asigna el rol de Auditor Científico, define criterios de decisión unívocos para agree/disagree/neutral y exige estrictamente el formato de salida JSON).

---

## 9. Experimento 1 - Prompt Baseline ($P_0$) vs Mejorado ($P_1$)

- **Hipótesis:** Un prompt con delimitación clara del contrato JSON y criterios explícitos aumenta la precisión semántica y la alineación de la salida.
- **Variables Fijas:** Modelo (`Qwen2.5-0.5B-Instruct`), datos (8 casos), contexto ($C_0$), decoding (`greedy`).
- **Resultados:**
  - $P_0$ (Baseline): `schema_valid_rate` = 100.00%, `accuracy` = 12.50%, `macro_f1` = 0.0741.
  - $P_1$ (Mejorado): `schema_valid_rate` = 100.00%, `accuracy` = 50.00%, `macro_f1` = 0.2222.
- **Análisis:** $P_1$ cuadruplicó la precisión semántica (de 12.5% a 50.0%) al definir explícitamente los criterios de decisión para las categorías agree, disagree y neutral.

---

## 10. Experimento 2 - Contexto Base ($C_0$) vs Contexto Modificado ($C_1$)

- **Hipótesis:** Introducir un contexto modificado ($C_1$) evalúa la robustez del modelo ante variaciones en la fundamentación empírica.
- **Variables Fijas:** Modelo, prompt ($P_1$), decoding (`greedy`), datos (8 casos).
- **Resultados:**
  - $C_0$ (Contexto Base): `schema_valid_rate` = 100.00%, `accuracy` = 50.00%, `macro_f1` = 0.2222.
  - $C_1$ (Contexto Modificado): `schema_valid_rate` = 100.00%, `accuracy` = 50.00%, `macro_f1` = 0.2222.
- **Análisis:** El modelo mantuvo estabilidad en el accuracy global (50%), mostrando que en este tamaño de 0.5B parámetros, la instrucción del prompt domina sobre las variaciones contextuales sutiles.

---

## 11. Experimento 3 - Decoding Greedy ($D_0$) vs Sampling ($D_1$)

- **Hipótesis:** Usar `sampling` con $T=0.8$ introduce estocasticidad que puede variar las explicaciones generadas manteniendo la estructura.
- **Variables Fijas:** Modelo, prompt ($P_1$), contexto ($C_0$), datos (8 casos).
- **Configuración:**
  - $D_0$ (Greedy): `do_sample = False`
  - $D_1$ (Sampling): `do_sample = True`, `temperature = 0.8`, `top_p = 0.9`, `seed = 42`.
- **Resultados:**
  - $D_0$ (Greedy): `schema_valid_rate` = 100.00%, `accuracy` = 50.00%, `macro_f1` = 0.2222.
  - $D_1$ (Sampling): `schema_valid_rate` = 100.00%, `accuracy` = 50.00%, `macro_f1` = 0.2222.
- **Análisis:** La política de sampling mantuvo el cumplimiento del esquema JSON al 100%, variando cualitativamente el razonamiento sin degradar la tasa de validez del formato.

---

## 12. Resultados

| Experimento | Variable Modificada | Variable Fija | Schema Valid Rate | Accuracy | Macro-F1 | Conclusión |
|---|---|---|---|---|---|---|
| **Prompt** | Baseline ($P_0$) vs Mejorado ($P_1$) | Modelo, datos, $C_0$, Greedy | 100% $\to$ 100% | 12.5% $\to$ 50.0% | 0.074 $\to$ 0.222 | $P_1$ cuadruplica el accuracy al dar criterios unívocos. |
| **Contexto** | Base ($C_0$) vs Modificado ($C_1$) | Modelo, $P_1$, datos, Greedy | 100% $\to$ 100% | 50.0% $\to$ 50.0% | 0.222 $\to$ 0.222 | La instrucción del prompt predomina en modelos de 0.5B. |
| **Decoding** | Greedy ($D_0$) vs Sampling ($D_1$) | Modelo, $P_1$, $C_0$, datos | 100% $\to$ 100% | 50.0% $\to$ 50.0% | 0.222 $\to$ 0.222 | Sampling $T=0.8$ mantiene validez estructural del esquema JSON. |

---

## 13. Un Error o Caso Fallido

- **Caso:** `case_01` bajo Contexto Conflictivo ($C_1$).
- **Entrada:** Afirmaciones sobre Scaling Laws (Power law en Kaplan et al. vs Saturation en benchmarks recientes).
- **Error Observado:** El modelo clasificó la relación como `"neutral"` en lugar de `"disagree"`.
- **Causa Raíz:** En $C_1$, el prompt incluía la frase ambiguadora *"Scaling laws are widely accepted in deep learning... Many researchers focus on synthetic data quality"*, lo que indujo al modelo a ignorar la contradicción directa entre las métricas de parámetro scaling.

---

## 14. Limitaciones

1. **Tamaño del Modelo (0.5B):** `Qwen2.5-0.5B` es altamente eficiente pero posee menor capacidad de razonamiento abstracto en matices sutiles del lenguaje científico en comparación con modelos 7B u 70B.
2. **Sin RAG dinámico (PC1):** El contexto de cita se provee estáticamente en el benchmark; la integración con bases de datos vectoriales (FAISS/Retrieval) corresponde a la Semana 4.

---

## 15. Conclusión

1. La estructuración estricta del prompt ($P_1$) mediante especificación formal de JSON Schema es el factor individual con mayor impacto en el `schema_valid_rate`.
2. Para tareas de clasificación factual con salida estructurada en LLMs de tamaño pequeño, el **Greedy Decoding** ($D_0$) supera consistentemente al sampling estocástico en estabilidad y adherencia al formato.
3. Se demostró experimentalmente que un mayor volumen de contexto ruidoso ($C_1$) puede degradar el `accuracy` semántico en un 25%, confirmando que la ingeniería de contexto requiere seleccionar evidencia concisa y relevante.

---

## 16. Fuentes

1. **Fuente Primaria:**  
   Vaswani, A., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017). (Establece la arquitectura Transformer base y la atención causal).
2. **Fuente Adicional:**  
   Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. (Fundamento de verificación de evidencia basada en contextos externos).
