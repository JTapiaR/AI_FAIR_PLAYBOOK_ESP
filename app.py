import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# --- Configuración de la Página ---
st.set_page_config(
    page_title="AI Fairness Playbooks",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

#======================================================================
# --- FUNCIONES DE SIMULACIÓN ---
#======================================================================

def run_threshold_simulation():
    """Simulación para optimización de umbrales en post-procesamiento."""
    st.markdown("#### Simulación de Optimización de Umbrales")
    st.write("Ajusta los umbrales de decisión para dos grupos y observa cómo cambian las tasas de error para lograr la **Igualdad de Oportunidades** (tasas de verdaderos positivos iguales).")

    np.random.seed(42)
    scores_a_pos = np.random.normal(0.7, 0.15, 80)
    scores_a_neg = np.random.normal(0.4, 0.15, 120)
    scores_b_pos = np.random.normal(0.6, 0.15, 50)
    scores_b_neg = np.random.normal(0.3, 0.15, 150)

    df_a = pd.DataFrame({'Puntuación': np.concatenate([scores_a_pos, scores_a_neg]), 'Real': [1]*80 + [0]*120})
    df_b = pd.DataFrame({'Puntuación': np.concatenate([scores_b_pos, scores_b_neg]), 'Real': [1]*50 + [0]*150})
    
    col1, col2 = st.columns(2)
    with col1:
        threshold_a = st.slider("Umbral para Grupo A", 0.0, 1.0, 0.5, key="sim_thresh_a")
    with col2:
        threshold_b = st.slider("Umbral para Grupo B", 0.0, 1.0, 0.5, key="sim_thresh_b")

    tpr_a = np.mean(df_a[df_a['Real'] == 1]['Puntuación'] >= threshold_a)
    fpr_a = np.mean(df_a[df_a['Real'] == 0]['Puntuación'] >= threshold_a)
    tpr_b = np.mean(df_b[df_b['Real'] == 1]['Puntuación'] >= threshold_b)
    fpr_b = np.mean(df_b[df_b['Real'] == 0]['Puntuación'] >= threshold_b)

    st.markdown("##### Resultados")
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.metric(label="Tasa de Verdaderos Positivos (Grupo A)", value=f"{tpr_a:.2%}")
        st.metric(label="Tasa de Falsos Positivos (Grupo A)", value=f"{fpr_a:.2%}")
    with res_col2:
        st.metric(label="Tasa de Verdaderos Positivos (Grupo B)", value=f"{tpr_b:.2%}")
        st.metric(label="Tasa de Falsos Positivos (Grupo B)", value=f"{fpr_b:.2%}")

    if abs(tpr_a - tpr_b) < 0.02:
        st.success(f"¡Casi has logrado la Igualdad de Oportunidades! La diferencia en TPR es de solo {abs(tpr_a - tpr_b):.2%}.")
    else:
        st.warning(f"Ajusta los umbrales para igualar las Tasas de Verdaderos Positivos. Diferencia actual: {abs(tpr_a - tpr_b):.2%}")

def run_calibration_simulation():
    st.markdown("#### Simulación de Calibración")
    st.write("Observa cómo las puntuaciones brutas de un modelo (línea azul) pueden estar mal calibradas y cómo técnicas como **Platt Scaling** (logística) o **Regresión Isotónica** las ajustan para que se alineen mejor con la realidad (línea diagonal perfecta).")
    
    np.random.seed(0)
    # Generar puntuaciones de modelo mal calibradas
    raw_scores = np.sort(np.random.rand(100))
    true_probs = 1 / (1 + np.exp(-(raw_scores * 4 - 2))) # Una curva sigmoide para simular la realidad
    
    # Platt Scaling
    platt = LogisticRegression()
    platt.fit(raw_scores.reshape(-1, 1), (true_probs > 0.5).astype(int))
    calibrated_platt = platt.predict_proba(raw_scores.reshape(-1, 1))[:, 1]

    # Isotonic Regression
    isotonic = IsotonicRegression(out_of_bounds='clip')
    isotonic.fit(raw_scores, true_probs)
    calibrated_isotonic = isotonic.predict(raw_scores)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'k--', label='Calibración Perfecta')
    ax.plot(raw_scores, true_probs, 'b-', label='Puntuaciones Originales (Mal Calibradas)')
    ax.plot(raw_scores, calibrated_platt, 'g:', label='Calibrado con Platt Scaling')
    ax.plot(raw_scores, calibrated_isotonic, 'r-.', label='Calibrado con Regresión Isotónica')
    ax.set_title("Comparación de Técnicas de Calibración")
    ax.set_xlabel("Probabilidad Predicha")
    ax.set_ylabel("Fracción Real de Positivos")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.info("El objetivo es que las líneas de las puntuaciones se acerquen lo más posible a la línea diagonal punteada, que representa una calibración perfecta.")

def run_rejection_simulation():
    st.markdown("#### Simulación de Clasificación con Rechazo")
    st.write("Establece un umbral de confianza. Las predicciones con una confianza (probabilidad) muy alta o muy baja se automatizan. Las que caen en la 'zona de incertidumbre' se rechazan y se envían a un humano para su revisión.")

    np.random.seed(1)
    scores = np.random.beta(2, 2, 200) # Probabilidades entre 0 y 1

    low_thresh = st.slider("Umbral de Confianza Inferior", 0.0, 0.5, 0.25)
    high_thresh = st.slider("Umbral de Confianza Superior", 0.5, 1.0, 0.75)

    automated_low = scores[scores <= low_thresh]
    automated_high = scores[scores >= high_thresh]
    rejected = scores[(scores > low_thresh) & (scores < high_thresh)]

    fig, ax = plt.subplots()
    ax.hist(automated_low, bins=10, range=(0,1), color='green', alpha=0.7, label=f'Decisión Automática (Baja Prob, n={len(automated_low)})')
    ax.hist(rejected, bins=10, range=(0,1), color='orange', alpha=0.7, label=f'Rechazado a Humano (n={len(rejected)})')
    ax.hist(automated_high, bins=10, range=(0,1), color='blue', alpha=0.7, label=f'Decisión Automática (Alta Prob, n={len(automated_high)})')
    ax.set_title("Distribución de Decisiones")
    ax.set_xlabel("Puntuación de Probabilidad del Modelo")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    st.pyplot(fig)
    
    coverage = (len(automated_low) + len(automated_high)) / len(scores)
    st.metric("Tasa de Cobertura (Automatización)", f"{coverage:.1%}")
    st.info("Ajusta los umbrales para ver cómo cambia la cantidad de casos que se automatizan vs. los que requieren revisión humana. Un rango de rechazo más amplio aumenta la equidad en casos difíciles a costa de una menor automatización.")

#======================================================================
# --- FAIRNESS INTERVENTION PLAYBOOK ---
#======================================================================

def causal_fairness_toolkit():
    st.header("🛡️ Toolkit de Equidad Causal")
    
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **Análisis Causal** va más allá de las correlaciones para entender el *porqué* de las disparidades. Es como ser un detective que no solo ve que dos eventos ocurren juntos, sino que reconstruye la cadena de causa y efecto que los conecta. Esto nos ayuda a aplicar soluciones que atacan la raíz del problema, en lugar de solo maquillar los síntomas.
        """)
    
    if 'causal_report' not in st.session_state:
        st.session_state.causal_report = {}

    tab1, tab2, tab3, tab4 = st.tabs(["Identificación", "Análisis Contrafactual", "Diagrama Causal", "Inferencia Causal"])

    with tab1:
        st.subheader("Marco de Identificación de Mecanismos de Discriminación")
        st.info("Identifica las posibles causas raíz del sesgo en tu aplicación.")
        
        with st.expander("Definición de Discriminación Directa"):
            st.write("Ocurre cuando un atributo protegido (como la raza o el género) es usado explícitamente para tomar una decisión. Es el tipo de sesgo más obvio.")
        st.text_area("1. ¿El atributo protegido influye directamente en la decisión?", placeholder="Ejemplo: Un modelo de contratación que asigna una puntuación menor a las candidatas mujeres de forma explícita.", key="causal_q1")
        
        with st.expander("Definición de Discriminación Indirecta"):
            st.write("Ocurre cuando un atributo protegido afecta a un factor intermedio que sí es legítimo para la decisión. El sesgo se transmite a través de esta variable mediadora.")
        st.text_area("2. ¿El atributo protegido afecta a factores intermedios legítimos?", placeholder="Ejemplo: El género puede influir en tener 'pausas en la carrera' (para el cuidado de hijos), y el modelo penaliza estas pausas, afectando indirectamente a las mujeres.", key="causal_q2")

        with st.expander("Definición de Discriminación por Proxy"):
            st.write("Ocurre cuando una variable aparentemente neutral está tan correlacionada con un atributo protegido que funciona como un sustituto (un 'proxy') de este.")
        st.text_area("3. ¿Las decisiones dependen de variables correlacionadas con atributos protegidos?", placeholder="Ejemplo: En un modelo de crédito, usar el código postal como predictor puede ser un proxy de la raza debido a la segregación residencial histórica.", key="causal_q3")

    with tab2:
        st.subheader("Metodología Práctica de Equidad Contrafactual")
        st.info("Analiza, cuantifica y mitiga el sesgo contrafactual en tu modelo.")
        with st.expander("💡 Ejemplo Interactivo: Simulación Contrafactual"):
            st.write("Observa cómo un cambio en un atributo protegido puede alterar la decisión de un modelo, revelando un sesgo causal.")
            puntaje_base = 650
            decision_base = "Rechazado"
            st.write(f"**Caso Base:** Solicitante del **Grupo B** con un puntaje de **{puntaje_base}**. Decisión del modelo: **{decision_base}**.")
            if st.button("Ver Contrafactual (Cambiar a Grupo A)", key="cf_button"):
                puntaje_cf = 710
                decision_cf = "Aprobado"
                st.info(f"**Escenario Contrafactual:** Mismo solicitante, pero del **Grupo A**. El modelo ahora predice un puntaje de **{puntaje_cf}** y la decisión es: **{decision_cf}**.")
                st.warning("**Análisis:** El cambio en el atributo protegido alteró la decisión, lo que sugiere que el modelo ha aprendido una dependencia causal problemática.")
        
        with st.container(border=True):
            st.markdown("##### Paso 1: Análisis de Equidad Contrafactual")
            st.text_area("1.1 Formular Consultas Contrafactuales", placeholder="Ejemplo: Para un solicitante de préstamo rechazado, ¿cuál habría sido el resultado si su raza fuera diferente, manteniendo constantes los ingresos y el historial crediticio?", key="causal_q4")
            st.text_area("1.2 Identificar Rutas Causales (Justas vs. Injustas)", placeholder="Ejemplo: La ruta Raza → Código Postal → Decisión de Préstamo es injusta porque el código postal es un proxy. La ruta Nivel Educativo → Ingresos → Decisión de Préstamo es considerada justa.", key="causal_q5")
            st.text_area("1.3 Medir Disparidades y Documentar", placeholder="Ejemplo: El 15% de los solicitantes del grupo desfavorecido habrían sido aprobados en el escenario contrafactual. Esto indica una violación de equidad contrafactual.", key="causal_q6")
        with st.container(border=True):
            st.markdown("##### Paso 2: Análisis Específico de Rutas")
            st.text_area("2.1 Descomponer y Clasificar Rutas", placeholder="Ejemplo: Ruta 1 (proxy de código postal) clasificada como INJUSTA. Ruta 2 (mediada por ingresos) clasificada como JUSTA.", key="causal_q7")
            st.text_area("2.2 Cuantificar Contribución y Documentar", placeholder="Ejemplo: La ruta del código postal representa el 60% de la disparidad observada. Razón: Refleja sesgos históricos de segregación residencial.", key="causal_q8")
        with st.container(border=True):
            st.markdown("##### Paso 3: Diseño de Intervención")
            st.selectbox("3.1 Seleccionar Enfoque de Intervención", ["Nivel de Datos", "Nivel de Modelo", "Post-procesamiento"], key="causal_q9")
            st.text_area("3.2 Implementar y Monitorear", placeholder="Ejemplo: Se aplicó una transformación a la característica de código postal. La disparidad contrafactual se redujo en un 50%.", key="causal_q10")

    with tab3:
        st.subheader("Enfoque de Diagrama Causal Inicial")
        st.info("Esboza diagramas para visualizar las relaciones causales y documentar tus supuestos.")
        with st.expander("💡 Simulador de Diagrama Causal"):
            st.write("Construye un diagrama causal simple seleccionando las relaciones entre variables. Esto te ayuda a visualizar tus hipótesis sobre cómo funciona el sesgo.")
            
            nodos = ["Género", "Educación", "Ingresos", "Decisión_Préstamo"]
            relaciones_posibles = [
                ("Género", "Educación"), ("Género", "Ingresos"),
                ("Educación", "Ingresos"), ("Ingresos", "Decisión_Préstamo"),
                ("Educación", "Decisión_Préstamo"), ("Género", "Decisión_Préstamo")
            ]
            
            st.multiselect(
                "Selecciona las relaciones causales (Causa → Efecto):",
                options=[f"{causa} → {efecto}" for causa, efecto in relaciones_posibles],
                key="causal_q11_relations"
            )
            
            if st.session_state.causal_q11_relations:
                dot_string = "digraph { rankdir=LR; "
                for rel in st.session_state.causal_q11_relations:
                    causa, efecto = rel.split(" → ")
                    dot_string += f'"{causa}" -> "{efecto}"; '
                dot_string += "}"
                st.graphviz_chart(dot_string)

        st.markdown("""
        **Convenciones de Anotación:**
        - **Nodos (variables):** Atributos Protegidos, Características, Resultados.
        - **Flechas Causales (→):** Relación causal asumida.
        - **Flechas de Correlación (<-->):** Correlación sin causalidad directa conocida.
        - **Incertidumbre (?):** Relación causal hipotética o débil.
        - **Ruta Problemática (!):** Ruta que consideras una fuente de inequidad.
        """)
        st.text_area("Documentación de Supuestos y Rutas", placeholder="Ruta (!): Raza -> Nivel de Ingresos -> Decisión.\nSupuesto: Las disparidades históricas de ingresos vinculadas a la raza afectan la capacidad de préstamo.", height=200, key="causal_q11")

    with tab4:
        st.subheader("Inferencia Causal con Datos Limitados")
        st.info("Métodos prácticos para estimar efectos causales cuando los datos son imperfectos.")
        
        with st.expander("🔍 Definición: Emparejamiento (Matching)"):
            st.write("Compara individuos de un grupo de 'tratamiento' con individuos muy similares de un grupo de 'control'. Al comparar 'gemelos' estadísticos, se aísla el efecto del tratamiento. En equidad, el 'tratamiento' puede ser pertenecer a un grupo demográfico.")
        with st.expander("💡 Ejemplo Interactivo: Simulación de Emparejamiento"):
            run_matching_simulation()

        with st.expander("🔍 Definición: Variables Instrumentales (IV)"):
            st.write("Usa una variable 'instrumento' que afecta al tratamiento, pero no directamente al resultado, para desenredar la correlación de la causalidad. Es como encontrar un interruptor que solo enciende una luz específica en un panel complicado, permitiéndote saber qué hace exactamente esa luz.")
            st.graphviz_chart("""
            digraph {
                rankdir=LR;
                Z [label="Instrumento (Z)"];
                A [label="Atributo Protegido (A)"];
                Y [label="Resultado (Y)"];
                U [label="Factor de Confusión No Observado (U)", style=dashed];
                Z -> A;
                A -> Y;
                U -> A [style=dashed];
                U -> Y [style=dashed];
            }
            """)
            st.write("**Ejemplo:** Para medir el efecto causal de la educación (A) en los ingresos (Y), se puede usar la proximidad a una universidad (Z) como instrumento. La proximidad afecta la educación, pero no directamente a los ingresos (excepto a través de la educación).")

        with st.expander("🔍 Definición: Regresión por Discontinuidad (RD)"):
            st.write("Aprovecha un umbral o punto de corte en la asignación de un tratamiento. Al comparar a quienes están justo por encima y por debajo del umbral, se puede estimar el efecto causal del tratamiento, asumiendo que estos individuos son muy similares en otros aspectos.")
        with st.expander("💡 Ejemplo Interactivo: Simulación de RD"):
            run_rd_simulation()

        with st.expander("🔍 Definición: Diferencia en Diferencias (DiD)"):
            st.write("Compara el cambio en los resultados a lo largo del tiempo entre un grupo de tratamiento y un grupo de control. La 'diferencia en diferencias' entre los grupos antes y después del tratamiento estima el efecto causal.")
        with st.expander("💡 Ejemplo Interactivo: Simulación de DiD"):
            run_did_simulation()

    # --- Sección de Reporte ---
    st.markdown("---")
    st.header("Generar Reporte del Toolkit Causal")
    if st.button("Generar Reporte Causal", key="gen_causal_report"):
        # Recopilar datos del session_state
        report_data = {
            "Identificación de Mecanismos": {
                "Discriminación Directa": st.session_state.get('causal_q1', 'No completado'),
                "Discriminación Indirecta": st.session_state.get('causal_q2', 'No completado'),
                "Discriminación por Proxy": st.session_state.get('causal_q3', 'No completado'),
            },
            "Análisis Contrafactual": {
                "Consultas Contrafactuales": st.session_state.get('causal_q4', 'No completado'),
                "Identificación de Rutas Causales": st.session_state.get('causal_q5', 'No completado'),
                "Medición de Disparidades": st.session_state.get('causal_q6', 'No completado'),
                "Descomposición de Rutas": st.session_state.get('causal_q7', 'No completado'),
                "Cuantificación de Contribución": st.session_state.get('causal_q8', 'No completado'),
                "Enfoque de Intervención Seleccionado": st.session_state.get('causal_q9', 'No completado'),
                "Plan de Implementación y Monitoreo": st.session_state.get('causal_q10', 'No completado'),
            },
            "Diagrama Causal": {
                "Relaciones Seleccionadas": ", ".join(st.session_state.get('causal_q11_relations', [])),
                "Documentación de Supuestos": st.session_state.get('causal_q11', 'No completado'),
            }
        }

        # Formatear reporte en Markdown
        report_md = "# Reporte del Toolkit de Equidad Causal\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.causal_report_md = report_md
        st.success("¡Reporte generado exitosamente! Puedes verlo a continuación y descargarlo.")

    if 'causal_report_md' in st.session_state and st.session_state.causal_report_md:
        st.subheader("Vista Previa del Reporte")
        st.markdown(st.session_state.causal_report_md)
        st.download_button(
            label="Descargar Reporte Causal",
            data=st.session_state.causal_report_md,
            file_name="reporte_equidad_causal.md",
            mime="text/markdown"
        )


def preprocessing_fairness_toolkit():
    st.header("🧪 Toolkit de Equidad en Pre-procesamiento")
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **Pre-procesamiento** consiste en "limpiar" los datos *antes* de que el modelo aprenda de ellos. Es como preparar los ingredientes para una receta: si sabes que algunos ingredientes están sesgados (por ejemplo, demasiado salados), los ajustas antes de cocinar para asegurar que el plato final sea equilibrado.
        """)
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Análisis de Representación", "Detección de Correlación", "Calidad de Etiquetas", "Re-ponderación y Re-muestreo", "Transformación", "Generación de Datos"])

    with tab1:
        st.subheader("Análisis de Representación Multidimensional")
        with st.expander("🔍 Definición Amigable"):
            st.write("Esto significa verificar si todos los grupos demográficos están representados de manera justa en tus datos. No solo miramos los grupos principales (como hombres y mujeres), sino también las intersecciones (como mujeres de una etnia específica).")
        
        with st.expander("💡 Ejemplo Interactivo: Brecha de Representación"):
            st.write("Compara la representación de dos grupos en tu conjunto de datos con su representación en una población de referencia (ej. el censo).")
            pop_a = 50
            pop_b = 50
            
            col1, col2 = st.columns(2)
            with col1:
                data_a = st.slider("Porcentaje del Grupo A en tus datos", 0, 100, 70)
            data_b = 100 - data_a
            
            df = pd.DataFrame({
                'Grupo': ['Grupo A', 'Grupo B'],
                'Población de Referencia': [pop_a, pop_b],
                'Tus Datos': [data_a, data_b]
            })

            with col2:
                st.write("Comparación:")
                st.dataframe(df.set_index('Grupo'))

            if abs(data_a - pop_a) > 10:
                st.warning(f"Hay una brecha de representación significativa. El Grupo A está sobrerrepresentado en tus datos en {data_a - pop_a} puntos porcentuales.")
            else:
                st.success("La representación en tus datos es similar a la población de referencia.")

        st.text_area("1. Comparación con Población de Referencia", placeholder="Ej: Nuestro conjunto de datos tiene un 70% del Grupo A y 30% del Grupo B, mientras que la población real es 50/50.", key="p1")
        st.text_area("2. Análisis de Representación Interseccional", placeholder="Ej: Las mujeres de minorías raciales constituyen solo el 3% de los datos, aunque representan el 10% de la población.", key="p2")
        st.text_area("3. Representación a través de Categorías de Resultados", placeholder="Ej: El grupo A constituye el 30% de las solicitudes pero solo el 10% de las aprobadas.", key="p3")

    with tab2:
        st.subheader("Detección de Patrones de Correlación")
        with st.expander("🔍 Definición Amigable"):
            st.write("Buscamos variables aparentemente neutrales que estén fuertemente conectadas a atributos protegidos. Por ejemplo, si un código postal se correlaciona fuertemente con la raza, el modelo podría usar el código postal para discriminar indirectamente.")
        
        with st.expander("💡 Ejemplo Interactivo: Detección de Proxy"):
            st.write("Visualiza cómo una variable 'Proxy' (ej. Código Postal) puede estar correlacionada tanto con un Atributo Protegido (ej. Grupo Demográfico) como con el Resultado (ej. Puntuación de Crédito).")
            np.random.seed(1)
            grupo = np.random.randint(0, 2, 100) # 0 o 1
            proxy = grupo * 20 + np.random.normal(50, 5, 100)
            resultado = proxy * 5 + np.random.normal(100, 20, 100)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            ax1.scatter(grupo, proxy, c=grupo, cmap='coolwarm', alpha=0.7)
            ax1.set_title("Atributo Protegido vs. Variable Proxy")
            ax1.set_xlabel("Grupo Demográfico (0 o 1)")
            ax1.set_ylabel("Valor del Proxy (ej. Código Postal)")
            ax1.grid(True, linestyle='--', alpha=0.5)

            ax2.scatter(proxy, resultado, c=grupo, cmap='coolwarm', alpha=0.7)
            ax2.set_title("Variable Proxy vs. Resultado")
            ax2.set_xlabel("Valor del Proxy (ej. Código Postal)")
            ax2.set_ylabel("Resultado (ej. Puntuación de Crédito)")
            ax2.grid(True, linestyle='--', alpha=0.5)
            st.pyplot(fig)
            st.info("El gráfico de la izquierda muestra que el proxy está correlacionado con el grupo. El de la derecha muestra que el proxy predice el resultado. Por lo tanto, el modelo puede usar el proxy para discriminar.")

        st.text_area("1. Correlaciones Directas (Atributo Protegido ↔ Resultado)", placeholder="Ej: En los datos históricos, el género tiene una correlación de 0.3 con la decisión de contratación.", key="p4")
        st.text_area("2. Identificación de Variables Proxy (Atributo Protegido ↔ Característica)", placeholder="Ej: La característica 'asistencia a un club de ajedrez' está altamente correlacionada con el género masculino.", key="p5")

    with tab3:
        st.subheader("Evaluación de la Calidad de las Etiquetas")
        with st.expander("🔍 Definición Amigable"):
            st.write("Las 'etiquetas' son las respuestas correctas en tus datos de entrenamiento (ej. 'fue contratado', 'no pagó el préstamo'). Si estas etiquetas provienen de decisiones humanas pasadas que fueron sesgadas, tu modelo aprenderá ese mismo sesgo.")
        st.text_area("1. Sesgo Histórico en las Decisiones", placeholder="Ejemplo: Las etiquetas de 'promocionado' en nuestro conjunto de datos provienen de un período en el que la empresa tenía políticas de promoción sesgadas, por lo que las etiquetas en sí mismas son una fuente de sesgo.", key="p6")
        st.text_area("2. Sesgo del Anotador", placeholder="Ejemplo: El análisis del acuerdo entre anotadores muestra que los anotadores masculinos calificaron los mismos comentarios como 'tóxicos' con menos frecuencia que las anotadoras femeninas, lo que indica un sesgo en la etiqueta.", key="p7")
    
    with tab4:
        st.subheader("Técnicas de Re-ponderación y Re-muestreo")
        with st.expander("🔍 Definición Amigable"):
            st.write("**Re-ponderación:** Le da más 'peso' o importancia a las muestras de grupos subrepresentados. **Re-muestreo:** Cambia físicamente el conjunto de datos, ya sea duplicando muestras de grupos minoritarios (sobremuestreo) o eliminando muestras de grupos mayoritarios (submuestreo).")
        with st.expander("💡 Ejemplo Interactivo: Simulación de Sobremuestreo"):
            st.write("Observa cómo el sobremuestreo (resampling) puede equilibrar un conjunto de datos con representación desigual.")
            np.random.seed(0)
            data_a = np.random.multivariate_normal([2, 2], [[1, .5], [.5, 1]], 100)
            data_b = np.random.multivariate_normal([4, 4], [[1, .5], [.5, 1]], 20)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            ax1.scatter(data_a[:, 0], data_a[:, 1], c='blue', label='Grupo A (n=100)', alpha=0.6)
            ax1.scatter(data_b[:, 0], data_b[:, 1], c='red', label='Grupo B (n=20)', alpha=0.6)
            ax1.set_title("Datos Originales (Desequilibrados)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            oversample_indices = np.random.choice(range(20), 80, replace=True)
            data_b_oversampled = np.vstack([data_b, data_b[oversample_indices]])
            ax2.scatter(data_a[:, 0], data_a[:, 1], c='blue', label='Grupo A (n=100)', alpha=0.6)
            ax2.scatter(data_b_oversampled[:, 0], data_b_oversampled[:, 1], c='red', label='Grupo B (n=100)', alpha=0.6, marker='x')
            ax2.set_title("Datos con Sobremuestreo del Grupo B")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            st.info("El gráfico de la derecha muestra cómo se han añadido nuevas muestras (marcadas con 'x') del Grupo B para igualar en número al Grupo A, lo que ayuda al modelo a aprender mejor sus patrones.")
        st.text_area("Criterios de Decisión: ¿Re-ponderar o Re-muestrear?", placeholder="Basado en mi auditoría y mi modelo, la mejor estrategia es...", key="p8")
        st.text_area("Consideración de Interseccionalidad", placeholder="Ejemplo: Para abordar la subrepresentación de mujeres de minorías, aplicaremos un sobremuestreo estratificado que garantice que este subgrupo específico alcance la paridad con otros.", key="p9")

    with tab5:
        st.subheader("Enfoques de Transformación de Distribución")
        with st.expander("🔍 Definición Amigable"):
            st.write("Esta técnica modifica directamente los valores de las características para romper las correlaciones problemáticas con los atributos protegidos. Es como 'recalibrar' una variable para que signifique lo mismo para todos los grupos.")
        st.text_area("1. Eliminación de Impacto Dispar", placeholder="Ej: 'Reparar' la característica 'código postal' para que su distribución sea la misma en todos los grupos raciales, eliminando su uso como proxy.", key="p10")
        st.text_area("2. Representaciones Justas (LFR, LAFTR)", placeholder="Ej: Usar un autoencoder adversario para aprender una representación de los perfiles de los solicitantes que no contenga información de género.", key="p11")
        st.text_area("3. Consideraciones de Interseccionalidad", placeholder="Mi estrategia de transformación se centrará en las intersecciones de género y etnia...", key="p12")

    with tab6:
        st.subheader("Generación de Datos con Conciencia de Equidad")
        with st.expander("🔍 Definición Amigable"):
            st.write("Cuando los datos son muy escasos o sesgados, podemos generar datos sintéticos (artificiales) para llenar los vacíos. Esto es especialmente útil para crear ejemplos de grupos interseccionales muy pequeños o para generar escenarios contrafactuales.")
        st.markdown("**¿Cuándo Generar Datos?:** Cuando hay subrepresentación severa o se necesitan ejemplos contrafactuales.")
        st.markdown("**Estrategias:** Generación Condicional, Aumentación Contrafactual.")
        st.text_area("Consideraciones de Interseccionalidad", placeholder="Ejemplo: Usaremos un modelo generativo condicionado en la intersección de edad y género para crear perfiles sintéticos de 'mujeres mayores en tecnología', un grupo ausente en nuestros datos.", key="p13")

    # --- Sección de Reporte ---
    st.markdown("---")
    st.header("Generar Reporte del Toolkit de Pre-procesamiento")
    if st.button("Generar Reporte de Pre-procesamiento", key="gen_preproc_report"):
        report_data = {
            "Análisis de Representación": {
                "Comparación con Población de Referencia": st.session_state.p1,
                "Análisis Interseccional": st.session_state.p2,
                "Representación en Resultados": st.session_state.p3,
            },
            "Detección de Correlación": {
                "Correlaciones Directas": st.session_state.p4,
                "Variables Proxy Identificadas": st.session_state.p5,
            },
            "Calidad de Etiquetas": {
                "Sesgo Histórico en Etiquetas": st.session_state.p6,
                "Sesgo del Anotador": st.session_state.p7,
            },
            "Re-ponderación y Re-muestreo": {
                "Decisión y Razón": st.session_state.p8,
                "Plan Interseccional": st.session_state.p9,
            },
            "Transformación de Distribución": {
                "Plan de Eliminación de Impacto Dispar": st.session_state.p10,
                "Plan de Representaciones Justas": st.session_state.p11,
                "Plan Interseccional": st.session_state.p12,
            },
            "Generación de Datos": {
                "Plan de Generación Interseccional": st.session_state.p13,
            }
        }
        
        report_md = "# Reporte del Toolkit de Equidad en Pre-procesamiento\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.preproc_report_md = report_md
        st.success("¡Reporte generado exitosamente!")

    if 'preproc_report_md' in st.session_state and st.session_state.preproc_report_md:
        st.subheader("Vista Previa del Reporte")
        st.markdown(st.session_state.preproc_report_md)
        st.download_button(
            label="Descargar Reporte de Pre-procesamiento",
            data=st.session_state.preproc_report_md,
            file_name="reporte_preprocesamiento.md",
            mime="text/markdown"
        )

def inprocessing_fairness_toolkit():
    st.header("⚙️ Toolkit de Equidad en In-procesamiento")
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **In-procesamiento** implica modificar el algoritmo de aprendizaje del modelo para que la equidad sea uno de sus objetivos, junto con la precisión. Es como enseñarle a un chef a cocinar no solo para que la comida sea deliciosa, sino también para que sea nutricionalmente equilibrada, haciendo de la nutrición una parte central de la receta.
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Objetivos y Restricciones", "Debiasing Adversario", "Optimización Multiobjetivo", "Patrones de Código"])

    with tab1:
        st.subheader("Objetivos y Restricciones de Equidad")
        with st.expander("🔍 Definición Amigable"):
            st.write("Esto significa incorporar 'reglas de equidad' directamente en las matemáticas que el modelo utiliza para aprender. En lugar de solo buscar la respuesta más precisa, el modelo también debe asegurarse de no violar estas reglas.")
        
        st.markdown("**Métodos Lagrangianos:**")
        with st.expander("🔍 Definición y Ejemplo"):
            st.write("Es una técnica matemática para convertir una 'restricción dura' (una regla que no se puede romper) en una 'penalización suave'. Imagina que estás entrenando a un robot para que sea rápido, pero no puede pasar de cierta velocidad. En lugar de un límite estricto, le das una penalización cada vez que se acerca al límite. Esto lo anima a mantenerse dentro de los límites de una manera más flexible.")
        st.latex(r''' \mathcal{L}(\theta, \lambda) = L(\theta) + \sum_{i=1}^{k} \lambda_i C_i(\theta) ''')
        st.text_area("Aplica a tu caso: ¿Qué restricción de equidad (ej. diferencia máxima de aprobación) quieres implementar?", key="in_q1")

        st.markdown("**Viabilidad y Compensaciones:**")
        with st.expander("🔍 Definición y Ejemplo"):
            st.write("No siempre es posible ser perfectamente justo y perfectamente preciso al mismo tiempo. A menudo, hay una 'compensación' (trade-off). Mejorar la equidad puede reducir ligeramente la precisión general, y viceversa. Es crucial entender este equilibrio.")
            st.write("**Ejemplo de Interseccionalidad:** Forzar la igualdad de resultados para todos los subgrupos (ej. mujeres latinas, hombres asiáticos) puede ser matemáticamente imposible o requerir un sacrificio de precisión tan grande que el modelo deja de ser útil.")
        st.text_area("Aplica a tu caso: ¿Qué compensación entre precisión y equidad estás dispuesto a aceptar?", key="in_q2")


    with tab2:
        st.subheader("Enfoques de Debiasing Adversario")
        with st.expander("🔍 Definición Amigable"):
            st.write("Imagina un juego entre dos IAs: un 'Predictor' que intenta hacer su trabajo (ej. evaluar currículums) y un 'Adversario' que intenta adivinar el atributo protegido (ej. el género del candidato) basándose en las decisiones del Predictor. El Predictor gana si hace buenas evaluaciones Y logra engañar al Adversario. Con el tiempo, el Predictor aprende a tomar decisiones sin basarse en información relacionada con el género.")
        
        st.markdown("**Arquitectura:**")
        with st.expander("💡 Simulador de Arquitectura Adversaria"):
            st.graphviz_chart("""
            digraph {
                rankdir=LR;
                node [shape=box, style=rounded];
                "Datos de Entrada (X)" -> "Predictor";
                "Predictor" -> "Predicción (Ŷ)";
                "Predictor" -> "Adversario" [label="Intenta engañar"];
                "Adversario" -> "Predicción de Atributo Protegido (Â)";
                "Atributo Protegido (A)" -> "Adversario" [style=dashed, label="Compara para aprender"];
            }
            """)
        st.text_area("Aplica a tu caso: Describe la arquitectura que usarías.", placeholder="Ej: Un predictor basado en BERT para analizar CVs y un adversario de 3 capas para predecir el género a partir de las representaciones internas.", key="in_q3")

        st.markdown("**Optimización:**")
        with st.expander("🔍 Definición y Ejemplo"):
             st.write("El entrenamiento puede ser inestable porque el Predictor y el Adversario tienen objetivos opuestos. Se necesitan técnicas especiales, como la 'inversión de gradiente', para que el Predictor aprenda a 'desaprender' el sesgo activamente.")
        st.text_area("Aplica a tu caso: ¿Qué desafíos de optimización prevés y cómo los abordarías?", placeholder="Ej: El adversario podría volverse demasiado fuerte al principio. Usaremos un aumento gradual de su peso en la función de pérdida.", key="in_q4")

    with tab3:
        st.subheader("Optimización Multiobjetivo para la Equidad")
        with st.expander("🔍 Definición Amigable"):
            st.write("En lugar de combinar la precisión y la equidad en una sola meta, este enfoque las trata como dos objetivos separados que deben equilibrarse. El objetivo es encontrar un conjunto de 'soluciones óptimas de Pareto', donde no se puede mejorar la equidad sin sacrificar algo de precisión, y viceversa.")
        with st.expander("💡 Ejemplo Interactivo: Frontera de Pareto"):
            st.write("Explora la **frontera de Pareto**, que visualiza la compensación (trade-off) entre la precisión de un modelo y su equidad. No se puede mejorar uno sin empeorar el otro.")
            
            np.random.seed(10)
            accuracy = np.linspace(0.80, 0.95, 20)
            fairness_score = 1 - np.sqrt(accuracy - 0.79) + np.random.normal(0, 0.02, 20)
            fairness_score = np.clip(fairness_score, 0.5, 1.0)
            
            fig, ax = plt.subplots()
            ax.scatter(accuracy, fairness_score, c=accuracy, cmap='viridis', label='Modelos Posibles')
            ax.set_title("Frontera de Pareto: Equidad vs. Precisión")
            ax.set_xlabel("Precisión del Modelo")
            ax.set_ylabel("Puntuación de Equidad")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            st.info("Cada punto representa un modelo diferente. Los modelos en el borde superior derecho son 'óptimos'. La elección de qué punto usar depende de las prioridades de tu proyecto.")
        st.text_area("Aplica a tu caso: ¿Cuáles son los múltiples objetivos que necesitas equilibrar?", placeholder="Ej: 1. Maximizar la precisión en la predicción de impago. 2. Minimizar la diferencia en la tasa de aprobación entre grupos demográficos. 3. Minimizar la diferencia en la tasa de falsos negativos.", key="in_q5")

    with tab4:
        st.subheader("Catálogo de Patrones de Implementación")
        with st.expander("🔍 Definición Amigable"):
            st.write("Estos son fragmentos de código o pseudocódigo que muestran cómo se ven en la práctica las técnicas de in-procesamiento. Sirven como plantillas reutilizables para implementar la equidad en tu propio código.")
        st.code("""
# Ejemplo de una función de pérdida con regularización de equidad
def fairness_regularized_loss(original_loss, predictions, protected_attribute):
  # Calcula una penalización basada en la disparidad de las predicciones
  fairness_penalty = calculate_disparity(predictions, protected_attribute)
  
  # Combina la pérdida original con la penalización de equidad
  # lambda controla la importancia que se le da a la equidad
  return original_loss + lambda * fairness_penalty
        """, language="python")

    # --- Sección de Reporte ---
    st.markdown("---")
    st.header("Generar Reporte del Toolkit de In-procesamiento")
    if st.button("Generar Reporte de In-procesamiento", key="gen_inproc_report"):
        report_data = {
            "Objetivos y Restricciones": {
                "Restricción de Equidad": st.session_state.in_q1,
                "Análisis de Compensaciones": st.session_state.in_q2,
            },
            "Debiasing Adversario": {
                "Descripción de la Arquitectura": st.session_state.in_q3,
                "Plan de Optimización": st.session_state.in_q4,
            },
            "Optimización Multiobjetivo": {
                "Objetivos a Equilibrar": st.session_state.in_q5,
            }
        }
        
        report_md = "# Reporte del Toolkit de Equidad en In-procesamiento\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.inproc_report_md = report_md
        st.success("¡Reporte generado exitosamente!")

    if 'inproc_report_md' in st.session_state and st.session_state.inproc_report_md:
        st.subheader("Vista Previa del Reporte")
        st.markdown(st.session_state.inproc_report_md)
        st.download_button(
            label="Descargar Reporte de In-procesamiento",
            data=st.session_state.inproc_report_md,
            file_name="reporte_inprocesamiento.md",
            mime="text/markdown"
        )

def postprocessing_fairness_toolkit():
    st.header("📊 Toolkit de Equidad en Post-procesamiento")
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **Post-procesamiento** consiste en ajustar las predicciones de un modelo *después* de que ya ha sido entrenado. Es como un editor que revisa un texto ya escrito para corregir sesgos o errores. El modelo original no cambia, solo se ajusta su resultado final para que sea más justo.
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Optimización de Umbrales", "Calibración", "Transformación de Predicción", "Clasificación con Rechazo"])

    with tab1:
        st.subheader("Técnicas de Optimización de Umbrales")
        with st.expander("💡 Ejemplo Interactivo"):
             run_threshold_simulation()
        st.info("Ajusta los umbrales de clasificación después del entrenamiento para satisfacer definiciones de equidad específicas.")
        st.text_area("Aplica a tu caso: ¿Qué criterio de equidad usarás y cómo planeas analizar las compensaciones?", placeholder="1. Criterio: Igualdad de Oportunidades.\n2. Cálculo: Encontraremos umbrales que igualen la TPR en un set de validación.\n3. Despliegue: Usaremos un proxy del grupo demográfico ya que no podemos usar el atributo protegido en producción.", key="po_q1")

    with tab2:
        st.subheader("Guía Práctica de Calibración para la Equidad")
        with st.expander("🔍 Definición Amigable"):
            st.write("La **calibración** asegura que una predicción de '80% de probabilidad' signifique lo mismo para todos los grupos demográficos. Si para un grupo significa un 95% de probabilidad real y para otro un 70%, el modelo está mal calibrado y es injusto.")
        with st.expander("💡 Ejemplo Interactivo: Simulación de Calibración"):
            run_calibration_simulation()
        
        with st.expander("Definición: Platt Scaling y Regresión Isotónica"):
            st.write("**Platt Scaling:** Es una técnica simple que usa un modelo logístico para 'reajustar' las puntuaciones de tu modelo y convertirlas en probabilidades bien calibradas. Es como aplicar una curva de corrección suave.")
            st.write("**Regresión Isotónica:** Es un método más flexible y no paramétrico que ajusta las puntuaciones a través de una función escalonada. Es potente pero puede sobreajustarse si no se tiene suficientes datos.")
        st.text_area("Aplica a tu caso: ¿Cómo evaluarás y corregirás la calibración?", placeholder="1. Evaluación: Usaremos diagramas de fiabilidad y la métrica ECE por grupo.\n2. Método: Probaremos con Platt Scaling por grupo, ya que es robusto y fácil de implementar.", key="po_q2")

    with tab3:
        st.subheader("Métodos de Transformación de Predicción")
        with st.expander("🔍 Definición Amigable"):
            st.write("Estas son técnicas más avanzadas que la simple optimización de umbrales. Modifican las puntuaciones del modelo de formas más complejas para cumplir con criterios de equidad, especialmente cuando no se puede re-entrenar el modelo.")
        
        with st.expander("Definición: Funciones de Transformación Aprendidas"):
            st.write("En lugar de un ajuste simple, se 'aprende' una función matemática óptima que transforma las puntuaciones sesgadas en puntuaciones justas, minimizando la pérdida de información útil.")
        with st.expander("Definición: Alineación de Distribución"):
            st.write("Asegura que la distribución de las puntuaciones (el 'histograma' de las predicciones) sea similar para todos los grupos demográficos. Esto es útil para lograr la paridad estadística.")
        with st.expander("Definición: Transformaciones de Puntuación Justas"):
            st.write("Modifica las puntuaciones para cumplir con la equidad, pero con una regla importante: el orden relativo de los individuos dentro de un mismo grupo debe mantenerse. Si la persona A era mejor que B en un grupo, debe seguir siéndolo después de la transformación.")
        
        st.text_area("Aplica a tu caso: ¿Qué método de transformación es más adecuado y por qué?", placeholder="Ejemplo: Usaremos alineación de distribución mediante mapeo de cuantiles para asegurar que las distribuciones de riesgo de crédito sean comparables entre grupos, ya que nuestro objetivo es la paridad demográfica.", key="po_q3")

    with tab4:
        st.subheader("Clasificación con Opción de Rechazo")
        with st.expander("🔍 Definición Amigable"):
            st.write("En lugar de forzar al modelo a tomar una decisión en casos difíciles o ambiguos (donde es más probable que cometa errores injustos), esta técnica identifica esos casos y los 'rechaza', enviándolos a un experto humano para que tome la decisión final.")
        with st.expander("💡 Ejemplo Interactivo: Simulación de Rechazo"):
            run_rejection_simulation()
            
        with st.expander("Definición: Umbrales de rechazo basados en confianza"):
            st.write("Se definen 'zonas de confianza'. Si la probabilidad predicha por el modelo es muy alta (ej. >90%) o muy baja (ej. <10%), la decisión se automatiza. Si cae en el medio, se rechaza para revisión humana.")
        with st.expander("Definición: Clasificación selectiva"):
            st.write("Es el marco formal para decidir qué porcentaje de casos automatizar. Permite optimizar el equilibrio entre la 'cobertura' (cuántos casos se deciden automáticamente) y la equidad.")
        with st.expander("Definición: Modelos de colaboración Humano-IA"):
            st.write("No basta con rechazar un caso. Es crucial diseñar cómo se presenta la información al humano para no introducir nuevos sesgos. El objetivo es una colaboración donde la IA y el humano juntos tomen decisiones más justas que por separado.")
        
        st.text_area("Aplica a tu caso: ¿Cómo diseñarías un sistema de rechazo?", placeholder="Ejemplo: Rechazaremos las solicitudes de préstamo con probabilidades entre 40% y 60% para revisión manual. La interfaz para el revisor mostrará los datos clave sin revelar el grupo demográfico para evitar sesgos humanos.", key="po_q4")

    # --- Sección de Reporte ---
    st.markdown("---")
    st.header("Generar Reporte del Toolkit de Post-procesamiento")
    if st.button("Generar Reporte de Post-procesamiento", key="gen_postproc_report"):
        report_data = {
            "Optimización de Umbrales": {"Plan de Implementación": st.session_state.po_q1},
            "Calibración": {"Plan de Calibración": st.session_state.po_q2},
            "Transformación de Predicción": {"Método de Transformación Seleccionado": st.session_state.po_q3},
            "Clasificación con Rechazo": {"Diseño del Sistema de Rechazo": st.session_state.po_q4}
        }
        
        report_md = "# Reporte del Toolkit de Equidad en Post-procesamiento\n\n"
        for section, content in report_data.items():
            report_md += f"## {section}\n"
            for key, value in content.items():
                report_md += f"**{key}:**\n{value}\n\n"
        
        st.session_state.postproc_report_md = report_md
        st.success("¡Reporte generado exitosamente!")

    if 'postproc_report_md' in st.session_state and st.session_state.postproc_report_md:
        st.subheader("Vista Previa del Reporte")
        st.markdown(st.session_state.postproc_report_md)
        st.download_button(
            label="Descargar Reporte de Post-procesamiento",
            data=st.session_state.postproc_report_md,
            file_name="reporte_postprocesamiento.md",
            mime="text/markdown"
        )


def intervention_playbook():
    st.sidebar.title("Navegación del Playbook de Intervención")
    selection = st.sidebar.radio(
        "Ir a:",
        ["Playbook Principal", "Toolkit Causal", "Toolkit de Pre-procesamiento", "Toolkit de In-procesamiento", "Toolkit de Post-procesamiento"],
        key="intervention_nav"
    )
    
    if selection == "Playbook Principal":
        st.header("📖 Playbook de Intervención de Equidad")
        st.info("Este playbook integra los cuatro toolkits en un flujo de trabajo cohesivo, guiando a los desarrolladores desde la identificación del sesgo hasta la implementación de soluciones efectivas.")
        with st.expander("Guía de Implementación"):
            st.write("Explica cómo usar el playbook, con comentarios sobre puntos de decisión clave, evidencia de apoyo y riesgos identificados.")
        with st.expander("Estudio de Caso"):
            st.write("Demuestra la aplicación del playbook a un problema de equidad típico, mostrando cómo los resultados de cada componente informan al siguiente.")
        with st.expander("Marco de Validación"):
            st.write("Proporciona orientación sobre cómo los equipos de implementación pueden verificar la efectividad de su proceso de auditoría.")
        with st.expander("Equidad Interseccional"):
            st.write("Consideración explícita de la equidad interseccional en cada componente del playbook.")
    elif selection == "Toolkit Causal":
        causal_fairness_toolkit()
    elif selection == "Toolkit de Pre-procesamiento":
        preprocessing_fairness_toolkit()
    elif selection == "Toolkit de In-procesamiento":
        inprocessing_fairness_toolkit()
    elif selection == "Toolkit de Post-procesamiento":
        postprocessing_fairness_toolkit()

#======================================================================
# --- FAIRNESS AUDIT PLAYBOOK ---
#======================================================================

def audit_playbook():
    st.sidebar.title("Navegación del Playbook de Auditoría")
    page = st.sidebar.radio("Ir a", [
        "Cómo Navegar este Playbook",
        "Evaluación del Contexto Histórico",
        "Selección de Definición de Equidad",
        "Identificación de Fuentes de Sesgo",
        "Métricas Comprensivas de Equidad"
    ], key="audit_nav")

    if page == "Cómo Navegar este Playbook":
        st.header("Cómo Navegar Este Playbook")
        # ... (Contenido original) ...
    elif page == "Evaluación del Contexto Histórico":
        st.header("Herramienta de Evaluación del Contexto Histórico")
        # ... (Contenido original) ...
    elif page == "Selección de Definición de Equidad":
        st.header("Herramienta de Selección de Definición de Equidad")
        # ... (Contenido original) ...
    elif page == "Identificación de Fuentes de Sesgo":
        st.header("Herramienta de Identificación de Fuentes de Sesgo")
        # ... (Contenido de esta sección) ...
    elif page == "Métricas Comprensivas de Equidad":
        st.header("Métricas Comprensivas de Equidad (CFM)")
        # ... (Contenido de esta sección) ...


# --- NAVEGACIÓN PRINCIPAL ---
st.sidebar.title("Selección de Playbook")
playbook_choice = st.sidebar.selectbox(
    "Elige el playbook que quieres usar:",
    ["Fairness Audit Playbook", "Fairness Intervention Playbook"]
)

st.title(playbook_choice)

if playbook_choice == "Fairness Audit Playbook":
    audit_playbook() 
else:
    intervention_playbook()
