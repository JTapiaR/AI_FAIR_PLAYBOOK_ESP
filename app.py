import streamlit as st
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def plot_simulation(df, title, x_label="Puntuación del Modelo", y_label="Densidad"):
    """Función auxiliar para graficar distribuciones."""
    fig, ax = plt.subplots()
    for group in df['Grupo'].unique():
        df[df['Grupo'] == group]['Puntuación'].plot(kind='density', ax=ax, label=group)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    st.pyplot(fig)

def run_threshold_simulation():
    """Simulación para optimización de umbrales en post-procesamiento."""
    st.markdown("#### Simulación de Optimización de Umbrales")
    st.write("Ajusta los umbrales de decisión para dos grupos y observa cómo cambian las tasas de error para lograr la **Igualdad de Oportunidades** (tasas de verdaderos positivos iguales).")

    # Generar datos simulados
    np.random.seed(42)
    scores_a_pos = np.random.normal(0.7, 0.15, 80)
    scores_a_neg = np.random.normal(0.4, 0.15, 120)
    scores_b_pos = np.random.normal(0.6, 0.15, 50)
    scores_b_neg = np.random.normal(0.3, 0.15, 150)

    df_a = pd.DataFrame({'Puntuación': np.concatenate([scores_a_pos, scores_a_neg]), 'Real': [1]*80 + [0]*120, 'Grupo': 'Grupo A'})
    df_b = pd.DataFrame({'Puntuación': np.concatenate([scores_b_pos, scores_b_neg]), 'Real': [1]*50 + [0]*150, 'Grupo': 'Grupo B'})
    
    col1, col2 = st.columns(2)
    with col1:
        threshold_a = st.slider("Umbral para Grupo A", 0.0, 1.0, 0.5, key="sim_thresh_a")
    with col2:
        threshold_b = st.slider("Umbral para Grupo B", 0.0, 1.0, 0.5, key="sim_thresh_b")

    # Calcular métricas
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

def run_matching_simulation():
    st.markdown("#### Simulación de Emparejamiento (Matching)")
    st.write("Compara dos grupos para estimar un efecto. El emparejamiento busca individuos 'similares' en ambos grupos para hacer una comparación más justa.")
    np.random.seed(0)
    # Grupo de Tratamiento
    x_treat = np.random.normal(5, 1.5, 50)
    y_treat = 2 * x_treat + 5 + np.random.normal(0, 2, 50)
    # Grupo de Control
    x_control = np.random.normal(3.5, 1.5, 50)
    y_control = 2 * x_control + np.random.normal(0, 2, 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    ax1.scatter(x_treat, y_treat, c='red', label='Tratamiento', alpha=0.7)
    ax1.scatter(x_control, y_control, c='blue', label='Control', alpha=0.7)
    ax1.set_title("Antes del Emparejamiento")
    ax1.set_xlabel("Característica (ej. Gasto previo)")
    ax1.set_ylabel("Resultado (ej. Compras)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Simular emparejamiento (encontrar puntos cercanos en X)
    matched_indices = [np.argmin(np.abs(x_c - x_treat)) for x_c in x_control]
    x_treat_matched = x_treat[matched_indices]
    y_treat_matched = y_treat[matched_indices]

    ax2.scatter(x_treat_matched, y_treat_matched, c='red', label='Tratamiento (Emparejado)', alpha=0.7)
    ax2.scatter(x_control, y_control, c='blue', label='Control', alpha=0.7)
    ax2.set_title("Después del Emparejamiento")
    ax2.set_xlabel("Característica (ej. Gasto previo)")
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    st.pyplot(fig)
    st.info("A la izquierda, los grupos no son directamente comparables. A la derecha, hemos seleccionado un subconjunto del grupo de tratamiento que es 'similar' al de control, permitiendo una estimación más justa del efecto del tratamiento.")

def run_rd_simulation():
    st.markdown("#### Simulación de Regresión por Discontinuidad (RD)")
    st.write("La RD se usa cuando un tratamiento se asigna basado en un umbral (ej. una calificación mínima para una beca). Se compara a los individuos justo por encima y por debajo del umbral para estimar el efecto del tratamiento.")
    np.random.seed(42)
    cutoff = st.slider("Valor del Umbral (Cutoff)", 40, 60, 50)
    
    x = np.linspace(0, 100, 200)
    y = 10 + 0.5 * x + np.random.normal(0, 5, 200)
    # Efecto del tratamiento
    treatment_effect = 15
    y[x >= cutoff] += treatment_effect

    fig, ax = plt.subplots()
    ax.scatter(x[x < cutoff], y[x < cutoff], c='blue', label='Control (No recibió tratamiento)')
    ax.scatter(x[x >= cutoff], y[x >= cutoff], c='red', label='Tratamiento')
    ax.axvline(x=cutoff, color='gray', linestyle='--', label=f'Umbral en {cutoff}')
    ax.set_title("Efecto del Tratamiento en el Umbral")
    ax.set_xlabel("Variable de asignación (ej. Calificación de examen)")
    ax.set_ylabel("Resultado (ej. Ingreso futuro)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    st.info(f"El 'salto' o discontinuidad en la línea de resultados en el punto del umbral ({cutoff}) es una estimación del efecto causal del tratamiento. Aquí, el efecto es de aproximadamente **{treatment_effect}** unidades.")

def run_did_simulation():
    st.markdown("#### Simulación de Diferencia en Diferencias (DiD)")
    st.write("DiD compara el cambio en los resultados a lo largo del tiempo entre un grupo que recibe un tratamiento y uno que no. Asume que ambos grupos habrían seguido 'tendencias paralelas' sin el tratamiento.")
    
    time = ['Antes', 'Después']
    # Grupo de Control: sin tratamiento
    control_outcomes = [20, 25] 
    # Grupo de Tratamiento: recibe tratamiento en el período 'Después'
    treat_outcomes = [15, 28]

    fig, ax = plt.subplots()
    ax.plot(time, control_outcomes, 'bo-', label='Grupo de Control (Observado)')
    ax.plot(time, treat_outcomes, 'ro-', label='Grupo de Tratamiento (Observado)')
    
    # Línea contrafactual: qué le habría pasado al grupo de tratamiento sin tratamiento
    counterfactual = [treat_outcomes[0], treat_outcomes[0] + (control_outcomes[1] - control_outcomes[0])]
    ax.plot(time, counterfactual, 'r--', label='Grupo de Tratamiento (Contrafactual)')
    
    ax.set_title("Estimación del Efecto del Tratamiento con DiD")
    ax.set_ylabel("Resultado")
    ax.set_ylim(10, 35)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)
    
    effect = treat_outcomes[1] - counterfactual[1]
    st.info(f"La línea punteada muestra la 'tendencia paralela' que el grupo de tratamiento habría seguido sin la intervención. La diferencia vertical entre la línea roja sólida y la punteada en el período 'Después' es el efecto del tratamiento, estimado en **{effect}** unidades.")


#======================================================================
# --- FAIRNESS INTERVENTION PLAYBOOK ---
#======================================================================

def causal_fairness_toolkit():
    st.header("🛡️ Toolkit de Equidad Causal")
    
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **Análisis Causal** va más allá de las correlaciones para entender el *porqué* de las disparidades. Es como ser un detective que no solo ve que dos eventos ocurren juntos, sino que reconstruye la cadena de causa y efecto que los conecta. Esto nos ayuda a aplicar soluciones que atacan la raíz del problema, en lugar de solo maquillar los síntomas.
        """)
    
    # Inicializar session_state para el reporte
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
        st.info("Examina las distribuciones demográficas para identificar brechas de representación.")
        st.text_area("1. Comparación con Población de Referencia", placeholder="Ej: Nuestro conjunto de datos tiene un 20% de mujeres en roles técnicos, mientras que el mercado laboral es del 35%.", key="p1")
        st.text_area("2. Análisis de Representación Interseccional", placeholder="Ej: Las mujeres de minorías raciales constituyen solo el 3% de los datos.", key="p2")
        st.text_area("3. Representación a través de Categorías de Resultados", placeholder="Ej: El grupo A constituye el 30% de las solicitudes pero solo el 10% de las aprobadas.", key="p3")

    with tab2:
        st.subheader("Detección de Patrones de Correlación")
        st.info("Identifica asociaciones problemáticas que podrían permitir la discriminación por proxy.")
        st.text_area("1. Correlaciones Directas (Atributo Protegido ↔ Resultado)", placeholder="Ej: El género tiene una correlación de 0.3 con la decisión de contratación.", key="p4")
        st.text_area("2. Identificación de Variables Proxy (Atributo Protegido ↔ Característica)", placeholder="Ej: La característica 'asistencia a un club de ajedrez' está altamente correlacionada con el género masculino.", key="p5")

    with tab3:
        st.subheader("Evaluación de la Calidad de las Etiquetas")
        st.info("Evalúa los sesgos potenciales en las etiquetas de entrenamiento.")
        st.text_area("1. Sesgo Histórico en las Decisiones", placeholder="Ej: Las etiquetas de 'promocionado' provienen de un período con políticas de promoción sesgadas.", key="p6")
        st.text_area("2. Sesgo del Anotador", placeholder="Ej: Los anotadores masculinos calificaron los mismos comentarios como 'tóxicos' con menos frecuencia que las anotadoras femeninas.", key="p7")
    
    with tab4:
        st.subheader("Técnicas de Re-ponderación y Re-muestreo")
        with st.expander("💡 Ejemplo Interactivo: Simulación de Sobremuestreo"):
            st.write("Observa cómo el sobremuestreo (resampling) puede equilibrar un conjunto de datos con representación desigual.")
            np.random.seed(0)
            data_a = np.random.multivariate_normal([2, 2], [[1, .5], [.5, 1]], 100)
            data_b = np.random.multivariate_normal([4, 4], [[1, .5], [.5, 1]], 20)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Gráfico Original
            ax1.scatter(data_a[:, 0], data_a[:, 1], c='blue', label='Grupo A (n=100)', alpha=0.6)
            ax1.scatter(data_b[:, 0], data_b[:, 1], c='red', label='Grupo B (n=20)', alpha=0.6)
            ax1.set_title("Datos Originales (Desequilibrados)")
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.5)

            # Gráfico con Sobremuestreo
            oversample_indices = np.random.choice(range(20), 80, replace=True)
            data_b_oversampled = np.vstack([data_b, data_b[oversample_indices]])
            ax2.scatter(data_a[:, 0], data_a[:, 1], c='blue', label='Grupo A (n=100)', alpha=0.6)
            ax2.scatter(data_b_oversampled[:, 0], data_b_oversampled[:, 1], c='red', label='Grupo B (n=100)', alpha=0.6, marker='x')
            ax2.set_title("Datos con Sobremuestreo del Grupo B")
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig)
            st.info("El gráfico de la derecha muestra cómo se han añadido nuevas muestras (marcadas con 'x') del Grupo B para igualar en número al Grupo A, lo que ayuda al modelo a aprender mejor sus patrones.")
        st.info("Aborda las disparidades de representación ajustando la influencia de las instancias de entrenamiento.")
        st.markdown("**Re-ponderación:** Asigna pesos a las muestras para dar más importancia a los grupos subrepresentados.")
        st.markdown("**Re-muestreo:** Modifica físicamente el conjunto de datos (sobre-muestreo o sub-muestreo).")
        st.text_area("Criterios de Decisión: ¿Re-ponderar o Re-muestrear?", placeholder="Basado en mi auditoría y mi modelo, la mejor estrategia es...", key="p8")
        st.text_area("Consideración de Interseccionalidad", placeholder="Mi plan para la interseccionalidad es...", key="p9")

    with tab5:
        st.subheader("Enfoques de Transformación de Distribución")
        st.info("Modifica el espacio de características para mitigar el sesgo.")
        st.text_area("1. Eliminación de Impacto Dispar", placeholder="Ej: 'Reparar' la característica 'código postal' para que su distribución sea la misma en todos los grupos raciales.", key="p10")
        st.text_area("2. Representaciones Justas (LFR, LAFTR)", placeholder="Ej: Usar un autoencoder adversario para aprender una representación que no contenga información de género.", key="p11")
        st.text_area("3. Consideraciones de Interseccionalidad", placeholder="Mi estrategia de transformación se centrará en las intersecciones de género y etnia...", key="p12")

    with tab6:
        st.subheader("Generación de Datos con Conciencia de Equidad")
        st.info("Crea datos sintéticos para mitigar patrones de sesgo.")
        st.markdown("**¿Cuándo Generar Datos?:** Cuando hay subrepresentación severa o se necesitan ejemplos contrafactuales.")
        st.markdown("**Estrategias:** Generación Condicional, Aumentación Contrafactual.")
        st.text_area("Consideraciones de Interseccionalidad", placeholder="Mi modelo generativo será condicionado en la intersección de edad y género para...", key="p13")

def inprocessing_fairness_toolkit():
    st.header("⚙️ Toolkit de Equidad en In-procesamiento")
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **In-procesamiento** implica modificar el algoritmo de aprendizaje del modelo para que la equidad sea uno de sus objetivos, junto con la precisión. Es como enseñarle a un chef a cocinar no solo para que la comida sea deliciosa, sino también para que sea nutricionalmente equilibrada, haciendo de la nutrición una parte central de la receta.
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Objetivos y Restricciones", "Debiasing Adversario", "Optimización Multiobjetivo", "Patrones de Código"])

    with tab1:
        st.subheader("Objetivos y Restricciones de Equidad")
        st.info("Incorpora la equidad directamente en la optimización del modelo.")
        st.markdown("**Métodos Lagrangianos:** Transforma restricciones duras en penalizaciones suaves en la función de pérdida.")
        st.latex(r''' \mathcal{L}(\theta, \lambda) = L(\theta) + \sum_{i=1}^{k} \lambda_i C_i(\theta) ''')
        st.markdown("**Viabilidad y Compensaciones:** Entiende la tensión entre equidad y rendimiento.")
        st.markdown("**Interseccionalidad:** Las restricciones deben considerar combinaciones de atributos.")

    with tab2:
        st.subheader("Enfoques de Debiasing Adversario")
        st.info("Usa aprendizaje adversario para que los modelos 'desaprendan' patrones discriminatorios.")
        st.markdown("**Arquitectura:** Un **Predictor** compite contra un **Adversario**. El predictor aprende a engañar al adversario, creando representaciones sin información del atributo protegido.")
        st.markdown("**Optimización:** El entrenamiento puede ser inestable. Requiere equilibrio de componentes, inversión de gradiente y entrenamiento progresivo.")
    
    with tab3:
        st.subheader("Optimización Multiobjetivo para la Equidad")
        with st.expander("💡 Ejemplo Interactivo: Frontera de Pareto"):
            st.write("Explora la **frontera de Pareto**, que visualiza la compensación (trade-off) entre la precisión de un modelo y su equidad. No se puede mejorar uno sin empeorar el otro.")
            
            # Datos simulados para la frontera
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
            st.info("Cada punto representa un modelo diferente. Los modelos en el borde superior derecho son 'óptimos': no puedes encontrar un modelo más justo sin sacrificar precisión, y viceversa. La elección de qué punto usar depende de las prioridades de tu proyecto.")
        st.info("Navega sistemáticamente las tensiones entre equidad y rendimiento.")
        st.text_area("Análisis y Definición de Objetivos", placeholder="Define tus métricas de rendimiento y criterios de equidad aquí.", key="i1")

    with tab4:
        st.subheader("Catálogo de Patrones de Implementación")
        st.code("""
def fairness_regularized_loss(original_loss, predictions, protected_attribute):
  fairness_penalty = calculate_disparity(predictions, protected_attribute)
  return original_loss + lambda * fairness_penalty
        """, language="python")

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
        st.text_area("Metodología de Implementación", placeholder="1. Seleccionar Criterio de Equidad (ej. igualdad de oportunidades)\n2. Calcular Umbrales en datos de validación\n3. Analizar Compensaciones y Desplegar", key="po1")

    with tab2:
        st.subheader("Guía Práctica de Calibración para la Equidad")
        st.info("Garantiza que las probabilidades predichas tengan un significado consistente en todos los grupos.")
        st.markdown("**Fundamentos:** Una mala calibración significa que la misma puntuación de riesgo representa diferentes niveles de riesgo real para diferentes grupos.")
        st.markdown("**Técnicas:** Platt Scaling, Regresión Isotónica.")
        st.text_area("Metodología de Implementación", placeholder="1. Evaluar Calibración (con ECE, MCE)\n2. Seleccionar Método\n3. Implementar y Validar", key="po2")

    with tab3:
        st.subheader("Métodos de Transformación de Predicción")
        st.info("Modifica las salidas del modelo para satisfacer restricciones de equidad complejas.")
        st.markdown("**Conceptos Clave:** Funciones de Transformación Aprendidas, Alineación de Distribución, Transformaciones de Puntuación Justas.")
        st.text_area("Metodología de Implementación", placeholder="1. Diseñar Transformación\n2. Aprender y Evaluar\n3. Considerar Interseccionalidad", key="po3")

    with tab4:
        st.subheader("Clasificación con Opción de Rechazo")
        st.info("Identifica predicciones inciertas y las difiere a juicio humano.")
        st.markdown("**Fundamentos:** Umbrales de rechazo basados en confianza, clasificación selectiva, modelos de colaboración Humano-IA.")
        st.text_area("Metodología de Implementación", placeholder="1. Estimar Confianza\n2. Optimizar Umbral de Rechazo\n3. Diseñar Flujo de Trabajo Humano-IA", key="po4")

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
        st.markdown("""
        **El Marco de Cuatro Componentes** – Sigue secuencialmente a través de:
        
        1. **Evaluación del Contexto Histórico (HCA)** – Descubre sesgos sistémicos y desequilibrios de poder en tu dominio.
        
        2. **Selección de Definición de Equidad (FDS)**
         – Elige las definiciones de equidad apropiadas basadas en tu contexto y objetivos.
        
        3. **Identificación de Fuentes de Sesgo (BSI)** – Identifica y prioriza las formas en que el sesgo puede entrar en tu sistema.
        
        4. **Métricas Comprensivas de Equidad (CFM)**
         – Implementa métricas cuantitativas para el monitoreo y la presentación de informes.

        **Consejos:**
        - Avanza por las secciones en orden, pero siéntete libre de retroceder si surgen nuevas ideas.
        - Usa los botones de **Guardar Resumen** en cada herramienta para registrar tus hallazgos.
        - Consulta los ejemplos incrustados en cada sección para ver cómo otros han aplicado estas herramientas.
        """)
    elif page == "Evaluación del Contexto Histórico":
        st.header("Herramienta de Evaluación del Contexto Histórico")
        with st.expander("🔍 Definición Amigable"):
            st.write("""
            El **Contexto Histórico** es el trasfondo social y cultural en el que se utilizará tu IA. Es importante porque los sesgos no nacen en los algoritmos, sino en la sociedad. Entender la historia de la discriminación en áreas como la banca o la contratación nos ayuda a anticipar dónde nuestra IA podría fallar y perpetuar injusticias pasadas.
            """)
        st.subheader("1. Cuestionario Estructurado")
        st.markdown("Esta sección te ayuda a descubrir patrones relevantes de discriminación histórica.")
        
        q1 = st.text_area("¿En qué dominio específico operará este sistema (ej. préstamos, contratación, salud)?")
        q2 = st.text_area("¿Cuál es la función específica del sistema o caso de uso dentro de ese dominio?")
        q3 = st.text_area("¿Cuáles son los patrones de discriminación histórica documentados en este dominio?")
        q4 = st.text_area("¿Qué fuentes de datos históricos se utilizan o se referencian en este sistema?")
        q5 = st.text_area("¿Cómo se definieron históricamente las categorías clave (ej. género, riesgo crediticio) y han evolucionado?")
        q6 = st.text_area("¿Cómo se midieron históricamente las variables (ej. ingresos, educación)? ¿Podrían codificar sesgos?")
        q7 = st.text_area("¿Han servido otras tecnologías para roles similares en este dominio? ¿Desafiaron o reforzaron las desigualdades?")
        q8 = st.text_area("¿Cómo podría la automatización amplificar los sesgos pasados o introducir nuevos riesgos en este dominio?")

        st.subheader("2. Matriz de Clasificación de Riesgos")
        st.markdown("""
        Para cada patrón histórico identificado, estima:
        - **Severidad**: Alto = impacta derechos/resultados de vida, Medio = afecta oportunidades/acceso a recursos, Bajo = impacto material limitado.
        - **Probabilidad**: Alta = probable que aparezca en sistemas similares, Media = posible, Baja = raro.
        - **Relevancia**: Alta = directamente relacionado con tu sistema, Media = afecta partes, Baja = periférico.
        """)
        matrix = st.text_area("Matriz de Clasificación de Riesgos (tabla Markdown)", height=200, placeholder="| Patrón | Severidad | Probabilidad | Relevancia | Puntuación (S×P×R) | Prioridad |\n|---|---|---|---|---|---|")

        if st.button("Guardar Resumen HCA"):
            summary = {
                "Cuestionario Estructurado": {
                    "Dominio": q1, "Función": q2, "Patrones Históricos": q3, "Fuentes de Datos": q4,
                    "Definiciones de Categoría": q5, "Riesgos de Medición": q6, "Sistemas Anteriores": q7, "Riesgos de Automatización": q8
                },
                "Matriz de Riesgos": matrix
            }
            summary_md = "# Resumen de Evaluación del Contexto Histórico\n"
            for section, answers in summary.items():
                summary_md += f"## {section}\n"
                if isinstance(answers, dict):
                    for k, v in answers.items():
                        summary_md += f"**{k}:** {v}\n\n"
                else:
                    summary_md += f"{answers}\n"
            
            st.subheader("Vista Previa del Resumen HCA")
            st.markdown(summary_md)
            st.download_button("Descargar Resumen HCA", summary_md, "HCA_summary.md", "text/markdown")
            st.success("Resumen de Evaluación del Contexto Histórico guardado.")

    elif page == "Selección de Definición de Equidad":
        st.header("Herramienta de Selección de Definición de Equidad")
        with st.expander("🔍 Definición Amigable"):
            st.write("""
            No existe una única "receta" para la equidad. Diferentes situaciones requieren diferentes tipos de justicia. Esta sección te ayuda a elegir la **definición de equidad** más adecuada para tu proyecto, como un médico que elige el tratamiento correcto para una enfermedad específica. Algunas definiciones buscan igualdad de resultados, otras igualdad de oportunidades, y la elección correcta depende de tu objetivo y del daño que intentas evitar.
            """)
        st.subheader("1. Catálogo de Definiciones de Equidad")
        st.markdown("""
        | Definición | Fórmula | Cuándo Usar | Ejemplo |
        |---|---|---|---|
        | Paridad Demográfica | P(Ŷ=1|A=a) = P(Ŷ=1|A=b) | Asegurar tasas de positivos iguales entre grupos. | Anuncios de universidad mostrados por igual a todos los géneros. |
        | Igualdad de Oportunidades | P(Ŷ=1|Y=1,A=a) = P(Ŷ=1|Y=1,A=b) | Minimizar falsos negativos entre individuos calificados. | Sensibilidad de prueba médica igual entre razas. |
        | Probabilidades Igualadas | P(Ŷ=1|Y=y,A=a) = P(Ŷ=1|Y=y,A=b) ∀ y | Equilibrar falsos positivos y negativos entre grupos. | Predicciones de reincidencia con tasas de error iguales. |
        | Calibración | P(Y=1|ŝ=s,A=a) = s | Cuando las puntuaciones predichas se exponen a los usuarios. | Puntuaciones de crédito calibradas para diferentes demografías. |
        | Equidad Contrafactual | Ŷ(x) = Ŷ(x') si A cambia | Requerir eliminación de sesgo causal relativo a rasgos sensibles. | Resultado sin cambios si solo cambia la raza en el perfil. |
        """)
        st.subheader("2. Árbol de Decisión para Selección")
        exclusion = st.radio("¿El HCA reveló exclusión sistémica de grupos protegidos?", ("Sí", "No"), key="fds1")
        error_harm = st.radio("¿Qué tipo de error es más dañino en tu contexto?", ("Falsos Negativos", "Falsos Positivos", "Ambos por igual"), key="fds2")
        score_usage = st.checkbox("¿Se usarán las salidas como puntuaciones (ej. riesgo, ranking)?", key="fds3")
        
        st.subheader("Definiciones Recomendadas")
        definitions = []
        if exclusion == "Sí": definitions.append("Paridad Demográfica")
        if error_harm == "Falsos Negativos": definitions.append("Igualdad de Oportunidades")
        elif error_harm == "Falsos Positivos": definitions.append("Igualdad Predictiva")
        elif error_harm == "Ambos por igual": definitions.append("Probabilidades Igualadas")
        if score_usage: definitions.append("Calibración")
        
        for d in definitions: st.markdown(f"- **{d}**")

    # (El resto de las secciones del Audit Playbook se pueden añadir aquí de manera similar)
    elif page == "Identificación de Fuentes de Sesgo":
        st.header("Herramienta de Identificación de Fuentes de Sesgo")
        # (Contenido de esta sección)
    elif page == "Métricas Comprensivas de Equidad":
        st.header("Métricas Comprensivas de Equidad (CFM)")
        # (Contenido de esta sección)


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
