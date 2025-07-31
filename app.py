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

#======================================================================
# --- FAIRNESS INTERVENTION PLAYBOOK ---
#======================================================================

def causal_fairness_toolkit():
    st.header("🛡️ Toolkit de Equidad Causal")
    
    with st.expander("🔍 Definición Amigable"):
        st.write("""
        El **Análisis Causal** va más allá de las correlaciones para entender el *porqué* de las disparidades. Es como ser un detective que no solo ve que dos eventos ocurren juntos, sino que reconstruye la cadena de causa y efecto que los conecta. Esto nos ayuda a aplicar soluciones que atacan la raíz del problema, en lugar de solo maquillar los síntomas.
        """)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Identificación", "Análisis Contrafactual", "Diagrama Causal", "Inferencia Causal"])

    with tab1:
        st.subheader("Marco de Identificación de Mecanismos de Discriminación")
        st.info("Identifica las posibles causas raíz del sesgo en tu aplicación.")
        st.text_area("1. Discriminación Directa: ¿El atributo protegido influye directamente en la decisión?", placeholder="Ej: ¿Se utiliza explícitamente el 'género' como una característica en el modelo?", key="c1")
        st.text_area("2. Discriminación Indirecta: ¿El atributo protegido afecta a factores intermedios legítimos?", placeholder="Ej: ¿El 'género' afecta a los 'años de experiencia' debido a pausas en la carrera?", key="c2")
        st.text_area("3. Discriminación por Proxy: ¿Las decisiones dependen de variables correlacionadas con atributos protegidos?", placeholder="Ej: ¿El 'código postal' se correlaciona con la 'raza' y se utiliza para predecir el riesgo?", key="c3")

    with tab2:
        st.subheader("Metodología Práctica de Equidad Contrafactual")
        st.info("Analiza, cuantifica y mitiga el sesgo contrafactual en tu modelo.")
        with st.expander("💡 Ejemplo Interactivo: Simulación Contrafactual"):
            st.write("Observa cómo un cambio en un atributo protegido puede alterar la decisión de un modelo, revelando un sesgo causal.")
            
            # Simulación
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
            st.text_area("1.1 Formular Consultas Contrafactuales", placeholder="Ej: Para un solicitante rechazado, ¿cuál habría sido el resultado si su raza fuera diferente, manteniendo constantes los ingresos?", key="c4")
            st.text_area("1.2 Identificar Rutas Causales (Justas vs. Injustas)", placeholder="Ej: La ruta Raza -> Código Postal -> Decisión es injusta.", key="c5")
            st.text_area("1.3 Medir Disparidades y Documentar", placeholder="Ej: El 15% de los solicitantes del grupo desfavorecido habrían sido aprobados en el escenario contrafactual.", key="c6")
        with st.container(border=True):
            st.markdown("##### Paso 2: Análisis Específico de Rutas")
            st.text_area("2.1 Descomponer y Clasificar Rutas", placeholder="Ej: Ruta 1 (proxy de código postal) clasificada como INJUSTA.", key="c7")
            st.text_area("2.2 Cuantificar Contribución y Documentar", placeholder="Ej: La ruta del código postal representa el 60% de la disparidad observada.", key="c8")
        with st.container(border=True):
            st.markdown("##### Paso 3: Diseño de Intervención")
            st.selectbox("3.1 Seleccionar Enfoque de Intervención", ["Nivel de Datos", "Nivel de Modelo", "Post-procesamiento"], key="c9")
            st.text_area("3.2 Implementar y Monitorear", placeholder="Ej: Se aplicó una transformación a la característica de código postal. La disparidad se redujo en un 50%.", key="c10")

    with tab3:
        st.subheader("Enfoque de Diagrama Causal Inicial")
        st.info("Esboza diagramas para visualizar las relaciones causales y documentar tus supuestos.")
        st.markdown("""
        **Convenciones de Anotación:**
        - **Nodos (variables):** Atributos Protegidos, Características, Resultados.
        - **Flechas Causales (→):** Relación causal asumida.
        - **Flechas de Correlación (<-->):** Correlación sin causalidad directa conocida.
        - **Incertidumbre (?):** Relación causal hipotética o débil.
        - **Ruta Problemática (!):** Ruta que consideras una fuente de inequidad.
        """)
        st.text_area("Documentación de Supuestos y Rutas", placeholder="Ruta (!): Raza -> Nivel de Ingresos -> Decisión.\nSupuesto: Las disparidades históricas de ingresos vinculadas a la raza afectan la capacidad de préstamo.", height=200, key="c11")

    with tab4:
        st.subheader("Inferencia Causal con Datos Limitados")
        st.info("Métodos prácticos para estimar efectos causales cuando los datos son imperfectos.")
        st.markdown("""
        **Métodos de Inferencia Observacional:**
        - **Matching (Emparejamiento):** Compara individuos similares de diferentes grupos.
        - **Variables Instrumentales (IV):** Usa una variable externa para aislar el efecto causal.
        - **Regresión por Discontinuidad:** Aprovecha umbrales naturales en los datos.
        - **Diferencia en Diferencias:** Compara cambios en el tiempo entre grupos.
        """)
        st.text_area("Análisis de Sensibilidad", placeholder="¿Qué tan fuerte tendría que ser una variable de confusión no medida para anular el efecto de sesgo que has encontrado?", key="c12")

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
