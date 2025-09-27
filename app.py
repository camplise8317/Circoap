import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel
import docx
import io
import os

# --- INICIALIZACIÓN DEL ESTADO DE LA SESIÓN ---
if 'stage' not in st.session_state:
    st.session_state.stage = "inspiration"
if 'inspiration_text' not in st.session_state:
    st.session_state.inspiration_text = ""
if 'final_context' not in st.session_state:
    st.session_state.final_context = ""
if 'processed_activity' not in st.session_state:
    st.session_state.processed_activity = None


# --- FUNCIÓN PRINCIPAL QUE ENVUELVE LA APP ---
def main():
    # --- CONFIGURACIÓN DE LA PÁGINA DE STREAMLIT ---
    st.set_page_config(
        page_title="Diseñador Pedagógico con Vertex AI",
        page_icon="🧠",
        layout="wide"
    )
    st.title("🤖 Compañero de Diseño Pedagógico con Vertex AI 🧠")
    st.markdown("Un co-piloto interactivo para crear experiencias de aprendizaje inmersivas.")

    # --- INICIALIZACIÓN Y CONFIGURACIÓN DE VERTEX AI ---
    st.sidebar.header("Configuración de Vertex AI")
    try:
        GCP_PROJECT_ID = os.environ.get("GCP_PROJECT")
        GCP_LOCATION = os.environ.get("GCP_LOCATION")

        if not GCP_PROJECT_ID or not GCP_LOCATION:
            st.sidebar.error("Variables de entorno GCP_PROJECT y GCP_LOCATION no encontradas.")
            st.error("Configura tus variables de entorno de Google Cloud para continuar.")
            st.stop()

        vertexai.init(project=GCP_PROJECT_ID, location=GCP_LOCATION)
        st.sidebar.success(f"✅ Conectado a Vertex AI\nProyecto: {GCP_PROJECT_ID}")
    except Exception as e:
        st.sidebar.error(f"Error al inicializar Vertex AI: {e}")
        st.error("No se pudo conectar con Vertex AI. Verifica la configuración del proyecto y tu autenticación.")
        st.stop()

    # --- BLOQUE DE CONFIGURACIÓN DE MODELOS EN LA BARRA LATERAL ---
    st.sidebar.subheader("Selección de Modelos")

    # Nota: Los nombres de los modelos son ejemplos. Ajústalos a los modelos disponibles en tu proyecto.
    vertex_ai_models = [
        "gemini-2.5",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite"
    ]

    # Guardamos los modelos seleccionados en el estado de la sesión
    st.session_state.gen_model_name = st.sidebar.selectbox(
        "**Modelo para Generación**",
        vertex_ai_models,
        index=1, # Por defecto 'flash'
        key="gen_vertex_name_sidebar"
    )

    st.session_state.audit_model_name = st.sidebar.selectbox(
        "**Modelo para Auditoría**",
        vertex_ai_models,
        index=0, # Por defecto 'pro'
        key="audit_vertex_name_sidebar",
        help="Se recomienda un modelo potente (ej. Pro) para la auditoría."
    )
    
    # --- DICCIONARIO DE LA TAXONOMÍA DE BLOOM (sin cambios) ---
    bloom_taxonomy_detallada = {
        "RECORDAR": { "definicion": "Recuperar conocimiento relevante de la memoria de largo plazo.", "subprocesos": { "Reconocer": {"nombres_alternativos": "Identificar", "definicion_ejemplo": "Localizar conocimiento..."}, "Evocar": {"nombres_alternativos": "Recuperar", "definicion_ejemplo": "Recuperar conocimiento..."} } },
        "COMPRENDER": { "definicion": "Construir significado a partir de contenidos educativos.", "subprocesos": { "Interpretar": {"nombres_alternativos": "Aclarar, parafrasear", "definicion_ejemplo": "Transformar de una forma de representación a otra..."}, "Ejemplificar": {"nombres_alternativos": "Ilustrar, citar casos", "definicion_ejemplo": "Poner un ejemplo específico..."}, "Clasificar": {"nombres_alternativos": "Categorizar", "definicion_ejemplo": "Determinar que algo pertenece a una categoría..."}, "Resumir": {"nombres_alternativos": "Abstraer, generalizar", "definicion_ejemplo": "Extraer el tema general..."}, "Inferir": {"nombres_alternativos": "Concluir, predecir", "definicion_ejemplo": "Sacar una conclusión lógica..."}, "Comparar": {"nombres_alternativos": "Contrastar, esquematizar", "definicion_ejemplo": "Detectar correspondencias..."}, "Explicar": {"nombres_alternativos": "Construir modelos", "definicion_ejemplo": "Construir un modelo de causa-efecto..."} } },
        "APLICAR": { "definicion": "Desarrolar o usar un procedimiento en una situación dada.", "subprocesos": { "Ejecutar": {"nombres_alternativos": "Llevar a cabo", "definicion_ejemplo": "Aplicar un procedimiento a una tarea familiar..."}, "Implementar": {"nombres_alternativos": "Utilizar", "definicion_ejemplo": "Aplicar un procedimiento a una tarea no familiar..."} } },
        "ANALIZAR": { "definicion": "Despiezar el material en sus partes constituyentes y determinar cómo se relacionan.", "subprocesos": { "Diferenciar": {"nombres_alternativos": "Discriminar, seleccionar", "definicion_ejemplo": "Distinguir las partes relevantes..."}, "Organizar": {"nombres_alternativos": "Integrar, estructurar", "definicion_ejemplo": "Determinar cómo encajan los elementos..."}, "Atribuir": {"nombres_alternativos": "Deconstruir", "definicion_ejemplo": "Determinar los puntos de vista, sesgos..."} } },
        "EVALUAR": { "definicion": "Formular juicios con base en criterios o parámetros.", "subprocesos": { "Verificar": {"nombres_alternativos": "Detectar, monitorear", "definicion_ejemplo": "Detectar inconsistencias o falacias..."}, "Criticar": {"nombres_alternativos": "Juzgar, argumentar", "definicion_ejemplo": "Detectar inconsistencias con base en criterios externos..."} } },
        "CREAR": { "definicion": "Agrupar elementos para formar un todo coherente o funcional.", "subprocesos": { "Generar": {"nombres_alternativos": "Formular hipótesis", "definicion_ejemplo": "Formular hipótesis alternativas..."}, "Planear": {"nombres_alternativos": "Diseñar", "definicion_ejemplo": "Idear un procedimiento..."}, "Producir": {"nombres_alternativos": "Construir", "definicion_ejemplo": "Inventar un producto..."} } }
    }


    # --- SISTEMA DE PROMPTS CENTRALIZADO ---
    def get_master_prompt_system(contexto_narrativo):
        bloom_text = ""
        for level, data in bloom_taxonomy_detallada.items():
            bloom_text += f"\n### {level}: {data['definicion']}\n"
            for subprocess, sub_data in data.get('subprocesos', {}).items():
                alt_names = sub_data.get('nombres_alternativos', '')
                bloom_text += f"- **{subprocess} ({alt_names}):** {sub_data.get('definicion_ejemplo', '')}\n"

        return f"""
# MODELO PEDAGÓGICO INTEGRAL
## CAPA 0: CONTEXTO NARRATIVO
Toda la actividad debe estar inmersa en esta historia:
---
{contexto_narrativo}
---
## CAPA 1: FILOSOFÍA (Círculos de Aprendizaje)
Entorno colaborativo. El facilitador guía con preguntas.
## CAPA 2: ESTRUCTURA (Bruner)
Viaje: Enactivo (hacer) -> Icónico (representar) -> Simbólico (abstraer).
## CAPA 3: COHESIÓN (Hilo Conductor)
El producto de una fase es el insumo de la siguiente.
## CAPA 4: DIFERENCIACIÓN (Piso Bajo, Techo Alto)
Accesible para todos, desafiante para los más avanzados.
## CAPA 5: INTENCIÓN COGNITIVA (Bloom)
Las tareas deben provocar procesos de pensamiento específicos y ascender en la taxonomía.
## CAPA 6: DETALLE DE PROCESOS COGNITIVOS
Usa los siguientes verbos y definiciones con precisión.
{bloom_text}
"""

    # --- Estructura de categorías (sin cambios) ---
    CATEGORIAS_ACTIVIDADES = {
        "Círculos de Matemática y Razonamiento": {"Edades": ["5 a 7 años", "8 a 11 años", "12 a 15 años"]},
        "Ciencias": {"Disciplinas": ["Física", "Química", "Biología"]},
        "Tecnología": {"Disciplinas": ["Programación", "Robótica"]}
    }

    # --- FUNCIONES DE UTILIDAD Y LÓGICA DE LA APP ---

    def leer_docx(file):
        try:
            doc = docx.Document(io.BytesIO(file.read()))
            return '\n'.join([para.text for para in doc.paragraphs])
        except Exception as e:
            st.error(f"Error al leer el archivo Word: {e}")
            return ""

    def generar_texto_con_vertex(model_name, prompt):
        try:
            modelo = GenerativeModel(model_name)
            response = modelo.generate_content(prompt)
            return response.text
        except Exception as e:
            st.error(f"Ocurrió un error al llamar al modelo {model_name} en Vertex AI: {e}")
            return None

    def set_stage(stage_name):
        st.session_state.stage = stage_name

    # --- FUNCIÓN DE AUDITORÍA (AHORA RECIBE EL NOMBRE DEL MODELO) ---
    def auditar_actividad(actividad_generada, nivel_salida_esperado, contexto_narrativo, audit_model_name):
        master_prompt_ref = get_master_prompt_system(contexto_narrativo)
        auditoria_prompt = f"""
        Eres un auditor experto en diseño instruccional. Audita RIGUROSAMENTE la siguiente actividad.
        --- MODELO PEDAGÓGICO DE REFERENCIA ---
        {master_prompt_ref}
        --- OBJETIVO COGNITIVO ---
        El nivel de salida esperado es **{nivel_salida_esperado}**.
        --- ACTIVIDAD A AUDITAR ---
        {actividad_generada}
        ---
        **VALIDACIÓN DE CRITERIOS (Responde con ✅/❌ y un comentario breve si es ❌):**
        1.  **Contexto Narrativo (Capa 0):** ¿La actividad está completamente inmersa en la historia y usa su lenguaje?
        2.  **Hilo Conductor (Capa 3):** ¿El producto de cada fase se usa explícitamente como insumo de la siguiente?
        3.  **Intención Cognitiva (Capa 5):** ¿La actividad culmina exitosamente en el nivel de **{nivel_salida_esperado}** en la fase simbólica?
        **DICTAMEN FINAL:** [✅ CUMPLE / ❌ RECHAZADO]
        **OBSERVACIONES FINALES:** [Si es ❌, sé específico en qué capa del modelo falló.]
        """
        return generar_texto_con_vertex(audit_model_name, auditoria_prompt)

    # --- FUNCIÓN DE GENERACIÓN CON CICLO DE AUDITORÍA (AHORA RECIBE LOS NOMBRES DE MODELOS) ---
    def generar_actividad_con_auditoria(params):
        master_prompt = get_master_prompt_system(params["contexto"])
        current_activity_text = ""
        audit_observations = ""
        max_attempts = 3
        attempt = 0

        gen_model = params["gen_model"]
        audit_model = params["audit_model"]

        while attempt < max_attempts:
            attempt += 1
            st.info(f"--- Generando/Refinando Actividad (Intento {attempt}/{max_attempts}) ---")

            prompt_generacion = f"""
            Eres un diseñador instruccional de élite. Crea una actividad de 1 hora siguiendo ESTRICTAMENTE el modelo.
            --- 1. ENTRADA ESTRATÉGICA ---
            - **Inspiración:** {params["inspiracion"]}
            - **Grupo:** {params["grupo"]}
            - **Nivel Entrada:** {params["nivel_entrada"]}
            - **Nivel Salida:** {params["nivel_salida"]}
            --- 2. MODELO PEDAGÓGICO ---
            {master_prompt}
            --- 3. FORMATO DE SALIDA ---
            **TÍTULO:** [Título creativo basado en CAPA 0]
            **OBJETIVOS DE APRENDIZAJE:** [2-3 objetivos culminando en {params['nivel_salida']}]
            **MATERIALES Y MONTAJE:** [Basado en CAPA 0]
            **HILO CONDUCTOR:** [Artefacto Enactivo -> Rep. Icónica -> Conclusión Simbólica]
            **DESARROLLO (60 MIN):**
            (Usa el lenguaje del contexto narrativo)
            ---
            **FASE 1: ENACTIVA (20 min) | Foco: APLICAR**
            - **Facilitador (Piso Bajo/Techo Alto):** [Invitaciones y desafíos]
            - **Interacciones Sociales:** [Negociación, debate, etc.]
            - **➡️ Producto Clave:** [Artefacto enactivo]
            ---
            **FASE 2: ICÓNICA (20 min) | Foco: ANALIZAR**
            - **Punto de Partida:** El Artefacto Enactivo.
            - **Facilitador (Piso Bajo/Techo Alto):** [Preguntas para representar y sistematizar]
            - **➡️ Producto Clave:** [Representación icónica]
            ---
            **FASE 3: SIMBÓLICA (15 min) | Foco: {params['nivel_salida']}**
            - **Punto de Partida:** La Representación Icónica.
            - **Facilitador (Piso Bajo/Techo Alto):** [Preguntas para generalizar y juzgar]
            - **➡️ Producto Clave:** [Conclusión simbólica]
            ---
            **CIERRE Y REFLEXIÓN (5 min):** [Conectar la misión con el aprendizaje]
            """
            
            if attempt > 1:
                prompt_generacion += f"\n--- RETROALIMENTACIÓN PARA REFINAMIENTO ---\nLa versión anterior fue rechazada. Observaciones del auditor: {audit_observations}\nPor favor, genera una nueva versión que corrija estos puntos.\n"

            current_activity_text = generar_texto_con_vertex(gen_model, prompt_generacion)
            if not current_activity_text:
                st.error("Fallo en la generación de texto.")
                break

            with st.expander(f"Ver Actividad Generada (Intento {attempt})", expanded=False):
                st.markdown(current_activity_text)

            auditoria_resultado = auditar_actividad(current_activity_text, params["nivel_salida"], params["contexto"], audit_model)
            if not auditoria_resultado:
                st.error("Fallo en la auditoría.")
                break

            with st.expander(f"Ver Resultado de Auditoría (Intento {attempt})", expanded=True):
                st.markdown(auditoria_resultado)

            if "✅ CUMPLE" in auditoria_resultado:
                st.success(f"¡Actividad generada y aprobada en el intento {attempt}!")
                return {"activity_text": current_activity_text, "status": "✅ CUMPLE", "observations": ""}
            else:
                observaciones_start = auditoria_resultado.find("OBSERVACIONES FINALES:")
                audit_observations = auditoria_resultado[observaciones_start:] if observaciones_start != -1 else "No se pudo extraer observaciones."
                st.warning("La actividad necesita refinamiento...")
        
        st.error(f"No se pudo generar una actividad aprobada después de {max_attempts} intentos.")
        return {"activity_text": current_activity_text, "status": "❌ RECHAZADO", "observations": audit_observations}

    # --- FUNCIÓN DE EXPORTACIÓN A WORD ---
    def exportar_actividad_a_word(activity_data):
        doc = docx.Document()
        doc.add_heading('Actividad de Aprendizaje Generada con IA', level=1)
        doc.add_heading('Actividad Generada', level=2)
        
        lines = activity_data.get("activity_text", "").split('\n')
        for line in lines:
            stripped_line = line.strip()
            if stripped_line.startswith("**TÍTULO") or stripped_line.startswith("**OBJETIVOS") or \
               stripped_line.startswith("**MATERIALES") or stripped_line.startswith("**EL HILO") or \
               stripped_line.startswith("**DESARROLLO") or stripped_line.startswith("**CIERRE"):
                p = doc.add_paragraph()
                run = p.add_run(stripped_line.replace("**", ""))
                run.bold = True
                run.font.size = docx.shared.Pt(14)
            elif stripped_line.startswith("**FASE"):
                p = doc.add_paragraph()
                run = p.add_run(stripped_line.replace("**", ""))
                run.bold = True
                run.font.size = docx.shared.Pt(12)
            else:
                doc.add_paragraph(stripped_line.replace("*", "").strip())
        
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    # --- INTERFAZ DE USUARIO POR ETAPAS ---

    if st.session_state.stage == "inspiration":
        st.header("ETAPA 1: El Punto de Partida 💡")
        tab1, tab2, tab3 = st.tabs(["🎯 Empezar con un Tema", "📝 Lluvia de Ideas", "📄 Subir un Archivo (.docx)"])
        with tab1:
            tema_foco_usuario = st.text_input("Tema central", placeholder="Ej: El ciclo del agua")
            if st.button("Usar este Tema", key="tema_btn"):
                if tema_foco_usuario:
                    st.session_state.inspiration_text = f"El tema central es: {tema_foco_usuario}."
                    set_stage("context")
                    st.rerun()
        with tab2:
            idea_box = st.text_area("Lluvia de ideas...", height=200, placeholder="Ej: Volcanes, construir un modelo...")
            if st.button("Usar estas Ideas", key="ideas_btn"):
                if idea_box:
                    st.session_state.inspiration_text = idea_box
                    set_stage("context")
                    st.rerun()
        with tab3:
            uploaded_file = st.file_uploader("Sube tu archivo .docx", type=['docx'])
            if st.button("Usar este Archivo", key="file_btn"):
                if uploaded_file:
                    st.session_state.inspiration_text = leer_docx(uploaded_file)
                    set_stage("context")
                    st.rerun()

    elif st.session_state.stage == "context":
        st.header("ETAPA 2: La Gran Historia 🎭")
        st.info("**Inspiración proporcionada:**")
        st.text_area("", value=st.session_state.inspiration_text, height=100, disabled=True)
        if st.button("🤖 IA, ¡sugiéreme un contexto!", type="primary"):
            with st.spinner("La IA está imaginando un universo... ✨"):
                # Usaremos un modelo rápido para sugerencias
                suggestion_model = "gemini-2.5-flash-lite"
                prompt_contexto = f"Basado en: '{st.session_state.inspiration_text}', genera 1 opción de contexto narrativo breve y creativo."
                sugerencia = generar_texto_con_vertex(suggestion_model, prompt_contexto)
                if sugerencia: st.session_state.final_context = sugerencia
        contexto_editable = st.text_area("Refina y personaliza el contexto:", value=st.session_state.get('final_context', ""), height=250)
        col1, col2 = st.columns([1,1])
        with col1:
            if st.button("✅ Usar este Contexto y Continuar"):
                if contexto_editable:
                    st.session_state.final_context = contexto_editable
                    set_stage("generation")
                    st.rerun()
        with col2:
            if st.button("Volver a Inspiración"):
                set_stage("inspiration")
                st.rerun()

    elif st.session_state.stage in ["generation", "refinement"]:
        st.header("ETAPA 3: Diseño y Refinamiento 🛠️")
        
        with st.expander("Ver y Editar Contexto Narrativo", expanded=False):
            st.markdown(st.session_state.final_context)
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader("1. Parámetros Pedagógicos")
            categoria_seleccionada = st.selectbox("Categoría", list(CATEGORIAS_ACTIVIDADES.keys()))
            if categoria_seleccionada == "Círculos de Matemática y Razonamiento":
                sub_options = CATEGORIAS_ACTIVIDADES[categoria_seleccionada]["Edades"]
            else:
                sub_options = CATEGORIAS_ACTIVIDADES[categoria_seleccionada]["Disciplinas"]
            subcategoria_seleccionada = st.selectbox("Grupo", sub_options)
        with col_p2:
            st.subheader("2. Alcance del Aprendizaje")
            nivel_entrada_usuario = st.text_input("Nivel de entrada", placeholder="Ej: Saben construir con bloques.")
            nivel_salida_usuario = st.selectbox("Máxima habilidad de Bloom", options=list(bloom_taxonomy_detallada.keys()), index=len(bloom_taxonomy_detallada) - 1)

        if st.button("🚀 Generar Actividad con Auditoría", type="primary"):
            if not all([st.session_state.final_context, nivel_entrada_usuario]):
                st.error("Por favor, define un contexto y completa los parámetros.")
            else:
                params = {
                    "inspiracion": st.session_state.inspiration_text,
                    "contexto": st.session_state.final_context,
                    "grupo": subcategoria_seleccionada,
                    "nivel_entrada": nivel_entrada_usuario,
                    "nivel_salida": nivel_salida_usuario,
                    "gen_model": st.session_state.gen_model_name,      # <-- Lee desde el estado de la sesión
                    "audit_model": st.session_state.audit_model_name   # <-- Lee desde el estado de la sesión
                }
                st.session_state.processed_activity = generar_actividad_con_auditoria(params)
                if st.session_state.processed_activity:
                    set_stage("refinement")
                    st.rerun()

        if st.session_state.stage == "refinement" and st.session_state.processed_activity:
            st.markdown("---")
            st.header("Actividad Generada (Versión Actual)")
            st.markdown(st.session_state.processed_activity["activity_text"])

            st.subheader("🗣️ Ciclo de Refinamiento Manual")
            feedback_usuario = st.text_area("Escribe tu feedback para mejorar la actividad:", placeholder="Ej: 'La fase icónica necesita un ejemplo más claro.'")
            if st.button("♻️ Refinar con mi Feedback"):
                if feedback_usuario:
                    with st.spinner("La IA está aplicando tus sugerencias... ✍️"):
                        prompt_refinamiento = f"""
                        Refina la actividad basándote en el feedback.
                        --- ACTIVIDAD ANTERIOR ---
                        {st.session_state.processed_activity['activity_text']}
                        --- FEEDBACK DEL USUARIO ---
                        {feedback_usuario}
                        --- TAREA ---
                        Genera la nueva versión completa de la actividad incorporando el feedback. Produce solo la actividad mejorada.
                        """
                        # Usa el modelo de generación seleccionado para el refinamiento
                        actividad_refinada = generar_texto_con_vertex(st.session_state.gen_model_name, prompt_refinamiento)
                        if actividad_refinada:
                            st.session_state.processed_activity['activity_text'] = actividad_refinada
                            st.rerun()
            
            st.markdown("---")
            st.subheader("✅ Exportar Actividad Final")
            word_buffer = exportar_actividad_a_word(st.session_state.processed_activity)
            st.download_button(
                label="Descargar Actividad en Word",
                data=word_buffer,
                file_name=f"actividad_{subcategoria_seleccionada.replace(' ', '_')}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

        if st.session_state.stage in ["context", "generation", "refinement"]:
            if st.button("Reiniciar y Empezar de Nuevo"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    main()
