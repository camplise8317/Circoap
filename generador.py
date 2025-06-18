import streamlit as st
import pandas as pd
import google.generativeai as genai
import PyPDF2
import docx
import re
import io
import openai

# --- Configuración de API Keys (Streamlit Secrets para despliegue, o input para desarrollo) ---
st.sidebar.header("Configuración de API Keys")

# Usar st.secrets si está disponible (para despliegue en Streamlit Cloud)
# gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
# openai_api_key = st.secrets.get("OPENAI_API_KEY", "")

# Para desarrollo local o si st.secrets no está configurado, usar text_input
gemini_api_key = st.sidebar.text_input("API Key de Google Gemini", type="password", 
                                        value=st.session_state.get("gemini_api_key", ""), 
                                        help="Obtén tu clave en https://aistudio.google.com/app/apikey")
openai_api_key = st.sidebar.text_input("API Key de OpenAI (para modelos GPT)", type="password", 
                                       value=st.session_state.get("openai_api_key", ""), 
                                       help="Obtén tu clave en https://platform.openai.com/account/api-keys")

# Guardar las claves en session_state para persistencia durante la sesión
if gemini_api_key:
    st.session_state["gemini_api_key"] = gemini_api_key
if openai_api_key:
    st.session_state["openai_api_key"] = openai_api_key


# Inicialización condicional de Gemini y OpenAI
gemini_config_ok = False
openai_config_ok = False

if gemini_api_key:
    try:
        genai.configure(api_key=gemini_api_key)
        gemini_config_ok = True
        st.sidebar.success("API Key de Gemini configurada.")
    except Exception as e:
        st.sidebar.error(f"Error al configurar la API Key de Gemini: {e}")
else:
    st.sidebar.warning("Por favor, ingresa tu API Key de Gemini para usar modelos Gemini.")

if openai_api_key:
    openai.api_key = openai_api_key
    openai_config_ok = True
    st.sidebar.success("API Key de OpenAI configurada.")
else:
    st.sidebar.warning("Por favor, ingresa tu API Key de OpenAI para usar modelos GPT.")

# --- Definición del contexto de Círculos de Aprendizaje (reemplaza la carga de PDF) ---
CIRCULOS_DE_APRENDIZAJE_CONTEXTO = """
Un círculo de aprendizaje se erige como un espacio de profunda reflexión y exploración intelectual, destinado a desentrañar los enigmas del mundo mediante la experiencia personal del descubrimiento. En CIRCOAP, no solo alentamos a los participantes a crear sus propias interrogantes, sino que también les brindamos las herramientas para trazar el camino hacia la comprensión del mundo que les rodea. Aquí, lejos de asumir el papel de meros educadores, nos convertimos en guías que dirigen el pensamiento de los jóvenes, llevándolos a superar el temor que a menudo suscitan las matemáticas y las ciencias, y a descubrir su potencial para generar nuevo conocimiento. La experiencia que ofrecemos es genuinamente única y suscita un entusiasmo inquebrantable en nuestros participantes.

Es fundamental subrayar que los círculos de aprendizaje no tienen la intención de reemplazar las aulas tradicionales ni se oponen a las metodologías convencionales empleadas en el entorno educativo. En cambio, su propósito es complementar estas metodologías al proporcionar un espacio adicional donde los estudiantes puedan desarrollar habilidades distintas, particularmente aquellas relacionadas con los procesos de razonamiento inherentes a cada área del conocimiento y el fomento de la creatividad. En CIRCOAP, no fomentamos la competencia entre los participantes; en su lugar, abogamos por un enfoque de aprendizaje colaborativo, donde el conocimiento se cultiva y construye de manera conjunta.

Los círculos de aprendizaje persiguen tres objetivos fundamentales:

Objetivo Académico/Científico/Epistemológico: Fomentamos el desarrollo de habilidades de pensamiento crítico, lógico, matemático, algorítmico y científico en los participantes. Priorizamos la formación de una sólida base de razonamiento por encima de la simple adquisición de contenidos o conceptos específicos, utilizando una metodología mayéutica que estimula el descubrimiento autónomo.

Objetivo Psicológico: Nos esforzamos en fortalecer la autoestima y empoderamiento de los participantes en el ámbito de la ciencia y la tecnología, superando de raíz los prejuicios existentes al respecto.

Objetivo Democrático: Nos empeñamos en extender los círculos de aprendizaje a una amplia gama de públicos, sin que las barreras socioeconómicas, geográficas, de género o raciales sean un impedimento. Aspiramos a representar de manera fiel y completa la diversidad de nuestra nación.

Metodología de los Círculos de Aprendizaje:

Cada círculo de aprendizaje se compone de una serie de sesiones, generalmente entre 6 y 8, que se enfocan en un tema específico a lo largo de todas las sesiones* (Hemos trabajado principalmente áreas STEM aunque puede extenderse a practicamente cualquier área). El líder del círculo tiene la encomienda de asegurarse de que los participantes comprendan el problema central y se involucren de manera profunda en su exploración. Durante estas sesiones, el objetivo primordial no es llegar a una respuesta correcta de inmediato, sino fomentar la formulación de preguntas, la creación de conjeturas y la colaboración en la búsqueda de soluciones. Algunas características distintivas de nuestros círculos incluyen:

Equidad dentro del círculo: Ningún participante recibe un trato preferencial, ya sea por motivos sociales, étnicos o de personalidad. El líder no se percibe como una autoridad, sino como un mediador.

Libertad de expresión: En las sesiones, se alienta a los participantes a presentar sus ideas, sin importar si son acertadas o no. El círculo gira en torno a la discusión de estas ideas, y son los participantes quienes ocupan la mayor parte del tiempo hablando.

No imposición de ideas: Nuestro enfoque se basa en la exploración y la discusión, a diferencia de la enseñanza tradicional, en la que los estudiantes aceptan pasivamente lo que dice el profesor. El líder puede proponer preguntas o guiar las discusiones de los participantes.

Empoderamiento de los participantes: En nuestros círculos, permitimos que los participantes tengan influencia en la dirección del aprendizaje, permitiéndoles decidir hacia dónde quieren llevar su exploración.

Espíritu Cooperativo: Los círculos de aprendizaje se caracterizan por fomentar un espíritu de colaboración entre los participantes. En este sentido, las actividades se diseñan con la intención de que sean discutidas de manera grupal, permitiendo que los participantes compartan sus ideas, perspectivas y conocimientos de manera efectiva. En general, nos es grato que cada grupo de trabajo esté compuesto por unos 5 participantes, lo que favorece la dinámica colaborativa y el intercambio de experiencias en un ambiente de apoyo mutuo. La cooperación y el intercambio de ideas son fundamentales para alcanzar un entendimiento más profundo y enriquecedor.

Formación de los líderes:

En CIRCOAP, reconocemos la importancia de preparar a nuestros líderes de círculos para que desempeñen su papel de manera efectiva y enriquecedora. Por esta razón, requerimos que todos los líderes participen en una formación que dura aproximadamente 10 horas. Durante esta formación, los líderes se sumergen en actividades de inmersión que les permiten experimentar de primera mano la dinámica de los círculos de aprendizaje. Se promueve un entorno de discusión activa y abierta en el que se exploran las dificultades principales que pueden surgir al liderar círculos.

Además, esta formación incluye prácticas reales en las que los líderes tienen la oportunidad de aplicar lo aprendido y adquirir experiencia directa en guiar a los participantes a través de las sesiones de círculo. Esta capacitación integral garantiza que nuestros líderes estén plenamente preparados para ofrecer una experiencia educativa de alta calidad que fomente el pensamiento crítico y la colaboración en los participantes.

*Esto puede ser una diferencia importante con otro proyectos de círculos ya que durante todas las sesiones del círculo se discute sobre el mismo tema en vez de estar proponiendo nuevos problemas en cada sesión.
"""

# Limitar la longitud del texto para evitar problemas con el límite de tokens del LLM
MAX_MANUAL_LENGTH = 15000
manual_reglas_texto = CIRCULOS_DE_APRENDIZAJE_CONTEXTO[:MAX_MANUAL_LENGTH]

# --- Estructura de categorías y subcategorías (reemplaza el Excel) ---
CATEGORIAS_ACTIVIDADES = {
    "Círculos de Matemática y Razonamiento": {
        "Edades": ["5 a 7 años", "8 a 11 años", "12 a 15 años"]
    },
    "Ciencias": {
        "Disciplinas": ["Física", "Química", "Biología"]
    },
    "Tecnología": {
        "Disciplinas": ["Programación", "Robótica"]
    }
}

# --- Función para generar texto con Gemini o GPT ---
def generar_texto_con_llm(model_type, model_name, prompt):
    """
    Generates text using the specified LLM (Gemini or GPT).
    """
    if model_type == "Gemini":
        if not gemini_config_ok:
            st.error("API Key de Gemini no configurada. No se puede generar texto con Gemini.")
            return None
        modelo = genai.GenerativeModel(model_name)
        response = modelo.generate_content(prompt)
        return response.text
    elif model_type == "GPT":
        if not openai_config_ok:
            st.error("API Key de OpenAI no configurada. No se puede generar texto con GPT.")
            return None
        client = openai.OpenAI(api_key=openai.api_key)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000 # Ajusta según necesidad para actividades más largas
            )
            return response.choices[0].message.content
        except openai.AuthenticationError:
            st.error("Error de autenticación con OpenAI. Verifica tu API Key.")
            return None
        except openai.APITimeoutError:
            st.error("Tiempo de espera agotado para la API de OpenAI. Intenta de nuevo.")
            return None
        except openai.APIConnectionError as e:
            st.error(f"Error de conexión con la API de OpenAI: {e}. Verifica tu conexión a internet.")
            return None
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al llamar a la API de OpenAI: {e}")
            return None
    return None

# --- Función para auditar la actividad generada ---
def auditar_actividad_circulo_aprendizaje(model_type, model_name, actividad_generada,
                                          categoria, subcategoria, tema_foco, manual_reglas_texto):
    """
    Audits a generated learning circle activity for compliance with specific criteria.
    """
    auditoria_prompt = f"""
    Eres un experto en validación de actividades didácticas para Círculos de Aprendizaje, especializado en las directrices de CIRCOAP.
    Tu tarea es AUDITAR RIGUROSAMENTE la siguiente actividad generada por un modelo de lenguaje.

    Debes verificar que la actividad cumpla con TODOS los siguientes criterios, prestando especial atención a la alineación con los parámetros proporcionados y a las reglas de formato y contenido.

    --- CRITERIOS DE AUDITORÍA ---
    1.  **Alineación General**: ¿La actividad se alinea a lo esperado para un Círculo de Aprendizaje según el contexto de CIRCOAP (énfasis en descubrimiento, reflexión, colaboración, no reemplazo del aula, etc.)?
    2.  **Fomenta la Discusión**: ¿La actividad está diseñada explícitamente para generar y sostener una discusión activa y profunda entre los niños? ¿Propone preguntas abiertas y situaciones que inviten al diálogo?
    3.  **Preguntas Retadoras**: ¿La actividad incluye preguntas que desafían el pensamiento de los participantes y los guían hacia el descubrimiento autónomo (metodología mayéutica)?
    4.  **Piso Bajo y Techo Alto**: ¿La actividad tiene un "piso bajo" (es accesible para todos los participantes sin importar su nivel inicial) y un "techo alto" (ofrece oportunidades para que los participantes más avanzados profundicen y exploren más allá)?
    5.  **Formato Claro**: ¿La actividad está bien estructurada, con secciones claras como Título, Objetivo, Materiales, Desarrollo de la Actividad, Preguntas para la Discusión, etc.?

    --- PARÁMETROS DE LA ACTIVIDAD ---
    - Categoría: {categoria}
    - Subcategoría/Edad: {subcategoria}
    - Tema de Foco: {tema_foco}

    --- CONTEXTO DE CÍRCULOS DE APRENDIZAJE CIRCOAP ---
    Las siguientes reglas y filosofía son de suma importancia para la calidad y pertinencia de la actividad. Debes asegurar que la actividad cumple con todas ellas.
    {manual_reglas_texto}
    --------------------------------------------------

    ACTIVIDAD A AUDITAR:
    --------------------
    {actividad_generada}
    --------------------

    Devuelve tu auditoría con este formato estructurado:

    VALIDACIÓN DE CRITERIOS:
    - Alineación General: [✅ / ❌] + Comentario (si ❌)
    - Fomenta la Discusión: [✅ / ❌] + Comentario (si ❌)
    - Preguntas Retadoras: [✅ / ❌] + Comentario (si ❌)
    - Piso Bajo y Techo Alto: [✅ / ❌] + Comentario (si ❌)
    - Formato Claro: [✅ / ❌] + Comentario (si ❌)

    DICTAMEN FINAL:
    [✅ CUMPLE TOTALMENTE / ⚠️ CUMPLE PARCIALMENTE / ❌ RECHAZADO]

    OBSERVACIONES FINALES:
    [Explica de forma concisa qué aspectos necesitan mejora, si el dictamen no es ✅. Si es ✅, puedes indicar "La actividad cumple con todos los criterios."]
    """
    return generar_texto_con_llm(model_type, model_name, auditoria_prompt)

# --- Función para generar actividades de Círculos de Aprendizaje ---
def generar_actividad_circulo_aprendizaje(gen_model_type, gen_model_name, audit_model_type, audit_model_name,
                                          categoria_seleccionada, subcategoria_seleccionada, tema_foco_usuario,
                                          manual_reglas_texto, informacion_adicional_usuario):
    """
    Generates a learning circle activity and refines it based on auditor feedback.
    """
    current_activity_text = ""
    auditoria_status = "❌ RECHAZADO"
    audit_observations = ""
    max_refinement_attempts = 3 # Máximo 3 intentos de refinamiento
    attempt = 0

    activity_final_data = None # Variable to store the final activity (approved or last audited version)

    # Classification details for the activity
    classification_details = {
        "Categoría": categoria_seleccionada,
        "Subcategoría/Edad": subcategoria_seleccionada,
        "Tema de Foco": tema_foco_usuario
    }

    while auditoria_status != "✅ CUMPLE TOTALMENTE" and attempt < max_refinement_attempts:
        attempt += 1
        st.info(f"--- Generando/Refinando Actividad (Intento {attempt}/{max_refinement_attempts}) ---")

        prompt_content_for_llm = f"""
        Eres un diseñador experto en actividades didácticas para Círculos de Aprendizaje de CIRCOAP,
        especializado en fomentar el descubrimiento, la discusión y el aprendizaje colaborativo.

        Tu tarea es construir una actividad didáctica de Círculo de Aprendizaje, enfocada en la discusión,
        que cumpla rigurosamente con los principios de CIRCOAP y los siguientes parámetros.

        --- PARÁMETROS DE LA ACTIVIDAD ---
        - Categoría principal: {categoria_seleccionada}
        - Subcategoría / Edad: {subcategoria_seleccionada}
        - Tema principal de foco para la actividad: {tema_foco_usuario}

        --- CARACTERÍSTICAS CLAVE DE LA ACTIVIDAD ---
        1.  **Enfocada en la discusión**: La actividad debe estar diseñada para provocar una discusión rica y profunda entre los niños. Incluye preguntas abiertas que estimulen el diálogo y la reflexión.
        2.  **Preguntas retadoras (mayéutica)**: Incorpora preguntas que guíen a los niños a formular sus propias interrogantes y a descubrir el conocimiento de forma autónoma.
        3.  **Piso bajo y techo alto**: La actividad debe ser accesible para todos los participantes ("piso bajo") y al mismo tiempo permitir a los más avanzados explorar y profundizar ("techo alto"). Proporciona sugerencias para facilitar la participación inicial y para extender el desafío.
        4.  **Colaborativa**: Diseña la actividad para que fomente el aprendizaje colaborativo, no la competencia. Los niños deben trabajar juntos para explorar el tema.
        5.  **Contextualizada y Relevante**: La actividad debe ser relevante para la edad y subcategoría seleccionada, y presentarse en un contexto significativo.

        --- CONTEXTO DE CÍRCULOS DE APRENDIZAJE CIRCOAP ---
        Considera y aplica estrictamente todas las directrices, filosofía y características de CIRCOAP contenidas en el siguiente manual.
        Esto es de suma importancia para la calidad y pertinencia de la actividad.

        Manual de Círculos de Aprendizaje CIRCOAP:
        {manual_reglas_texto}
        ----------------------------------------------------

        --- INFORMACIÓN ADICIONAL PROPORCIONADA POR EL USUARIO ---
        {informacion_adicional_usuario if informacion_adicional_usuario else "No se proporcionó información adicional."}
        ----------------------------------------------------------

        --- FORMATO ESPERADO DE SALIDA ---
        TITULO DE LA ACTIVIDAD: [Título claro y atractivo]
        OBJETIVO: [Qué se espera que los participantes logren o descubran]
        MATERIALES: [Lista de materiales necesarios, si los hay. Si no, poner 'Ninguno'.]
        DESARROLLO DE LA ACTIVIDAD: [Descripción paso a paso de la actividad. Incluye el contexto, la situación o el problema a explorar. Asegúrate de incluir los elementos de "piso bajo" y "techo alto" aquí.]
        PREGUNTAS PARA LA DISCUSIÓN: [Lista de preguntas abiertas que guíen la reflexión y el debate.]
        CIERRE Y REFLEXIÓN: [Cómo concluir la sesión y consolidar el aprendizaje colaborativo.]
        """

        # If not the first attempt, add audit observations for refinement
        if attempt > 1:
            prompt_content_for_llm += f"""
            --- RETROALIMENTACIÓN DE AUDITORÍA PARA REFINAMIENTO ---
            La actividad anterior no cumplió con todos los criterios. Por favor, revisa las siguientes observaciones y mejora la actividad para abordarlas.
            Observaciones del Auditor:
            {audit_observations}
            ---------------------------------------------------
            """
            # Add the previous item for the LLM to reformulate
            prompt_content_for_llm += f"""
            --- ACTIVIDAD ANTERIOR A REFINAR ---
            {current_activity_text}
            -------------------------------
            """
        try:
            with st.spinner(f"Generando contenido con IA ({gen_model_type} - {gen_model_name}, Intento {attempt})..."):
                full_llm_response = generar_texto_con_llm(gen_model_type, gen_model_name, prompt_content_for_llm)

                if full_llm_response is None:
                    st.error(f"Fallo en la generación de texto con {gen_model_type} ({gen_model_name}).")
                    auditoria_status = "❌ RECHAZADO (Error de Generación)"
                    audit_observations = "El modelo de generación no pudo producir una respuesta válida."
                    break

                current_activity_text = full_llm_response # The full response is the activity text for now

                st.subheader(f"Actividad Generada/Refinada (Intento {attempt}):")
                st.markdown(current_activity_text)
                st.markdown("---")

            with st.spinner(f"Auditando actividad ({audit_model_type} - {audit_model_name}, Intento {attempt})..."):
                auditoria_resultado = auditar_actividad_circulo_aprendizaje(
                    audit_model_type, audit_model_name,
                    actividad_generada=current_activity_text,
                    categoria=categoria_seleccionada,
                    subcategoria=subcategoria_seleccionada,
                    tema_foco=tema_foco_usuario,
                    manual_reglas_texto=manual_reglas_texto
                )
                if auditoria_resultado is None:
                    st.error(f"Fallo en la auditoría con {audit_model_type} ({audit_model_name}).")
                    auditoria_status = "❌ RECHAZADO (Error de Auditoría)"
                    audit_observations = "El modelo de auditoría no pudo producir una respuesta válida."
                    break

                st.subheader("Resultado de Auditoría:")
                st.markdown(auditoria_resultado)
                st.markdown("---")

            # --- Extract DICTAMEN FINAL more robustly ---
            # Search for the specific verdict strings directly in the audit result
            if "✅ CUMPLE TOTALMENTE" in auditoria_resultado:
                auditoria_status = "✅ CUMPLE TOTALMENTE"
            elif "⚠️ CUMPLE PARCIALMENTE" in auditoria_resultado:
                auditoria_status = "⚠️ CUMPLE PARCIALMENTE"
            elif "❌ RECHAZADO" in auditoria_resultado:
                auditoria_status = "❌ RECHAZADO"
            else:
                auditoria_status = "❌ RECHAZADO (formato de dictamen inesperado)"
            
            observaciones_start = auditoria_resultado.find("OBSERVACIONES FINALES:")
            if observaciones_start != -1:
                audit_observations = auditoria_resultado[observaciones_start + len("OBSERVACIONES FINALES:"):].strip()
            else:
                audit_observations = "No se pudieron extraer observaciones específicas del auditor. Posiblemente un error de formato en la respuesta del auditor."
            
            st.info(f"Dictamen extraído: {auditoria_status}. Observaciones: {audit_observations[:100]}...")

            # Save activity data, including final audit status and observations
            activity_final_data = {
                "activity_text": current_activity_text,
                "classification": classification_details,
                "final_audit_status": auditoria_status,
                "final_audit_observations": audit_observations
            }

            if auditoria_status == "✅ CUMPLE TOTALMENTE":
                st.success(f"¡La actividad ha sido auditada y CUMPLE TOTALMENTE en el intento {attempt}!")
                break
            else:
                st.warning(f"La actividad necesita refinamiento. Dictamen: {auditoria_status}. Intentando de nuevo...")

        except Exception as e:
            st.error(f"Error durante la generación o auditoría (intento {attempt}): {e}")
            audit_observations = f"Error técnico durante la generación: {e}. Por favor, corrige este problema."
            auditoria_status = "❌ RECHAZADO (error técnico)"
            activity_final_data = {
                "activity_text": current_activity_text if current_activity_text else "No se pudo generar la actividad debido a un error técnico.",
                "classification": classification_details,
                "final_audit_status": auditoria_status,
                "final_audit_observations": audit_observations
            }
            break

    if activity_final_data is None:
        st.error(f"No se pudo generar ninguna actividad después de {max_refinement_attempts} intentos debido a fallas en la generación/auditoría.")
        return []

    return [activity_final_data] # Always return a list with the last processed activity.

# --- Función para exportar actividades a un documento Word ---
def exportar_actividad_a_word(actividades_procesadas_list):
    """
    Exports a list of processed activities to a Word document (.docx) in memory,
    including their classification details and the final audit verdict.
    Returns: BytesIO object of the document.
    """
    doc = docx.Document()
    
    doc.add_heading('Actividades de Círculos de Aprendizaje Generadas y Auditadas', level=1)
    doc.add_paragraph('Este documento contiene las actividades generadas por el sistema de IA y sus resultados de auditoría.')
    doc.add_paragraph('')

    if not actividades_procesadas_list:
        doc.add_paragraph('No se procesaron actividades para este informe.')

    for i, activity_data in enumerate(actividades_procesadas_list):
        activity_text = activity_data["activity_text"]
        classification = activity_data["classification"]
        final_audit_status = activity_data.get("final_audit_status", "N/A")
        final_audit_observations = activity_data.get("final_audit_observations", "No hay observaciones finales de auditoría.")

        doc.add_heading(f'Actividad #{i+1}', level=2)
        
        # Add classification details
        doc.add_paragraph('--- Clasificación de la Actividad ---')
        for key, value in classification.items():
            p = doc.add_paragraph()
            run = p.add_run(f"{key}: ")
            run.bold = True
            p.add_run(str(value))

        doc.add_paragraph('')
        
        # Add the activity text and its formatting
        lines = activity_text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for common headings in the activity text and format them
            if line.upper().startswith("TITULO DE LA ACTIVIDAD:"):
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = docx.shared.Pt(14)
            elif line.upper().startswith(("OBJETIVO:", "MATERIALES:", "DESARROLLO DE LA ACTIVIDAD:",
                                        "PREGUNTAS PARA LA DISCUSIÓN:", "CIERRE Y REFLEXIÓN:")):
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
                run.font.size = docx.shared.Pt(12)
            elif line.startswith("VALIDACIÓN DE CRITERIOS:") or line.startswith("DICTAMEN FINAL:") or line.startswith("OBSERVACIONES FINALES:"):
                p = doc.add_paragraph()
                run = p.add_run(line)
                run.bold = True
            elif line.startswith("✅") or line.startswith("⚠️") or line.startswith("❌"):
                p = doc.add_paragraph(line)
                p.paragraph_format.left_indent = docx.shared.Inches(0.25)
            else:
                doc.add_paragraph(line)
        
        # Add the final verdict and audit observations for EACH activity
        doc.add_paragraph('')
        p = doc.add_paragraph()
        run = p.add_run("--- Resultado Final de Auditoría ---")
        run.bold = True
        doc.add_paragraph(f"**DICTAMEN FINAL:** {final_audit_status}")
        doc.add_paragraph(f"**OBSERVACIONES FINALES:** {final_audit_observations}")
        doc.add_paragraph('')

        doc.add_page_break() # Separate each activity with a page break

    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer

# --- Interfaz de Usuario de Streamlit ---
st.title("📚 Generador y Auditor de Actividades para Círculos de Aprendizaje con IA 🧠")
st.markdown("Esta aplicación genera actividades didácticas enfocadas en la discusión para círculos de aprendizaje y las audita automáticamente.")

st.sidebar.info(f"Contexto de Círculos de Aprendizaje cargado. Longitud: {len(manual_reglas_texto)} caracteres.")

# --- Selección de Modelos ---
st.sidebar.header("Configuración de Modelos de IA")

# Generador (Default to GPT as per request)
st.sidebar.subheader("Modelo para Generación de Actividades")
# User requested GPT to be used for generation
gen_model_type = st.sidebar.radio("Tipo de Modelo", ["GPT", "Gemini"], key="gen_model_type") 
gen_model_name = ""
if gen_model_type == "GPT":
    gen_model_name = st.sidebar.selectbox("Nombre del Modelo GPT", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="gen_gpt_name")
else: # Gemini
    gen_model_name = st.sidebar.selectbox("Nombre del Modelo Gemini", ["gemini-1.5-flash", "gemini-1.5-pro"], key="gen_gemini_name")


# Auditor
st.sidebar.subheader("Modelo para Auditoría de Actividades")
audit_model_type = st.sidebar.radio("Tipo de Modelo", ["Gemini", "GPT"], key="audit_model_type")
audit_model_name = ""
if audit_model_type == "Gemini":
    audit_model_name = st.sidebar.selectbox("Nombre del Modelo Gemini", ["gemini-1.5-flash", "gemini-1.5-pro"], key="audit_gemini_name")
else: # GPT
    audit_model_name = st.sidebar.selectbox("Nombre del Modelo GPT", ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"], key="audit_gpt_name")


# --- Lógica Principal de la Aplicación ---
if gemini_config_ok or openai_config_ok:
    st.header("Selecciona los Criterios para la Actividad")

    # Dropdown for main category
    categoria_seleccionada = st.selectbox(
        "Categoría",
        list(CATEGORIAS_ACTIVIDADES.keys()),
        key="categoria_sel"
    )

    # Conditional dropdown for subcategory/age/discipline
    sub_options = []
    if categoria_seleccionada == "Círculos de Matemática y Razonamiento":
        sub_options = CATEGORIAS_ACTIVIDADES[categoria_seleccionada]["Edades"]
        subcategoria_label = "Rango de Edad"
    else: # Ciencias or Tecnología
        sub_options = CATEGORIAS_ACTIVIDADES[categoria_seleccionada]["Disciplinas"]
        subcategoria_label = "Disciplina"
    
    subcategoria_seleccionada = st.selectbox(
        subcategoria_label,
        sub_options,
        key="subcategoria_sel"
    )

    # Input for focus topic
    tema_foco_usuario = st.text_input(
        """Tema específico o foco para la actividad (ej. "La fotosíntesis en plantas", "Resolución de problemas con fracciones", "Introducción a la robótica con sensores")""",
        key="tema_foco_text"
    )
    if not tema_foco_usuario:
        st.warning("Por favor, ingresa un tema de foco para la actividad.")


    # --- Información Adicional del Usuario ---
    st.subheader("Información Adicional para la Actividad")
    opcion_info_adicional = st.radio(
        "¿Deseas proporcionar alguna información adicional o contexto para la generación de la actividad?",
        ("No", "Sí"),
        key="info_ad_radio"
    )
    informacion_adicional_usuario = ""
    if opcion_info_adicional == "Sí":
        informacion_adicional_usuario = st.text_area("Escribe la información adicional que deseas incluir:", key="info_ad_text")

    # --- Botón para Generar y Auditar ---
    if st.button("Generar y Auditar Actividad"):
        if not tema_foco_usuario:
            st.error("Por favor, ingresa un tema de foco válido para generar la actividad.")
        elif (gen_model_type == "Gemini" and not gemini_config_ok) or (gen_model_type == "GPT" and not openai_config_ok):
            st.error(f"Por favor, configura la API Key para el modelo de generación ({gen_model_type}).")
        elif (audit_model_type == "Gemini" and not gemini_config_ok) or (audit_model_type == "GPT" and not openai_config_ok):
            st.error(f"Por favor, configura la API Key para el modelo de auditoría ({audit_model_type}).")
        else:
            st.markdown("---")
            st.info("Iniciando generación y auditoría de la actividad. Esto puede tardar unos momentos...")

            # Call the function to generate and audit the activity
            activity_processed_individual = generar_actividad_circulo_aprendizaje(
                gen_model_type, gen_model_name, audit_model_type, audit_model_name,
                categoria_seleccionada, subcategoria_seleccionada, tema_foco_usuario,
                manual_reglas_texto, informacion_adicional_usuario
            )

            # Store the processing result in the session state
            if activity_processed_individual:
                st.session_state['last_processed_activity_data'] = activity_processed_individual[0]
                
                if activity_processed_individual[0].get('final_audit_status') == "✅ CUMPLE TOTALMENTE":
                    st.success("¡Actividad generada y aprobada por el auditor! Lista para exportar.")
                else:
                    st.warning(f"Actividad generada pero NO aprobada por el auditor. Dictamen final: {activity_processed_individual[0].get('final_audit_status')}. Se guardará la última versión con observaciones.")
                
                st.subheader("Última Actividad Procesada:")
                st.markdown(activity_processed_individual[0]['activity_text'])
                st.write("--- Clasificación ---")
                for key, value in activity_processed_individual[0]['classification'].items():
                    st.write(f"- **{key}**: {value}")
                
                st.write("--- Resultado Final de Auditoría ---")
                st.write(f"**DICTAMEN FINAL:** {activity_processed_individual[0]['final_audit_status']}")
                st.write(f"**OBSERVACIONES FINALES:** {activity_processed_individual[0]['final_audit_observations']}")
                st.markdown("---")

            else:
                st.error("No se pudo generar ni procesar la actividad. Verifica tus entradas y la conexión a la IA.")
                st.session_state['last_processed_activity_data'] = None
    
    # --- Sección de Exportación a Word (Siempre visible al final) ---
    st.header("Exportar a Documento Word")

    if 'last_processed_activity_data' in st.session_state and st.session_state['last_processed_activity_data'] is not None:
        st.write("Hay una actividad procesada disponible para exportar (aprobada o la última versión con observaciones).")
        nombre_archivo_word = st.text_input("Ingresa el nombre deseado para el archivo Word (sin la extensión .docx):", key="word_filename_activity")
        
        if nombre_archivo_word:
            # Create a list with the single processed activity for the export function
            activities_to_export = [st.session_state['last_processed_activity_data']]
            
            word_buffer = exportar_actividad_a_word(activities_to_export)
            
            st.download_button(
                label="Descargar Documento Word",
                data=word_buffer,
                file_name=f"{nombre_archivo_word}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingprocessingml.document"
            )
            st.info("Haz clic en el botón de arriba para descargar tu archivo Word. Se guardará en la carpeta de descargas de tu navegador.")
        else:
            st.warning("Por favor, ingresa un nombre para el archivo Word para habilitar la descarga.")
    else:
        st.info("No hay actividades procesadas disponibles para exportar a Word en este momento.")
        st.write("Genera y audita una actividad para que esté disponible aquí.")

elif not (gemini_config_ok or openai_config_ok):
    st.info("Por favor, ingresa al menos una API Key de Gemini o OpenAI en la barra lateral para comenzar.")
