# Módulo de estado: define el esquema de datos compartido entre el supervisor y los especialistas
# Regla de oro: cada campo del estado tiene UN solo dueño (un nodo que lo escribe); los demás solo lo leen

# Importa add para usarlo como reducer: concatena las listas en vez de reemplazarlas
from operator import add
# Importa los tipos para anotar el estado y los contratos
from typing import Annotated, Any, Literal

# Importa HumanMessage para armar el mensaje inicial del usuario
from langchain_core.messages import HumanMessage
# Importa MessagesState: ya trae el campo "messages" con el reducer add_messages
from langgraph.graph import MessagesState
# Importa BaseModel y Field para definir contratos validados (los especialistas no pueden devolver cualquier cosa)
from pydantic import BaseModel, Field

# Nombres de los nodos a los que puede ir el supervisor (se usa también para tipar las aristas condicionales)
Destino = Literal["investigador", "analista", "sintetizador"]
# Etiquetas posibles de sentimiento
Etiqueta = Literal["positivo", "neutro", "negativo"]


# Un dato encontrado por el investigador: viene siempre de un documento real de la base de conocimiento
class Hallazgo(BaseModel):
    # Identificador del documento (sirve de procedencia: de dónde salió el dato)
    doc_id: str = Field(min_length=1)
    # Producto sobre el que habla el documento
    producto: str = Field(min_length=1)
    # Tema del documento (entrega, calidad, precio o soporte)
    categoria: str = Field(min_length=1)
    # Puntaje que dio el cliente, de 1 a 5
    rating: int = Field(ge=1, le=5)
    # Texto de la reseña
    texto: str = Field(min_length=1)


# Contrato de salida del investigador: debe traer al menos un hallazgo
class ResultadoInvestigacion(BaseModel):
    # Instrucción que recibió el investigador
    consulta: str = Field(min_length=3)
    # Documentos encontrados (mínimo uno: sin datos no hay nada que analizar)
    hallazgos: list[Hallazgo] = Field(min_length=1)
    # Resumen breve escrito por el LLM del investigador
    resumen: str = Field(min_length=1)


# Sentimiento calculado para un documento puntual
class SentimientoDoc(BaseModel):
    # Documento al que pertenece el puntaje
    doc_id: str = Field(min_length=1)
    # Puntaje de sentimiento entre -1 (muy negativo) y 1 (muy positivo)
    puntaje: float = Field(ge=-1, le=1)
    # Etiqueta derivada del puntaje
    etiqueta: Etiqueta


# Contrato de salida del analista: números calculados por herramientas + conclusiones del LLM
class ResultadoAnalisis(BaseModel):
    # Un sentimiento por cada documento analizado
    sentimientos: list[SentimientoDoc] = Field(min_length=1)
    # Promedio del puntaje de sentimiento
    puntaje_promedio: float = Field(ge=-1, le=1)
    # Promedio del rating (1 a 5) de los documentos
    rating_promedio: float = Field(ge=1, le=5)
    # Cantidad de documentos analizados
    cantidad_analizada: int = Field(gt=0)
    # Etiqueta global derivada del puntaje promedio
    sentimiento_global: Etiqueta
    # Conclusiones escritas por el LLM del analista
    conclusiones: str = Field(min_length=1)


# Decisión estructurada del supervisor: el LLM está obligado a elegir uno de los nodos válidos
class Decision(BaseModel):
    # A qué nodo ir ahora (el Literal impide inventar nombres de nodos)
    siguiente: Destino = Field(description="Próximo nodo: investigador, analista o sintetizador.")
    # Instrucción AUTOCONTENIDA para el especialista (campo obligatorio: sin valor por defecto, para que los proveedores con esquema estricto lo acepten)
    instruccion: str = Field(description="Instrucción autocontenida para el especialista; vacía si el destino es sintetizador.")
    # Razón breve de la decisión (queda en la traza para auditar)
    razon: str = Field(min_length=1, description="Una frase que explica por qué se eligió ese paso.")


# Estado compartido del orquestador
class EstadoOrquestador(MessagesState):
    # --- Entrada (la completa la API al crear la tarea) ---
    # Destinatario al que hay que enviar el informe ("" = solo responder, sin efectos secundarios)
    notificar_a: str
    # --- Escribe el supervisor ---
    # Próximo nodo elegido (lo lee la arista condicional)
    siguiente: str
    # Instrucción específica para el especialista que va a trabajar
    instruccion: str
    # Cantidad de decisiones que ya tomó el supervisor (contador anti "supervisor infinito")
    pasos_supervisor: int
    # Historial auditable de decisiones (propuesta del LLM, decisión final y guardarraíl aplicado)
    decisiones: Annotated[list[dict[str, Any]], add]
    # --- Escribe el investigador ---
    # Resultado crudo de la investigación (se valida en el nodo de validación)
    investigacion: dict[str, Any]
    # Veces que se ejecutó el investigador
    intentos_investigador: int
    # --- Escribe el analista ---
    # Resultado crudo del análisis (se valida en el nodo de validación)
    analisis: dict[str, Any]
    # Veces que se ejecutó el analista
    intentos_analista: int
    # --- Escriben los dos especialistas (de a uno por vez) ---
    # Último especialista que trabajó (le dice a la validación qué revisar)
    ultimo_agente: str
    # Registro de quién contribuyó con qué (el reducer add concatena, nunca pisa)
    contribuciones: Annotated[list[dict[str, Any]], add]
    # --- Escribe la validación ---
    # True si la investigación pasó el contrato y las reglas de negocio
    investigacion_valida: bool
    # True si el análisis pasó el contrato y las reglas de negocio
    analisis_valido: bool
    # Motivo específico del rechazo de la investigación ("" si está bien)
    error_investigacion: str
    # Motivo específico del rechazo del análisis ("" si está bien)
    error_analisis: str
    # --- Escribe el sintetizador ---
    # Respuesta final para el usuario
    respuesta_final: str
    # --- Escribe la aprobación humana (solo en tareas críticas) ---
    # Decisión del revisor: {"estado": "aprobada" | "rechazada", "comentario": str}
    aprobacion: dict[str, Any]
    # --- Escribe el despacho (la única acción con efectos secundarios) ---
    # Resultado del envío del informe ("" si la tarea no era crítica)
    despacho: str


# Arma el estado inicial de una ejecución: todos los campos con su valor neutro
def estado_inicial(pregunta: str, notificar_a: str = "") -> EstadoOrquestador:
    return {
        "messages": [HumanMessage(content=pregunta)],
        "notificar_a": notificar_a,
        "siguiente": "",
        "instruccion": "",
        "pasos_supervisor": 0,
        "decisiones": [],
        "investigacion": {},
        "intentos_investigador": 0,
        "analisis": {},
        "intentos_analista": 0,
        "ultimo_agente": "",
        "contribuciones": [],
        "investigacion_valida": False,
        "analisis_valido": False,
        "error_investigacion": "",
        "error_analisis": "",
        "respuesta_final": "",
        "aprobacion": {},
        "despacho": "",
    }
