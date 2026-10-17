# Módulo de traza: convierte lo que pasa en el grafo en una traza legible (JSON y texto)

# Importa Sequence para tipar las listas de mensajes
from collections.abc import Sequence
# Importa Any para tipar los diccionarios de la traza
from typing import Any

# Importa los tipos de mensajes que genera un especialista
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

# Largo máximo del resultado de una herramienta en la traza (evita traza ilegible)
MAX_RESULTADO = 220


# Normaliza el contenido de un mensaje a texto (Anthropic a veces devuelve una lista de bloques)
def extraer_texto(mensaje: BaseMessage) -> str:
    # Toma el contenido crudo del mensaje
    contenido = mensaje.content
    # Si ya es texto, lo devuelve tal cual
    if isinstance(contenido, str):
        return contenido
    # Si es una lista de bloques, junta el texto de cada bloque
    partes: list[str] = []
    for bloque in contenido:
        # Un bloque puede ser un texto suelto
        if isinstance(bloque, str):
            partes.append(bloque)
        # O un diccionario de tipo "text"
        elif bloque.get("type") == "text":
            partes.append(str(bloque.get("text", "")))
    # Devuelve el texto completo
    return "".join(partes)


# Recorta un texto largo agregando puntos suspensivos
def recortar(texto: str, maximo: int = MAX_RESULTADO) -> str:
    return texto if len(texto) <= maximo else texto[: maximo - 1] + "…"


# Convierte los mensajes de un especialista en pasos de traza (decisión de herramienta, observación y respuesta)
def extraer_pasos(mensajes: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    # Lista donde se acumulan los pasos
    pasos: list[dict[str, Any]] = []
    # Recorre los mensajes en orden
    for mensaje in mensajes:
        # Mensaje del modelo
        if isinstance(mensaje, AIMessage):
            # Si pidió herramientas, registra una decisión por cada llamada
            if mensaje.tool_calls:
                for llamada in mensaje.tool_calls:
                    pasos.append({"tipo": "decide_herramienta", "herramienta": llamada["name"], "argumentos": llamada["args"]})
            # Si no pidió herramientas, es su respuesta final
            else:
                pasos.append({"tipo": "respuesta", "contenido": recortar(extraer_texto(mensaje).strip())})
        # Resultado de una herramienta (la "observación")
        elif isinstance(mensaje, ToolMessage):
            pasos.append({"tipo": "herramienta_devuelve", "herramienta": mensaje.name, "resultado": recortar(extraer_texto(mensaje))})
    # Devuelve los pasos en orden
    return pasos


# Describe en texto los pasos internos de un especialista (con sangría)
def _formatear_pasos(pasos: Sequence[dict[str, Any]]) -> list[str]:
    lineas: list[str] = []
    for paso in pasos:
        # Decisión de usar una herramienta
        if paso["tipo"] == "decide_herramienta":
            argumentos = ", ".join(f"{clave}={valor!r}" for clave, valor in paso["argumentos"].items())
            lineas.append(f"      · usa la herramienta {paso['herramienta']}({recortar(argumentos, 120)})")
        # Resultado de la herramienta
        elif paso["tipo"] == "herramienta_devuelve":
            lineas.append(f"      · la herramienta devuelve: {paso['resultado']}")
        # Respuesta final del especialista
        else:
            lineas.append(f"      · responde: {paso['contenido']}")
    return lineas


# Convierte la actualización de UN nodo (lo que escribió en el estado) en líneas de texto
def formatear_evento(nodo: str, cambios: dict[str, Any]) -> list[str]:
    lineas: list[str] = []
    # El supervisor deja su decisión en "decisiones"
    if nodo == "supervisor":
        decision = cambios["decisiones"][0]
        lineas.append(f"[supervisor] paso {decision['paso']}: el LLM propone → {decision['propuesta']}; decisión final → {decision['final']}")
        lineas.append(f"      razón: {decision['razon']}")
        # Si un guardarraíl corrigió la propuesta del LLM, lo deja a la vista
        if decision["guardarrail"]:
            lineas.append(f"      guardarraíl: {decision['guardarrail']}")
        # Muestra la instrucción que recibirá el especialista
        if decision["instruccion"]:
            lineas.append(f"      instrucción enviada: {recortar(decision['instruccion'].replace(chr(10), ' '), 300)}")
    # Los especialistas dejan su aporte en "contribuciones"
    elif nodo in ("investigador", "analista"):
        aporte = cambios["contribuciones"][0]
        lineas.append(f"[{nodo}] intento {aporte['intento']}: {aporte['resumen']}")
        lineas.extend(_formatear_pasos(aporte["pasos"]))
        if aporte["error"]:
            lineas.append(f"      ERROR: {aporte['error']}")
    # La validación deja su veredicto en "contribuciones"
    elif nodo == "validacion":
        veredicto = cambios["contribuciones"][0]
        marca = "OK" if veredicto["valido"] else "RECHAZADO"
        lineas.append(f"[validación] {veredicto['sobre']}: {marca} — {veredicto['resumen']}")
    # El sintetizador deja la respuesta final
    elif nodo == "sintetizador":
        lineas.append(f"[sintetizador] respuesta final: {cambios['respuesta_final']}")
    # La aprobación humana deja la decisión del revisor
    elif nodo == "aprobacion":
        lineas.append(f"[aprobación humana] decisión del revisor: {cambios['aprobacion']['estado']} {cambios['aprobacion']['comentario']}".rstrip())
    # El despacho deja el resultado del envío
    elif nodo == "despacho":
        lineas.append(f"[despacho] {cambios['despacho']}")
    return lineas
