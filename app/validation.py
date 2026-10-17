# Módulo de validación: el "árbitro" determinista. Revisa lo que entregó cada especialista ANTES de que el supervisor decida
# No usa LLM: son reglas de código, así que el veredicto es siempre el mismo y el mensaje de error es específico

# Importa Any para tipar los diccionarios de estado
from typing import Any

# Importa ValidationError para convertir los errores de Pydantic en texto claro
from pydantic import ValidationError

# Importa la base de conocimiento para comprobar la procedencia de los datos
from app import knowledge_base
# Importa los contratos
from app.state import EstadoOrquestador, ResultadoAnalisis, ResultadoInvestigacion

# Diferencia máxima tolerada entre un promedio informado y el recalculado
TOLERANCIA = 0.011


# Convierte un ValidationError de Pydantic en una frase corta y accionable
def resumir_error(error: ValidationError) -> str:
    # Una parte por cada campo inválido: "campo: mensaje"
    partes = [f"{'.'.join(str(p) for p in e['loc']) or 'resultado'}: {e['msg']}" for e in error.errors()]
    # Une las partes (máximo 4 para no llenar el prompt del reintento)
    return "; ".join(partes[:4])


# Valida la investigación: contrato + procedencia de cada hallazgo. Devuelve "" si está bien o el motivo del rechazo
def validar_investigacion(investigacion: dict[str, Any]) -> str:
    # Si el especialista no entregó nada (falló o no encontró datos)
    if not investigacion:
        return "el investigador no entregó ningún resultado"
    # Valida el contrato con Pydantic
    try:
        resultado = ResultadoInvestigacion.model_validate(investigacion)
    except ValidationError as error:
        # Sin reseñas hay dos causas muy distintas: la fuente se cayó (falla técnica, conviene reintentar) o no hay datos (no conviene insistir)
        if not investigacion.get("hallazgos") and investigacion.get("fuente_caida"):
            return "la fuente de reseñas no respondió (falla técnica, no falta de datos): reintentá la misma búsqueda; si vuelve a fallar, no hay forma de responder y hay que informar la falla"
        if not investigacion.get("hallazgos"):
            return "no se encontró ninguna reseña sobre lo pedido: probá sinónimos del mismo producto o tema; si no existe en la base, no hay datos y no hay que reemplazarlo por otro tema"
        return resumir_error(error)
    # Regla de procedencia: cada hallazgo tiene que existir realmente en la base y coincidir con su texto
    for hallazgo in resultado.hallazgos:
        documento = knowledge_base.obtener(hallazgo.doc_id)
        if documento is None or documento.texto != hallazgo.texto:
            return f"el hallazgo '{hallazgo.doc_id}' no coincide con ningún documento real de la base"
    return ""


# True si el valor informado coincide (dentro de la tolerancia) con alguno de los promedios recalculados
def _coincide(informado: float, candidatos: list[float]) -> bool:
    return any(abs(informado - candidato) <= TOLERANCIA for candidato in candidatos)


# Valida el análisis: contrato + coherencia con la investigación (el árbitro recalcula los números)
def validar_analisis(analisis: dict[str, Any], investigacion: dict[str, Any]) -> str:
    # Si el especialista no entregó nada
    if not analisis:
        return "el analista no entregó ningún resultado"
    # Valida el contrato con Pydantic
    try:
        resultado = ResultadoAnalisis.model_validate(analisis)
    except ValidationError as error:
        return f"faltan o son inválidos estos datos del análisis → {resumir_error(error)}. Recordá calcular_estadisticas para 'sentimiento' y para 'rating'"
    # Datos de la investigación contra los que se contrasta (ya validados antes de llegar acá)
    try:
        hallazgos = ResultadoInvestigacion.model_validate(investigacion).hallazgos
    # Sin una investigación válida no hay contra qué contrastar (el supervisor no debería llegar acá)
    except ValidationError:
        return "no se puede validar el análisis porque no hay una investigación válida"
    # Regla de cobertura: debe haber analizado exactamente las reseñas encontradas
    esperados = {h.doc_id for h in hallazgos}
    analizados = {s.doc_id for s in resultado.sentimientos}
    if analizados != esperados:
        faltan = sorted(esperados - analizados)
        sobran = sorted(analizados - esperados)
        return f"el análisis no cubre las mismas reseñas que la investigación (faltan: {faltan}, sobran: {sobran})"
    # Regla de consistencia del rating: el árbitro recalcula el promedio con los datos reales
    # Se acepta el promedio global o el de cualquier producto (en una comparación el analista calcula uno por producto)
    ratings_por_producto: dict[str, list[float]] = {}
    for h in hallazgos:
        ratings_por_producto.setdefault(h.producto, []).append(float(h.rating))
    rating_global = sum(h.rating for h in hallazgos) / len(hallazgos)
    candidatos_rating = [rating_global, *(sum(v) / len(v) for v in ratings_por_producto.values())]
    if not _coincide(resultado.rating_promedio, candidatos_rating):
        return f"rating_promedio informado ({resultado.rating_promedio}) no coincide con el real ({rating_global:.2f}); recalculalo con los ratings exactos de las reseñas"
    # Regla de consistencia del sentimiento: recalcula el promedio con los puntajes por documento (global o por producto)
    producto_de = {h.doc_id: h.producto for h in hallazgos}
    puntajes_por_producto: dict[str, list[float]] = {}
    for s in resultado.sentimientos:
        puntajes_por_producto.setdefault(producto_de[s.doc_id], []).append(s.puntaje)
    sentimiento_global = sum(s.puntaje for s in resultado.sentimientos) / len(resultado.sentimientos)
    candidatos_sentimiento = [sentimiento_global, *(sum(v) / len(v) for v in puntajes_por_producto.values())]
    if not _coincide(resultado.puntaje_promedio, candidatos_sentimiento):
        return f"puntaje_promedio informado ({resultado.puntaje_promedio}) no coincide con el de los puntajes por reseña ({sentimiento_global:.2f}); recalculalo con los puntajes exactos"
    return ""


# Nodo de validación: revisa al último especialista que trabajó y deja el veredicto en el estado
async def nodo_validacion(state: EstadoOrquestador) -> dict[str, Any]:
    # Averigua a quién hay que revisar
    agente = state["ultimo_agente"]
    # Revisa la investigación
    if agente == "investigador":
        error = validar_investigacion(state.get("investigacion", {}))
        cambios: dict[str, Any] = {"investigacion_valida": error == "", "error_investigacion": error}
    # Revisa el análisis (contrastándolo con la investigación)
    else:
        error = validar_analisis(state.get("analisis", {}), state.get("investigacion", {}))
        cambios = {"analisis_valido": error == "", "error_analisis": error}
    # Deja el veredicto también en el registro de contribuciones
    cambios["contribuciones"] = [
        {"agente": "validacion", "sobre": agente, "valido": error == "", "resumen": error or "cumple el contrato y las reglas de negocio"}
    ]
    return cambios
