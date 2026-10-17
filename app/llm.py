# Módulo del LLM: crea el modelo de chat (OpenAI o Anthropic) a partir de la configuración

# Importa BaseChatModel para tipar el modelo que devuelve la fábrica
from langchain_core.language_models.chat_models import BaseChatModel
# Importa SecretStr para que la API key no se imprima nunca por accidente
from pydantic import SecretStr

# Importa la configuración validada
from app.config import Settings


# Crea el LLM del proveedor elegido en la configuración
def crear_llm(configuracion: Settings) -> BaseChatModel:
    # Convierte la API key en un SecretStr (su repr es '**********')
    clave = SecretStr(configuracion.api_key)

    # Caso OpenAI
    if configuracion.proveedor == "openai":
        # Importa el cliente de OpenAI solo cuando se usa
        from langchain_openai import ChatOpenAI

        # temperature=0 para que las decisiones de herramientas sean lo más estables posible;
        # max_retries=3 reintenta solos los errores transitorios de la API (red, rate limit)
        return ChatOpenAI(model=configuracion.modelo, api_key=clave, temperature=0, max_retries=3)

    # Caso Anthropic
    # Importa el cliente de Anthropic solo cuando se usa
    from langchain_anthropic import ChatAnthropic

    # No se pasa temperature: algunos modelos de Anthropic rechazan ese parámetro (bug visto en la Pre-entrega 1);
    # timeout=None y stop=None son los valores por defecto, se escriben porque el tipado de langchain-anthropic los exige
    return ChatAnthropic(
        model_name=configuracion.modelo,
        api_key=clave,
        max_retries=3,
        timeout=None,
        stop=None,
    )


# Indica si un error del LLM es de credenciales (401/403): no se arregla reintentando ni degradando la respuesta
def es_error_de_credenciales(error: BaseException) -> bool:
    # Recorre la cadena de causas (el error original puede venir envuelto en otro)
    actual: BaseException | None = error
    while actual is not None:
        # Los clientes de OpenAI y Anthropic exponen el código HTTP en status_code
        if getattr(actual, "status_code", None) in (401, 403):
            return True
        actual = actual.__cause__
    return False
