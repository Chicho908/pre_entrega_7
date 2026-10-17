# Módulo de la base de conocimiento: reseñas de clientes simuladas y una búsqueda por palabras clave
# Es la "fuente externa" del investigador (la consigna permite una búsqueda simulada en lugar de Tavily)

# Importa asyncio para simular la latencia de una fuente externa sin bloquear el programa
import asyncio
# Importa re para separar un texto en palabras
import re
# Importa unicodedata para quitar tildes y comparar "envío" con "envio"
import unicodedata
# Importa dataclass para definir cada documento como un objeto inmutable
from dataclasses import dataclass


# Un documento de la base: una reseña de un cliente
@dataclass(frozen=True)
class Documento:
    # Identificador único
    id: str
    # Producto reseñado
    producto: str
    # Tema de la reseña
    categoria: str
    # Puntaje del cliente (1 a 5)
    rating: int
    # Texto de la reseña
    texto: str


# Reseñas simuladas (simples y fáciles de verificar a mano)
DOCUMENTOS: tuple[Documento, ...] = (
    Documento("d01", "auriculares Pulse", "entrega", 5, "Llegaron en dos días, el envío fue rápido y el empaque perfecto. Excelente entrega."),
    Documento("d02", "auriculares Pulse", "entrega", 2, "El envío fue muy lento, llegaron tarde y la caja estaba golpeada. Pésima entrega."),
    Documento("d03", "auriculares Pulse", "calidad", 4, "El sonido es muy bueno y la batería dura mucho. Recomiendo estos auriculares."),
    Documento("d04", "auriculares Pulse", "calidad", 1, "Dejaron de funcionar a la semana. Producto defectuoso y muy decepcionante."),
    Documento("d05", "auriculares Pulse", "soporte", 3, "El soporte respondió en un día y resolvió el reclamo, pero la demora fue larga."),
    Documento("d06", "zapatillas Runner", "calidad", 5, "Muy cómodas, excelente calidad y el talle es perfecto. Las recomiendo."),
    Documento("d07", "zapatillas Runner", "entrega", 4, "El envío llegó a tiempo y rápido. Buen servicio."),
    Documento("d08", "zapatillas Runner", "precio", 2, "Son buenas pero muy caras para lo que ofrecen. Precio decepcionante."),
    Documento("d09", "cafetera Barista", "calidad", 5, "Hace un café genial, es fácil de usar y de limpiar. Excelente compra."),
    Documento("d10", "cafetera Barista", "entrega", 1, "La cafetera llegó rota y tardaron semanas. Entrega pésima, nunca más."),
    Documento("d11", "cafetera Barista", "soporte", 4, "El soporte fue amable y rápido, cambiaron el producto sin problemas."),
    Documento("d12", "teclado Keyra", "precio", 4, "Buen precio y teclas cómodas, lo recomiendo para trabajar."),
)

# Palabras que no aportan a la búsqueda (se ignoran)
PALABRAS_VACIAS = frozenset(
    {"que", "los", "las", "del", "una", "uno", "por", "con", "para", "como", "sobre", "dicen", "opinan",
     "cliente", "clientes", "cual", "cuales", "son", "les", "sus", "mas", "hay", "este", "esta", "analiza",
     "analizar", "investiga", "investigar", "busca", "buscar", "resenas", "opiniones", "general"}
)


# Normaliza un texto: minúsculas y sin tildes
def normalizar(texto: str) -> str:
    # Separa cada letra de su tilde (á -> a + ´)
    descompuesto = unicodedata.normalize("NFD", texto.lower())
    # Descarta las marcas de tilde y deja solo las letras
    return "".join(caracter for caracter in descompuesto if unicodedata.category(caracter) != "Mn")


# Convierte un texto en raíces de palabra (primeros 5 caracteres) para que "entrega" y "entregas" coincidan
def raices(texto: str) -> set[str]:
    # Separa en palabras y se queda con las que tienen contenido
    palabras = re.findall(r"[a-z0-9]+", normalizar(texto))
    # Descarta palabras cortas o vacías y recorta cada una a su raíz
    return {palabra[:5] for palabra in palabras if len(palabra) >= 3 and palabra not in PALABRAS_VACIAS}


# Devuelve un documento por id, o None si no existe (se usa también para validar la procedencia)
def obtener(doc_id: str) -> Documento | None:
    # Recorre los documentos buscando el id
    for documento in DOCUMENTOS:
        if documento.id == doc_id:
            return documento
    return None


# Busca documentos que contengan TODAS las palabras clave de la consulta; devuelve los más relevantes primero
# (si una palabra no aparece en un documento, ese documento no sirve: es mejor no encontrar nada que devolver datos de otro tema)
async def buscar(consulta: str, max_resultados: int = 5) -> list[Documento]:
    # Simula la latencia de una fuente externa (cede el control al event loop)
    await asyncio.sleep(0)
    # Obtiene las raíces de la consulta
    buscadas = raices(consulta)
    # Una consulta sin palabras con contenido no puede buscar nada
    if not buscadas:
        return []
    # Lista donde se acumulan (puntaje, documento)
    puntuados: list[tuple[int, Documento]] = []
    for documento in DOCUMENTOS:
        # Palabras del documento: producto, categoría y texto
        etiquetas = raices(f"{documento.producto} {documento.categoria}")
        palabras_doc = etiquetas | raices(documento.texto)
        # Solo sirve si TODAS las palabras buscadas aparecen en el documento
        if not buscadas <= palabras_doc:
            continue
        # Producto y categoría pesan el doble que las palabras del texto
        puntaje = 2 * len(buscadas & etiquetas) + len(buscadas & raices(documento.texto))
        puntuados.append((puntaje, documento))
    # Ordena por puntaje descendente (y por id para que el orden sea estable)
    puntuados.sort(key=lambda par: (-par[0], par[1].id))
    # Devuelve solo los documentos, recortados al máximo pedido
    return [documento for _, documento in puntuados[:max_resultados]]
