# AINA Translator – Microservicio de Traducción del Catalán

Microservicio HTTP que traduce texto en **catalán** a **en, es, fr, pt, it, de** usando los modelos oficiales de [Projecte AINA](https://huggingface.co/projecte-aina) en Hugging Face.

## Arquitectura

```
Client  ──POST /translate──▶  FastAPI (uvicorn :8000)
                                 │
                    ┌────────────┼────────────────┐
                    ▼            ▼                 ▼
              ca→en model   ca→es model  ...  ca→de model
              (GPU fp16 or CPU fp32)
```

Los 6 modelos se cargan en memoria al arrancar. Cada request reutiliza los modelos ya cargados → latencia baja.

---

## Modelos utilizados

| Target | Modelo HF |
|--------|-----------|
| en | `projecte-aina/aina-translator-ca-en` |
| es | `projecte-aina/aina-translator-ca-es` |
| fr | `projecte-aina/aina-translator-ca-fr` |
| pt | `projecte-aina/aina-translator-ca-pt` |
| it | `projecte-aina/aina-translator-ca-it` |
| de | `projecte-aina/aina-translator-ca-de` |

---

## 1. Construir la imagen Docker

```bash
# Desde el directorio del proyecto
docker build -t ghcr.io/<TU_USUARIO>/aina-translator:latest .
```

> **Nota:** La imagen base usa CUDA 12.1. Si no tienes GPU, el servicio degrada automáticamente a CPU (fp32), pero el cold-start será más lento.

### Alternativa: imagen base PyTorch

Si hay problemas con la instalación de torch sobre la imagen CUDA, puedes cambiar el `FROM` en el Dockerfile a:

```dockerfile
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
```

Y eliminar la línea `RUN pip install torch ...` del Dockerfile.

---

## 2. Subir la imagen

### GitHub Container Registry (GHCR)

```bash
# Autenticarse (necesitas un PAT con permisos write:packages)
echo $GITHUB_TOKEN | docker login ghcr.io -u <TU_USUARIO> --password-stdin

# Push
docker push ghcr.io/<TU_USUARIO>/aina-translator:latest
```

### Docker Hub (alternativa)

```bash
docker tag ghcr.io/<TU_USUARIO>/aina-translator:latest <TU_USUARIO>/aina-translator:latest
docker login
docker push <TU_USUARIO>/aina-translator:latest
```

---

## 3. Desplegar en RunPod

1. Ve a [runpod.io](https://runpod.io) → **Pods** → **+ Deploy**
2. Selecciona una GPU (recomendado: **RTX 3090 / RTX 4090 / A40** con ≥24 GB VRAM)
3. En **Container Image**: `ghcr.io/<TU_USUARIO>/aina-translator:latest`
4. En **Expose HTTP Ports**: `8000`
5. (Opcional) Volumen persistente en `/app/.hf_cache` para cachear modelos entre reinicios
6. **Deploy**

### Obtener el endpoint

Una vez el pod esté en estado **Running**, tu endpoint será:

```
https://<POD_ID>-8000.proxy.runpod.net
```

Lo puedes ver en la UI de RunPod → Pods → Connect → HTTP.

---

## 4. Probar el servicio

### Health check

```bash
curl https://<POD_ID>-8000.proxy.runpod.net/health
```

### Traducción con curl

```bash
curl -X POST https://<POD_ID>-8000.proxy.runpod.net/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Bon dia a tothom",
    "targets": ["en", "es", "fr", "pt", "it", "de"]
  }'
```

Respuesta esperada:

```json
{
  "en": "Good morning everyone",
  "es": "Buenos días a todos",
  "fr": "Bonjour à tous",
  "pt": "Bom dia a todos",
  "it": "Buongiorno a tutti",
  "de": "Guten Morgen allerseits"
}
```

### Traducción con Python

```python
import requests

URL = "https://<POD_ID>-8000.proxy.runpod.net"

r = requests.post(
    f"{URL}/translate",
    json={
        "text": "Bon dia",
        "targets": ["en", "es"],
    },
)
print(r.json())
# {"en": "Good morning", "es": "Buenos días"}
```

### Parámetros opcionales

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `max_new_tokens` | 256 | Máximo de tokens generados por traducción |
| `num_beams` | 1 | Anchura de beam search (1 = greedy, rápido; 4–5 = mejor calidad, más lento) |

Ejemplo con beam search:

```bash
curl -X POST .../translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "El projecte va avançant bé",
    "targets": ["en"],
    "num_beams": 4
  }'
```

---

## 5. Rendimiento y costes

### Tiempos de referencia

| Fase | GPU (RTX 4090) | CPU |
|------|----------------|-----|
| Cold start (descarga modelos) | ~5–10 min | ~10–15 min |
| Warm start (modelos cacheados) | ~30–60 s | ~2–3 min |
| Traducción (1 idioma, ~20 tokens) | ~100–300 ms | ~2–5 s |
| Traducción (6 idiomas, ~20 tokens) | ~0.5–1.5 s | ~15–30 s |

### VRAM

Cada modelo ocupa ~300–500 MB en fp16. Con 6 modelos: **~2–3 GB de VRAM**. Cualquier GPU con ≥4 GB es suficiente.

### Costes RunPod (orientativos)

| GPU | Coste/hora (aprox.) |
|-----|---------------------|
| RTX 3090 | ~$0.22/h |
| RTX 4090 | ~$0.39/h |
| A40 (48 GB) | ~$0.39/h |

> **Consejo:** Usa **Start/Stop** en la UI de RunPod para no pagar tiempo idle. Con la API de RunPod puedes automatizar start/stop programáticamente.

---

## 6. Mejores prácticas

- **Cache de modelos:** Los modelos de HF se cachean en `/app/.hf_cache` dentro del contenedor. Si montas un volumen persistente ahí, los reinicios serán mucho más rápidos (warm start).
- **Cold start:** La primera vez que arranque el pod, descargará ~6 modelos desde HF (~2–3 GB total). Mantén el pod en running o usa volúmenes persistentes.
- **Escalado horizontal:** Para más throughput, despliega múltiples pods y balancea con un proxy/load balancer.
- **Seguridad:** El endpoint de RunPod es público por defecto. Para producción, añade autenticación (API key header, JWT, etc.) o usa RunPod Serverless con auth integrado.
- **Calidad vs. velocidad:** `num_beams=1` (greedy) es el más rápido. Para traducciones de mayor calidad, usa `num_beams=4` o `num_beams=5` a costa de mayor latencia.

---

## Estructura del proyecto

```
aina-translator/
├── app.py              # FastAPI server
├── requirements.txt    # Python dependencies
├── Dockerfile          # GPU-ready container
└── README.md           # This file
```

---

## Documentación interactiva

Una vez desplegado, accede a la doc de la API en:

```
https://<POD_ID>-8000.proxy.runpod.net/docs
```

(Swagger UI generada automáticamente por FastAPI)
