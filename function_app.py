import azure.functions as func
import os, json
from azurefunctions.extensions.http.fastapi import Request, StreamingResponse, HTMLResponse
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage
from azure.ai.inference.models import UserMessage
from azure.core.credentials import AzureKeyCredential

# Azure Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

client = ChatCompletionsClient(
    endpoint="https://models.inference.ai.azure.com",
    credential=AzureKeyCredential('API KEY'),
)

# Get data from Azure Open AI
async def stream_processor(response):
    for chunk in response:
        if chunk.choices:
            yield json.dumps({
                        "content": chunk.choices[0].delta.content
                    }, ensure_ascii=False) + "\n"


# HTTP streaming Azure Function
@app.route(route="chat", methods=[func.HttpMethod.POST])
async def stream_openai_text(req: Request) -> StreamingResponse:
    req_body = await req.json()
    prompt = req_body.get('prompt')
    model = req_body.get('model')  # Get the model parameter from the request body

    response = client.complete(
        messages=[
            SystemMessage(content=""""""),
            UserMessage(content=prompt),
        ],
        model=model,
        temperature=0.8,
        max_tokens=2048,
        top_p=0.1,
        stream=True
    )

    return StreamingResponse(stream_processor(response), media_type="application/x-ndjson")


# Serve the index.html file
@app.route(route="home", methods=[func.HttpMethod.GET])
async def serve_index(req: Request) -> HTMLResponse:
    with open(os.path.join(os.path.dirname(__file__), 'index.html'), 'r') as file:
        html_content = file.read()
    return HTMLResponse(content=html_content, status_code=200)
