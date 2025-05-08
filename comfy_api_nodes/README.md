# ComfyUI API Nodes

## Introduction 

Below are a collection of nodes that work by calling external APIs. More information available in our [docs](https://docs.comfy.org/tutorials/api-nodes/overview#api-nodes).

## Development

While developing, you should be testing against the Staging environment. To test against staging:

**Install ComfyUI_frontend**

Follow the instructions [here](https://github.com/Comfy-Org/ComfyUI_frontend) to start the frontend server. By default, it will connect to Staging authentication. 

> **Hint:** If you use --front-end-version argument for ComfyUI, it will use production authentication.

```bash
python run main.py --comfy-api-base https://stagingapi.comfy.org
```

API stubs are generated through automatic codegen tools from OpenAPI definitions. Since the Comfy Org OpenAPI definition contains many things from the Comfy Registry as well, we use redocly/cli to filter out only the paths relevant for API nodes.

### Redocly Instructions 

**Tip**
When developing locally, use the `redocly-dev.yaml` file to generate pydantic models. This lets you use stubs for APIs that are not marked `Released` yet.

Before your API node PR merges, make sure to add the `Released` tag to the `openapi.yaml` file and test in staging.

```bash
# Download the OpenAPI file from prod server.
curl -o openapi.yaml https://stagingapi.comfy.org/openapi

# Filter out unneeded API definitions.
npm install -g @redocly/cli
redocly bundle openapi.yaml --output filtered-openapi.yaml --config comfy_api_nodes/redocly-dev.yaml --remove-unused-components

# Generate the pydantic datamodels for validation.
datamodel-codegen --use-subclass-enum --field-constraints --strict-types bytes --input filtered-openapi.yaml --output comfy_api_nodes/apis/__init__.py --output-model-type pydantic_v2.BaseModel

```
