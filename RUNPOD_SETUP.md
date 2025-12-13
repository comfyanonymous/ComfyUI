# RunPod Deployment Configuration

This document describes how to configure environment variables for ComfyUI when deploying to RunPod Serverless.

## Environment Variables Setup

### Required: Google API Key for Nano Banana

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Navigate to **Templates** → Select your ComfyUI template → **Edit**
3. Scroll to **Environment Variables** section
4. Add the following variable:
   - **Key**: `GOOGLE_API_KEY`
   - **Value**: Your Google AI API key (get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

### Optional: Vertex AI Configuration

If you prefer to use Vertex AI instead of the Google AI API:

- **Key**: `PROJECT_ID`
- **Value**: Your GCP project ID
- **Key**: `LOCATION`
- **Value**: `us-central1` (or your preferred region)

**Note:** Vertex AI requires Application Default Credentials (ADC) to be configured in your container image.

### Optional: OpenTelemetry Configuration

For observability and monitoring:

- **Key**: `OTEL_EXPORTER_OTLP_ENDPOINT`
- **Value**: `http://your-otel-collector:4317`
- **Key**: `OTEL_SERVICE_NAME`
- **Value**: `comfyui`
- **Key**: `OTEL_RESOURCE_ATTRIBUTES`
- **Value**: `service.name=comfyui`

## How It Works

The ComfyUI Docker image is configured to read environment variables directly from `os.environ`, which means:

1. **Local Development**: Variables are loaded from `.env` file via `docker-compose.yml`
2. **RunPod Deployment**: Variables are set in the RunPod template interface
3. **Code Compatibility**: No code changes needed—both methods use `os.environ`

## Verification

After setting the environment variables in RunPod:

1. Deploy or restart your serverless endpoint
2. Check the logs for "GOOGLE_API_KEY" to verify it's set
3. Test Nano Banana nodes to confirm authentication works

## Differences from Local Setup

| Aspect | Local (Docker Compose) | RunPod Serverless |
|--------|----------------------|-------------------|
| **Config Method** | `.env` file | RunPod template UI |
| **GPU Support** | Optional (can run CPU mode) | Typically GPU-enabled |
| **Persistence** | Docker volumes | RunPod storage |
| **API Key Storage** | Local `.env` (gitignored) | RunPod environment vars |

## Security Notes

- Environment variables in RunPod are encrypted at rest
- Never commit `.env` files with real API keys to Git
- Rotate your API keys periodically
- Use different API keys for development vs production if possible

