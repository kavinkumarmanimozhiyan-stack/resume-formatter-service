# Integrating another frontend

Use this backend through its `/api` base URL. Local development uses
`http://localhost:8000/api`; production should use your deployed backend URL,
for example `https://api.example.com/api`.

## 1. Allow the frontend origin

Copy `.env.example` to `.env` if needed, then set `ALLOWED_ORIGINS` to a
comma-separated list of exact frontend origins. Do not include a trailing
slash.

```env
ALLOWED_ORIGINS=http://localhost:5173,https://app.example.com
```

Restart the backend after changing environment variables:

```powershell
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## 2. Configure the frontend

For a Vite frontend, set an environment variable in `.env`:

```env
VITE_RESUME_API_URL=http://localhost:8000/api
```

Use it in a shared client module:

```js
const API_URL = import.meta.env.VITE_RESUME_API_URL;

export async function generateResume({ templateFile, resumeFile, outputFormat = 'pdf' }) {
  const formData = new FormData();
  formData.append('format_template', templateFile);
  formData.append('content_file', resumeFile);
  formData.append('output_format', outputFormat);

  const response = await fetch(`${API_URL}/generate-resume`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) throw new Error((await response.json()).error || 'Resume generation failed');
  return response;
}
```

Do not set the `Content-Type` header manually when sending `FormData`; the
browser supplies the multipart boundary.

## API endpoints

| Purpose | Method and path | Request |
| --- | --- | --- |
| Health check | `GET /api/health` | none |
| Generate a PDF/HTML resume | `POST /api/generate-resume` | multipart: `format_template`, `content_file`, optional `output_format` |
| Generate HTML for preview/Word export | `POST /api/generate-resume-html` | same multipart fields |
| Upload a reusable template | `POST /api/templates` | multipart: `file`, optional `name` |
| List saved templates | `GET /api/templates` | none |
| Delete template | `DELETE /api/templates/{template_id}` | none |
| Rename template | `PATCH /api/templates/{template_id}` | JSON: `{ "name": "..." }` |
| Manage LLM settings | `/api/llm-settings` | JSON; see Swagger docs |

Interactive request/response schemas are available at `http://localhost:8000/docs`.

## Downloading a generated file

The generate endpoint returns a file response. In a browser, turn it into a
download like this:

```js
const response = await generateResume({ templateFile, resumeFile });
const blob = await response.blob();
const url = URL.createObjectURL(blob);
const link = Object.assign(document.createElement('a'), { href: url, download: 'formatted-resume.pdf' });
link.click();
URL.revokeObjectURL(url);
```

## Deployment notes

- Keep provider and Cloudinary secrets only in the backend environment; never place them in frontend variables.
- If the frontend and backend use different domains, configure `ALLOWED_ORIGINS` before deploying or browsers will block requests with CORS errors.
- Existing unprefixed endpoints remain available for the bundled frontend. New integrations should use `/api`.
