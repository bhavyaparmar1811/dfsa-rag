// All calls go to /api/*: the Vite dev server forwards them to FastAPI and adds
// the X-API-Key header (vite.config.ts), so no key lives in the browser.

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
    readonly traceId: string | null = null,
  ) {
    super(detail);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "The API is not reachable. Is FastAPI running on port 8000?");
  }
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    let detail = res.statusText;
    if (typeof body?.detail === "string") detail = body.detail;
    else if (Array.isArray(body?.detail))
      detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    if (res.status === 502 || res.status === 504)
      detail = "The API is not reachable. Is FastAPI running on port 8000?";
    throw new ApiError(res.status, detail, body?.trace_id ?? null);
  }
  return body as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) =>
    request<T>(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
};
