export type ParamSpec = {
  type: 'int' | 'float' | 'str' | 'bool' | 'enum'
  default: unknown
  values?: string[] | null
  description?: string | null
  flag?: string | null
}

export type Recipe = {
  id: string
  name: string
  description?: string | null
  engine: string
  source: Record<string, unknown>
  served_name: string
  vram_gb: number
  disk_gb?: number | null
  capabilities: string[]
  params: Record<string, ParamSpec>
  extra_args: string[]
  origin: string
  deployment_status: string | null
}

export type Deployment = {
  id: string
  recipe_id: string
  name: string
  engine: string
  served_name: string
  capabilities: string[]
  params: Record<string, unknown>
  status: 'pending' | 'downloading' | 'starting' | 'ready' | 'stopping' | 'stopped' | 'failed'
  error: string | null
  created_at: number
  ready_at: number | null
  progress: { done_bytes: number; total_bytes: number | null } | null
  endpoint: { base_url: string; model: string }
}

export type Gpu = {
  index: number
  name: string
  memory_total_mib: number
  memory_used_mib: number
  memory_free_mib: number
  utilization_pct: number | null
}

export type Status = {
  public_url: string
  openai_base_url: string
  pod_id: string | null
  gpus: Gpu[]
  disk: { path: string; total_bytes: number; used_bytes: number; free_bytes: number }
  recipes_repo: string | null
  recipes_remote_error: string | null
  dev: boolean
}

const KEY = 'mp.apiKey'

export function getKey(): string {
  try {
    return localStorage.getItem(KEY) ?? ''
  } catch {
    return ''
  }
}

export function setKey(key: string | null) {
  try {
    if (key) localStorage.setItem(KEY, key)
    else localStorage.removeItem(KEY)
  } catch {
    /* storage unavailable: key lives only in memory for this session */
  }
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

export async function api<T>(path: string, init: RequestInit = {}, key = getKey()): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: { Authorization: `Bearer ${key}`, ...(init.headers ?? {}) },
  })
  if (!res.ok) {
    let msg = res.statusText
    try {
      const body = await res.json()
      msg = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail ?? body)
    } catch {
      /* non-JSON error */
    }
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

export const post = <T,>(path: string, body?: unknown) =>
  api<T>(path, {
    method: 'POST',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

export function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '?'
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)} MB`
  return `${(n / 1e3).toFixed(0)} KB`
}
