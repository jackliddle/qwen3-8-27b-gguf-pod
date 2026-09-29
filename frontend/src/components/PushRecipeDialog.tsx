import { useState } from 'react'
import { api } from '../api'

const TEMPLATE = `id: my-model
name: My model
engine: llamacpp        # llamacpp | vllm | ollama
source:
  hf_repo: owner/repo-GGUF
  files: ["*Q4_K_M.gguf"]
served_name: my-model
vram_gb: 8
capabilities: [chat]
params:
  ctx_size: {type: int, default: 8192}
`

export function PushRecipeDialog({ onClose, onPushed }: { onClose: () => void; onPushed: () => void }) {
  const [text, setText] = useState(TEMPLATE)
  const [error, setError] = useState<string | null>(null)

  const submit = async () => {
    setError(null)
    try {
      await api('/api/recipes', { method: 'POST', headers: { 'Content-Type': 'text/plain' }, body: text })
      onPushed()
      onClose()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="overlay" role="dialog" aria-modal aria-labelledby="push-title" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="dialog wide">
        <h2 id="push-title">Add recipe</h2>
        <p className="muted">
          Stored on this pod only (gone when it's terminated). To keep it, commit it to <code>recipes/</code> in the repo.
        </p>
        <textarea className="yaml" value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} rows={18} />
        {error && <p className="error-text">{error}</p>}
        <div className="actions">
          <span className="spacer" />
          <button className="btn btn-ghost" onClick={onClose}>
            Cancel
          </button>
          <button className="btn btn-primary" onClick={submit}>
            Save recipe
          </button>
        </div>
      </div>
    </div>
  )
}
