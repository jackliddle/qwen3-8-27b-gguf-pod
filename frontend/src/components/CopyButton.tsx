import { useState } from 'react'

export function CopyButton({ text, label = 'Copy' }: { text: string; label?: string }) {
  const [done, setDone] = useState(false)
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text)
    } catch {
      // Clipboard API needs a secure context; fall back to a hidden textarea.
      const ta = document.createElement('textarea')
      ta.value = text
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      ta.remove()
    }
    setDone(true)
    setTimeout(() => setDone(false), 1200)
  }
  return (
    <button className={`btn btn-ghost btn-xs copy ${done ? 'copied' : ''}`} onClick={copy} type="button">
      {done ? 'Copied' : label}
    </button>
  )
}
