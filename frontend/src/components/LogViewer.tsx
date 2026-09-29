import { useEffect, useRef, useState } from 'react'
import { getKey } from '../api'

// Remounted per deployment attempt (parent keys cards by created_at), so no reset needed.
export function LogViewer({ depId }: { depId: string }) {
  const [lines, setLines] = useState<string[]>([])
  const [follow, setFollow] = useState(true)
  const boxRef = useRef<HTMLPreElement>(null)

  useEffect(() => {
    // EventSource can't send headers, so the key goes in the query string.
    const es = new EventSource(`/api/deployments/${encodeURIComponent(depId)}/logs/stream?key=${encodeURIComponent(getKey())}`)
    es.onmessage = (ev) => {
      const { line } = JSON.parse(ev.data) as { line: string }
      setLines((prev) => (prev.length > 3000 ? [...prev.slice(-2500), line] : [...prev, line]))
    }
    return () => es.close()
  }, [depId])

  useEffect(() => {
    const el = boxRef.current
    if (el && follow) el.scrollTop = el.scrollHeight
  }, [lines, follow])

  const onScroll = () => {
    const el = boxRef.current
    if (el) setFollow(el.scrollHeight - el.scrollTop - el.clientHeight < 40)
  }

  return (
    <pre className="logs" ref={boxRef} onScroll={onScroll}>
      {lines.length ? lines.join('\n') : 'Waiting for output…'}
    </pre>
  )
}
