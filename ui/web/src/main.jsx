import React, { useState } from 'react'
import { createRoot } from 'react-dom/client'

function App() {
  const [q, setQ] = useState('')
  const [res, setRes] = useState(null)
  const [loading, setLoading] = useState(false)
  const [streamingResponse, setStreamingResponse] = useState('')
  const [api, setApi] = useState('http://localhost:8001')
  const [error, setError] = useState(null)
  const [health, setHealth] = useState(null)
  const [largeSlide, setLargeSlide] = useState(null)
  const [selectedFile, setSelectedFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [ingestionState, setIngestionState] = useState('none') // 'none', 'ingesting', 'complete', 'error'
  const [ingestionProgress, setIngestionProgress] = useState('')

  async function checkHealth() {
    try {
      const r = await fetch(api + '/health')
      const h = await r.json()
      setHealth(h)
      setError(null)
    } catch (err) {
      setHealth(null)
      setError('Cannot connect to API server')
    }
  }

  async function ask(e){
    e.preventDefault()
    if(!q.trim()) return
    setLoading(true)
    setError(null)
    setStreamingResponse('')

    await askStreaming()
  }

  async function askStreaming(){
    try {
      const response = await fetch(api + '/ask/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q })
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`)
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      setRes(null) // Clear previous results
      setStreamingResponse('') // Clear previous streaming text

      // Add timeout to prevent hanging
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error('Streaming timeout after 60 seconds')), 60000)
      )

      const readPromise = (async () => {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const lines = buffer.split('\n')
          buffer = lines.pop() // Keep incomplete line in buffer

          for (const line of lines) {
            if (line.trim() === '' || !line.startsWith('data: ')) continue

            try {
              const data = JSON.parse(line.slice(6))

              if (data.type === 'metadata') {
                // Handle metadata if needed
                console.log('Stream metadata:', data)
              } else if (data.type === 'error') {
                throw new Error(data.message || 'Server error during streaming')
              } else if (data.type === 'complete') {
                // Final result with citations and slides
                setRes({
                  answer_md: data.answer_md,
                  citations: data.citations,
                  slides: data.slides,
                  page_content: data.page_content
                })
                // Update streaming response with final cleaned version
                if (data.answer_md) {
                  setStreamingResponse(data.answer_md)
                }

                // Check if response seems incomplete
                if (data.is_complete === false) {
                  console.warn('Response may be incomplete - token limit reached')
                  setError('Response may be incomplete due to length limits. Try rephrasing your question.')
                }

                return // Exit the loop when complete
              } else if (data.type === 'token' && data.char !== undefined) {
                // Stream individual characters for smooth display
                setStreamingResponse(prev => {
                  const newText = prev + data.char
                  
                  // Detect if we're getting repetitive content
                  if (newText.length > 100) {
                    const lastChunk = newText.slice(-50)
                    const beforeLastChunk = newText.slice(-100, -50)
                    
                    // If last 50 chars appear in previous 50 chars, likely repetition
                    if (lastChunk.length > 20 && beforeLastChunk.includes(lastChunk.slice(0, 20))) {
                      console.warn('Detected repetitive streaming, stopping display')
                      return prev // Don't add more
                    }
                  }
                  
                  return newText
                })
              }
            } catch (e) {
              console.error('Error parsing stream data:', e, 'Line:', line)
            }
          }
        }
      })()

      try {
        await Promise.race([readPromise, timeoutPromise])
      } catch (err) {
        console.error('Streaming error:', err)
        if (err.message.includes('timeout')) {
          setError('Response timed out. The answer may be incomplete.')
        } else {
          setError(err.message || 'Streaming error occurred')
        }
      }

    } catch (err) {
      console.error(err)
      setError(err.message || 'Streaming error occurred')
    } finally {
      setLoading(false)
    }
  }

  async function uploadPDF() {
    if (!selectedFile) return

    setUploading(true)
    setUploadStatus(null)
    setError(null)
    setIngestionState('ingesting')
    setIngestionProgress('Starting ingestion process...')

    try {
      const formData = new FormData()
      formData.append('file', selectedFile)

      setIngestionProgress('Uploading PDF...')
      const r = await fetch(api + '/ingest', {
        method: 'POST',
        body: formData
      })

      if (!r.ok) {
        throw new Error(`HTTP ${r.status}: ${r.statusText}`)
      }

      setIngestionProgress('Processing PDF pages...')
      const result = await r.json()

      setIngestionProgress('Extracting text content...')
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate processing time

      setIngestionProgress('Building search indexes...')
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate processing time

      setIngestionProgress('Finalizing...')
      await new Promise(resolve => setTimeout(resolve, 500)) // Simulate processing time

      setUploadStatus('PDF uploaded and processed successfully!')
      setIngestionState('complete')

      // Refresh health check to show updated status
      await checkHealth()

    } catch (err) {
      console.error(err)
      setError(err.message || 'Upload failed')
      setUploadStatus(null)
      setIngestionState('error')
      setIngestionProgress('')
    } finally {
      setUploading(false)
    }
  }

  // Check health on mount and API URL change
  React.useEffect(() => {
    checkHealth()
  }, [api])

  // Helper function to check if system is ready
  const isSystemReady = health && health.index && health.pages && ingestionState === 'complete'

  // Helper function to reset to initial state
  const resetToInitial = () => {
    setIngestionState('none')
    setIngestionProgress('')
    setUploadStatus(null)
    setSelectedFile(null)
    setError(null)
  }

  return (
    <div style={{fontFamily:'system-ui, sans-serif', padding:20, maxWidth: 960, margin:'0 auto'}}>
      {/* API Configuration - always shown */}
      <div style={{marginBottom:20, padding:15, backgroundColor:'#f5f5f5', borderRadius:8, border:'1px solid #ddd'}}>
        <h3 style={{margin:0, marginBottom:10, color:'#333'}}>API Configuration</h3>
        <div style={{display:'flex', gap:10, alignItems:'center'}}>
          <label style={{minWidth:80}}>API URL:</label>
          <input
            value={api}
            onChange={e=>setApi(e.target.value)}
            style={{flex:1, padding:8, border:'1px solid #ccc', borderRadius:4}}
            placeholder="http://localhost:8001"
          />
          <button onClick={checkHealth} style={{padding:'8px 16px', backgroundColor:'#007bff', color:'white', border:'none', borderRadius:4, cursor:'pointer'}}>
            Check
          </button>
        </div>

        {/* API Status */}
        <div style={{marginTop:10, padding:8, backgroundColor: health ? '#e8f5e8' : '#ffe8e8', borderRadius:4, fontSize:14}}>
          <strong>Status: </strong>
          {error ? (
            <span style={{color:'#f44336'}}>{error}</span>
          ) : health ? (
            <span style={{color:'#4caf50'}}>
              Connected (Index: {health.index ? '✓' : '✗'}, Pages: {health.pages ? '✓' : '✗'})
            </span>
          ) : (
            <span>Checking...</span>
          )}
        </div>
      </div>

      {/* Show Setup Interface when not ready */}
      {ingestionState !== 'complete' && (
        <>
          {/* PDF Upload Section */}
          <div style={{marginBottom:30, padding:20, backgroundColor:'#f0f8ff', borderRadius:8, border:'1px solid #007bff'}}>
            <h3 style={{margin:0, marginBottom:15, color:'#007bff'}}>📄 Setup: Upload PDF Manual</h3>
            <p style={{margin:0, marginBottom:15, color:'#666'}}>
              Upload a PDF manual to enable the RAG system. The document will be processed and indexed for search.
            </p>

            <div style={{display:'flex', gap:15, alignItems:'center', flexWrap:'wrap'}}>
              <div style={{flex:1, minWidth:'200px'}}>
                <input
                  type="file"
                  accept=".pdf"
                  onChange={(e) => setSelectedFile(e.target.files[0])}
                  style={{width:'100%', padding:8, border:'1px solid #ccc', borderRadius:4}}
                  disabled={uploading}
                />
              </div>
              <button
                onClick={uploadPDF}
                disabled={!selectedFile || uploading || !health}
                style={{
                  padding:'10px 20px',
                  backgroundColor: (!selectedFile || uploading || !health) ? '#ccc' : '#007bff',
                  color:'white',
                  border:'none',
                  borderRadius:4,
                  cursor: (!selectedFile || uploading || !health) ? 'not-allowed' : 'pointer'
                }}
              >
                {uploading ? 'Uploading...' : 'Upload & Process PDF'}
              </button>
            </div>

            {selectedFile && (
              <div style={{marginTop:10, fontSize:14, color:'#666'}}>
                Selected: <strong>{selectedFile.name}</strong> ({Math.round(selectedFile.size / 1024)} KB)
              </div>
            )}

            {uploadStatus && (
              <div style={{marginTop:10, padding:10, backgroundColor:'#e8f5e8', borderRadius:4, color:'#4caf50'}}>
                <strong>✓ {uploadStatus}</strong>
              </div>
            )}
          </div>

          {/* Ingestion Progress */}
          {ingestionState === 'ingesting' && (
            <div style={{marginBottom:30, padding:20, backgroundColor:'#fff3cd', borderRadius:8, border:'1px solid #ffc107'}}>
              <h3 style={{margin:0, marginBottom:15, color:'#856404'}}>🔄 Processing Document</h3>
              <div style={{display:'flex', alignItems:'center', gap:15, marginBottom:15}}>
                <div style={{
                  display:'inline-flex',
                  gap:2
                }}>
                  {[0,1,2].map(i => (
                    <span key={i} style={{
                      width:8,
                      height:8,
                      backgroundColor:'#ffc107',
                      borderRadius:'50%',
                      animation:`pulse 1.4s infinite ease-in-out ${i * 0.16}s`
                    }}></span>
                  ))}
                </div>
                <span style={{color:'#856404', fontSize:14}}>{ingestionProgress}</span>
              </div>
              <p style={{margin:0, color:'#856404', fontSize:14}}>
                Please wait while we process your PDF. This may take a few minutes depending on the document size.
              </p>
            </div>
          )}

          {/* Error State */}
          {ingestionState === 'error' && (
            <div style={{marginBottom:30, padding:20, backgroundColor:'#f8d7da', borderRadius:8, border:'1px solid #f5c6cb'}}>
              <h3 style={{margin:0, marginBottom:15, color:'#721c24'}}>❌ Processing Failed</h3>
              <p style={{margin:0, marginBottom:15, color:'#721c24'}}>
                There was an error processing your PDF. This could be due to:
              </p>
              <ul style={{margin:0, marginBottom:15, color:'#721c24'}}>
                <li>Corrupted or invalid PDF file</li>
                <li>PDF with complex formatting or images that can't be processed</li>
                <li>Insufficient disk space or permissions</li>
                <li>Server configuration issues</li>
              </ul>
              <p style={{margin:0, marginBottom:15, color:'#721c24', fontSize:14}}>
                <strong>Error details:</strong> {error}
              </p>
              <div style={{display:'flex', gap:10}}>
                <button
                  onClick={resetToInitial}
                  style={{
                    padding:'8px 16px',
                    backgroundColor:'#dc3545',
                    color:'white',
                    border:'none',
                    borderRadius:4,
                    cursor:'pointer'
                  }}
                >
                  Try Again
                </button>
                <button
                  onClick={() => window.location.reload()}
                  style={{
                    padding:'8px 16px',
                    backgroundColor:'#6c757d',
                    color:'white',
                    border:'none',
                    borderRadius:4,
                    cursor:'pointer'
                  }}
                >
                  Refresh Page
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Show Main RAG Interface only when system is ready */}
      {isSystemReady && (
        <>
          <h1>Articula RAG Demo</h1>
          <p style={{color:'#666', fontSize:'16px', lineHeight:'1.5'}}>
            Just describe your issue or ask a question. Responses stream in real-time as they're generated.
          </p>

          <form onSubmit={ask} style={{display:'flex', gap:8, marginBottom:10}}>
            <input
              value={q}
              onChange={e=>setQ(e.target.value)}
              placeholder="e.g., I'm getting M_ERR_SYSTEM_OPEN, how do I fix it?"
              style={{flex:1, padding:10}}
              disabled={loading}
            />
            <button disabled={loading} style={{padding:'10px 16px'}}>
              {loading ? 'Generating...' : 'Ask'}
            </button>
          </form>

          {error && (
            <div style={{marginBottom:20, padding:10, backgroundColor:'#ffe8e8', border:'1px solid #f44336', borderRadius:4, color:'#f44336'}}>
              <strong>Error:</strong> {error}
            </div>
          )}

          {loading && !streamingResponse && (
            <div style={{display:'grid', gridTemplateColumns:'2fr 1fr', gap:20, marginTop:20}}>
              <div>
                <h3>Answer</h3>
                <div style={{whiteSpace:'pre-wrap', backgroundColor:'#f9f9f9', padding:15, borderRadius:4, position:'relative'}}>
                  <div style={{display:'flex', alignItems:'center', gap:8}}>
                    <span style={{color:'#999'}}>Generating response...</span>
                    <div style={{
                      display:'inline-flex',
                      gap:2
                    }}>
                      {[0,1,2].map(i => (
                        <span key={i} style={{
                          width:6,
                          height:6,
                          backgroundColor:'#007bff',
                          borderRadius:'50%',
                          animation:`pulse 1.4s infinite ease-in-out ${i * 0.16}s`
                        }}></span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {streamingResponse && (
            <div style={{display:'grid', gridTemplateColumns:'2fr 1fr', gap:20, marginTop:20}}>
              <div>
                <h3>Answer</h3>
                <div style={{whiteSpace:'pre-wrap', backgroundColor:'#f9f9f9', padding:15, borderRadius:4, position:'relative'}}>
                  {streamingResponse}
                  {loading && (
                    <span style={{
                      display:'inline-block',
                      width:8,
                      height:16,
                      backgroundColor:'#007bff',
                      marginLeft:2,
                      animation:'blink 1s infinite'
                    }}></span>
                  )}
                </div>
                {res && res.citations && (
                  <>
                    <h4>Citations</h4>
                    <ul style={{backgroundColor:'#f0f8ff', padding:15, borderRadius:4}}>
                      {res.citations?.map((c,i)=>(
                        <li key={i}>p. {c.page}</li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
              {res && res.slides && (
                <div>
                  <h3>Relevant slides</h3>
                  <div style={{display:'flex', flexDirection:'column', gap:10}}>
                    {res.slides?.map((s,i)=>(
                      <figure key={i} style={{margin:0}}>
                        <img
                          src={api + s.image_url}
                          alt={s.alt}
                          style={{
                            width:'100%',
                            border:'1px solid #ddd',
                            borderRadius:4,
                            cursor:'pointer',
                            transition:'transform 0.2s'
                          }}
                          onClick={() => setLargeSlide(s)}
                          onMouseOver={(e) => e.target.style.transform = 'scale(1.02)'}
                          onMouseOut={(e) => e.target.style.transform = 'scale(1)'}
                          onError={(e) => {e.target.style.display='none'}}
                        />
                        <figcaption style={{fontSize:12, textAlign:'center', marginTop:5}}>
                          p. {s.page} - Click to enlarge
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Add CSS animations for loading indicators */}
      <style>{`
        @keyframes blink {
          0%, 50% { opacity: 1; }
          51%, 100% { opacity: 0; }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.4; transform: scale(0.8); }
          50% { opacity: 1; transform: scale(1.2); }
        }
      `}</style>

      {/* Large slide viewer */}
      {largeSlide && (
        <div style={{
          position:'fixed',
          bottom:0,
          left:0,
          right:0,
          top:0,
          backgroundColor:'rgba(0,0,0,0.8)',
          display:'flex',
          alignItems:'center',
          justifyContent:'center',
          zIndex:1000,
          cursor:'pointer'
        }} onClick={() => setLargeSlide(null)}>
          <div style={{
            backgroundColor:'white',
            padding:20,
            borderRadius:8,
            maxWidth:'90%',
            maxHeight:'90%',
            position:'relative'
          }}>
            <button
              onClick={(e) => {e.stopPropagation(); setLargeSlide(null);}}
              style={{
                position:'absolute',
                top:10,
                right:10,
                background:'none',
                border:'none',
                fontSize:24,
                cursor:'pointer',
                color:'#666'
              }}
            >
              ×
            </button>
            <img
              src={api + largeSlide.image_url}
              alt={largeSlide.alt}
              style={{
                maxWidth:'100%',
                maxHeight:'80vh',
                border:'1px solid #ddd',
                borderRadius:4
              }}
              onError={(e) => {e.target.style.display='none'}}
            />
            <div style={{
              textAlign:'center',
              marginTop:10,
              fontSize:14,
              color:'#666'
            }}>
              {largeSlide.alt}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

createRoot(document.getElementById('root')).render(<App/>)
