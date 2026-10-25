document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chatMessages');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');

    // Функция добавления сообщения
    function addMessage(text, sender, sources = [], ragData = null) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}-message`;

        let contentHTML = `<div class="message-content">${text}`;

        // 1. Блок источников (карточки)
        if (sources && sources.length > 0) {
            contentHTML += `<div class="sources-container">
                <div class="sources-title">Источники</div>`;

            sources.forEach(src => {
                const sourceName = src.source || 'Документ';
                const chunkNum = src.chunk !== undefined ? ` (чанк ${src.chunk})` : '';
                const title = `${sourceName}${chunkNum}`;

                contentHTML += `<span class="source-card">
                    <i class="fa-regular fa-file-lines"></i>
                    ${title}
                </span>`;
            });
            contentHTML += `</div>`;
        }

        // 2. Блок логики RAG
        if (ragData && ragData.keywords) {
            contentHTML += `<div class="rag-logic-box">
                <div class="rag-logic-title">⚙️ Логика работы RAG</div>
                
                <div class="rag-section">
                    <b>Ключевые слова:</b> ${ragData.keywords.join(', ')}
                </div>`;

            const renderChunks = (title, chunks) => {
                if (!chunks || chunks.length === 0) return '';
                let html = `<div class="rag-section"><b>${title}:</b></div>`;
                chunks.forEach((chunk, idx) => {
                    let scoreInfo = '';
                    if (chunk.rrf_score !== undefined) scoreInfo = `RRF: ${chunk.rrf_score.toFixed(3)}`;
                    else if (chunk.bm25_score !== undefined) scoreInfo = `BM25: ${chunk.bm25_score.toFixed(3)}`;
                    else if (chunk.score !== undefined) scoreInfo = `Emb: ${chunk.score.toFixed(3)}`;

                    html += `<div class="rag-chunk" onclick="this.classList.toggle('expanded')">
                        <span class="rag-chunk-source">${idx + 1}. ${chunk.source} (чанк ${chunk.chunk}) — ${scoreInfo}</span>
                        <div class="rag-chunk-text">${chunk.text}</div>
                    </div>`;
                });
                return html;
            };

            contentHTML += renderChunks('Embedding-поиск', ragData.embedding_chunks);
            contentHTML += renderChunks('BM25-поиск', ragData.keyword_chunks);
            contentHTML += renderChunks('Финальный набор (RRF)', ragData.final_chunks);

            contentHTML += `</div>`;
        }

        contentHTML += `</div>`;
        msgDiv.innerHTML = contentHTML;
        chatMessages.appendChild(msgDiv);

        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    // Функция отправки запроса
    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        addMessage(text, 'user');
        userInput.value = '';
        userInput.disabled = true;

        const typingDiv = document.createElement('div');
        typingDiv.className = 'message bot-message';
        typingDiv.id = 'typingIndicator';
        typingDiv.innerHTML = `<div class="typing-indicator"><span></span><span></span><span></span></div>`;
        chatMessages.appendChild(typingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: text })
            });

            if (!response.ok) throw new Error('Ошибка сервера');

            const data = await response.json();
            typingDiv.remove();

            const sources = data.chunks || [];
            addMessage(data.answer || 'Нет ответа', 'bot', sources, data);

        } catch (error) {
            typingDiv.remove();
            addMessage("Извините, произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже.", 'bot');
            console.error(error);
        } finally {
            userInput.disabled = false;
            userInput.focus();
        }
    }

    sendBtn.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
});
