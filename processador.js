function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

// --- FUNÇÃO 1: WORDPIECE (A "Inteligência" do BERT) ---
function tokenizeWordPiece(text, vocab) {
    const words = text.toLowerCase()
        .replace(/([.,!?])/g, ' $1 ')
        .trim()
        .split(/\s+/);
    
    const resultIds = [];
    for (let word of words) {
        if (vocab[word]) {
            resultIds.push(BigInt(vocab[word]));
            continue;
        }

        let start = 0;
        let found = false;
        while (start < word.length) {
            let end = word.length;
            let curSubstrId = null;
            while (start < end) {
                let substr = (start === 0) ? word.substring(start, end) : "##" + word.substring(start, end);
                if (vocab[substr]) {
                    curSubstrId = BigInt(vocab[substr]);
                    break;
                }
                end--;
            }
            if (curSubstrId === null) {
                resultIds.push(BigInt(vocab['[UNK]'] || 100));
                break;
            }
            resultIds.push(curSubstrId);
            start = end;
        }
    }
    return resultIds;
}

// --- FUNÇÃO 2: SOFTMAX (Converte números da IA em Score de 0 a 1) ---
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// --- FUNÇÃO 3: O PROCESSADOR COMPLETO ---
async function processarLinhasComClassificador(linhas, session, vocab) {
    const resultados = [];
    const maxLength = 128;
    const BATCH_SIZE = 16; // Processamos 16 linhas por vez

    // Filtramos primeiro as linhas válidas para não gastar IA com lixo
    const linhasValidas = linhas
        .map(l => l.trim())
        .filter(t => t.length >= 5 && /[:\-\t=|—]/.test(t));

    // Processamos em grupos (Batches)
    for (let i = 0; i < linhasValidas.length; i += BATCH_SIZE) {
        const batchAtual = linhasValidas.slice(i, i + BATCH_SIZE);
        const atualBatchSize = batchAtual.length;

        // Criamos os buffers para o tamanho exato deste batch
        const inputIdsData = new BigInt64Array(atualBatchSize * maxLength);
        const attentionMaskData = new BigInt64Array(atualBatchSize * maxLength);

        // Preenchemos os dados de todas as linhas do batch
        batchAtual.forEach((t, index) => {
            const tokenIds = tokenizeWordPiece(t, vocab);
            const offset = index * maxLength;

            inputIdsData[offset] = 101n; // [CLS]
            let pos = 1;
            for (const id of tokenIds) {
                if (pos >= maxLength - 1) break;
                inputIdsData[offset + pos] = id;
                attentionMaskData[offset + pos] = 1n;
                pos++;
            }
            inputIdsData[offset + pos] = 102n; // [SEP]
            attentionMaskData[offset] = 1n;
            attentionMaskData[offset + pos] = 1n;
        });

        // Criamos os Tensores para o BATCH inteiro [N, 128]
        const tensorIds = new ort.Tensor('int64', inputIdsData, [atualBatchSize, maxLength]);
        const tensorMask = new ort.Tensor('int64', attentionMaskData, [atualBatchSize, maxLength]);

        try {
            // EXECUÇÃO ÚNICA PARA O LOTE
            const output = await session.run({
                input_ids: tensorIds,
                attention_mask: tensorMask
            });

            const outputData = output[session.outputNames[0]].data;
            const numLabels = outputData.length / atualBatchSize;

            // Processamos os resultados de cada linha do lote
            batchAtual.forEach((t, index) => {
                // Extrai os logits desta linha específica
                const inicio = index * numLabels;
                const fim = inicio + numLabels;
                const logits = Array.from(outputData.slice(inicio, fim));
                
                const scores = softmax(logits);
                const scoreConfianca = Math.max(...scores);

                // --- SUA LÓGICA DE EXTRAÇÃO MANTIDA ---
                const divisorMatch = t.match(/[:\t=|—]| - /);
                if (divisorMatch) {
                    const indiceDivisor = divisorMatch.index;
                    const chaveBruta = t.substring(0, indiceDivisor).trim();
                    const valor = t.substring(indiceDivisor + divisorMatch[0].length).trim();

                    let chaveLimpa = limparChave(chaveBruta);
                    const palavrasNaChave = chaveLimpa.split(/\s+/).filter(p => p.length > 0).length;

                    if (scoreConfianca > 0.3 && palavrasNaChave > 0 && palavrasNaChave <= 3 && valor.length > 5) {
                        resultados.push({
                            palavra: chaveLimpa.toUpperCase(),
                            dica: valor
                        });
                    }
                }
            });
        } catch (e) {
            console.warn("Erro no processamento do lote:", e);
        }
    }
    return resultados;
}
