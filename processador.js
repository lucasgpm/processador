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
    const BATCH_SIZE = 8; 
    const TETO_MAX_LENGTH = 128;

    // MANTIDO: Filtro original exatamente como você tinha
    const linhasValidas = linhas
        .map(l => l.trim())
        .filter(t => t.length >= 5 && /[:\-\t=|—]/.test(t));

    const totalBatches = Math.ceil(linhasValidas.length / BATCH_SIZE);

    for (let i = 0; i < linhasValidas.length; i += BATCH_SIZE) {
        const batchAtual = linhasValidas.slice(i, i + BATCH_SIZE);
        const atualBatchSize = batchAtual.length;

        // --- DINÂMICO: Tokenizamos antes para saber o tamanho real do lote ---
        const tokensDoBatch = batchAtual.map(t => tokenizeWordPiece(t, vocab));
        const maiorLinhaNoBatch = Math.max(...tokensDoBatch.map(t => t.length));
        
        // Ajusta o tamanho do "contêiner" para o tamanho da maior frase do lote
        const dynamicMaxLength = Math.min(maiorLinhaNoBatch + 2, TETO_MAX_LENGTH);

        // --- OTIMIZAÇÃO: Alocação precisa de memória ---
        const inputIdsData = new BigInt64Array(atualBatchSize * dynamicMaxLength);
        const attentionMaskData = new BigInt64Array(atualBatchSize * dynamicMaxLength);

        batchAtual.forEach((t, index) => {
            const tokenIds = tokensDoBatch[index];
            const offset = index * dynamicMaxLength;

            inputIdsData[offset] = 101n; // [CLS]
            let pos = 1;
            for (const id of tokenIds) {
                if (pos >= dynamicMaxLength - 1) break;
                inputIdsData[offset + pos] = id;
                attentionMaskData[offset + pos] = 1n;
                pos++;
            }
            inputIdsData[offset + pos] = 102n; // [SEP]
            attentionMaskData[offset] = 1n;
            attentionMaskData[offset + pos] = 1n;
        });

        try {
            const output = await session.run({
                input_ids: new ort.Tensor('int64', inputIdsData, [atualBatchSize, dynamicMaxLength]),
                attention_mask: new ort.Tensor('int64', attentionMaskData, [atualBatchSize, dynamicMaxLength])
            });

            const outputData = output[session.outputNames[0]].data;
            const numLabels = outputData.length / atualBatchSize;

            batchAtual.forEach((t, index) => {
                const inicio = index * numLabels;
                const logits = Array.from(outputData.slice(inicio, inicio + numLabels));
                const scores = softmax(logits);
                const scoreConfianca = Math.max(...scores);

                // MANTIDO: Lógica de divisão e limpeza original
                const divisorMatch = t.match(/[:\t=|—]| - /);
                if (divisorMatch) {
                    const indiceDivisor = divisorMatch.index;
                    const chaveBruta = t.substring(0, indiceDivisor).trim();
                    const valor = t.substring(indiceDivisor + divisorMatch[0].length).trim();

                    const chaveLimpa = limparChave(chaveBruta);
                    const palavrasNaChave = chaveLimpa.split(/\s+/).filter(p => p.length > 0).length;

                    // MANTIDO: Seus filtros de validação exatos
                    if (scoreConfianca > 0.3 && palavrasNaChave > 0 && palavrasNaChave <= 3 && valor.length > 5) {
                        resultados.push({
                            palavra: chaveLimpa.toUpperCase(),
                            dica: valor
                        });
                    }
                }
            });

            // Notifica o progresso para a barra de carregamento
            const progresso = Math.round(((i / BATCH_SIZE) + 1) / totalBatches * 100);
            self.postMessage({ tipo: 'PROGRESSO', valor: Math.min(progresso, 100) });

        } catch (e) {
            console.error("Erro no processamento do lote:", e);
        }
    }
    return resultados;
}
