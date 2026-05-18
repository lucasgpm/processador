// MANTIDO: Mesmo nome da função utilitária original
function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ') // Mantido padrão de trocar traço por espaço           
        .trim();
}

// MANTIDO: Mesmo nome da função do tokenizador
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

// MANTIDO: Mesmo nome da função softmax
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// MANTIDO: Mesmo nome da função principal chamada pelo Worker original
async function processarLinhasComClassificador(linhas, session, vocab) {
    const resultados = [];
    const BATCH_SIZE = 4; // Lotes menores dão mais precisão para palavras isoladas
    const TETO_MAX_LENGTH = 128;

    // Pré-filtro físico para Caça-Palavras (Garante que a linha tenha tamanho viável e não seja uma frase longa)
    const linhasValidas = linhas
        .map(l => l.trim())
        .filter(t => t.length >= 3 && t.length <= 16 && !t.includes(" ") && !t.includes("."));

    const totalBatches = Math.ceil(linhasValidas.length / BATCH_SIZE);

    for (let i = 0; i < linhasValidas.length; i += BATCH_SIZE) {
        const batchAtual = linhasValidas.slice(i, i + BATCH_SIZE);
        const atualBatchSize = batchAtual.length;

        const tokensDoBatch = batchAtual.map(t => tokenizeWordPiece(t, vocab));
        const maiorLinhaNoBatch = Math.max(...tokensDoBatch.map(t => t.length));
        const dynamicMaxLength = Math.min(maiorLinhaNoBatch + 2, TETO_MAX_LENGTH);

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

                // --- Cálculo de Entropia Semântica ---
                const maxScore = Math.max(...scores);
                const minScore = Math.min(...scores);
                const margemCerteza = maxScore - minScore;

                // Executa a limpeza idêntica
                const chaveLimpa = limparChave(t);
                const palavraUpper = chaveLimpa.toUpperCase();

                // Se a IA tiver convicção sobre a palavra (evita títulos/lixo estrutural)
                if (margemCerteza > 0.20) {
                    const tokens = tokensDoBatch[index];
                    const contemDesconhecido = tokens.includes(100n); // Filtra lixo de caracteres / código [UNK]

                    if (!contemDesconhecido && /^[A-ZÁ-Ý]+$/i.test(palavraUpper)) {
                        // AGORA: Adiciona apenas a String direta no array resultados, atendendo ao Caça-Palavras
                        resultados.push(palavraUpper);
                    }
                } else {
                    console.log(`🗑️ [IA FILTRO] Removido por incerteza semântica (Título/Lixo): "${t}" (Certeza: ${margemCerteza.toFixed(4)})`);
                }
            });

            const progresso = Math.round(((i / BATCH_SIZE) + 1) / totalBatches * 100);
            self.postMessage({ tipo: 'PROGRESSO', valor: Math.min(progresso, 100) });

        } catch (e) {
            console.error("Erro no processamento do lote:", e);
        }
    }

    console.log("🎯 [IA SUCESSO] Vetor limpo para Caça-Palavras gerado:", resultados);
    return resultados;
}
