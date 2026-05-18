function limparChaveCaca(texto) {
    return texto
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .trim();
}

async function processarLinhasCacaPalavras(linhas, session, vocab) {
    const resultados = [];
    const BATCH_SIZE = 8; 
    const TETO_MAX_LENGTH = 128;

    // Pega qualquer linha que tenha pelo menos 2 caracteres de texto útil
    const linhasValidas = linhas
        .map(l => l.trim())
        .filter(t => t.length >= 2);

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
                const scoreConfianca = Math.max(...scores);

                // Se a IA confia no texto, limpamos e adicionamos direto no array
                if (scoreConfianca > 0.3) {
                    const palavraBruta = limparChaveCaca(t);
                    if (palavraBruta.length > 0) {
                        resultados.push(palavraBruta); // Retorna apenas a string direta!
                    }
                }
            });

            const progresso = Math.round(((i / BATCH_SIZE) + 1) / totalBatches * 100);
            self.postMessage({ tipo: 'PROGRESSO', valor: Math.min(progresso, 100) });

        } catch (e) {
            console.error("Erro no lote:", e);
        }
    }
    return resultados;
}
