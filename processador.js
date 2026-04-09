function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

async function processarLinhasComClassificador(linhas, session, vocab) {
    const limpas = linhas.map(l => l.trim()).filter(l => l.length > 5);
    const resultados = [];
    const maxLength = 128;

    // --- ALOCAÇÃO ÚNICA (Fora do loop) ---
    // Criamos os buffers uma vez só para economizar memória e CPU
    const inputIdsData = new BigInt64Array(maxLength);
    const attentionMaskData = new BigInt64Array(maxLength);

    // Criamos os Tensores apontando para esses mesmos buffers
    const tensorIds = new self.ort.Tensor('int64', inputIdsData, [1, maxLength]);
    const tensorMask = new self.ort.Tensor('int64', attentionMaskData, [1, maxLength]);

    console.time("🚀 Performance Máxima");

    for (const linha of limpas) {
        try {
            // Reinicia os buffers sem criar novos objetos
            inputIdsData.fill(0n);
            attentionMaskData.fill(0n);

            const tokens = linha.toLowerCase().split(/\s+/);
            
            inputIdsData[0] = 101n; // [CLS]
            let pos = 1;
            
            for (const token of tokens) {
                if (pos >= maxLength - 1) break;
                const id = vocab[token] || vocab['[UNK]'] || 100;
                inputIdsData[pos] = BigInt(id);
                attentionMaskData[pos] = 1n;
                pos++;
            }
            
            inputIdsData[pos] = 102n; // [SEP]
            attentionMaskData[0] = 1n; // CLS
            attentionMaskData[pos] = 1n; // SEP

            // Execução: O ONNX vai ler os dados atualizados nos buffers que já mapeamos
            const output = await session.run({
                input_ids: tensorIds,
                attention_mask: tensorMask
            });

            const outputName = session.outputNames[0];
            const logits = output[outputName];
            
            resultados.push({ 
                texto: linha, 
                score: Array.from(logits.data) 
            });

        } catch (e) {
            console.warn(`Erro na linha: ${linha}`, e);
        }
    }

    console.timeEnd("🚀 Performance Máxima");
    return resultados;
}
