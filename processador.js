function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

// Adicionamos 'vocab' como parâmetro
async function processarLinhasComClassificador(linhas, session, vocab) {
    const limpas = linhas.map(l => l.trim()).filter(l => l.length > 5);
    const resultados = [];
    const maxLength = 128;

    console.time("🚀 Performance Máxima");

    for (const linha of limpas) {
        try {
            // Tokenização manual ultra rápida
            const tokens = linha.toLowerCase().split(/\s+/);
            const inputIds = new BigUint64Array(maxLength).fill(0n);
            const attentionMask = new BigUint64Array(maxLength).fill(0n);

            inputIds[0] = 101n; // [CLS]
            let pos = 1;
            for (const token of tokens) {
                if (pos >= maxLength - 1) break;
                const id = vocab[token] || vocab['[UNK]'] || 100;
                inputIds[pos] = BigInt(id);
                attentionMask[pos] = 1n;
                pos++;
            }
            inputIds[pos] = 102n; // [SEP]
            attentionMask[0] = 1n; // CLS mask
            attentionMask[pos] = 1n; // SEP mask

            // Criação de Tensores sem overhead
            const tensorIds = new ort.Tensor('int64', inputIds, [1, maxLength]);
            const tensorMask = new ort.Tensor('int64', attentionMask, [1, maxLength]);

            // Execução da IA
            const output = await session.run({
                input_ids: tensorIds,
                attention_mask: tensorMask
            });

            // Extração rápida (Logits)
            const logits = output.logits || output[Object.keys(output)[0]];
            
            // Enviamos como Array normal para o Worker não travar no postMessage
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
