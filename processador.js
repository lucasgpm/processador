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

    console.time("⏱️ Processamento Total");

    for (const linha of limpas) {
        try {
            // --- TOKENIZAÇÃO MANUAL SIMPLIFICADA ---
            // Se você não tem uma lib de tokenização, fazemos o básico:
            // 1. Converte para minúsculo e separa por palavras (ou usa o vocab)
            const tokens = linha.toLowerCase().split(/\s+/);
            
            // 2. Converte palavras em IDs usando o vocab que baixamos
            // Adicionamos o ID 101 (CLS) no início e 102 (SEP) no fim (padrão BERT)
            const inputIds = [101]; 
            for (const token of tokens) {
                const id = vocab[token] || vocab['[UNK]'] || 100; // 100 é o padrão para desconhecido
                inputIds.push(id);
            }
            inputIds.push(102);

            // 3. Truncamento e Padding para 128 tokens
            const maxLength = 128;
            const finalIds = inputIds.slice(0, maxLength);
            while (finalIds.length < maxLength) finalIds.push(0); // Padding com 0

            const attentionMask = finalIds.map(id => id > 0 ? 1n : 0n);
            const bigIntIds = finalIds.map(id => BigInt(id));

            // 4. Cria os Tensores para o ONNX
            const tensorIds = new ort.Tensor('int64', BigUint64Array.from(bigIntIds), [1, maxLength]);
            const tensorMask = new ort.Tensor('int64', BigUint64Array.from(attentionMask), [1, maxLength]);

            // Roda a IA
            const output = await session.run({
                input_ids: tensorIds,
                attention_mask: tensorMask
            });

            resultados.push({ texto: linha, raw: output });

        } catch (e) {
            console.warn(`Erro na linha: ${linha}`, e);
        }
    }

    console.timeEnd("⏱️ Processamento Total");
    return resultados;
}
