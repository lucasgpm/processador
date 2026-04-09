// Apenas a lógica de limpeza pura (sem transformers aqui)
function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

// Essa função agora será usada PELO WORKER
async function processarLinhasComClassificador(linhas, session) {
    // Nota: Para classificação real, precisaríamos tokenizar. 
    // Como teste de fogo para ver se o modelo CARREGA, tente rodar um input vazio:
    
    const resultados = [];
    for (const linha of linhas) {
        // Exemplo simplificado de input para o DistilBERT (ids fictícios para teste)
        const inputIds = new BigInt64Array(128).fill(0n); // Exemplo de tamanho fixo
        const attentionMask = new BigInt64Array(128).fill(1n);
        
        const feeds = {
            input_ids: new ort.Tensor('int64', inputIds, [1, 128]),
            attention_mask: new ort.Tensor('int64', attentionMask, [1, 128])
        };

        const output = await session.run(feeds);
        resultados.push({ texto: linha, raw: output });
    }
    return resultados;
}
