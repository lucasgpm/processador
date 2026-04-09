// Apenas a lógica de limpeza pura (sem transformers aqui)
function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

async function processarLinhasComClassificador(linhas, session) {
    const limpas = linhas.map(l => l.trim()).filter(l => l !== "");
    if (limpas.length === 0) return [];

    // Em vez de for individual, vamos processar em blocos (ex: 10 em 10)
    // Para simplificar agora, vamos apenas logar o tempo:
    console.time("Tempo de Inferência");
    
    const resultados = await Promise.all(limpas.map(async (linha) => {
        const inputIds = new BigInt64Array(128).fill(0n);
        const attentionMask = new BigInt64Array(128).fill(1n);
        const feeds = {
            input_ids: new ort.Tensor('int64', inputIds, [1, 128]),
            attention_mask: new ort.Tensor('int64', attentionMask, [1, 128])
        };
        const output = await session.run(feeds);
        return { texto: linha, raw: output };
    }));

    console.timeEnd("Tempo de Inferência");
    return resultados;
}
