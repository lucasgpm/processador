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
    const limpas = linhas.map(l => l.trim()).filter(l => l.length > 5);
    
    // Processamos em sequência ou pequenos grupos para não travar a GPU
    const resultados = [];
    for (const linha of limpas) {
        // O MÁGICA: Converte texto real em números que a IA entende
        const { input_ids, attention_mask } = await tokenizer(linha, {
            padding: true,
            truncation: true,
            maxLength: 128
        });

        const feeds = {
            input_ids: input_ids,
            attention_mask: attention_mask
        };

        const output = await session.run(feeds);
        resultados.push({ texto: linha, raw: output });
    }
    return resultados;
}
