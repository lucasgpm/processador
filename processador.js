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
    const resultados = [];

    console.time("⏱️ Processamento Total");

    // IMPORTANTE: Use um loop for normal. 
    // O Promise.all em 32 linhas de uma vez causa o "Loading Infinito" na GPU.
    for (const linha of limpas) {
        try {
            // Transforma o texto em números (input_ids e attention_mask)
            const inputs = await tokenizer(linha, {
                padding: true,
                truncation: true,
                max_length: 128
            });

            // Roda a IA com os dados reais
            const output = await session.run({
                input_ids: inputs.input_ids,
                attention_mask: inputs.attention_mask
            });

            resultados.push({ texto: linha, raw: output });
        } catch (e) {
            console.warn(`Erro na linha: ${linha}`, e);
        }
    }

    console.timeEnd("⏱️ Processamento Total");
    return resultados;
}
