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
    const resultados = [];
    console.log("🧐 Iniciando loop de processamento para", linhas.length, "linhas.");

    for (const linha of linhas) {
        if (!linha.trim()) continue;

        const inputIds = new BigInt64Array(128).fill(0n);
        const attentionMask = new BigInt64Array(128).fill(1n);
        
        const feeds = {
            input_ids: new ort.Tensor('int64', inputIds, [1, 128]),
            attention_mask: new ort.Tensor('int64', attentionMask, [1, 128])
        };

        try {
            const output = await session.run(feeds);
            console.log("🔍 Output bruto da IA para a linha:", linha, output);

            // O ONNX puro retorna um objeto com chaves (ex: output.logits ou output.last_hidden_state)
            // Precisamos garantir que estamos enviando algo que o modal.js entenda
            resultados.push({ 
                texto: linha, 
                raw: output,
                verificacao: "OK" 
            });
        } catch (e) {
            console.error("❌ Falha na inferência da linha:", linha, e);
        }
    }
    console.log("📤 Enviando resultados finais para o script principal:", resultados);
    return resultados;
}
