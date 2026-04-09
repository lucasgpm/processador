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
    // 1. Limpeza inicial
    const limpas = linhas.map(l => l.trim()).filter(l => l !== "");
    
    if (limpas.length === 0) {
        console.warn("⚠️ Nenhuma linha válida para processar.");
        return [];
    }

    console.log(`🧐 Iniciando processamento paralelo de ${limpas.length} linhas...`);
    console.time("⏱️ Tempo Total de Inferência");

    try {
        // 2. Processamento paralelo com logs individuais
        const resultados = await Promise.all(limpas.map(async (linha, index) => {
            const inicioLinha = performance.now();
            
            // Simulação de Tensores (Ajuste conforme seu modelo pedir)
            const inputIds = new BigInt64Array(128).fill(0n);
            const attentionMask = new BigInt64Array(128).fill(1n);
            
            const feeds = {
                input_ids: new ort.Tensor('int64', inputIds, [1, 128]),
                attention_mask: new ort.Tensor('int64', attentionMask, [1, 128])
            };

            try {
                const output = await session.run(feeds);
                const fimLinha = performance.now();
                
                // Log de sucesso por linha (útil para ver se alguma específica trava)
                console.log(`✅ [Linha ${index}] Processada em ${(fimLinha - inicioLinha).toFixed(2)}ms: "${linha.substring(0, 30)}..."`);
                
                return { 
                    texto: linha, 
                    raw: output, 
                    status: "sucesso" 
                };
            } catch (error) {
                console.error(`❌ Erro na linha ${index}:`, linha, error);
                return { texto: linha, error: error.message, status: "erro" };
            }
        }));

        console.timeEnd("⏱️ Tempo Total de Inferência");
        console.log("📤 Enviando resultados consolidados:", resultados);
        return resultados;

    } catch (err) {
        console.error("🚨 Erro crítico no loop de processamento:", err);
        console.timeEnd("⏱️ Tempo Total de Inferência");
        throw err;
    }
}
