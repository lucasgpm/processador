function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

// --- FUNÇÃO 1: WORDPIECE (A "Inteligência" do BERT) ---
function tokenizeWordPiece(text, vocab) {
    const words = text.toLowerCase()
        .replace(/([.,!?])/g, ' $1 ')
        .trim()
        .split(/\s+/);
    
    const resultIds = [];
    for (let word of words) {
        if (vocab[word]) {
            resultIds.push(BigInt(vocab[word]));
            continue;
        }

        let start = 0;
        let found = false;
        while (start < word.length) {
            let end = word.length;
            let curSubstrId = null;
            while (start < end) {
                let substr = (start === 0) ? word.substring(start, end) : "##" + word.substring(start, end);
                if (vocab[substr]) {
                    curSubstrId = BigInt(vocab[substr]);
                    break;
                }
                end--;
            }
            if (curSubstrId === null) {
                resultIds.push(BigInt(vocab['[UNK]'] || 100));
                break;
            }
            resultIds.push(curSubstrId);
            start = end;
        }
    }
    return resultIds;
}

// --- FUNÇÃO 2: SOFTMAX (Converte números da IA em Score de 0 a 1) ---
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// --- FUNÇÃO 3: O PROCESSADOR COMPLETO ---
async function processarLinhasComClassificador(linhas, session, vocab) {
    const resultados = [];
    const maxLength = 128;

    // Alocação única de Tensores para Performance
    const inputIdsData = new BigInt64Array(maxLength);
    const attentionMaskData = new BigInt64Array(maxLength);
    const tensorIds = new ort.Tensor('int64', inputIdsData, [1, maxLength]);
    const tensorMask = new ort.Tensor('int64', attentionMaskData, [1, maxLength]);

    for (const linha of linhas) {
        const t = linha.trim();
        
        // FILTRO DE ESTRUTURA (Seu código original)
        if (t.length < 5 || !/[:\-\t=|—]/.test(t)) continue;

        try {
            // Reinicia buffers
            inputIdsData.fill(0n);
            attentionMaskData.fill(0n);

            const tokenIds = tokenizeWordPiece(t, vocab);
            
            inputIdsData[0] = 101n; // [CLS]
            let pos = 1;
            for (const id of tokenIds) {
                if (pos >= maxLength - 1) break;
                inputIdsData[pos] = id;
                attentionMaskData[pos] = 1n;
                pos++;
            }
            inputIdsData[pos] = 102n; // [SEP]
            attentionMaskData[0] = 1n;
            attentionMaskData[pos] = 1n;

            // Executa IA
            const output = await session.run({
                input_ids: tensorIds,
                attention_mask: tensorMask
            });

            // Converte saída para score de confiança (0 a 1)
            const rawLogits = Array.from(output[session.outputNames[0]].data);
            const scores = softmax(rawLogits);
            const scoreConfianca = Math.max(...scores); 

            // --- SUA LÓGICA DE EXTRAÇÃO (IDÊNTICA À ORIGINAL) ---
            const divisorMatch = t.match(/[:\t=|—]| - /); 
            if (!divisorMatch) continue;

            const indiceDivisor = divisorMatch.index;
            const chaveBruta = t.substring(0, indiceDivisor).trim();
            const valor = t.substring(indiceDivisor + divisorMatch[0].length).trim();

            let chaveLimpa = chaveBruta
                .replace(/\(.*?\)/g, '')           
                .replace(/->|=>|^\d+[\s.]*/g, '')  
                .replace(/["'«»]/g, '')            
                .replace(/[-—_]/g, ' ')            
                .trim();

            const palavrasNaChave = chaveLimpa.split(/\s+/).filter(p => p.length > 0).length;

            // FILTRO DE ELITE
            if (scoreConfianca > 0.3 && palavrasNaChave > 0 && palavrasNaChave <= 3 && valor.length > 5) {
                resultados.push({
                    palavra: chaveLimpa.toUpperCase(),
                    dica: valor
                });
            }
        } catch (e) {
            console.warn("Erro ao processar linha:", t, e);
        }
    }
    return resultados;
}
