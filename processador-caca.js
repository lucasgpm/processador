// MANTIDO: Mesmo nome da função utilitária original
function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')           
        .trim();
}

// MANTIDO: Mesmo nome da função do tokenizador original
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

// MANTIDO: Mesmo nome da função softmax original
function softmax(logits) {
    const maxLogit = Math.max(...logits);
    const scores = logits.map(l => Math.exp(l - maxLogit));
    const sum = scores.reduce((a, b) => a + b, 0);
    return scores.map(s => s / sum);
}

// MANTIDO: Nome idêntico para não quebrar a comunicação com o Worker original
async function processarLinhasComClassificador(linhas, session, vocab) {
    const resultados = [];
    const BATCH_SIZE = 4; 
    const TETO_MAX_LENGTH = 128;

    // 1. TRITURADOR UNIVERSAL (Nivelamento de estrutura)
    // Se o usuário colar palavras separadas por vírgulas na mesma linha, transformamos em linhas separadas.
    const linhasExpandidas = [];
    linhas.forEach(linha => {
        let texto = linha.trim();
        if (!texto) return;

        // Se a linha tem estrutura de dica (Ex: "1. Prioridade: Defina..."), isola a palavra antes do divisor
        const divisorMatch = texto.match(/[:|—]|\s-\s/);
        if (divisorMatch) {
            texto = texto.substring(0, divisorMatch.index).trim();
        }

        // Se a linha contiver palavras separadas por vírgulas ou ponto e vírgula, divide em sublinhas
        if (texto.includes(',') || texto.includes(';')) {
            const subPalavras = texto.split(/[,;]+/);
            subPalavras.forEach(sp => linhasExpandidas.push(sp.trim()));
        } else {
            linhasExpandidas.push(texto);
        }
    });

    // 2. HIGIENIZAÇÃO SENSÍVEL AO CONTEXTO
    const linhasLimpas = linhasExpandidas
        .map(l => {
            return l
                // Remove marcações de Markdown comuns de IA (como **, _, *, `)
                .replace(/[\*_`~]/g, '')
                // Remove numerações, marcadores, parênteses de quantidade (Ex: "(15)") e pontuações das pontas
                .replace(/^[\s\d.,•\-\*#•§+–—\(]+|[\s.,:;!?\-\+–—\)]+$/g, '')
                .trim();
        })
        // Filtro físico final: O termo isolado não pode ter espaços internos e deve ter tamanho válido de tabuleiro
        .filter(t => t.length >= 2 && t.length <= 16 && !t.includes(" "));

    const totalBatches = Math.ceil(linhasLimpas.length / BATCH_SIZE);

    for (let i = 0; i < linhasLimpas.length; i += BATCH_SIZE) {
        const batchAtual = linhasLimpas.slice(i, i + BATCH_SIZE);
        const atualBatchSize = batchAtual.length;

        const tokensDoBatch = batchAtual.map(t => tokenizeWordPiece(t, vocab));
        const maiorLinhaNoBatch = Math.max(...tokensDoBatch.map(t => t.length));
        const dynamicMaxLength = Math.min(maiorLinhaNoBatch + 2, TETO_MAX_LENGTH);

        const inputIdsData = new BigInt64Array(atualBatchSize * dynamicMaxLength);
        const attentionMaskData = new BigInt64Array(atualBatchSize * dynamicMaxLength);

        batchAtual.forEach((t, index) => {
            const tokenIds = tokensDoBatch[index];
            const offset = index * dynamicMaxLength;
            inputIdsData[offset] = 101n; // [CLS]
            let pos = 1;
            for (const id of tokenIds) {
                if (pos >= dynamicMaxLength - 1) break;
                inputIdsData[offset + pos] = id;
                attentionMaskData[offset + pos] = 1n;
                pos++;
            }
            inputIdsData[offset + pos] = 102n; // [SEP]
            attentionMaskData[offset] = 1n;
            attentionMaskData[offset + pos] = 1n;
        });

        try {
            // 3. INFERÊNCIA DA IA
            const output = await session.run({
                input_ids: new ort.Tensor('int64', inputIdsData, [atualBatchSize, dynamicMaxLength]),
                attention_mask: new ort.Tensor('int64', attentionMaskData, [atualBatchSize, dynamicMaxLength])
            });

            const outputData = output[session.outputNames[0]].data;
            const numLabels = outputData.length / atualBatchSize;

            batchAtual.forEach((t, index) => {
                const inicio = index * numLabels;
                const logits = Array.from(outputData.slice(inicio, inicio + numLabels));
                const scores = softmax(logits);

                const maxScore = Math.max(...scores);
                const minScore = Math.min(...scores);
                const margemCerteza = maxScore - minScore;

                // Aqui usamos a sua função original de limpeza de caracteres
                const chaveLimpa = limparChave(t);
                
                const ehLetraPura = /^\p{L}+$/u.test(chaveLimpa);
                const tokens = tokensDoBatch[index];
                const contemDesconhecido = tokens.includes(100n);

                // A IA valida se o termo purificado tem força semântica real (jogando fora conjunções ou resíduos)
                if (margemCerteza > 0.12 && ehLetraPura && !contemDesconhecido) {
                    resultados.push(chaveLimpa.toUpperCase());
                }
            });

            const progresso = Math.round(((i / BATCH_SIZE) + 1) / totalBatches * 100);
            self.postMessage({ tipo: 'PROGRESSO', valor: Math.min(progresso, 100) });

        } catch (e) {
            console.error("Erro no lote de IA do Caça-Palavras:", e);
        }
    }

    self.postMessage({ tipo: 'PROGRESSO', valor: 100 });
    return resultados;
}
