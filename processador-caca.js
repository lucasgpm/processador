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
    
    // 1. Pré-filtro universal por estrutura física
    // Mantém linhas que contenham apenas uma palavra (sem espaços ou pontos internos)
    // O limite de caracteres considera que ideogramas (como chinês/japonês) formam palavras com menos caracteres (ex: 2)
    const linhasCandidatas = lines = linhas
        .map(l => l.trim())
        .filter(t => t.length >= 2 && t.length <= 16 && !t.includes(" ") && !t.includes("."));

    const totalLinhas = linhasCandidatas.length;

    linhasCandidatas.forEach((linha, index) => {
        const chaveLimpa = limparChave(linha);
        
        // 2. Validação por Vocabulário Multilíngue do BERT (104 idiomas nativos)
        const palavraLower = chaveLimpa.toLowerCase();
        const tokens = tokenizeWordPiece(palavraLower, vocab);
        
        // Se a palavra gera um token desconhecido [UNK] (ID 100), significa que é lixo ou ruído gráfico
        const contemDesconhecido = tokens.includes(100n);

        // 3. Validação Ortográfica Global (Unicode)
        // \p{L} aceita QUALQUER letra de QUALQUER alfabeto do planeta (Árabe, Cirílico, Ideogramas, Kanji, etc.)
        // a flag 'u' no final ativa o suporte completo a caracteres Unicode do JavaScript
        const ehPalavraValida = /^\p{L}+$/u.test(chaveLimpa);

        // Se a palavra existe no dicionário global e é um conjunto puro de letras/ideogramas, ela entra
        if (!contemDesconhecido && ehPalavraValida) {
            // Converte para UpperCase apenas se o alfabeto suportar caixa alta (evita bugs em escritas asiáticas)
            resultados.push(chaveLimpa.toUpperCase());
        } else {
            console.log(`🗑️ [FILTRO] Linha descartada por não ser uma palavra válida globalmente: "${linha}"`);
        }

        // Atualiza o progresso da barra para cada palavra processada
        if (totalLinhas > 0) {
            const progresso = Math.round(((index + 1) / totalLinhas) * 100);
            self.postMessage({ tipo: 'PROGRESSO', valor: progresso });
        }
    });

    // Garante o fechamento da barra de progresso enviando 100% ao final
    self.postMessage({ tipo: 'PROGRESSO', valor: 100 });

    console.log("🎯 [SUCESSO] Vetor limpo e internacional gerado:", resultados);
    return resultados;
}
