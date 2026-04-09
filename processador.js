// Apenas a lógica de limpeza pura (sem transformers aqui)
export function limparChave(chaveBruta) {
    return chaveBruta
        .replace(/\(.*?\)/g, '')           
        .replace(/->|=>|^\d+[\s.]*/g, '')  
        .replace(/["'«»]/g, '')            
        .replace(/[-—_]/g, ' ')            
        .trim();
}

// Essa função agora será usada PELO WORKER
export async function processarLinhasComClassificador(linhas, classifier) {
    const resultadoLimpo = [];
    for (const linha of linhas) {
        const t = linha.trim();
        if (t.length < 5 || !/[:\-\t=|—]/.test(t)) continue;

        const analysis = await classifier(t);
        const scoreConfianca = analysis[0].score;

        const divisorMatch = t.match(/[:\t=|—]| - /); 
        if (!divisorMatch) continue;

        const chaveBruta = t.substring(0, divisorMatch.index).trim();
        const valor = t.substring(divisorMatch.index + divisorMatch[0].length).trim();
        
        let chaveLimpa = limparChave(chaveBruta);
        const palavrasNaChave = chaveLimpa.split(/\s+/).filter(p => p.length > 0).length;

        if (scoreConfianca > 0.3 && palavrasNaChave > 0 && palavrasNaChave <= 3 && valor.length > 5) {
            resultadoLimpo.push({
                palavra: chaveLimpa.toUpperCase(),
                dica: valor
            });
        }
    }
    return resultadoLimpo;
}
