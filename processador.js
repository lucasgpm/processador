import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// CONFIGURAÇÃO CRUCIAL: Desativa a busca local e aponta para o CDN remoto
env.allowLocalModels = false;
env.remoteHost = 'https://huggingface.co/';
env.remotePathComponent = 'models/';

let classificadorInstancia = null;

async function obterClassificador() {
    if (!classificadorInstancia) {
        // Agora ele vai direto na fonte sem dar erro 404 no seu console
        classificadorInstancia = await pipeline(
            'text-classification', 
            'Xenova/distilbert-base-multilingual-cased-sentiments-student'
        );
    }
    return classificadorInstancia;
}

export async function processarCapturaInteligente(entradaSuja) {
    const classifier = await obterClassificador();
    const linhas = entradaSuja.split('\n');
    const resultadoLimpo = [];

    for (const linha of linhas) {
        const t = linha.trim();
        if (t.length < 5 || !/[:\-\t=|—]/.test(t)) continue;

        const analysis = await classifier(t);
        const scoreConfianca = analysis[0].score;

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

        if (scoreConfianca > 0.3 && palavrasNaChave > 0 && palavrasNaChave <= 3 && valor.length > 5) {
            resultadoLimpo.push({
                palavra: chaveLimpa.toUpperCase(),
                dica: valor
            });
        }
    }
    return resultadoLimpo;
}
