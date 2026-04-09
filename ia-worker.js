import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

// URL base do seu GitHub Pages
const BASE_URL = 'https://lucasgpm.github.io/processador/';

// Configurações globais simplificadas para evitar duplicação de URL
env.allowRemoteModels = true;
env.allowLocalModels = false;

let classificador;
let processarLinhasComClassificador;

/**
 * Reconstrói o arquivo do modelo a partir dos pedaços binários
 */
async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do modelo...");
    
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome);
        if (!res.ok) throw new Error(`Erro ao baixar parte: ${nome}`);
        return res.arrayBuffer();
    }));

    const totalLength = buffers.reduce((acc, b) => acc + b.byteLength, 0);
    const combined = new Uint8Array(totalLength);
    let offset = 0;
    for (const b of buffers) {
        combined.set(new Uint8Array(b), offset);
        offset += b.byteLength;
    }
    return combined;
}

/**
 * Carrega o Tokenizer, a Configuração e o Pipeline
 */
const carregarIA = async () => {
    if (!classificador) {
        // 1. Obtém o buffer do modelo reconstruído
        const modelBuffer = await reconstruirCerebroIA();

        console.log("🔄 Inicializando Tokenizer e Configurações...");
        
        const modeloPath = `${BASE_URL}meu-modelo/`;

        // 2. Carrega o Tokenizer usando a URL da pasta (evita erro t.replace)
        const tokenizer = await AutoTokenizer.from_pretrained(modeloPath);

        // 3. Carrega o config.json manualmente para evitar erro de model type null
        const configRes = await fetch(`${modeloPath}config.json`);
        if (!configRes.ok) throw new Error("Não foi possível carregar o config.json");
        const configData = await configRes.json();

        console.log("🔄 Montando Pipeline com Buffer...");

        // 4. Inicializa o pipeline injetando o buffer e o objeto de configuração
        classificador = await pipeline('text-classification', modelBuffer, {
            tokenizer: tokenizer,
            config: configData,
            quantized: true
        });

        console.log("🚀 IA Carregada com sucesso!");
        
        // 5. Importa o script de processamento lógico
        const modulo = await import(`${BASE_URL}processador.js`);
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
    return classificador;
};

/**
 * Listener de mensagens do Worker
 */
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            
            if (tipo === 'PRELOAD') {
                self.postMessage({ tipo: 'PRONTO' });
            }
            
            if (tipo === 'PROCESSAR' && texto) {
                if (typeof processarLinhasComClassificador === 'function') {
                    const dados = await processarLinhasComClassificador(texto.split('\n'), classificador);
                    self.postMessage({ tipo: 'RESULTADO', dados });
                } else {
                    throw new Error("Função de processamento não encontrada no processador.js");
                }
            }
        }
    } catch (err) {
        console.error("❌ Erro no Worker:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
