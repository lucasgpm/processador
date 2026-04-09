import { pipeline, env, AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

// 1. CONFIGURAÇÃO DO AMBIENTE
env.allowRemoteModels = true; 
env.allowLocalModels = false;
env.remoteHost = BASE_URL;       // Diz onde é a "casa" dos arquivos
env.remotePathTemplate = '{model}'; // FORÇA a lib a NÃO colocar "/resolve/main/" no link

let classificador;
let processarLinhasComClassificador;

async function reconstruirCerebroIA() {
    console.log("🧠 Baixando pedaços do modelo...");
    
    // USAR URL ABSOLUTA AQUI
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    const buffers = await Promise.all(partes.map(async (nome) => {
        const res = await fetch(path + nome); // Fetch agora tem a URL completa
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

const carregarIA = async () => {
    if (!classificador) {
        // 1. Reconstrói o arquivo .bin
        const modelBuffer = await reconstruirCerebroIA();

        console.log("🔄 Inicializando Tokenizer e Configurações...");
        
        // Caminho para a pasta onde estão os JSONs
        const modeloPath = `${BASE_URL}meu-modelo/`;

        // Carregamos o Tokenizer passando a URL da PASTA
        // O erro t.replace rolou porque a lib se confundiu com os argumentos
        const tokenizer = await AutoTokenizer.from_pretrained(modeloPath);

        // Carregamos a config manualmente para garantir que o pipeline não se perca
        const configRes = await fetch(`${modeloPath}config.json`);
        if (!configRes.ok) throw new Error("Não foi possível carregar o config.json");
        const configData = await configRes.json();

        console.log("🔄 Montando Pipeline com Buffer...");

        // O SEGREDO: Passamos o buffer, e nas opções passamos o objeto config
        classificador = await pipeline('text-classification', modelBuffer, {
            tokenizer: tokenizer,
            config: configData,
            quantized: true
        });

        console.log("🚀 IA Carregada com sucesso!");
        
        const modulo = await import(`${BASE_URL}processador.js`);
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
    return classificador;
};

// Listener de mensagens do Worker
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
