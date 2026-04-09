import { pipeline, env, AutoTokenizer, RawModel } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

// CONFIGURAÇÃO DE SEGURANÇA:
// Deixamos 'local' como true para ela aceitar os buffers que vamos injetar.
env.allowRemoteModels = false; 
env.allowLocalModels = true; 

let classificador;
let processarLinhasComClassificador;

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

const carregarIA = async () => {
    if (!classificador) {
        const modelBuffer = await reconstruirCerebroIA();
        const modeloPath = `${BASE_URL}meu-modelo/`;

        console.log("📂 Baixando configurações...");
        const [configRes, tokenizerRes, tokenizerConfigRes] = await Promise.all([
            fetch(`${modeloPath}config.json`),
            fetch(`${modeloPath}tokenizer.json`),
            fetch(`${modeloPath}tokenizer_config.json`)
        ]);

        const configData = await configRes.json();
        const tokenizerData = await tokenizerRes.json();
        const tokenizerConfigData = await tokenizerConfigRes.json();

        console.log("🔄 Inicializando componentes manualmente...");
        const tokenizer = new AutoTokenizer(tokenizerConfigData, tokenizerData);
        
        // Em vez de usar pipeline(), usamos o RawModel para carregar o buffer diretamente
        // Isso evita que a lib tente tratar o buffer como uma URL (erro t.replace)
        const model = await RawModel.from_pretrained('meu-modelo', {
            model_data: modelBuffer,
            config: configData,
            quantized: true
        });

        console.log("🚀 Criando Pipeline de Classificação...");
        
        // Agora criamos o pipeline passando o modelo e tokenizer JÁ INSTANCIADOS
        classificador = await pipeline('text-classification', model, {
            tokenizer: tokenizer,
            config: configData
        });

        console.log("✅ IA Carregada com sucesso!");
        
        const modulo = await import(`${BASE_URL}processador.js`);
        processarLinhasComClassificador = modulo.processarLinhasComClassificador;
    }
    return classificador;
};

self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            if (tipo === 'PRELOAD') self.postMessage({ tipo: 'PRONTO' });
            if (tipo === 'PROCESSAR' && texto) {
                if (typeof processarLinhasComClassificador === 'function') {
                    const dados = await processarLinhasComClassificador(texto.split('\n'), classificador);
                    self.postMessage({ tipo: 'RESULTADO', dados });
                } else {
                    throw new Error("Função de processamento não encontrada.");
                }
            }
        }
    } catch (err) {
        console.error("❌ Erro fatal no Worker:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
