// 1. Carrega a biblioteca de um CDN alternativo que registra melhor o objeto global
importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');

// 2. Variáveis globais
let tokenizer;
let session;
let carregando = false;
const BASE_URL = 'https://lucasgpm.github.io/processador/';

async function configurarTokenizer() {
    // Tentamos todas as variações de nome que o Xenova assume em Workers
    const lib = self.transformers || self.Transformers || (self.Xenova && self.Xenova.transformers);
    
    if (!lib) {
        console.log("Variáveis disponíveis no self:", Object.keys(self));
        throw new Error("Biblioteca Transformers não detectada. Verifique o importScripts.");
    }

    if (!tokenizer) {
        console.log("📝 Lendo arquivos do Tokenizer (vocab, special_tokens, json)...");
        
        // Configurações para ler do seu GitHub
        lib.env.allowLocalModels = false;
        lib.env.allowRemoteModels = true;
        
        // Aqui ele vai buscar automaticamente: 
        // tokenizer.json, tokenizer_config.json, vocab.txt e special_tokens_map.json
        tokenizer = await lib.AutoTokenizer.from_pretrained(BASE_URL);
        console.log("✅ Tokenizer configurado com sucesso!");
    }
}

// 4. Reconstrói o modelo a partir dos chunks
async function reconstruirModelo() {
    console.log("🧠 Reconstruindo modelo binário...");
    const path = `${BASE_URL}onnx/chunks/`; 
    const partes = ['model_part_0.bin', 'model_part_1.bin', 'model_part_2.bin'];
    
    try {
        const buffers = await Promise.all(partes.map(async (nome) => {
            const res = await fetch(path + nome);
            if (!res.ok) throw new Error(`Falha ao carregar ${nome}`);
            return res.arrayBuffer();
        }));

        const totalLength = buffers.reduce((acc, b) => acc + b.byteLength, 0);
        const combined = new Uint8Array(totalLength);
        let offset = 0;
        for (const b of buffers) {
            combined.set(new Uint8Array(b), offset);
            offset += b.byteLength;
        }
        return combined.buffer;
    } catch (error) {
        throw new Error("Erro na reconstrução: " + error.message);
    }
}

// 5. Inicialização da IA (Sessão + Tokenizer)
const carregarIA = async () => {
    if (session && tokenizer) return;
    if (carregando) {
        while (carregando) { await new Promise(r => setTimeout(r, 500)); }
        return;
    }

    carregando = true;
    try {
        self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        
        // Carrega Modelo
        const modelBuffer = await reconstruirModelo();
        console.log("🚀 Iniciando sessão ONNX (WebGPU/WASM)...");
        
        session = await self.ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['webgpu', 'wasm'],
            graphOptimizationLevel: 'all'
        });

        // Carrega Tokenizer
        await configurarTokenizer();
        
        console.log("✅ Motor e Tokenizer prontos para uso!");
    } catch (e) {
        console.error("❌ Falha crítica no carregamento:", e);
        self.postMessage({ tipo: 'ERRO', mensagem: e.message });
    } finally {
        carregando = false;
    }
};

// 6. Processamento das linhas
async function processarLinhasComClassificador(linhas, session) {
    const limpas = linhas.map(l => l.trim()).filter(l => l.length > 5);
    const resultados = [];

    for (const linha of limpas) {
        try {
            // Gera os tensores reais a partir do texto
            const { input_ids, attention_mask } = await tokenizer(linha, {
                padding: true,
                truncation: true,
                maxLength: 128
            });

            const output = await session.run({ input_ids, attention_mask });
            resultados.push({ texto: linha, raw: output });
        } catch (e) {
            console.warn("Ignorando linha por erro de processamento:", linha);
        }
    }
    return resultados;
}

// 7. Listener de Mensagens
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
        await carregarIA();
        if (tipo === 'PRELOAD') self.postMessage({ tipo: 'PRONTO' });
        if (tipo === 'PROCESSAR' && texto) {
            const dados = await processarLinhasComClassificador(texto.split('\n'), session);
            self.postMessage({ tipo: 'RESULTADO', dados });
        }
    }
};
