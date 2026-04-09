let tokenizer;
let session;
let carregando = false;
const BASE_URL = 'https://lucasgpm.github.io/processador/';

async function configurarTokenizer() {
    // Agora que injetamos via Firebase, o objeto 'transformers' deve existir
    const lib = self.transformers || self.Xenova;

    if (!lib) {
        console.log("Objetos disponíveis no self:", Object.keys(self));
        throw new Error("Biblioteca Transformers não encontrada no escopo global.");
    }

    if (!tokenizer) {
        console.log("📝 Carregando Tokenizer do GitHub...");
        
        // Bloqueia tentativas de ir no Hugging Face
        lib.env.allowLocalModels = true;
        lib.env.allowRemoteModels = false;
        lib.env.localModelPath = BASE_URL;

        // Tenta carregar os 4 arquivos (json e txt) da sua BASE_URL
        tokenizer = await lib.AutoTokenizer.from_pretrained(BASE_URL);
        console.log("✅ Tokenizer carregado!");
    }
}

// ... restante do código (reconstruirModelo, carregarIA, etc) ...

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
