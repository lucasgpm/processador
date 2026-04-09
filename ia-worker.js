importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');

// Agora o 'transformers' fica disponível no escopo global do Worker
// Vamos extrair o AutoTokenizer dele
const { AutoTokenizer } = self.Transformers;

let tokenizer;

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';

const BASE_URL = 'https://lucasgpm.github.io/processador/';

let session;

/**
 * Reconstrói o modelo a partir dos chunks binários
 */
async function reconstruirModelo() {
    console.log("🧠 Reconstruindo modelo para o ONNX Runtime...");
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
        return combined.buffer; // Retorna o ArrayBuffer puro
    } catch (error) {
        throw new Error("Erro na reconstrução do modelo: " + error.message);
    }
}

/**
 * Inicializa a IA e importa o processador lógico
 */
let carregando = false; 

const carregarIA = async () => {
    if (session) return;
    if (carregando) {
        // Se já estiver carregando, espera um pouco para não duplicar
        while (carregando) { await new Promise(r => setTimeout(r, 500)); }
        return;
    }

    carregando = true;
    try {
        self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/';
        const modelBuffer = await reconstruirModelo();
        
        console.log("🚀 Iniciando sessão ONNX...");
        
        // Tentativa de WebGPU (Rápido) ou WASM (Fallback)
        const options = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' };
        session = await self.ort.InferenceSession.create(modelBuffer, options);
        
        // CARREGA O TOKENIZADOR DO GITHUB (Busca tokenizer.json e tokenizer_config.json)
        tokenizer = await AutoTokenizer.from_pretrained(BASE_URL);
        
        console.log("✅ IA e Tokenizador prontos!");
    } catch (e) {
        console.error("Falha ao carregar motor:", e);
    } finally {
        carregando = false;
    }
};

/**
 * Listener de mensagens
 */
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    try {
        if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
            await carregarIA();
            if (tipo === 'PRELOAD') self.postMessage({ tipo: 'PRONTO' });
            if (tipo === 'PROCESSAR' && texto) {
                // Chama a função que agora está no escopo global do Worker
                const dados = await processarLinhasComClassificador(texto.split('\n'), session);
                self.postMessage({ tipo: 'RESULTADO', dados });
            }
        }
    } catch (err) {
        console.error("❌ Erro no motor ONNX:", err);
        self.postMessage({ tipo: 'ERRO', mensagem: err.message });
    }
};
