/* IA-WORKER.JS - Versão Final Estabilizada
   Suporta injeção via Blob e resolve conflitos de escopo.
*/

// As linhas abaixo são ignoradas quando injetamos via Firebase, 
// mas mantidas para compatibilidade com o GitHub.
if (typeof importScripts === 'function') {
    try {
        importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/ort.min.js');
        importScripts('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2/dist/transformers.min.js');
    } catch (e) {
        console.warn("Aviso: importScripts ignorado (carregamento via injeção local)");
    }
}

let tokenizer;
let session;
let carregando = false;
const BASE_URL = 'https://lucasgpm.github.io/processador/';

// 1. Configuração do Tokenizer com "Pesca" de objetos globais
async function configurarTokenizer() {
    let lib = self.transformers || self.Xenova;

    if (!lib) {
        console.log("🔍 Remontando objeto Transformers...");
        lib = {
            AutoTokenizer: self.AutoTokenizer || self.__webpack_exports__AutoTokenizer,
            env: self.env || self.__webpack_exports__env
        };
    }

    if (!lib.AutoTokenizer) {
        throw new Error("Biblioteca Transformers não encontrada no escopo.");
    }

    if (!tokenizer) {
        console.log("📝 Carregando Tokenizer do GitHub...");
        
        // --- AJUSTE DE PERMISSÃO AQUI ---
        if (lib.env) {
            lib.env.allowLocalModels = true;
            // Permitimos "remoto" porque o GitHub para o localhost é considerado remoto
            lib.env.allowRemoteModels = true; 
            lib.env.localModelPath = BASE_URL;
        }

        try {
            // Tentamos carregar explicitamente da URL do GitHub
            tokenizer = await lib.AutoTokenizer.from_pretrained(BASE_URL, {
                // Forçamos a lib a não procurar no cache do navegador se der erro
                use_cache: false 
            });
            console.log("✅ Tokenizer carregado!");
        } catch (err) {
            console.error("Erro específico no from_pretrained:", err);
            throw err;
        }
    }
}

// 2. Reconstrução dos chunks binários
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

// 3. Inicialização da Sessão ONNX (Modo Conservador)
const carregarIA = async () => {
    if (session && tokenizer) return;
    
    if (carregando) {
        while (carregando) { await new Promise(r => setTimeout(r, 500)); }
        return;
    }

    carregando = true;
    try {
        // --- TRAVAS DE MEMÓRIA (Evita memory access out of bounds) ---
        if (self.ort) {
            self.ort.env.wasm.numThreads = 1;
            self.ort.env.wasm.simd = false; 
            self.ort.env.wasm.proxy = false;
            self.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';
        }
        
        const modelBuffer = await reconstruirModelo();
        console.log("🚀 Iniciando sessão ONNX (WASM)...");
        
        // Usamos apenas WASM para garantir que rode em qualquer navegador sem Cross-Origin-Isolation
        session = await self.ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });

        await configurarTokenizer();
        
        console.log("✅ Motor e Tokenizer prontos!");
    } catch (e) {
        console.error("❌ Falha crítica no carregamento:", e);
        self.postMessage({ tipo: 'ERRO', mensagem: e.message });
    } finally {
        carregando = false;
    }
};

// 4. Listener de Mensagens Principal
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    
    if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
        await carregarIA();
        
        if (tipo === 'PRELOAD') {
            self.postMessage({ tipo: 'PRONTO' });
        }
        
        if (tipo === 'PROCESSAR' && texto) {
            // Assume que 'processarLinhasComClassificador' foi injetado via processador.js
            if (typeof processarLinhasComClassificador === 'function') {
                const dados = await processarLinhasComClassificador(texto.split('\n'), session);
                self.postMessage({ tipo: 'RESULTADO', dados });
            } else {
                self.postMessage({ tipo: 'ERRO', mensagem: "Função de processamento não encontrada." });
            }
        }
    }
};
