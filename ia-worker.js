/* IA-WORKER.JS - VERSÃO PURA (SEM TRANSFORMERS)
   Usa apenas ONNX Runtime e Tokenizer Manual
*/

let session;
let vocab;
let carregando = false;
const BASE_URL = 'https://lucasgpm.github.io/processador/';

// 1. Carrega o vocabulário e os tokens especiais do seu GitHub
async function carregarTokenizerManual() {
    if (vocab) return;
    console.log("📝 Carregando Vocabulário Manual (JSON)...");
    
    const [resVocab, resTokens] = await Promise.all([
        fetch(`${BASE_URL}tokenizer.json`),
        fetch(`${BASE_URL}tokenizer_config.json`)
    ]);

    const dataVocab = await resVocab.json();
    // Dependendo da estrutura do seu tokenizer.json:
    vocab = dataVocab.model ? dataVocab.model.vocab : dataVocab;
    
    console.log("✅ Vocabulário carregado!");
}

// 2. Reconstrói o modelo a partir dos chunks (Sua lógica de chunks)
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

// 3. Inicialização da IA
const carregarIA = async () => {
    if (session && vocab) return;
    if (carregando) return;
    carregando = true;

    try {
        // Redundância de performance dentro do Worker
        if (self.ort) {
            self.ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4;
            self.ort.env.wasm.simd = true;
        }

        const [modelBuffer] = await Promise.all([
            reconstruirModelo(),
            carregarTokenizerManual()
        ]);

        console.log("🚀 Iniciando Motor ONNX Turbo...");
        session = await self.ort.InferenceSession.create(modelBuffer, {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all', // Otimiza o grafo do modelo para ser mais rápido
            enableCpuMemAccessOptimizations: true
        });

        console.log("✅ Sistema em Potência Máxima!");
    } catch (e) {
        console.error("❌ Erro no Turbo:", e);
        self.postMessage({ tipo: 'ERRO', mensagem: e.message });
    } finally {
        carregando = false;
    }
};

// 4. Listener de Mensagens
self.onmessage = async (e) => {
    const { tipo, texto } = e.data;
    
    if (tipo === 'PRELOAD' || tipo === 'PROCESSAR') {
        await carregarIA();
        
        if (tipo === 'PRELOAD') {
            self.postMessage({ tipo: 'PRONTO' });
        }
        
        if (tipo === 'PROCESSAR' && texto) {
            // Aqui você usa a sua função que tokeniza manualmente usando o objeto 'vocab'
            // enviando para o ONNX (session)
            if (typeof processarLinhasComClassificador === 'function') {
                const dados = await processarLinhasComClassificador(texto.split('\n'), session, vocab);
                self.postMessage({ tipo: 'RESULTADO', dados });
            }
        }
    }
};
